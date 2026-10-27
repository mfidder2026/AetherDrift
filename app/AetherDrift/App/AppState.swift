import Foundation
import Observation
import os

struct Toast: Identifiable, Equatable {
    enum Action: Equatable {
        case openModels
        case retryFast
    }

    let id = UUID()
    let message: String
    var hint: String?
    var action: Action?
    var actionLabel: String?
}

struct DownloadState: Equatable {
    var bytes: Int64
    var total: Int64
    var fraction: Double? { total > 0 ? min(1, Double(bytes) / Double(total)) : nil }
}

/// App-wide state: engine connection, models, downloads, settings and notifications.
@MainActor
@Observable
final class AppState {
    let supervisor: EngineSupervisor
    let live = LiveStats()
    let log = LogStore()
    let chat: ChatStore

    private(set) var system: SystemInfo?
    private(set) var models: ModelsResponse?
    private(set) var settings: EngineSettings? {
        didSet {
            // The app owns the settings; the engine gets them again after every (re)start.
            updateSharingActivity()
            guard var copy = settings else { return }
            copy.requiresReload = nil
            if let data = try? API.encoder.encode(copy) {
                UserDefaults.standard.set(data, forKey: "engineSettings")
            }
        }
    }
    private(set) var downloads: [String: DownloadState] = [:]
    private(set) var toasts: [Toast] = []
    var showModelPalette = false
    var showConnect = false
    @ObservationIgnored var quickChat: QuickChatController?
    @ObservationIgnored private var hotKey: GlobalHotKey?

    /// nil = follow sharing: a Mac that serves other devices keeps running in the menu bar.
    var keepRunningSetting: Bool? = UserDefaults.standard.object(forKey: "keepRunning") as? Bool {
        didSet { UserDefaults.standard.set(keepRunningSetting, forKey: "keepRunning") }
    }
    var keepRunning: Bool { keepRunningSetting ?? isSharing }

    var quickChatEnabled = UserDefaults.standard.object(forKey: "quickChatHotKey") as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(quickChatEnabled, forKey: "quickChatHotKey")
            updateHotKey()
        }
    }
    /// The network API key (from the Keychain); nil until the engine has been connected.
    private(set) var apiKey: String?
    var showLog = UserDefaults.standard.bool(forKey: "showLog") {
        didSet { UserDefaults.standard.set(showLog, forKey: "showLog") }
    }
    var showInspector = UserDefaults.standard.object(forKey: "showInspector") as? Bool ?? true {
        didSet { UserDefaults.standard.set(showInspector, forKey: "showInspector") }
    }
    var onboardingDone = UserDefaults.standard.bool(forKey: "onboardingDone") {
        didSet { UserDefaults.standard.set(onboardingDone, forKey: "onboardingDone") }
    }
    private(set) var lastModelID = UserDefaults.standard.string(forKey: "lastModelID") {
        didSet { UserDefaults.standard.set(lastModelID, forKey: "lastModelID") }
    }
    /// Loads this model as soon as its download completes.
    var loadAfterDownload: String?
    /// Switches to this model once its download completes (a starter model runs meanwhile).
    var switchAfterDownload: String?

    @ObservationIgnored private var eventsTask: Task<Void, Never>?
    @ObservationIgnored private var lastSeq: Int?
    @ObservationIgnored private let logger = Logger(subsystem: "app.fremen.AetherDrift", category: "app")

    init(supervisor: EngineSupervisor = EngineSupervisor(), db: AppDatabase) {
        self.supervisor = supervisor
        self.chat = ChatStore(db: db) { await supervisor.readyClient() }
        chat.currentModel = { [weak self] in
            (self?.live.snapshot?.model.id, self?.live.snapshot?.model.mode)
        }
    }

    // MARK: Lifecycle

    func start() {
        quickChat = QuickChatController(state: self)
        updateHotKey()
        supervisor.start()
        Task { await connectLoop() }
        #if DEBUG
        Task { await autopilotChat() }
        Task { await autopilotShow() }
        #endif
    }

    private func connectLoop() async {
        var seen = -1
        while !Task.isCancelled {
            let client = await supervisor.readyClient()
            if supervisor.generation != seen {
                seen = supervisor.generation
                await connect(client)
            }
            try? await Task.sleep(for: .milliseconds(500))
        }
    }

    #if DEBUG
    /// Development only: AETHER_AUTOPILOT_SHOW=connect|quickchat opens that UI once the engine is up.
    private func autopilotShow() async {
        guard let what = ProcessInfo.processInfo.environment["AETHER_AUTOPILOT_SHOW"] else { return }
        _ = await supervisor.readyClient()
        try? await Task.sleep(for: .seconds(2))
        switch what {
        case "connect": showConnect = true
        case "connect-share":
            await setShare(true)
            showConnect = true
        case "quickchat": quickChat?.show()
        default: break
        }
    }

    /// Autopilot (development only): send a chat message once a model is ready.
    private func autopilotChat() async {
        guard let text = ProcessInfo.processInfo.environment["AETHER_AUTOPILOT_CHAT"] else { return }
        while live.snapshot?.model.state != "ready" || !onboardingDone {
            try? await Task.sleep(for: .milliseconds(200))
        }
        try? await Task.sleep(for: .seconds(1))
        chat.newConversation()
        chat.draft = text
        chat.send()
    }
    #endif

    private func connect(_ client: EngineClient) async {
        lastSeq = nil  // a new engine process numbers its events from the start again
        live.connect(client)
        eventsTask?.cancel()
        eventsTask = Task { [weak self] in
            while !Task.isCancelled {
                do {
                    for try await event in client.eventStream(since: self?.lastSeq) {
                        self?.handle(event)
                    }
                } catch {}
                try? await Task.sleep(for: .seconds(1))
            }
        }
        system = try? await client.system()
        // The key goes first, so a saved "share on" never opens the network without it.
        let key = apiKey ?? APIKeyStore.key()
        apiKey = key
        try? await client.setAPIKey(key)
        if let data = UserDefaults.standard.data(forKey: "engineSettings"),
           var saved = try? API.decoder.decode(EngineSettings.self, from: data) {
            // Before 0.1 builds had an automatic reserve, the engine always reported 16 GB and
            // the app saved it. Turn that inherited value back into "automatic" once.
            if !UserDefaults.standard.bool(forKey: "osReserveMigrated") {
                if saved.osReserveGb == 16 { saved.osReserveGb = nil }
                UserDefaults.standard.set(true, forKey: "osReserveMigrated")
            }
            if let applied = try? await client.replaceSettings(saved) {
                settings = applied
            } else {
                settings = try? await client.settings()
            }
        } else {
            settings = try? await client.settings()
        }
        await refreshModels()
        if onboardingDone, let last = lastModelID,
           (try? await client.stats())?.model.state == "none" {
            await load(last)
        }
    }

    // MARK: Models

    var modelState: StatsSnapshot.Model? { live.snapshot?.model }

    func refreshModels() async {
        guard let client = supervisor.client else { return }
        do {
            models = try await client.models()
        } catch {
            logger.error("models refresh failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    func load(_ id: String) async {
        guard let client = supervisor.client else { return }
        do {
            try await client.loadModel(id)
            lastModelID = id
        } catch let error as APIError {
            show(error)
        } catch {}
    }

    func download(_ repo: String, thenLoad: Bool) async {
        guard let client = supervisor.client else { return }
        downloads[repo] = DownloadState(bytes: 0, total: 0)
        if thenLoad { loadAfterDownload = repo }
        do {
            try await client.download(repo)
        } catch let error as APIError {
            downloads[repo] = nil
            show(error)
        } catch {}
    }

    func cancelDownload(_ repo: String) async {
        try? await supervisor.client?.cancelDownload(repo)
        downloads[repo] = nil
        if loadAfterDownload == repo { loadAfterDownload = nil }
    }

    func refreshSystemAndModels() async {
        guard let client = supervisor.client else { return }
        system = try? await client.system()
        await refreshModels()
    }

    /// nil = automatic: a quarter of RAM, 4–16 GB.
    func updateOSReserve(_ gb: Double?) async {
        guard let client = supervisor.client else { return }
        do {
            settings = try await client.updateSettings(["os_reserve_gb": gb ?? NSNull()])
            await refreshModels()
        } catch let error as APIError {
            show(error)
        } catch {}
    }

    /// nil = automatic. Applies on the next model load; the current model is reloaded.
    func updateWeightsBudget(_ gb: Double?) async {
        guard let client = supervisor.client else { return }
        do {
            let s = try await client.updateSettings(["weights_budget_gb": gb ?? NSNull()])
            settings = s
            if s.requiresReload == true, let id = modelState?.id {
                try await client.unloadModel()
                await load(id)
            }
            await refreshModels()
        } catch let error as APIError {
            show(error)
        } catch {}
    }

    private func updateHotKey() {
        hotKey?.unregister()
        hotKey = nil
        guard quickChatEnabled, quickChat != nil else { return }
        hotKey = GlobalHotKey { [weak self] in self?.quickChat?.toggle() }
        if hotKey == nil { logger.error("could not register ⌥Space") }
    }

    // MARK: Sharing

    @ObservationIgnored private var sharingActivity: NSObjectProtocol?

    /// A shared Mac must stay reachable: no idle sleep while sharing is on (the display may
    /// still sleep). The engine itself also stays awake while it generates.
    private func updateSharingActivity() {
        if isSharing, sharingActivity == nil {
            sharingActivity = ProcessInfo.processInfo.beginActivity(
                options: [.idleSystemSleepDisabled],
                reason: "AetherDrift deelt een model op het netwerk")
        } else if !isSharing, let activity = sharingActivity {
            ProcessInfo.processInfo.endActivity(activity)
            sharingActivity = nil
        }
    }

    var isSharing: Bool { settings?.share ?? false }

    /// Sharing never reloads the model (start-prompt-v2 §8.8).
    func setShare(_ on: Bool) async {
        guard let client = supervisor.client else { return }
        do {
            settings = try await client.updateSettings(["share": on])
        } catch let error as APIError {
            show(error)
        } catch {}
    }

    func regenerateAPIKey() async {
        guard let key = APIKeyStore.regenerate() else {
            toast(Toast(message: String(localized: "De nieuwe sleutel kon niet in de sleutelhanger bewaard worden."),
                        hint: String(localized: "De huidige sleutel blijft geldig.")))
            return
        }
        apiKey = key
        try? await supervisor.client?.setAPIKey(key)
    }

    func setMode(_ mode: String) async {
        guard let client = supervisor.client else { return }
        do {
            let s = try await client.updateSettings(["mode": mode])
            settings = s
            if s.requiresReload == true, let id = modelState?.id {
                try await client.unloadModel()
                await load(id)
            }
            await refreshModels()
        } catch let error as APIError {
            show(error)
        } catch {}
    }

    // MARK: Events

    private func handle(_ e: EngineEvent) {
        lastSeq = e.seq
        log.add(e)
        switch e.type {
        case "model_loaded":
            if let m = e.model { lastModelID = m }
            Task { await refreshModels() }
        case "model_unloaded":
            Task { await refreshModels() }
        case "download_progress":
            if let repo = e.repo { downloads[repo] = DownloadState(bytes: e.bytes ?? 0, total: e.total ?? 0) }
        case "download_done":
            guard let repo = e.repo else { return }
            downloads[repo] = nil
            Task {
                await refreshModels()
                if loadAfterDownload == repo {
                    loadAfterDownload = nil
                    await load(repo)
                } else if switchAfterDownload == repo {
                    switchAfterDownload = nil
                    await load(repo)  // queued behind a running answer, never interrupts it
                    let name = repo.split(separator: "/").last.map(String.init) ?? repo
                    toast(Toast(message: String(localized: "\(name) is gedownload en wordt geladen.")))
                }
            }
        case "download_failed":
            if let repo = e.repo { downloads[repo] = nil }
            show(APIError(code: e.code ?? "download_interrupted", message: e.message ?? "", hint: nil))
        case "error":
            show(APIError(code: e.code ?? "internal", message: e.message ?? "Onbekende fout.", hint: e.hint))
        case "pressure":
            if e.level == "warning" {
                toast(Toast(message: String(localized: "Geheugen krap."), hint: e.action))
            } else if e.level == "critical" {
                toast(Toast(message: String(localized: "Je Mac had het geheugen nodig."), hint: e.action,
                            action: .retryFast, actionLabel: String(localized: "Modus Snel")))
            }
        default:
            break
        }
    }

    // MARK: Notifications

    func show(_ error: APIError) {
        let f = FriendlyError(error)
        toast(Toast(message: f.message, hint: f.hint, action: f.action, actionLabel: f.actionLabel))
    }

    func toast(_ t: Toast) {
        toasts.append(t)
        Task {
            try? await Task.sleep(for: .seconds(6))
            dismiss(t.id)
        }
    }

    func dismiss(_ id: UUID) {
        toasts.removeAll { $0.id == id }
    }

    func perform(_ action: Toast.Action) {
        switch action {
        case .openModels: showModelPalette = true
        case .retryFast: Task { await setMode("snel") }
        }
    }
}

/// Error codes → plain language with an action (start-prompt-v2 §8.11).
struct FriendlyError {
    let message: String
    let hint: String?
    let action: Toast.Action?
    let actionLabel: String?

    init(_ e: APIError) {
        switch e.code {
        case "model_too_large":
            (message, hint, action, actionLabel) = (e.message, e.hint, .openModels, String(localized: "Kies een kleiner model"))
        case "model_not_loaded":
            (message, hint, action, actionLabel) = (String(localized: "Er is nog geen model geladen."), nil, .openModels, String(localized: "Kies een model"))
        case "memory":
            (message, hint, action, actionLabel) = (String(localized: "Je Mac had het geheugen nodig, dus het antwoord is ingekort."),
                                                    nil, .retryFast, String(localized: "Modus Snel"))
        case "download_interrupted":
            (message, hint, action, actionLabel) = (String(localized: "Download onderbroken."),
                                                    String(localized: "Probeer het opnieuw zodra je weer online bent."), nil, nil)
        case "engine_unreachable":
            (message, hint, action, actionLabel) = (String(localized: "De engine is even niet bereikbaar."),
                                                    String(localized: "Hij wordt automatisch herstart. Je gesprek is bewaard."), nil, nil)
        case "context_too_long":
            (message, hint, action, actionLabel) = (e.message, e.hint, nil, nil)
        default:
            (message, hint, action, actionLabel) = (e.message.isEmpty ? String(localized: "Er ging iets mis.") : e.message, e.hint, nil, nil)
        }
    }
}
