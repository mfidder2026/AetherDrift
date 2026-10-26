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
        supervisor.start()
        Task { await connectLoop() }
        #if DEBUG
        Task { await autopilotChat() }
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
        if let data = UserDefaults.standard.data(forKey: "engineSettings"),
           let saved = try? API.decoder.decode(EngineSettings.self, from: data) {
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

    func updateOSReserve(_ gb: Double) async {
        guard let client = supervisor.client else { return }
        do {
            settings = try await client.updateSettings(["os_reserve_gb": gb])
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
                }
            }
        case "download_failed":
            if let repo = e.repo { downloads[repo] = nil }
            show(APIError(code: e.code ?? "download_interrupted", message: e.message ?? "", hint: nil))
        case "error":
            show(APIError(code: e.code ?? "internal", message: e.message ?? "Onbekende fout.", hint: e.hint))
        case "pressure":
            if e.level == "warning" {
                toast(Toast(message: "Geheugen krap.", hint: e.action))
            } else if e.level == "critical" {
                toast(Toast(message: "Je Mac had het geheugen nodig.", hint: e.action,
                            action: .retryFast, actionLabel: "Modus Snel"))
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
            (message, hint, action, actionLabel) = (e.message, e.hint, .openModels, "Kies een kleiner model")
        case "model_not_loaded":
            (message, hint, action, actionLabel) = ("Er is nog geen model geladen.", nil, .openModels, "Kies een model")
        case "memory":
            (message, hint, action, actionLabel) = ("Je Mac had het geheugen nodig, dus het antwoord is ingekort.",
                                                    nil, .retryFast, "Modus Snel")
        case "download_interrupted":
            (message, hint, action, actionLabel) = ("Download onderbroken.",
                                                    "Probeer het opnieuw zodra je weer online bent.", nil, nil)
        case "engine_unreachable":
            (message, hint, action, actionLabel) = ("De engine is even niet bereikbaar.",
                                                    "Hij wordt automatisch herstart. Je gesprek is bewaard.", nil, nil)
        case "context_too_long":
            (message, hint, action, actionLabel) = (e.message, e.hint, nil, nil)
        default:
            (message, hint, action, actionLabel) = (e.message.isEmpty ? "Er ging iets mis." : e.message, e.hint, nil, nil)
        }
    }
}
