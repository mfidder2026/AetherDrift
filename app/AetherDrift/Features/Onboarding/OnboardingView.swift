import SwiftUI

/// First launch in at most three steps (start-prompt-v2 §8.2): check → choose → download & start.
struct OnboardingView: View {
    @Environment(AppState.self) private var state
    @State private var step = 0
    @State private var chosen: String?
    @State private var ssdTested = false

    var body: some View {
        VStack(spacing: 0) {
            Group {
                switch step {
                case 0: welcome
                case 1: choose
                default: starting
                }
            }
            .frame(maxWidth: 620)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(32)
            HStack(spacing: 6) {
                ForEach(0..<3) { i in
                    Circle().fill(i == step ? Color.accentColor : Color.secondary.opacity(0.3)).frame(width: 7, height: 7)
                }
            }
            .padding(.bottom, 20)
            .accessibilityLabel("Stap \(step + 1) van 3")
        }
    }

    // MARK: Step 1: welcome & check

    private var welcome: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Welkom bij AetherDrift").font(.largeTitle.bold())
            Text("Grote AI-modellen, gewoon op je eigen Mac. We controleren even je Mac.")
                .font(.title3).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 10) {
                checkRow("Engine", value: engineText, done: state.supervisor.isReady)
                checkRow("Chip", value: state.system?.chip, done: state.system != nil)
                checkRow("Geheugen", value: state.system.map { Format.gb($0.ramBytes) }, done: state.system != nil)
                checkRow("SSD-snelheid", value: state.system?.ssdReadBytesPerS.map {
                    Format.number($0 / 1e9, digits: 1) + " GB/s"
                }, done: state.system?.ssdReadBytesPerS != nil)
            }
            .padding(16)
            .background(.quinary, in: RoundedRectangle(cornerRadius: 12))
            HStack {
                Spacer()
                Button("Volgende") { step = 1 }
                    .keyboardShortcut(.defaultAction)
                    .controlSize(.large)
                    .disabled(!state.supervisor.isReady || state.models == nil)
            }
        }
        .task(id: state.supervisor.generation) { await runSSDTest() }
        #if DEBUG
        .task(id: state.models?.local.count) {
            // Autopilot (development only): walk the same path as the buttons.
            guard let model = ProcessInfo.processInfo.environment["AETHER_AUTOPILOT_ONBOARD"],
                  state.supervisor.isReady, state.models != nil, step == 0 else { return }
            try? await Task.sleep(for: .seconds(2))
            step = 1
            chosen = model
            try? await Task.sleep(for: .seconds(2))
            step = 2
            await start(model)
        }
        #endif
    }

    private var engineText: String {
        switch state.supervisor.status {
        case .ready: "Klaar"
        case .installing: "Installeren (eenmalig, ± 1 minuut)…"
        case .failed(let m): "Fout: \(m.prefix(120))"
        default: "Starten…"
        }
    }

    private func checkRow(_ label: String, value: String?, done: Bool) -> some View {
        HStack {
            Image(systemName: done ? "checkmark.circle.fill" : "circle.dotted")
                .foregroundStyle(done ? .green : .secondary)
            Text(label)
            Spacer()
            if let value { Text(value).foregroundStyle(.secondary) } else { ProgressView().controlSize(.small) }
        }
    }

    private func runSSDTest() async {
        guard !ssdTested, state.system?.ssdReadBytesPerS == nil else { return }
        let client = await state.supervisor.readyClient()
        ssdTested = true
        if (try? await client.ssdTest()) != nil {
            await state.refreshSystemAndModels()
        }
    }

    // MARK: Step 2: choose a model

    private var choose: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Kies een model").font(.largeTitle.bold())
            Text("Aanbevolen voor deze Mac. Je kunt later altijd wisselen (⌘K).")
                .foregroundStyle(.secondary)
            VStack(spacing: 10) {
                ForEach(state.models?.local ?? []) { m in
                    optionRow(id: m.id, title: m.id.split(separator: "/").last.map(String.init) ?? m.id,
                              detail: "Al op deze Mac · \(Format.bytes(m.sizeBytes))", fit: m.fit)
                }
                ForEach((state.models?.catalog ?? []).filter { !$0.downloaded }) { c in
                    optionRow(id: c.repo, title: c.name, detail: "\(c.description) · \(Format.bytes(c.sizeBytes))",
                              fit: c.fit, recommended: c.role == "starter")
                }
            }
            HStack {
                Button("Terug") { step = 0 }
                Spacer()
                Button("Starten") {
                    guard let chosen else { return }
                    step = 2
                    Task { await start(chosen) }
                }
                .keyboardShortcut(.defaultAction)
                .controlSize(.large)
                .disabled(chosen == nil)
            }
        }
        .onAppear {
            chosen = chosen ?? state.models?.local.first(where: { $0.fit?.loadable == true })?.id
                ?? state.models?.catalog.first(where: { $0.role == "starter" })?.repo
        }
    }

    private func optionRow(id: String, title: String, detail: String, fit: Fit?, recommended: Bool = false) -> some View {
        let loadable = fit?.loadable ?? false
        return Button {
            if loadable { chosen = id }
        } label: {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: chosen == id ? "largecircle.fill.circle" : "circle")
                    .foregroundStyle(loadable ? Color.accentColor : .secondary)
                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text(title).font(.headline)
                        if recommended { Text("Snel starten").font(.caption).foregroundStyle(.secondary) }
                    }
                    Text(detail).font(.callout).foregroundStyle(.secondary)
                    if let tps = fit?.expectedTps {
                        Text("Verwachte snelheid: ± \(Format.tps(tps)) (geschat)").font(.caption).foregroundStyle(.tertiary)
                    }
                    if !loadable {
                        Text("Vraagt de Drift Engine, die nog in ontwikkeling is.").font(.caption).foregroundStyle(.orange)
                    }
                }
                Spacer()
                FitBadge(fit: fit)
            }
            .padding(12)
            .contentShape(Rectangle())
            .background(chosen == id ? Color.accentColor.opacity(0.08) : .clear, in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.quaternary))
        }
        .buttonStyle(.plain)
        .opacity(loadable ? 1 : 0.6)
    }

    // MARK: Step 3: download & start

    private var starting: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Bijna klaar").font(.largeTitle.bold())
            if let chosen, let dl = state.downloads[chosen] {
                Text("Model downloaden…").font(.title3)
                if let f = dl.fraction {
                    ProgressView(value: f)
                    Text("\(Format.bytes(dl.bytes)) van \(Format.bytes(dl.total))").font(.caption).foregroundStyle(.secondary)
                } else {
                    ProgressView()
                }
            } else if state.modelState?.state == "loading" {
                Text("Model laden…").font(.title3)
                ProgressView(value: state.modelState?.progress ?? 0)
            } else if state.modelState?.state == "error" {
                Text("Het model kon niet geladen worden.").foregroundStyle(.orange)
                Button("Ander model kiezen") { step = 1 }
            } else {
                ProgressView()
            }
            HStack {
                Button("Terug") {
                    if let chosen { Task { await state.cancelDownload(chosen) } }
                    step = 1
                }
                Spacer()
            }
        }
        .onChange(of: state.modelState?.state) { _, new in
            if new == "ready" { state.onboardingDone = true }
        }
    }

    private func start(_ id: String) async {
        let isLocal = state.models?.local.contains { $0.id == id } ?? false
        if isLocal {
            await state.load(id)
        } else {
            await state.download(id, thenLoad: true)
        }
    }
}
