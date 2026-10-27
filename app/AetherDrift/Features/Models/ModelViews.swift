import AppKit
import SwiftUI

/// Toolbar chip: status dot + model name; opens the model palette (⌘K).
struct ModelChip: View {
    @Environment(AppState.self) private var state

    var body: some View {
        Button {
            state.showModelPalette = true
        } label: {
            HStack(spacing: 6) {
                Circle().fill(dotColor).frame(width: 8, height: 8)
                Text(title).lineLimit(1)
                if let progress = state.modelState?.progress, state.modelState?.state == "loading" {
                    Text(progress.formatted(.percent.precision(.fractionLength(0))))
                        .foregroundStyle(.secondary).monospacedDigit()
                }
                Image(systemName: "chevron.down").font(.caption2).foregroundStyle(.secondary)
            }
        }
        .help("Kies model (⌘K)")
        .accessibilityLabel("Model: \(title)")
    }

    private var title: String {
        guard state.supervisor.isReady, let m = state.modelState else { return String(localized: "Engine start…") }
        guard let id = m.id else { return String(localized: "Geen model") }
        let name = id.split(separator: "/").last.map(String.init) ?? id
        return m.mode.map { "\(name) · \(Format.modeName($0))" } ?? name
    }

    private var dotColor: Color {
        guard state.supervisor.isReady else { return .gray }
        return switch state.modelState?.state {
        case "ready": .green
        case "loading": .orange
        case "error": .red
        default: .gray
        }
    }
}

struct ModePicker: View {
    @Environment(AppState.self) private var state

    var body: some View {
        Menu(title) {
            ForEach(["snel", "balans", "lange_context"], id: \.self) { mode in
                Button {
                    Task { await state.setMode(mode) }
                } label: {
                    if state.settings?.mode == mode {
                        Label(Format.presetName(mode), systemImage: "checkmark")
                    } else {
                        Text(Format.presetName(mode))
                    }
                }
            }
            Divider()
            Toggle("Turbo voor Drift-MoE", isOn: Binding(
                get: { state.settings?.turbo ?? false },
                set: { on in Task { await state.setTurbo(on) } }
            ))
            .help("Ongeveer twee keer zo snel bij modellen als GLM die via Drift draaien, maar iets minder nauwkeurig")
        }
        .fixedSize()
        .help("Snel: hoogste snelheid · Balans: normaal · Lange context: lange documenten")
        .disabled(state.settings == nil)
    }

    private var title: String {
        let preset = Format.presetName(state.settings?.mode ?? "balans")
        return state.settings?.turbo == true ? String(localized: "\(preset) · Turbo") : preset
    }
}

/// ⌘K: search, switch, download.
struct ModelPalette: View {
    @Environment(AppState.self) private var state
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Zoek een model", text: $query).textFieldStyle(.plain).font(.title3)
            }
            .padding(14)
            Divider()
            List {
                Section("Op deze Mac") {
                    if local.isEmpty {
                        Text("Nog geen modellen. Download er hieronder één.").foregroundStyle(.secondary)
                    }
                    ForEach(local) { model in
                        LocalModelRow(model: model) {
                            Task { await state.load(model.id) }
                            dismiss()
                        }
                    }
                }
                Section("Downloaden") {
                    ForEach(catalog) { entry in CatalogRow(entry: entry) }
                }
            }
            .listStyle(.inset)
            Divider()
            HStack {
                Button("Map kiezen…") { pickFolder() }
                Text("of sleep een modelmap hierheen").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Sluiten") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            .padding(12)
        }
        .frame(width: 600, height: 520)
        .task { await state.refreshModels() }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            Task { await state.load(url.path) }
            dismiss()
            return true
        }
    }

    private var local: [LocalModel] {
        (state.models?.local ?? []).filter { query.isEmpty || $0.id.localizedCaseInsensitiveContains(query) }
    }

    private var catalog: [CatalogEntry] {
        (state.models?.catalog ?? []).filter {
            !$0.downloaded && (query.isEmpty || $0.name.localizedCaseInsensitiveContains(query))
        }
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.message = "Kies een map met een MLX-model (config.json + .safetensors)"
        if panel.runModal() == .OK, let url = panel.url {
            Task { await state.load(url.path) }
            dismiss()
        }
    }
}

struct FitBadge: View {
    let fit: Fit?

    var body: some View {
        if let fit {
            Text(label(fit))
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(color(fit).opacity(0.15), in: Capsule())
                .foregroundStyle(color(fit))
        }
    }

    private func label(_ f: Fit) -> String {
        switch f.mode {
        case "resident": "Past in RAM"
        case "too_large": "Te groot"
        default: f.loadable ? "Drift" : "Drift · binnenkort"
        }
    }

    private func color(_ f: Fit) -> Color {
        switch f.mode {
        case "resident": .green
        case "too_large": .red
        default: .purple
        }
    }
}

private func speedText(_ fit: Fit?) -> String? {
    guard let tps = fit?.expectedTps else { return nil }
    return String(localized: "± \(Format.tps(tps)) (geschat)")
}

private struct LocalModelRow: View {
    let model: LocalModel
    let onLoad: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.id.split(separator: "/").last.map(String.init) ?? model.id)
                HStack(spacing: 6) {
                    Text(Format.bytes(model.sizeBytes))
                    if let s = speedText(model.fit) { Text(s) }
                }
                .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            FitBadge(fit: model.fit)
            if model.loaded {
                Text("Geladen").font(.caption).foregroundStyle(.secondary)
            } else {
                Button("Laden", action: onLoad).disabled(model.fit?.loadable == false)
            }
        }
        .padding(.vertical, 2)
    }
}

private struct CatalogRow: View {
    @Environment(AppState.self) private var state
    let entry: CatalogEntry

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.name)
                Text(entry.description).font(.caption).foregroundStyle(.secondary)
                HStack(spacing: 6) {
                    Text(Format.bytes(entry.sizeBytes))
                    if let s = speedText(entry.fit) { Text(s) }
                }
                .font(.caption).foregroundStyle(.tertiary)
            }
            Spacer()
            FitBadge(fit: entry.fit)
            if let dl = state.downloads[entry.repo] {
                DownloadProgress(state: dl) { Task { await state.cancelDownload(entry.repo) } }
            } else {
                Button("Download") { Task { await state.download(entry.repo, thenLoad: true) } }
                    .disabled(!entry.fit.loadable)
                    .help(entry.fit.loadable ? "Downloaden en laden"
                                             : "Dit model past niet op deze Mac, ook niet via Drift.")
            }
        }
        .padding(.vertical, 2)
    }
}

struct DownloadProgress: View {
    let state: DownloadState
    let onCancel: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            if let f = state.fraction {
                ProgressView(value: f).frame(width: 90)
                Text(f.formatted(.percent.precision(.fractionLength(0)))).font(.caption).monospacedDigit()
            } else {
                ProgressView().controlSize(.small)
            }
            Button(action: onCancel) { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.borderless)
                .accessibilityLabel("Annuleer download")
        }
    }
}
