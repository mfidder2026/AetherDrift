import ServiceManagement
import SwiftUI

/// Menu bar status and actions (start-prompt-v2 §8.9).
struct MenuBarContent: View {
    @Environment(AppState.self) private var state
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        let model = state.modelState
        if let id = model?.id {
            Text("Model: \(id.split(separator: "/").last.map(String.init) ?? id)")
        } else {
            Text("Geen model geladen")
        }
        if let tps = state.live.snapshot?.generation.decodeTps, state.chat.isGenerating {
            Text(Format.tps(tps))
        }
        Text(state.isSharing ? "Delen: aan" : "Delen: uit")
        Divider()
        Button("Open AetherDrift") { openMain() }
        Button("Nieuw gesprek") {
            state.chat.newConversation()
            openMain()
        }
        Button("Snelchat") { state.quickChat?.show() }
            .keyboardShortcut(.space, modifiers: .option)
        Button("Stop generatie") { state.chat.stop() }
            .disabled(!state.chat.isGenerating)
        Toggle("Delen op netwerk", isOn: Binding(
            get: { state.isSharing },
            set: { on in Task { await state.setShare(on) } }
        ))
        .disabled(!state.supervisor.isReady)
        Divider()
        Button("Stop AetherDrift") { NSApp.terminate(nil) }
            .keyboardShortcut("q")
    }

    private func openMain() {
        openWindow(id: "main")
        NSApp.activate()
    }
}

struct MenuBarLabel: View {
    @Environment(AppState.self) private var state

    var body: some View {
        Image(systemName: state.chat.isGenerating ? "sparkles" : "sparkle")
            .accessibilityLabel(state.chat.isGenerating ? "AetherDrift genereert" : "AetherDrift")
    }
}

/// Starting at login through SMAppService (macOS 13+).
enum LoginItem {
    static var isEnabled: Bool { SMAppService.mainApp.status == .enabled }

    static func set(_ on: Bool) throws {
        if on {
            try SMAppService.mainApp.register()
        } else {
            try SMAppService.mainApp.unregister()
        }
    }
}
