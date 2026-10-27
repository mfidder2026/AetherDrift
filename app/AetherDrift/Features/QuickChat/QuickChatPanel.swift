import AppKit
import SwiftUI

/// Floating quick-chat window à la Spotlight (start-prompt-v2 §8.9). Enter sends, Esc closes.
/// A quick question becomes a normal conversation, so it is also in the main window's history.
@MainActor
final class QuickChatController {
    private var panel: NSPanel?
    private let state: AppState

    init(state: AppState) {
        self.state = state
    }

    func toggle() {
        if let panel, panel.isVisible {
            panel.orderOut(nil)
        } else {
            show()
        }
    }

    func show() {
        let panel = self.panel ?? makePanel()
        self.panel = panel
        if let screen = NSScreen.main {
            let f = screen.visibleFrame
            panel.setFrameTopLeftPoint(NSPoint(x: f.midX - panel.frame.width / 2, y: f.maxY - f.height * 0.2))
        }
        NSApp.activate()
        panel.makeKeyAndOrderFront(nil)
    }

    func close() {
        panel?.orderOut(nil)
    }

    private func makePanel() -> NSPanel {
        let panel = QuickPanel(
            contentRect: NSRect(x: 0, y: 0, width: 640, height: 60),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered, defer: false
        )
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.isMovableByWindowBackground = true
        panel.level = .floating
        panel.hidesOnDeactivate = false
        panel.isReleasedWhenClosed = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        let root = QuickChatView(close: { [weak self] in self?.close() })
            .environment(state)
        // The panel follows the content: one line while asking, taller once an answer streams.
        let controller = NSHostingController(rootView: root)
        controller.sizingOptions = [.preferredContentSize]
        panel.contentViewController = controller
        return panel
    }
}

private final class QuickPanel: NSPanel {
    override var canBecomeKey: Bool { true }

    override func cancelOperation(_ sender: Any?) {
        orderOut(nil)
    }
}

struct QuickChatView: View {
    @Environment(AppState.self) private var state
    @Environment(\.openWindow) private var openWindow
    @State private var text = ""
    @State private var asked = false
    @FocusState private var focused: Bool
    let close: () -> Void

    var body: some View {
        let chat = state.chat
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Image(systemName: "sparkle").foregroundStyle(.secondary)
                TextField("Vraag iets aan \(state.modelState?.id?.split(separator: "/").last.map(String.init) ?? "AetherDrift")…",
                          text: $text)
                    .textFieldStyle(.plain)
                    .font(.title3)
                    .focused($focused)
                    .onSubmit(send)
                if chat.isGenerating {
                    Button("Stop") { chat.stop() }
                        .keyboardShortcut(".", modifiers: .command)
                }
            }
            if asked, let answer = chat.items.last, answer.role == "assistant" {
                Divider()
                ScrollView {
                    MessageView(item: answer, isLast: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 360)
                HStack {
                    Spacer()
                    Button("Open in AetherDrift") {
                        openWindow(id: "main")
                        NSApp.activate()
                        close()
                    }
                }
            }
        }
        .padding(16)
        .frame(width: 640)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .onAppear {
            focused = true
            if !chat.isGenerating { asked = false }
        }
        .onExitCommand(perform: close)
    }

    private func send() {
        let q = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty, !state.chat.isGenerating else { return }
        state.chat.newConversation()
        state.chat.draft = q
        state.chat.send()
        text = ""
        asked = true
    }
}
