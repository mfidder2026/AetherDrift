import AppKit
import SwiftUI

struct ConversationList: View {
    @Environment(AppState.self) private var state

    var body: some View {
        let chat = state.chat
        List(selection: Binding(get: { chat.selectedID }, set: { chat.select($0) })) {
            ForEach(chat.conversations) { c in
                Text(c.title)
                    .lineLimit(1)
                    .tag(c.id)
                    .contextMenu {
                        Button("Verwijder", role: .destructive) { if let id = c.id { chat.delete(id) } }
                    }
            }
        }
        .safeAreaInset(edge: .top) {
            Button {
                chat.newConversation()
            } label: {
                Label("Nieuw gesprek", systemImage: "square.and.pencil")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.borderless)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .help("Nieuw gesprek (⌘N)")
        }
        .overlay {
            if chat.conversations.isEmpty {
                Text("Nog geen gesprekken").foregroundStyle(.tertiary)
            }
        }
    }
}

struct ChatView: View {
    @Environment(AppState.self) private var state

    var body: some View {
        let chat = state.chat
        VStack(spacing: 0) {
            if chat.items.isEmpty {
                EmptyChat()
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 18) {
                            ForEach(chat.items) { item in
                                MessageView(item: item, isLast: item === chat.items.last)
                                    .id(item.id)
                            }
                            Color.clear.frame(height: 1).id("bottom")
                        }
                        .padding(.horizontal, 24)
                        .padding(.vertical, 18)
                        .frame(maxWidth: 860)
                        .frame(maxWidth: .infinity)
                    }
                    .defaultScrollAnchor(.bottom)
                    .onChange(of: chat.items.count) {
                        withAnimation(.snappy) { proxy.scrollTo("bottom", anchor: .bottom) }
                    }
                }
            }
            Composer()
        }
    }
}

private struct EmptyChat: View {
    @Environment(AppState.self) private var state

    var body: some View {
        VStack(spacing: 10) {
            Spacer()
            Image(systemName: "sparkles").font(.system(size: 34)).foregroundStyle(.tertiary)
            Text("Waar kan ik mee helpen?").font(.title3)
            if let model = state.modelState, model.state == "ready", let id = model.id {
                Text(id.split(separator: "/").last.map(String.init) ?? id)
                    .font(.callout).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }
}

struct MessageView: View {
    @Environment(AppState.self) private var state
    let item: ChatItem
    let isLast: Bool
    @State private var hovering = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(item.role == "user" ? "Jij" : "AetherDrift")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            content
            footer
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onHover { hovering = $0 }
        .accessibilityElement(children: .contain)
    }

    @ViewBuilder private var content: some View {
        switch item.state {
        case .waiting:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(waitingText).foregroundStyle(.secondary)
            }
            LiveDriftStrip()
        case .failed(let error):
            let f = FriendlyError(error)
            VStack(alignment: .leading, spacing: 6) {
                Label(f.message, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                if let hint = f.hint { Text(hint).font(.callout).foregroundStyle(.secondary) }
                HStack {
                    if let action = f.action, let label = f.actionLabel {
                        Button(label) { state.perform(action) }
                    }
                    if isLast { Button("Opnieuw proberen") { state.chat.regenerate() } }
                }
                .controlSize(.small)
            }
        case .complete, .streaming:
            if item.role == "user" {
                Text(item.text)
                    .textSelection(.enabled)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(.quinary, in: RoundedRectangle(cornerRadius: 10))
            } else {
                if let thinking = item.thinking {
                    ThinkingView(text: thinking, active: item.isThinking && item.state == .streaming)
                }
                MarkdownView(blocks: item.blocks)
                if item.state == .complete, item.isThinking {
                    NoAnswerNotice(isLast: isLast)
                }
                if item.state == .streaming {
                    Rectangle().frame(width: 7, height: 15).foregroundStyle(.secondary)
                        .accessibilityHidden(true)
                    LiveDriftStrip()
                }
            }
        }
    }

    private var waitingText: String {
        switch state.supervisor.status {
        case .ready:
            if state.modelState?.state == "loading" { return String(localized: "Wacht op model…") }
            // Via Drift the prompt takes a while: show how far it has come.
            if let s = state.live.snapshot, s.model.mode?.hasPrefix("drift") == true,
               let layer = s.drift.currentLayer, let total = s.drift.layersTotal {
                return String(localized: "Vraag wordt gelezen · laag \(layer + 1) van \(total)")
            }
            return String(localized: "Bezig…")
        default:
            return String(localized: "Wacht op de engine…")
        }
    }

    @ViewBuilder private var footer: some View {
        if item.role == "assistant", item.state == .complete {
            HStack(spacing: 12) {
                if let meta = item.meta {
                    Text(footerText(meta)).font(.caption).foregroundStyle(.tertiary).monospacedDigit()
                }
                if hovering || isLast {
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(item.text, forType: .string)
                    } label: { Image(systemName: "doc.on.doc") }
                        .help("Kopieer antwoord")
                        .accessibilityLabel("Kopieer antwoord")
                    if isLast {
                        Button { state.chat.regenerate() } label: { Image(systemName: "arrow.clockwise") }
                            .help("Opnieuw genereren")
                            .accessibilityLabel("Opnieuw genereren")
                    }
                }
            }
            .buttonStyle(.borderless)
            .font(.caption)
        }
    }

    private func footerText(_ m: MessageMeta) -> String {
        var parts: [String] = []
        if let tps = m.decodeTps { parts.append(Format.tps(tps)) }
        if let n = m.completionTokens { parts.append(String(localized: "\(n) tokens")) }
        if let mode = m.mode { parts.append(Format.modeName(mode)) }
        if m.finishReason == "cancelled" { parts.append(String(localized: "gestopt")) }
        if m.finishReason == "memory" { parts.append(String(localized: "ingekort (geheugen)")) }
        if m.finishReason == "length" { parts.append(String(localized: "afgebroken (maximale lengte)")) }
        return parts.joined(separator: " · ")
    }
}

struct Composer: View {
    @Environment(AppState.self) private var state
    @FocusState private var focused: Bool

    var body: some View {
        @Bindable var chat = state.chat
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Stel een vraag…", text: $chat.draft, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...10)
                .focused($focused)
                .onKeyPress(.return, phases: .down) { press in
                    if press.modifiers.contains(.shift) || press.modifiers.contains(.option) {
                        chat.draft += "\n"
                    } else {
                        chat.send()
                    }
                    return .handled
                }
                .onKeyPress(.upArrow) {
                    guard chat.draft.isEmpty else { return .ignored }
                    chat.editLast()
                    return .handled
                }
                .accessibilityLabel("Bericht")
            ThinkingMenu()
            if chat.isGenerating {
                Button { chat.stop() } label: {
                    Image(systemName: "stop.circle.fill").font(.title2)
                }
                .buttonStyle(.borderless)
                .help("Stop (⌘.)")
                .accessibilityLabel("Stop generatie")
            } else {
                Button { chat.send() } label: {
                    Image(systemName: "arrow.up.circle.fill").font(.title2)
                }
                .buttonStyle(.borderless)
                .disabled(chat.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .help("Versturen (Enter)")
                .accessibilityLabel("Versturen")
            }
        }
        .padding(10)
        .background(.background, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary))
        .padding(.horizontal, 24)
        .padding(.bottom, 16)
        .padding(.top, 6)
        .frame(maxWidth: 860)
        .onAppear { focused = true }
    }
}

/// Quick access to the thinking level; the same setting as in Instellingen.
private struct ThinkingMenu: View {
    @Environment(AppState.self) private var state

    var body: some View {
        let level = state.settings?.thinking ?? "auto"
        Menu {
            Picker("Nadenken", selection: Binding(
                get: { level },
                set: { new in Task { await state.setThinking(new) } }
            )) {
                ForEach(Format.thinkingLevels, id: \.self) { Text(Format.thinkingName($0)).tag($0) }
            }
            .pickerStyle(.inline)
        } label: {
            Image(systemName: level == "off" ? "lightbulb.slash" : "lightbulb")
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .disabled(state.settings == nil)
        .help("Nadenken: \(Format.thinkingName(level))")
        .accessibilityLabel("Nadenken: \(Format.thinkingName(level))")
    }
}

/// The reply ended inside the reasoning block: there is no answer to show.
private struct NoAnswerNotice: View {
    @Environment(AppState.self) private var state
    let isLast: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Geen antwoord: het model was nog aan het nadenken toen het stopte.",
                  systemImage: "lightbulb")
                .foregroundStyle(.orange)
            if isLast, state.settings?.thinking != "off" {
                Button("Zet nadenken uit en probeer opnieuw") {
                    Task {
                        await state.setThinking("off")
                        state.chat.regenerate()
                    }
                }
                .controlSize(.small)
            }
        }
    }
}

/// Collapsed reasoning ("Denkproces"); open by choice, not by default.
private struct ThinkingView: View {
    let text: String
    let active: Bool
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            Text(text)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)
        } label: {
            HStack(spacing: 6) {
                if active { ProgressView().controlSize(.mini) }
                Text(active ? "Denkt na…" : "Denkproces")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}
