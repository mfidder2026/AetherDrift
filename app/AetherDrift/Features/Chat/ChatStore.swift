import Foundation
import Observation
import os

/// One message on screen. A class so streaming updates re-render only this message,
/// not the whole list (start-prompt-v2 §8.14).
@MainActor
@Observable
final class ChatItem: Identifiable {
    enum State: Equatable {
        case complete
        case waiting  // sent, no token yet (engine starting or model loading)
        case streaming
        case failed(APIError)
    }

    let id = UUID()
    var dbID: Int64?
    let role: String
    private(set) var text: String
    private(set) var blocks: [MarkdownBlock]
    /// Reasoning from <think>…</think>, shown collapsed; nil when the model did not think.
    private(set) var thinking: String?
    private(set) var isThinking = false
    var meta: MessageMeta?
    var state: State
    @ObservationIgnored private var renderer = StreamingMarkdown()

    init(role: String, text: String, dbID: Int64? = nil, meta: MessageMeta? = nil, state: State = .complete) {
        self.role = role
        self.text = text
        self.dbID = dbID
        self.meta = meta
        self.state = state
        self.blocks = []
        if role == "assistant" { render() }
    }

    func append(_ chunk: String) {
        text += chunk
        render()
    }

    private func render() {
        let parts = ReasoningSplit(text)
        thinking = parts.thinking
        isThinking = parts.isOpen
        blocks = renderer.update(parts.answer)
    }
}

@MainActor
@Observable
final class ChatStore {
    private(set) var conversations: [Conversation] = []
    private(set) var selectedID: Int64?
    private(set) var items: [ChatItem] = []
    private(set) var isGenerating = false
    var draft = ""

    @ObservationIgnored private let db: AppDatabase
    @ObservationIgnored private let engine: () async -> EngineClient
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var generationID: String?
    @ObservationIgnored private var client: EngineClient?
    @ObservationIgnored private let log = Logger(subsystem: "app.fremen.AetherDrift", category: "chat")
    /// Model info for the footer under each answer.
    @ObservationIgnored var currentModel: () -> (id: String?, mode: String?) = { (nil, nil) }

    init(db: AppDatabase, engine: @escaping () async -> EngineClient) {
        self.db = db
        self.engine = engine
        reloadConversations()
        if let first = conversations.first?.id { select(first) }
    }

    // MARK: Conversations

    func reloadConversations() {
        conversations = (try? db.conversations()) ?? []
    }

    func select(_ id: Int64?) {
        guard id != selectedID || items.isEmpty else { return }
        stop()
        selectedID = id
        guard let id else {
            items = []
            return
        }
        let stored = (try? db.messages(in: id)) ?? []
        items = stored.map { ChatItem(role: $0.role, text: $0.content, dbID: $0.id, meta: $0.decodedMeta) }
    }

    func newConversation() {
        select(nil)
        draft = ""
    }

    func delete(_ id: Int64) {
        try? db.deleteConversation(id)
        if id == selectedID { select(nil) }
        reloadConversations()
    }

    nonisolated static func title(for text: String) -> String {
        let words = text.split(whereSeparator: \.isWhitespace).prefix(8).joined(separator: " ")
        return words.count > 60 ? String(words.prefix(57)) + "…" : (words.isEmpty ? "Nieuw gesprek" : words)
    }

    // MARK: Sending

    func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isGenerating else { return }
        draft = ""
        do {
            if selectedID == nil {
                let c = try db.createConversation(title: Self.title(for: text))
                selectedID = c.id
            }
            guard let cid = selectedID else { return }
            let saved = try db.save(StoredMessage(id: nil, conversationId: cid, role: "user", content: text,
                                                  createdAt: .now, meta: nil))
            items.append(ChatItem(role: "user", text: text, dbID: saved.id))
            reloadConversations()
        } catch {
            log.error("saving message failed: \(error.localizedDescription, privacy: .public)")
        }
        generate()
    }

    func regenerate() {
        guard !isGenerating, let last = items.last, last.role == "assistant" else { return }
        if let dbID = last.dbID, let cid = selectedID { try? db.deleteMessages(in: cid, from: dbID) }
        items.removeLast()
        generate()
    }

    /// ↑ in an empty composer: take the last question back for editing.
    func editLast() {
        guard !isGenerating, let idx = items.lastIndex(where: { $0.role == "user" }) else { return }
        let item = items[idx]
        if let dbID = item.dbID, let cid = selectedID { try? db.deleteMessages(in: cid, from: dbID) }
        items.removeSubrange(idx...)
        draft = item.text
    }

    func stop() {
        guard isGenerating else { return }
        if let id = generationID, let client {
            Task { try? await client.cancelGeneration(id) }
        } else {
            task?.cancel()
        }
    }

    private func generate() {
        let history = items.compactMap { item -> ChatRequestMessage? in
            if case .failed = item.state { return nil }
            return ChatRequestMessage(role: item.role, content: item.text)
        }
        let answer = ChatItem(role: "assistant", text: "", state: .waiting)
        items.append(answer)
        isGenerating = true
        generationID = nil
        let conversationID = selectedID

        task = Task { [weak self] in
            guard let self else { return }
            let client = await engine()
            self.client = client
            let request = ChatRequest(model: "default", messages: history, stream: true,
                                      streamOptions: .init(includeUsage: true), temperature: nil, maxTokens: nil)
            let buffer = TokenBuffer(item: answer)
            var meta = MessageMeta(model: currentModel().id)
            meta.mode = currentModel().mode
            do {
                for try await event in client.chat(request) {
                    switch event {
                    case .started(let id):
                        generationID = id
                    case .token(let t):
                        if answer.state == .waiting { answer.state = .streaming }
                        buffer.add(t)
                    case .finished(let reason, let timings):
                        meta.finishReason = reason
                        meta.decodeTps = timings?.decodeTps
                    case .usage(let usage):
                        meta.completionTokens = usage.completionTokens
                    }
                }
                buffer.flush()
                answer.state = .complete
            } catch let error as APIError {
                buffer.flush()
                answer.state = answer.text.isEmpty ? .failed(error) : .complete
                meta.errorCode = error.code
            } catch {
                buffer.flush()  // cancelled before the engine assigned an id
                answer.state = .complete
                meta.finishReason = "cancelled"
            }
            answer.meta = meta
            if answer.text.isEmpty, answer.state == .complete {
                items.removeAll { $0 === answer }  // stopped before the first token
            }
            if let cid = conversationID, !answer.text.isEmpty {
                let json = (try? JSONEncoder().encode(meta)).map { String(decoding: $0, as: UTF8.self) }
                let saved = try? db.save(StoredMessage(id: nil, conversationId: cid, role: "assistant",
                                                       content: answer.text, createdAt: .now, meta: json))
                answer.dbID = saved?.id
                reloadConversations()
            }
            isGenerating = false
            generationID = nil
        }
    }
}

/// Coalesces streamed tokens: at most one UI update per frame, and pending text is still
/// shown when the next token is slow to arrive.
@MainActor
private final class TokenBuffer {
    static let frame: Duration = .milliseconds(16)
    private let item: ChatItem
    private var pending = ""
    private var scheduled: Task<Void, Never>?

    init(item: ChatItem) { self.item = item }

    func add(_ text: String) {
        pending += text
        guard scheduled == nil else { return }
        scheduled = Task { [weak self] in
            try? await Task.sleep(for: Self.frame)
            self?.flush()
        }
    }

    func flush() {
        scheduled?.cancel()
        scheduled = nil
        guard !pending.isEmpty else { return }
        item.append(pending)
        pending = ""
    }
}

/// Splits a reply into reasoning (`<think>…</think>`) and the answer.
struct ReasoningSplit: Equatable {
    let thinking: String?
    let answer: String
    let isOpen: Bool  // still thinking

    init(_ text: String) {
        let trimmed = text.drop { $0.isWhitespace }
        guard trimmed.hasPrefix("<think>") else {
            (thinking, answer, isOpen) = (nil, text, false)
            return
        }
        let body = trimmed.dropFirst("<think>".count)
        if let end = body.range(of: "</think>") {
            thinking = body[..<end.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
            answer = body[end.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines)
            isOpen = false
        } else {
            thinking = body.trimmingCharacters(in: .whitespacesAndNewlines)
            answer = ""
            isOpen = true
        }
    }
}
