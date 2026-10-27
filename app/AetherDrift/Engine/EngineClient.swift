import Foundation

/// HTTP/SSE client for the engine. The app uses exactly the public API (§2 principle 6).
final class EngineClient: Sendable {
    let baseURL: URL
    private let session: URLSession

    init(port: Int, host: String = "127.0.0.1") {
        baseURL = URL(string: "http://\(host):\(port)")!
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 3600  // streams can be long
        config.httpMaximumConnectionsPerHost = 8
        session = URLSession(configuration: config)
    }

    // MARK: Plain requests

    func health(timeout: TimeInterval = 2) async throws -> Health {
        var req = request("/health")
        req.timeoutInterval = timeout
        return try await send(req)
    }

    func system() async throws -> SystemInfo { try await send(request("/aether/v1/system")) }

    func ssdTest() async throws -> SystemInfo {
        try await send(request("/aether/v1/system/ssd-test", method: "POST"))
    }

    func models() async throws -> ModelsResponse { try await send(request("/aether/v1/models")) }

    func stats() async throws -> StatsSnapshot { try await send(request("/aether/v1/stats")) }

    func loadModel(_ id: String) async throws {
        try await sendIgnoringBody(request("/aether/v1/models/load", method: "POST", json: ["id": id]))
    }

    func unloadModel() async throws {
        try await sendIgnoringBody(request("/aether/v1/models/unload", method: "POST"))
    }

    func download(_ repo: String) async throws {
        try await sendIgnoringBody(request("/aether/v1/models/download", method: "POST", json: ["repo": repo]))
    }

    func cancelDownload(_ repo: String) async throws {
        try await sendIgnoringBody(request("/aether/v1/models/download/\(repo)", method: "DELETE"))
    }

    func cancelGeneration(_ id: String) async throws {
        try await sendIgnoringBody(request("/aether/v1/generations/\(id)/cancel", method: "POST"))
    }

    func setAPIKey(_ key: String) async throws {
        try await sendIgnoringBody(request("/aether/v1/api-key", method: "PUT", json: ["key": key]))
    }

    func clients() async throws -> [NetworkClient] {
        let r: NetworkClients = try await send(request("/aether/v1/clients"))
        return r.clients
    }

    func settings() async throws -> EngineSettings { try await send(request("/aether/v1/config")) }

    func replaceSettings(_ settings: EngineSettings) async throws -> EngineSettings {
        var req = request("/aether/v1/config", method: "PUT")
        var copy = settings
        copy.requiresReload = nil
        req.httpBody = try API.encoder.encode(copy)
        return try await send(req)
    }

    func updateSettings(_ changes: [String: any Sendable]) async throws -> EngineSettings {
        var req = request("/aether/v1/config", method: "PUT")
        req.httpBody = try JSONSerialization.data(withJSONObject: changes)
        return try await send(req)
    }

    // MARK: Streams

    func chat(_ body: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error> {
        var req = request("/v1/chat/completions", method: "POST")
        req.httpBody = try? API.encoder.encode(body)
        return sse(req) { payload, yield in
            guard payload != "[DONE]" else { return }
            let chunk = try API.decoder.decode(ChatChunk.self, from: Data(payload.utf8))
            if let error = chunk.error { throw error }
            if let id = chunk.aether?.generationId { yield(.started(generationID: id)) }
            if let usage = chunk.usage { yield(.usage(usage)) }
            for choice in chunk.choices ?? [] {
                if let text = choice.delta.content, !text.isEmpty { yield(.token(text)) }
                if let reason = choice.finishReason {
                    yield(.finished(reason: reason, timings: chunk.aether?.timings))
                }
            }
        }
    }

    func statsStream() -> AsyncThrowingStream<StatsSnapshot, Error> {
        sse(request("/aether/v1/stats/stream")) { payload, yield in
            yield(try API.decoder.decode(StatsSnapshot.self, from: Data(payload.utf8)))
        }
    }

    func eventStream(since: Int?) -> AsyncThrowingStream<EngineEvent, Error> {
        let path = since.map { "/aether/v1/events/stream?since=\($0)" } ?? "/aether/v1/events/stream"
        return sse(request(path)) { payload, yield in
            // Skip events this app version cannot read instead of dropping the stream.
            if let event = try? API.decoder.decode(EngineEvent.self, from: Data(payload.utf8)) {
                yield(event)
            }
        }
    }

    // MARK: Plumbing

    private func request(_ path: String, method: String = "GET", json: [String: String]? = nil) -> URLRequest {
        var req = URLRequest(url: URL(string: path, relativeTo: baseURL)!)
        req.httpMethod = method
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let json { req.httpBody = try? JSONSerialization.data(withJSONObject: json) }
        return req
    }

    private func send<T: Decodable>(_ req: URLRequest) async throws -> T {
        let (data, response) = try await perform(req)
        try Self.check(response, data)
        return try API.decoder.decode(T.self, from: data)
    }

    private func sendIgnoringBody(_ req: URLRequest) async throws {
        let (data, response) = try await perform(req)
        try Self.check(response, data)
    }

    private func perform(_ req: URLRequest) async throws -> (Data, URLResponse) {
        do {
            return try await session.data(for: req)
        } catch let error as URLError {
            throw APIError.transport(error.localizedDescription)
        }
    }

    private static func check(_ response: URLResponse, _ data: Data) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            if let env = try? API.decoder.decode(APIErrorEnvelope.self, from: data) { throw env.error }
            throw APIError(code: "http_\(http.statusCode)", message: "De engine gaf fout \(http.statusCode).", hint: nil)
        }
    }

    private func sse<T: Sendable>(
        _ req: URLRequest,
        handle: @escaping @Sendable (String, (T) -> Void) throws -> Void
    ) -> AsyncThrowingStream<T, Error> {
        let session = self.session
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let (bytes, response) = try await session.bytes(for: req)
                    if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                        var body = Data()
                        for try await b in bytes { body.append(b) }
                        try Self.check(response, body)
                    }
                    var parser = SSEParser()
                    for try await byte in bytes {
                        if let payload = parser.feed(byte) {
                            try handle(payload) { continuation.yield($0) }
                        }
                    }
                    continuation.finish()
                } catch let error as URLError where error.code == .cancelled {
                    continuation.finish()
                } catch let error as URLError {
                    continuation.finish(throwing: APIError.transport(error.localizedDescription))
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
