import XCTest
@testable import AetherDrift

final class SSEParserTests: XCTestCase {
    func testSplitsEventsAcrossArbitraryChunks() {
        var p = SSEParser()
        let stream = Array("data: {\"a\":1}\n\ndata: [DONE]\n\n".utf8)
        var events: [String] = []
        for chunk in stride(from: 0, to: stream.count, by: 3) {
            events += p.feed(stream[chunk..<min(chunk + 3, stream.count)])
        }
        XCTAssertEqual(events, ["{\"a\":1}", "[DONE]"])
    }

    func testCRLFCommentsAndMultilineData() {
        var p = SSEParser()
        let events = p.feed(Array(": keepalive\r\ndata: one\r\ndata: two\r\n\r\n".utf8))
        XCTAssertEqual(events, ["one\ntwo"])
    }
}

final class APIDecodingTests: XCTestCase {
    func testChatChunkWithFinishAndTimings() throws {
        let json = #"{"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"aether":{"timings":{"ttft_ms":12,"prefill_tps":900.5,"decode_tps":512.3}}}"#
        let chunk = try API.decoder.decode(ChatChunk.self, from: Data(json.utf8))
        XCTAssertEqual(chunk.choices?.first?.finishReason, "stop")
        XCTAssertEqual(chunk.aether?.timings?.decodeTps, 512.3)
    }

    func testStatsSnapshotDecodes() throws {
        let json = #"""
        {"ts":1.0,"model":{"id":"m","state":"ready","mode":"resident","progress":1.0},
         "memory":{"budget_bytes":100,"weights_bytes":10,"kv_bytes":0,"mlx_active_bytes":1,"mlx_cache_bytes":0,
                   "system_used_bytes":5,"system_total_bytes":9,"swap_used_bytes":0,"pressure":"normal"},
         "drift":{"layers_total":null,"current_layer":null,"layer_residency":[],"hit_rate":null,
                  "read_bytes_per_s":0.0,"read_p95_ms":null,"write_bytes_per_s":0.0},
         "generation":{"active":false,"decode_tps":null,"prefill_tps":null,"ttft_ms":null,
                       "context_tokens":0,"context_max":32768,"queue_depth":0}}
        """#
        let s = try API.decoder.decode(StatsSnapshot.self, from: Data(json.utf8))
        XCTAssertEqual(s.model.state, "ready")
        XCTAssertEqual(s.generation.contextMax, 32768)
    }

    func testErrorEnvelope() throws {
        let json = #"{"error":{"code":"model_too_large","message":"te groot","hint":null}}"#
        let e = try API.decoder.decode(APIErrorEnvelope.self, from: Data(json.utf8)).error
        XCTAssertEqual(e.code, "model_too_large")
        XCTAssertEqual(FriendlyError(e).action, .openModels)
    }
}

final class MarkdownTests: XCTestCase {
    func testBlocks() {
        let blocks = MarkdownRenderer.blocks(from: "# Kop\n\nTekst met **vet**.\n\n```swift\nlet a = 1\n```\n\n- een\n- twee")
        XCTAssertEqual(blocks.count, 4)
        if case .code(let lang, let code) = blocks[2] {
            XCTAssertEqual(lang, "swift")
            XCTAssertEqual(code, "let a = 1")
        } else { XCTFail("expected code block") }
    }

    func testStreamingMatchesFullParse() {
        let text = "Intro.\n\n```python\nprint(1)\n\nprint(2)\n```\n\nNa de code.\n\n1. a\n2. b\n"
        var s = StreamingMarkdown()
        var last: [MarkdownBlock] = []
        for end in text.indices {
            last = s.update(String(text[..<end]))
        }
        last = s.update(text)
        XCTAssertEqual(last, MarkdownRenderer.blocks(from: text))
    }

    func testBlankLineInsideFenceIsNotABoundary() {
        var s = StreamingMarkdown()
        _ = s.update("```\na\n\nb")
        XCTAssertTrue(s.stableBlocks.isEmpty)
    }

    func testResetOnEditedText() {
        var s = StreamingMarkdown()
        _ = s.update("Eerste.\n\nTweede")
        let blocks = s.update("Ander.\n\n")
        XCTAssertEqual(blocks, MarkdownRenderer.blocks(from: "Ander.\n\n"))
    }
}

final class DatabaseTests: XCTestCase {
    func testConversationRoundtripAndCascade() throws {
        let db = try AppDatabase.inMemory()
        let c = try db.createConversation(title: "Test")
        let m = try db.save(StoredMessage(id: nil, conversationId: c.id!, role: "user", content: "hoi",
                                          createdAt: .now, meta: nil))
        XCTAssertEqual(try db.messages(in: c.id!).map(\.content), ["hoi"])
        try db.deleteMessages(in: c.id!, from: m.id!)
        XCTAssertTrue(try db.messages(in: c.id!).isEmpty)
        try db.deleteConversation(c.id!)
        XCTAssertTrue(try db.conversations().isEmpty)
    }

    func testTitleFromFirstWords() {
        XCTAssertEqual(ChatStore.title(for: "  Wat is de hoofdstad van Frankrijk eigenlijk precies vandaag?"),
                       "Wat is de hoofdstad van Frankrijk eigenlijk precies")
        XCTAssertEqual(ChatStore.title(for: "   "), "Nieuw gesprek")
    }
}

@MainActor
final class SupervisorTests: XCTestCase {
    /// A fake engine: prints a ready line, then crashes after a moment. The supervisor must
    /// see the crash and start it again.
    func testRestartsAfterCrash() async throws {
        let script = #"echo '{"event":"ready","port":4242,"version":"t","pid":1}'; sleep 0.3; exit 3"#
        let launches = Counter()
        let hooks = EngineSupervisor.Hooks(
            findExternal: { nil },
            prepare: { _, _ in
                launches.increment()
                return LaunchSpec(executable: URL(filePath: "/bin/sh"), arguments: ["-c", script], environment: [:])
            },
            healthCheck: { _ in true }
        )
        let sup = EngineSupervisor(hooks: hooks)
        sup.start()
        let deadline = Date.now.addingTimeInterval(15)
        while launches.value < 2 && Date.now < deadline {
            try await Task.sleep(for: .milliseconds(50))
        }
        sup.stop()
        XCTAssertGreaterThanOrEqual(launches.value, 2)
        XCTAssertGreaterThanOrEqual(sup.generation, 1)
    }

    func testReadyClientResolvesWithReportedPort() async throws {
        let hooks = EngineSupervisor.Hooks(
            findExternal: { nil },
            prepare: { _, _ in
                LaunchSpec(executable: URL(filePath: "/bin/sh"),
                           arguments: ["-c", #"echo noise; echo '{"event":"ready","port":5151}'; sleep 5"#],
                           environment: [:])
            },
            healthCheck: { _ in true }
        )
        let sup = EngineSupervisor(hooks: hooks)
        sup.start()
        let client = await sup.readyClient()
        XCTAssertEqual(client.baseURL.port, 5151)
        XCTAssertEqual(sup.status, .ready(port: 5151, external: false))
        sup.stop()
    }

    func testStartFailureIsReportedAndRetried() async throws {
        let launches = Counter()
        let hooks = EngineSupervisor.Hooks(
            findExternal: { nil },
            prepare: { _, _ in
                launches.increment()
                return LaunchSpec(executable: URL(filePath: "/bin/sh"),
                                  arguments: ["-c", "echo boom >&2; exit 1"], environment: [:])
            },
            healthCheck: { _ in true }
        )
        let sup = EngineSupervisor(hooks: hooks)
        sup.start()
        let deadline = Date.now.addingTimeInterval(10)
        while launches.value < 2 && Date.now < deadline {
            try await Task.sleep(for: .milliseconds(50))
        }
        if case .ready = sup.status { XCTFail("must not report ready") }
        sup.stop()
        XCTAssertGreaterThanOrEqual(launches.value, 2)
    }
}

final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    func increment() { lock.withLock { n += 1 } }
    var value: Int { lock.withLock { n } }
}

final class ReasoningSplitTests: XCTestCase {
    func testSplit() {
        XCTAssertEqual(ReasoningSplit("Gewoon antwoord").answer, "Gewoon antwoord")
        let open = ReasoningSplit("<think>\nhmm")
        XCTAssertTrue(open.isOpen)
        XCTAssertEqual(open.thinking, "hmm")
        XCTAssertEqual(open.answer, "")
        let done = ReasoningSplit("<think>plan</think>\n\n**Antwoord**")
        XCTAssertFalse(done.isOpen)
        XCTAssertEqual(done.thinking, "plan")
        XCTAssertEqual(done.answer, "**Antwoord**")
    }
}

final class EngineSettingsEncodingTests: XCTestCase {
    func testAutomaticLimitsAreSentAsNull() throws {
        let s = EngineSettings(mode: "balans", share: false, allowClientModelSwitch: false, osReserveGb: nil,
                               engineOverheadGb: 2, weightsBudgetGb: nil, prefillStepSize: 2048,
                               promptCacheEntries: 3, requiresReload: nil)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: API.encoder.encode(s)) as? [String: Any])
        XCTAssertTrue(json["os_reserve_gb"] is NSNull)
        XCTAssertTrue(json["weights_budget_gb"] is NSNull)
        XCTAssertNil(json["requires_reload"])
    }

    func testNetworkAddressPutsKeyInFragment() {
        let a = NetworkAddress(hostName: "studio.local", ip: "192.168.1.20", port: 8080)
        XCTAssertEqual(a.baseURL, "http://studio.local:8080/v1")
        XCTAssertEqual(a.webchatURL(key: "abc-_1"), "http://192.168.1.20:8080/#key=abc-_1")
    }
}
