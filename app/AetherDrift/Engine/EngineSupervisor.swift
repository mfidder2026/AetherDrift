import Foundation
import Observation
import os

/// Starts the engine, waits for its ready line, watches its health and restarts it with
/// backoff (start-prompt-v2 §8.13). The UI never waits on any of this to appear.
@MainActor
@Observable
final class EngineSupervisor {
    enum Status: Equatable {
        case idle
        case installing(String)
        case starting
        case ready(port: Int, external: Bool)
        case restarting(attempt: Int)
        case failed(String)
    }

    struct Hooks: Sendable {
        /// Returns a running external engine's port, if any (development mode).
        var findExternal: @Sendable () async -> Int?
        /// Installs the runtime if needed and returns how to start the engine.
        var prepare: @Sendable (_ port: Int, _ progress: @escaping @Sendable (String) -> Void) async throws -> LaunchSpec
        var healthCheck: @Sendable (EngineClient) async -> Bool
    }

    nonisolated static let preferredPort = 8080
    static let backoff: [Duration] = [.seconds(1), .seconds(2), .seconds(5), .seconds(10), .seconds(30)]
    static let readyTimeout: Duration = .seconds(30)
    static let healthInterval: Duration = .seconds(2)
    static let allowedMisses = 3
    static let stableAfter: Duration = .seconds(60)

    private(set) var status: Status = .idle
    private(set) var client: EngineClient?
    /// Increases every time a (new) engine becomes ready; observers reconnect streams on change.
    private(set) var generation = 0

    private let hooks: Hooks
    private let log = Logger(subsystem: "app.fremen.AetherDrift", category: "supervisor")
    private var process: Process?
    private var attempt = 0
    private var stopping = false
    private var monitor: Task<Void, Never>?
    private var readyWaiters: [CheckedContinuation<EngineClient, Never>] = []

    init(hooks: Hooks = .live) {
        self.hooks = hooks
    }

    var isReady: Bool {
        if case .ready = status { return true }
        return false
    }

    /// Suspends until an engine is ready.
    func readyClient() async -> EngineClient {
        if isReady, let client { return client }
        return await withCheckedContinuation { readyWaiters.append($0) }
    }

    func start() {
        guard monitor == nil else { return }
        stopping = false
        monitor = Task { await run() }
    }

    func stop() {
        stopping = true
        monitor?.cancel()
        monitor = nil
        if let process, process.isRunning {
            process.terminate()
        }
        process = nil
    }

    /// Kill and restart (used by tests and the "Herstart engine" action).
    func restart() {
        guard let process else { return }
        Task { await Self.shutDown(process) }
    }

    // MARK: - Loop

    private func run() async {
        while !Task.isCancelled && !stopping {
            if let port = await hooks.findExternal() {
                becomeReady(port: port, external: true)
                await watch(process: nil)
                if stopping { return }
                continue
            }
            do {
                try await launchOnce()
                await watch(process: process)
            } catch {
                log.error("engine start failed: \(error.localizedDescription, privacy: .public)")
                client = nil
                if attempt >= Self.backoff.count {
                    status = .failed(error.localizedDescription)
                }
            }
            if stopping || Task.isCancelled { return }
            let delay = Self.backoff[min(attempt, Self.backoff.count - 1)]
            attempt += 1
            if case .failed = status {} else { status = .restarting(attempt: attempt) }
            try? await Task.sleep(for: delay)
        }
    }

    private func launchOnce() async throws {
        status = attempt == 0 ? .starting : .restarting(attempt: attempt)
        let spec = try await hooks.prepare(Self.preferredPort) { [weak self] line in
            Task { @MainActor in
                guard let self, !self.isReady else { return }
                self.status = .installing(line)
            }
        }
        if case .installing = status { status = .starting }

        let p = Process()
        p.executableURL = spec.executable
        p.arguments = spec.arguments
        p.environment = spec.environment
        let out = Pipe()
        let err = Pipe()
        p.standardOutput = out
        p.standardError = err
        p.standardInput = Pipe()
        let tail = OutputTail()
        err.fileHandleForReading.readabilityHandler = { h in
            let d = h.availableData
            if !d.isEmpty { tail.append(String(decoding: d, as: UTF8.self)) }
        }
        let ready = ReadyLineReader(pipe: out)
        try p.run()
        process = p

        let port = try await ready.wait(timeout: Self.readyTimeout) {
            !p.isRunning
        }
        guard let port else {
            await Self.shutDown(p)
            throw RuntimeError("De engine startte niet.\n\(tail.text.suffix(1500))")
        }
        becomeReady(port: port, external: false)
    }

    private func becomeReady(port: Int, external: Bool) {
        let c = EngineClient(port: port)
        client = c
        status = .ready(port: port, external: external)
        generation += 1
        let waiters = readyWaiters
        readyWaiters.removeAll()
        waiters.forEach { $0.resume(returning: c) }
        log.info("engine ready on port \(port) (external: \(external))")
    }

    /// Returns when the engine is gone or unhealthy.
    private func watch(process: Process?) async {
        guard let client else { return }
        var misses = 0
        let readySince = ContinuousClock.now
        while !Task.isCancelled && !stopping {
            try? await Task.sleep(for: Self.healthInterval)
            if let process, !process.isRunning {
                log.error("engine exited with status \(process.terminationStatus)")
                break
            }
            if await hooks.healthCheck(client) {
                misses = 0
                if ContinuousClock.now - readySince > Self.stableAfter { attempt = 0 }
            } else {
                misses += 1
                if misses >= Self.allowedMisses {
                    log.error("engine missed \(misses) health checks")
                    if let process { await Self.shutDown(process) }
                    break
                }
            }
        }
        self.client = nil
    }

    /// SIGTERM, then SIGKILL if the engine has not exited after `grace`. A new engine must
    /// never start while the old one still holds its memory (up to ~100 GB).
    static func shutDown(_ process: Process, grace: Duration = .seconds(10)) async {
        guard process.isRunning else { return }
        process.terminate()
        let deadline = ContinuousClock.now + grace
        while process.isRunning && ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(100))
        }
        if process.isRunning {
            kill(process.processIdentifier, SIGKILL)
            while process.isRunning { try? await Task.sleep(for: .milliseconds(50)) }
        }
    }
}

/// Reads stdout until the engine's single JSON ready line.
private final class ReadyLineReader: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer = ""
    private var port: Int?

    init(pipe: Pipe) {
        pipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let d = h.availableData
            guard let self, !d.isEmpty else { return }
            self.lock.withLock {
                self.buffer += String(decoding: d, as: UTF8.self)
                while let nl = self.buffer.firstIndex(of: "\n") {
                    let line = String(self.buffer[..<nl])
                    self.buffer.removeSubrange(...nl)
                    if let data = line.data(using: .utf8),
                       let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       obj["event"] as? String == "ready",
                       let port = obj["port"] as? Int {
                        self.port = port
                    }
                }
            }
        }
    }

    func wait(timeout: Duration, exited: @escaping @Sendable () -> Bool) async throws -> Int? {
        let deadline = ContinuousClock.now + timeout
        while ContinuousClock.now < deadline {
            if let port = lock.withLock({ port }) { return port }
            if exited() { return nil }
            try await Task.sleep(for: .milliseconds(20))
        }
        return nil
    }
}

extension EngineSupervisor.Hooks {
    static let live = EngineSupervisor.Hooks(
        findExternal: {
            let c = EngineClient(port: EngineSupervisor.preferredPort)
            guard ProcessInfo.processInfo.environment["AETHER_NO_EXTERNAL"] == nil,
                  let h = try? await c.health(timeout: 0.3), h.status == "ok" else { return nil }
            return EngineSupervisor.preferredPort
        },
        prepare: { port, progress in
            let runtime = try EngineRuntime.locate()
            if !runtime.isInstalled {
                progress("Engine installeren (eenmalig)…")
                try await runtime.install(log: progress)
            }
            return runtime.launchSpec(port: port, parentPID: ProcessInfo.processInfo.processIdentifier)
        },
        healthCheck: { client in (try? await client.health(timeout: 2)) != nil }
    )
}
