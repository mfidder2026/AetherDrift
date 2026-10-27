import CryptoKit
import Foundation

/// Locates the bundled engine and uv, installs the Python runtime once, and builds the
/// command line that starts the engine (start-prompt-v2 §7.5, §8.13).
struct EngineRuntime: Sendable {
    let engineDir: URL
    let uv: URL
    let supportDir: URL
    /// Release builds carry their own Python and packages (scripts/bundle-runtime.sh); then
    /// nothing is installed and no internet is needed.
    var bundledRuntime: URL?

    var runtimeDir: URL { supportDir.appending(path: "runtime") }
    var venv: URL { runtimeDir.appending(path: "venv") }
    var python: URL {
        bundledRuntime.map { $0.appending(path: "python/bin/python3.12") } ?? venv.appending(path: "bin/python")
    }

    static func locate() throws -> EngineRuntime {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appending(path: "AetherDrift")
        let env = ProcessInfo.processInfo.environment
        let resources = Bundle.main.resourceURL!
        let engine = env["AETHER_ENGINE_DIR"].map { URL(filePath: $0) } ?? resources.appending(path: "engine")
        let uv = env["AETHER_UV"].map { URL(filePath: $0) } ?? resources.appending(path: "bin/uv")
        guard FileManager.default.fileExists(atPath: engine.appending(path: "aether/__main__.py").path) else {
            throw RuntimeError("De engine ontbreekt in de app (\(engine.path)).")
        }
        guard FileManager.default.isExecutableFile(atPath: uv.path) else {
            throw RuntimeError("uv ontbreekt in de app (\(uv.path)).")
        }
        let bundled = resources.appending(path: "runtime")
        let hasBundled = env["AETHER_ENGINE_DIR"] == nil
            && FileManager.default.isExecutableFile(atPath: bundled.appending(path: "python/bin/python3.12").path)
        return EngineRuntime(engineDir: engine, uv: uv, supportDir: support, bundledRuntime: hasBundled ? bundled : nil)
    }

    /// The installed runtime is tied to the exact lock file it was built from.
    private var marker: URL {
        let lock = (try? Data(contentsOf: engineDir.appending(path: "uv.lock"))) ?? Data()
        let hash = SHA256.hash(data: lock).prefix(8).map { String(format: "%02x", $0) }.joined()
        return runtimeDir.appending(path: "installed-\(hash)")
    }

    var isInstalled: Bool {
        if bundledRuntime != nil { return true }
        return FileManager.default.fileExists(atPath: marker.path)
            && FileManager.default.isExecutableFile(atPath: python.path)
    }

    /// One-time `uv sync` into Application Support. Needs internet the first time.
    func install(log: @escaping @Sendable (String) -> Void) async throws {
        try FileManager.default.createDirectory(at: runtimeDir, withIntermediateDirectories: true)
        let process = Process()
        process.executableURL = uv
        process.arguments = ["sync", "--frozen", "--no-dev", "--no-install-project", "--project", engineDir.path]
        var env = ProcessInfo.processInfo.environment
        env["UV_PROJECT_ENVIRONMENT"] = venv.path
        env["UV_PYTHON_INSTALL_DIR"] = runtimeDir.appending(path: "python").path
        env["UV_CACHE_DIR"] = runtimeDir.appending(path: "cache").path
        env["UV_PYTHON_PREFERENCE"] = "only-managed"
        env["UV_NO_PROGRESS"] = "1"
        process.environment = env
        let errPipe = Pipe()
        process.standardError = errPipe
        process.standardOutput = FileHandle.nullDevice
        let tail = OutputTail()
        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let text = String(decoding: data, as: UTF8.self)
            tail.append(text)
            for line in text.split(separator: "\n") { log(String(line)) }
        }
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            process.terminationHandler = { p in
                errPipe.fileHandleForReading.readabilityHandler = nil
                if p.terminationStatus == 0 {
                    cont.resume()
                } else {
                    cont.resume(throwing: RuntimeError("Installatie mislukt:\n\(tail.text)"))
                }
            }
            do { try process.run() } catch { cont.resume(throwing: error) }
        }
        FileManager.default.createFile(atPath: marker.path, contents: Data())
    }

    func launchSpec(port: Int, parentPID: Int32) -> LaunchSpec {
        var env = ProcessInfo.processInfo.environment
        if let bundledRuntime {
            env["PYTHONPATH"] = engineDir.path + ":" + bundledRuntime.appending(path: "site-packages").path
            env["PYTHONNOUSERSITE"] = "1"
            env.removeValue(forKey: "PYTHONHOME")
        } else {
            env["PYTHONPATH"] = engineDir.path
        }
        env["PYTHONDONTWRITEBYTECODE"] = "1"  // never write into the signed app bundle
        env["PYTHONUNBUFFERED"] = "1"
        return LaunchSpec(
            executable: python,
            arguments: ["-m", "aether", "--port", String(port), "--parent-pid", String(parentPID),
                        "--data-dir", supportDir.path],
            environment: env
        )
    }
}

struct LaunchSpec: Sendable {
    var executable: URL
    var arguments: [String]
    var environment: [String: String]
}

struct RuntimeError: Error, LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

/// Keeps the last few KB of a process's output for error messages.
final class OutputTail: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer = ""

    func append(_ text: String) {
        lock.withLock {
            buffer += text
            if buffer.count > 4000 { buffer = String(buffer.suffix(4000)) }
        }
    }

    var text: String { lock.withLock { buffer } }
}
