import Foundation
import Observation

/// Live statistics from /aether/v1/stats/stream. Kept apart from chat state so 4 Hz updates
/// only re-render the views that show them (start-prompt-v2 §8.14).
@MainActor
@Observable
final class LiveStats {
    static let historyLength = 240  // 60 s at 4 Hz

    private(set) var snapshot: StatsSnapshot?
    private(set) var tpsHistory: [Double] = []
    private(set) var readHistory: [Double] = []
    @ObservationIgnored private var task: Task<Void, Never>?

    func connect(_ client: EngineClient) {
        task?.cancel()
        task = Task { [weak self] in
            while !Task.isCancelled {
                do {
                    for try await snap in client.statsStream() {
                        self?.ingest(snap)
                    }
                } catch {}
                try? await Task.sleep(for: .seconds(1))  // engine restarting: retry
            }
        }
    }

    func disconnect() {
        task?.cancel()
        task = nil
    }

    func ingest(_ snap: StatsSnapshot) {
        snapshot = snap
        Self.push(&tpsHistory, snap.generation.active ? (snap.generation.decodeTps ?? 0) : 0)
        Self.push(&readHistory, snap.drift.readBytesPerS / 1e9)
    }

    private static func push(_ history: inout [Double], _ value: Double) {
        history.append(value)
        if history.count > historyLength { history.removeFirst(history.count - historyLength) }
    }
}
