import Foundation

// Mirrors the engine API (start-prompt-v2 §7). JSON keys are snake_case; the shared
// coders below convert to and from camelCase.

enum API {
    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        return e
    }()
}

struct APIError: Error, Decodable, Equatable, Sendable {
    let code: String
    let message: String
    let hint: String?

    static func transport(_ message: String) -> APIError {
        APIError(code: "engine_unreachable", message: message, hint: nil)
    }
}

struct APIErrorEnvelope: Decodable {
    let error: APIError
}

struct Health: Decodable, Sendable {
    let status: String
    let version: String
    let modelState: String
}

struct SystemInfo: Decodable, Sendable, Equatable {
    let chip: String
    let ramBytes: Int64
    let metalWorkingSetBytes: Int64
    let macos: String
    let ssdReadBytesPerS: Double?
}

struct Fit: Decodable, Sendable, Hashable {
    let mode: String  // resident | drift-dense | drift-moe | too_large
    let requiredMinBytes: Int64
    let availableBytes: Int64
    let expectedTps: Double?
    let estimated: Bool
    let loadable: Bool
}

struct LocalModel: Decodable, Sendable, Identifiable, Hashable {
    let id: String
    let path: String
    let source: String
    let sizeBytes: Int64
    let modelType: String?
    let isMoe: Bool
    let loaded: Bool
    let fit: Fit?
}

struct CatalogEntry: Decodable, Sendable, Identifiable, Hashable {
    let repo: String
    let name: String
    let role: String
    let description: String
    let sizeBytes: Int64
    let downloaded: Bool
    let fit: Fit
    var id: String { repo }
}

struct ModelsResponse: Decodable, Sendable {
    let local: [LocalModel]
    let catalog: [CatalogEntry]
}

struct StatsSnapshot: Decodable, Sendable, Equatable {
    struct Model: Decodable, Sendable, Equatable {
        let id: String?
        let state: String  // none | loading | ready | error
        let mode: String?
        let progress: Double?
    }

    struct Memory: Decodable, Sendable, Equatable {
        let budgetBytes: Int64?
        let weightsBytes: Int64
        let kvBytes: Int64
        let mlxActiveBytes: Int64
        let mlxCacheBytes: Int64
        let systemUsedBytes: Int64
        let systemTotalBytes: Int64
        let swapUsedBytes: Int64
        let pressure: String
        let mlxPeakBytes: Int64?
    }

    struct Drift: Decodable, Sendable, Equatable {
        let layersTotal: Int?
        let currentLayer: Int?
        let layerResidency: [Double]
        let hitRate: Double?
        let readBytesPerS: Double
        let readP95Ms: Double?
        let writeBytesPerS: Double
        let pinnedLayers: Int?
        let stallSeconds: Double?
        let expertSlots: Int?
        let expertsTotal: Int?
    }

    struct Generation: Decodable, Sendable, Equatable {
        let active: Bool
        let decodeTps: Double?
        let prefillTps: Double?
        let ttftMs: Double?
        let contextTokens: Int
        let contextMax: Int?
        let queueDepth: Int
    }

    let ts: Double
    let model: Model
    let memory: Memory
    let drift: Drift
    let generation: Generation
}

/// One event from /aether/v1/events/stream. Events are heterogeneous; only the fields the
/// app uses are decoded.
struct EngineEvent: Decodable, Sendable {
    let seq: Int
    let ts: Double
    let type: String
    let model: String?
    let repo: String?
    let progress: Double?
    let bytes: Int64?
    let total: Int64?
    let code: String?
    let message: String?
    let hint: String?
    let level: String?
    let action: String?
    let mode: String?
    let ms: Int?
    let loads: Int?
    let evicts: Int?
    let unit: String?
    let pinned: Bool?
    let pinnedLayers: Int?
    let layers: Int?
    let slots: Int?
    let residentBytes: Int64?
    let readBytesPerS: Double?
    let finishReason: String?
    let completionTokens: Int?
    let decodeTps: Double?
    let windowMs: Int?
    let experts: Int?
    let prefill: Bool?
}

struct EngineSettings: Codable, Sendable, Equatable {
    var mode: String
    var share: Bool
    var allowClientModelSwitch: Bool
    var osReserveGb: Double
    var engineOverheadGb: Double
    var weightsBudgetGb: Double?
    var prefillStepSize: Int
    var promptCacheEntries: Int
    var requiresReload: Bool?
}

// MARK: - Chat

struct ChatRequestMessage: Encodable, Sendable {
    let role: String
    let content: String
}

struct ChatRequest: Encodable, Sendable {
    struct StreamOptions: Encodable, Sendable { let includeUsage: Bool }
    let model: String
    let messages: [ChatRequestMessage]
    let stream: Bool
    let streamOptions: StreamOptions
    let temperature: Double?
    let maxTokens: Int?
}

struct Usage: Decodable, Sendable, Equatable {
    struct Details: Decodable, Sendable, Equatable { let cachedTokens: Int }
    let promptTokens: Int
    let completionTokens: Int
    let totalTokens: Int
    let promptTokensDetails: Details?
}

struct Timings: Decodable, Sendable, Equatable {
    let ttftMs: Double?
    let prefillTps: Double?
    let decodeTps: Double?
}

struct ChatChunk: Decodable, Sendable {
    struct Choice: Decodable, Sendable {
        struct Delta: Decodable, Sendable {
            let role: String?
            let content: String?
        }
        let delta: Delta
        let finishReason: String?
    }

    struct Aether: Decodable, Sendable {
        let generationId: String?
        let timings: Timings?
    }

    let choices: [Choice]?
    let usage: Usage?
    let aether: Aether?
    let error: APIError?
}

enum ChatEvent: Sendable, Equatable {
    case started(generationID: String)
    case token(String)
    case finished(reason: String, timings: Timings?)
    case usage(Usage)
}
