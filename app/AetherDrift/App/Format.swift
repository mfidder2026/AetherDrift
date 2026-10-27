import Foundation

enum Format {
    static func bytes(_ value: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: value, countStyle: .memory)
    }

    static func gb(_ value: Int64, digits: Int = 0) -> String {
        (Double(value) / 1_073_741_824).formatted(.number.precision(.fractionLength(digits))) + " GB"
    }

    static func tps(_ value: Double?) -> String {
        guard let value else { return "—" }
        return value.formatted(.number.precision(.fractionLength(value < 10 ? 1 : 0))) + " tok/s"
    }

    static func number(_ value: Double?, digits: Int = 1) -> String {
        guard let value else { return "—" }
        return value.formatted(.number.precision(.fractionLength(digits)))
    }

    static func tokens(_ value: Int) -> String {
        value >= 1000 ? (Double(value) / 1000).formatted(.number.precision(.fractionLength(1))) + "k" : "\(value)"
    }

    static func modeName(_ mode: String?) -> String {
        switch mode {
        case "resident": String(localized: "In RAM")
        case "drift-dense", "drift-moe": String(localized: "Drift")
        case "too_large": String(localized: "Te groot")
        default: "—"
        }
    }

    static let thinkingLevels = ["auto", "off", "low", "high", "max"]

    static func thinkingName(_ level: String?) -> String {
        switch level {
        case "off": String(localized: "Uit")
        case "low": String(localized: "Kort")
        case "high": String(localized: "Grondig")
        case "max": String(localized: "Maximaal")
        default: String(localized: "Automatisch")
        }
    }

    static func presetName(_ preset: String) -> String {
        switch preset {
        case "snel": String(localized: "Snel")
        case "lange_context": String(localized: "Lange context")
        default: String(localized: "Balans")
        }
    }
}
