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
        case "resident": "In RAM"
        case "drift-dense", "drift-moe": "Drift"
        case "too_large": "Te groot"
        default: "—"
        }
    }

    static func presetName(_ preset: String) -> String {
        switch preset {
        case "snel": "Snel"
        case "lange_context": "Lange context"
        default: "Balans"
        }
    }
}
