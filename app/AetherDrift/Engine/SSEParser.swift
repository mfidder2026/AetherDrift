import Foundation

/// Incremental Server-Sent Events parser. Feed it bytes as they arrive; it returns the
/// `data` payload of every event that is complete. Handles \n and \r\n, multi-line data
/// and comment lines.
struct SSEParser {
    private var line: [UInt8] = []
    private var dataLines: [String] = []

    mutating func feed(_ byte: UInt8) -> String? {
        if byte == UInt8(ascii: "\n") {
            if line.last == UInt8(ascii: "\r") { line.removeLast() }
            defer { line.removeAll(keepingCapacity: true) }
            return endOfLine()
        }
        line.append(byte)
        return nil
    }

    mutating func feed<S: Sequence>(_ bytes: S) -> [String] where S.Element == UInt8 {
        var events: [String] = []
        for b in bytes {
            if let e = feed(b) { events.append(e) }
        }
        return events
    }

    private mutating func endOfLine() -> String? {
        if line.isEmpty {
            guard !dataLines.isEmpty else { return nil }
            defer { dataLines.removeAll() }
            return dataLines.joined(separator: "\n")
        }
        if line.first == UInt8(ascii: ":") { return nil }  // comment
        let text = String(decoding: line, as: UTF8.self)
        if text.hasPrefix("data:") {
            var value = text.dropFirst(5)
            if value.first == " " { value = value.dropFirst() }
            dataLines.append(String(value))
        }
        return nil
    }
}
