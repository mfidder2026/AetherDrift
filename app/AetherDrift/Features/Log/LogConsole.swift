import AppKit
import Observation
import SwiftUI

struct LogLine: Identifiable, Equatable, Sendable {
    enum Kind: String, CaseIterable, Sendable {
        case load = "Laden"
        case evict = "Vrijgeven"
        case warning = "Waarschuwing"
        case error = "Fout"
        case info = "Overig"
    }

    let id: Int
    let time: Date
    let kind: Kind
    let tag: String
    let text: String

    /// Engine event → one readable line (start-prompt-v2 §8.7). nil = not worth showing.
    init?(_ e: EngineEvent) {
        func gb(_ b: Int64?) -> String { Format.number(Double(b ?? 0) / 1_073_741_824, digits: 2) + " GB" }
        func layer(_ u: String?) -> String {
            guard let u, u.hasPrefix("L"), let n = Int(u.dropFirst()) else { return u ?? "?" }
            return "laag \(n)"
        }
        id = e.seq
        time = Date(timeIntervalSince1970: e.ts)
        switch e.type {
        case "load" where e.experts != nil:
            let what = "\(e.experts ?? 0) expert\(e.experts == 1 ? "" : "s")\(e.prefill == true ? " (prefill)" : "")"
            (kind, tag, text) = (.load, "LOAD", "\(layer(e.unit)) · \(what) · \(gb(e.bytes))")
        case "evict" where e.experts != nil:
            (kind, tag, text) = (.evict, "EVICT", "\(layer(e.unit)) · \(e.experts ?? 0) experts vrijgegeven")
        case "load":
            let ms = e.ms ?? 0
            let speed = ms > 0 ? " (" + Format.number(Double(e.bytes ?? 0) / Double(ms) / 1e6, digits: 1) + " GB/s)" : ""
            (kind, tag, text) = (.load, "LOAD", "\(layer(e.unit))\(e.pinned == true ? " (vast)" : "") · \(gb(e.bytes)) · \(ms) ms\(speed)")
        case "evict":
            (kind, tag, text) = (.evict, "EVICT", "\(layer(e.unit)) · \(gb(e.bytes)) vrijgegeven")
        case "summary":
            (kind, tag, text) = (.load, "SUMMARY", "\(e.loads ?? 0) loads, \(e.evicts ?? 0) evicts · \(gb(e.bytes)) in \(e.windowMs ?? 0) ms")
        case "unpinned":
            (kind, tag, text) = (.evict, "UNPIN", "\(gb(e.bytes)) vastgezette lagen vrijgegeven, nog \(e.pinnedLayers ?? 0) vast")
        case "pressure":
            let level = e.level == "critical" ? "kritiek" : e.level == "warning" ? "krap" : "normaal"
            (kind, tag, text) = (e.level == "normal" ? .info : .warning, "WARN",
                                 "Geheugendruk \(level)" + (e.action.map { ": \($0)" } ?? ""))
        case "drift_plan" where e.experts != nil:
            (kind, tag, text) = (.info, "DRIFT", "\(e.layers ?? 0) MoE-lagen, \(e.slots ?? 0) van \(e.experts ?? 0) experts per laag in RAM · \(gb(e.residentBytes))")
        case "drift_plan":
            (kind, tag, text) = (.info, "DRIFT", "\(e.layers ?? 0) lagen, \(e.pinnedLayers ?? 0) vast, \(e.slots ?? 0) slots · \(gb(e.residentBytes)) in RAM")
        case "model_loaded":
            (kind, tag, text) = (.load, "MODEL", "\(e.model ?? "") geladen (\(Format.modeName(e.mode))) · \(gb(e.bytes)) in \(e.ms ?? 0) ms")
        case "model_unloaded":
            (kind, tag, text) = (.info, "MODEL", "\(e.model ?? "") ontladen")
        case "generation_done":
            (kind, tag, text) = (.info, "GEN", "\(e.completionTokens ?? 0) tokens · \(Format.tps(e.decodeTps)) · \(e.finishReason ?? "")")
        case "download_done":
            (kind, tag, text) = (.info, "DOWNLOAD", "\(e.repo ?? "") klaar")
        case "download_failed":
            (kind, tag, text) = (.warning, "DOWNLOAD", "\(e.repo ?? ""): \(e.message ?? "mislukt")")
        case "error":
            (kind, tag, text) = (.error, "ERROR", e.message ?? e.code ?? "onbekende fout")
        default:
            return nil  // progress events etc.: shown elsewhere
        }
    }

    var formatted: String {
        "\(time.formatted(.dateTime.hour().minute().second().secondFraction(.fractional(3))))  \(tag.padding(toLength: 8, withPad: " ", startingAt: 0)) \(text)"
    }
}

/// Keeps the last lines for the console (the full log is on disk).
@MainActor
@Observable
final class LogStore {
    static let maxLines = 2000
    private(set) var lines: [LogLine] = []
    var paused = false
    @ObservationIgnored private var pending: [LogLine] = []

    func add(_ e: EngineEvent) {
        guard let line = LogLine(e) else { return }
        if paused {
            pending.append(line)
            return
        }
        lines.append(line)
        if lines.count > Self.maxLines { lines.removeFirst(lines.count - Self.maxLines) }
    }

    func resume() {
        paused = false
        lines += pending
        pending.removeAll()
        if lines.count > Self.maxLines { lines.removeFirst(lines.count - Self.maxLines) }
    }

    func clear() { lines.removeAll() }
}

struct LogConsole: View {
    @Environment(AppState.self) private var state
    @State private var filter: Set<LogLine.Kind> = Set(LogLine.Kind.allCases)
    @State private var query = ""

    var body: some View {
        let log = state.log
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Text("Log").font(.headline)
                ForEach(LogLine.Kind.allCases, id: \.self) { kind in
                    Toggle(kind.rawValue, isOn: Binding(
                        get: { filter.contains(kind) },
                        set: { on in if on { filter.insert(kind) } else { filter.remove(kind) } }
                    ))
                    .toggleStyle(.button)
                    .controlSize(.small)
                }
                TextField("Zoek", text: $query).textFieldStyle(.roundedBorder).frame(maxWidth: 160).controlSize(.small)
                Spacer()
                Button(log.paused ? "Hervat" : "Pauzeer") {
                    if log.paused { log.resume() } else { log.paused = true }
                }
                .controlSize(.small)
                Button("Kopieer") { copy() }.controlSize(.small)
                Button("Open logbestand") {
                    let url = FileManager.default.homeDirectoryForCurrentUser.appending(path: "Library/Logs/AetherDrift/engine.log")
                    NSWorkspace.shared.open(url)
                }
                .controlSize(.small)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            Divider()
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 1) {
                        ForEach(visible) { line in
                            Text(line.formatted)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(color(line.kind))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .id(line.id)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 4)
                    .textSelection(.enabled)
                }
                .onChange(of: log.lines.last?.id) { _, last in
                    if let last, !log.paused { proxy.scrollTo(last, anchor: .bottom) }
                }
            }
        }
        .frame(height: 210)
        .background(.background)
    }

    private var visible: [LogLine] {
        state.log.lines.filter {
            filter.contains($0.kind) && (query.isEmpty || $0.text.localizedCaseInsensitiveContains(query))
        }
    }

    private func color(_ kind: LogLine.Kind) -> Color {
        switch kind {
        case .warning: .orange
        case .error: .red
        case .evict: .secondary
        default: .primary
        }
    }

    private func copy() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(visible.map(\.formatted).joined(separator: "\n"), forType: .string)
    }
}
