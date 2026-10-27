import SwiftUI

/// Live panel (start-prompt-v2 §8.6). Every value comes from the engine; unknown shows "—".
struct LiveInspector: View {
    @Environment(AppState.self) private var state

    var body: some View {
        let live = state.live
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("Live").font(.headline)
                if let s = live.snapshot {
                    memory(s)
                    Divider()
                    StatRow(label: "Tokens/s", value: Format.number(s.generation.decodeTps, digits: 1),
                            accessibility: String(localized: "Tokens per seconde"))
                    Sparkline(values: live.tpsHistory).frame(height: 28)
                    StatRow(label: "Prefill", value: s.generation.prefillTps.map { Format.number($0, digits: 0) + " tok/s" } ?? "—")
                    StatRow(label: "Eerste token", value: s.generation.ttftMs.map { Format.number($0, digits: 0) + " ms" } ?? "—")
                    StatRow(label: "Context", value: context(s.generation))
                    Divider()
                    StatRow(label: "SSD-lezen", value: Format.number(s.drift.readBytesPerS / 1e9, digits: 1) + " GB/s")
                    Sparkline(values: live.readHistory).frame(height: 28)
                    StatRow(label: "Cache-hit", value: s.drift.hitRate.map {
                        $0.formatted(.percent.precision(.fractionLength(0)))
                    } ?? "—")
                    StatRow(label: "Laag", value: layer(s.drift))
                    if !s.drift.layerResidency.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Drift-kaart").font(.caption).foregroundStyle(.secondary)
                            DriftMap(residency: s.drift.layerResidency, current: s.drift.currentLayer)
                            if let slots = s.drift.expertSlots, let total = s.drift.expertsTotal {
                                Text("\(slots) van \(total) experts per laag in RAM; de rest komt van de SSD wanneer nodig")
                                    .font(.caption).foregroundStyle(.secondary)
                            } else if let pinned = s.drift.pinnedLayers, let total = s.drift.layersTotal {
                                Text("\(pinned) van \(total) lagen vast in RAM, de rest stroomt van de SSD")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                    Divider()
                    StatRow(label: "Modus", value: Format.modeName(s.model.mode))
                    StatRow(label: "Geheugendruk", value: pressure(s.memory.pressure))
                    StatRow(label: "Wachtrij", value: "\(s.generation.queueDepth)")
                } else {
                    Text("—").foregroundStyle(.tertiary)
                }
                Divider()
                Button(state.showLog ? "Verberg log" : "Toon log (⌘L)") { state.showLog.toggle() }
                    .buttonStyle(.link)
            }
            .padding(16)
        }
    }

    @ViewBuilder private func memory(_ s: StatsSnapshot) -> some View {
        let used = s.memory.systemUsedBytes
        let total = s.memory.systemTotalBytes
        VStack(alignment: .leading, spacing: 4) {
            StatRow(label: "RAM", value: "\(Format.gb(used)) / \(Format.gb(total))",
                    accessibility: String(localized: "RAM-gebruik \(used / 1_073_741_824) van \(total / 1_073_741_824) gigabyte"))
            ProgressView(value: Double(used), total: Double(max(total, 1)))
                .tint(s.memory.pressure == "normal" ? .accentColor : .orange)
            Text("Model \(Format.gb(s.memory.weightsBytes, digits: 1)) · KV \(Format.gb(s.memory.kvBytes, digits: 1))")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func context(_ g: StatsSnapshot.Generation) -> String {
        guard let max = g.contextMax else { return "—" }
        return "\(Format.tokens(g.contextTokens)) / \(Format.tokens(max))"
    }

    private func layer(_ d: StatsSnapshot.Drift) -> String {
        guard let total = d.layersTotal, let current = d.currentLayer else { return "—" }
        return "\(current + 1) / \(total)"
    }

    private func pressure(_ p: String) -> String {
        switch p {
        case "warning": String(localized: "Krap")
        case "critical": String(localized: "Kritiek")
        default: String(localized: "Normaal")
        }
    }
}

private struct StatRow: View {
    let label: LocalizedStringKey
    let value: String
    var accessibility: String?

    var body: some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).monospacedDigit()
        }
        .font(.callout)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(accessibility.map { Text($0) } ?? Text(label) + Text(verbatim: ": \(value)"))
    }
}

struct Sparkline: View {
    let values: [Double]

    var body: some View {
        Canvas { ctx, size in
            guard values.count > 1, let maxV = values.max(), maxV > 0 else {
                var base = Path()
                base.move(to: CGPoint(x: 0, y: size.height - 1))
                base.addLine(to: CGPoint(x: size.width, y: size.height - 1))
                ctx.stroke(base, with: .style(.quaternary), lineWidth: 1)
                return
            }
            let step = size.width / CGFloat(LiveStats.historyLength - 1)
            let offset = CGFloat(LiveStats.historyLength - values.count) * step
            var path = Path()
            for (i, v) in values.enumerated() {
                let p = CGPoint(x: offset + CGFloat(i) * step, y: size.height - 1 - CGFloat(v / maxV) * (size.height - 2))
                i == 0 ? path.move(to: p) : path.addLine(to: p)
            }
            ctx.stroke(path, with: .style(.tint), lineWidth: 1.5)
        }
        .accessibilityHidden(true)
    }
}
