import SwiftUI

/// One cell per layer (grouped above 128 layers). Fill = how much of the layer is in RAM,
/// outline = the layer being computed, soft pulse = being read from the SSD
/// (start-prompt-v2 §8.6). Only real engine data; no decorative motion.
struct DriftMap: View {
    let residency: [Double]
    let current: Int?
    var compact = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 15, paused: reduceMotion || !hasLoading)) { context in
            Canvas { ctx, size in
                draw(ctx, size: size, phase: context.date.timeIntervalSinceReferenceDate)
            }
        }
        .frame(height: compact ? 5 : 26)
        .accessibilityElement()
        .accessibilityLabel(accessibilityText)
    }

    private var cells: [Double] {
        let max = 128
        guard residency.count > max else { return residency }
        let group = Int((Double(residency.count) / Double(max)).rounded(.up))
        return stride(from: 0, to: residency.count, by: group).map { start in
            let slice = residency[start..<min(start + group, residency.count)]
            return slice.reduce(0, +) / Double(slice.count)
        }
    }

    private var groupSize: Int {
        residency.count > 128 ? Int((Double(residency.count) / 128).rounded(.up)) : 1
    }

    private var hasLoading: Bool { residency.contains { $0 > 0 && $0 < 1 } }

    private func draw(_ ctx: GraphicsContext, size: CGSize, phase: Double) {
        let values = cells
        guard !values.isEmpty else { return }
        let gap: CGFloat = compact ? 1 : 2
        let w = (size.width - gap * CGFloat(values.count - 1)) / CGFloat(values.count)
        let currentCell = current.map { $0 / groupSize }
        let pulse = reduceMotion ? 0.6 : 0.45 + 0.35 * (sin(phase * 6) + 1) / 2
        for (i, v) in values.enumerated() {
            let rect = CGRect(x: CGFloat(i) * (w + gap), y: 0, width: w, height: size.height)
            let path = Path(roundedRect: rect, cornerRadius: compact ? 1 : 2)
            ctx.fill(path, with: .color(.secondary.opacity(0.18)))
            if v >= 1 {
                ctx.fill(path, with: .color(.accentColor))
            } else if v > 0 {
                ctx.fill(path, with: .color(.accentColor.opacity(pulse)))
            }
            if i == currentCell {
                ctx.stroke(path, with: .color(.primary), lineWidth: compact ? 1 : 1.5)
            }
        }
    }

    private var accessibilityText: String {
        let inRam = residency.filter { $0 >= 1 }.count
        let layer = current.map { ", huidige laag \($0 + 1)" } ?? ""
        return "Drift-kaart: \(inRam) van \(residency.count) lagen in het geheugen\(layer)"
    }
}

/// Real forward-pass progress under an answer that is still coming in (Drift mode only).
struct LiveDriftStrip: View {
    @Environment(AppState.self) private var state

    var body: some View {
        if let s = state.live.snapshot, s.model.mode?.hasPrefix("drift") == true, !s.drift.layerResidency.isEmpty {
            DriftMap(residency: s.drift.layerResidency, current: s.drift.currentLayer, compact: true)
                .frame(maxWidth: 320)
                .help("Laag \((s.drift.currentLayer ?? 0) + 1) van \(s.drift.layersTotal ?? 0)")
        }
    }
}
