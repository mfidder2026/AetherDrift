import Foundation
import Markdown

/// Rendered block of a chat message.
indirect enum MarkdownBlock: Equatable, Sendable {
    case paragraph(AttributedString)
    case heading(level: Int, AttributedString)
    case code(language: String?, String)
    case list(ordered: Bool, start: Int, items: [[MarkdownBlock]])
    case quote([MarkdownBlock])
    case table(header: [AttributedString], rows: [[AttributedString]])
    case rule
}

enum MarkdownRenderer {
    static func blocks(from text: String) -> [MarkdownBlock] {
        let document = Document(parsing: text, options: [.disableSmartOpts])
        return document.children.compactMap(block)
    }

    private static func block(_ markup: Markup) -> MarkdownBlock? {
        switch markup {
        case let p as Paragraph:
            return .paragraph(inlines(p))
        case let h as Heading:
            return .heading(level: h.level, inlines(h))
        case let c as CodeBlock:
            var code = c.code
            if code.hasSuffix("\n") { code.removeLast() }
            return .code(language: c.language.flatMap { $0.isEmpty ? nil : $0 }, code)
        case let l as UnorderedList:
            return .list(ordered: false, start: 1, items: l.listItems.map { $0.children.compactMap(block) })
        case let l as OrderedList:
            return .list(ordered: true, start: Int(l.startIndex),
                         items: l.listItems.map { $0.children.compactMap(block) })
        case let q as BlockQuote:
            return .quote(q.children.compactMap(block))
        case let t as Markdown.Table:
            let header = t.head.cells.map { inlines($0) }
            let rows = t.body.rows.map { row in Array(row.cells.map { inlines($0) }) }
            return .table(header: Array(header), rows: Array(rows))
        case is ThematicBreak:
            return .rule
        case let html as HTMLBlock:
            return .paragraph(AttributedString(html.rawHTML))
        default:
            let plain = markup.format()
            return plain.isEmpty ? nil : .paragraph(AttributedString(plain))
        }
    }

    static func inlines(_ markup: Markup) -> AttributedString {
        var out = AttributedString()
        for child in markup.children { out += inline(child, intent: []) }
        return out
    }

    private static func inline(_ markup: Markup, intent: InlinePresentationIntent) -> AttributedString {
        func styled(_ s: String, _ extra: InlinePresentationIntent = []) -> AttributedString {
            var a = AttributedString(s)
            let combined = intent.union(extra)
            if !combined.isEmpty { a.inlinePresentationIntent = combined }
            return a
        }
        func children(_ m: Markup, _ extra: InlinePresentationIntent) -> AttributedString {
            var a = AttributedString()
            for c in m.children { a += inline(c, intent: intent.union(extra)) }
            return a
        }
        switch markup {
        case let t as Markdown.Text: return styled(t.string)
        case is SoftBreak: return styled(" ")
        case is LineBreak: return styled("\n")
        case let c as InlineCode: return styled(c.code, .code)
        case let e as Emphasis: return children(e, .emphasized)
        case let s as Strong: return children(s, .stronglyEmphasized)
        case let s as Strikethrough: return children(s, .strikethrough)
        case let l as Markdown.Link:
            var a = children(l, [])
            if let dest = l.destination, let url = URL(string: dest) { a.link = url }
            return a
        case let i as Markdown.Image: return styled(i.plainText)
        case let h as InlineHTML: return styled(h.rawHTML)
        default: return children(markup, [])
        }
    }
}

/// Incremental rendering for streamed text: completed blocks are parsed once and cached;
/// only the trailing, still-open block is parsed again on each update.
struct StreamingMarkdown {
    private(set) var stableBlocks: [MarkdownBlock] = []
    private var stableEnd = 0  // UTF-8 offset up to which stableBlocks are final
    private var source = ""
    private var fenceOpen = false

    mutating func update(_ text: String) -> [MarkdownBlock] {
        if text.utf8.count < stableEnd || !text.utf8.starts(with: source.utf8.prefix(stableEnd)) {
            self = StreamingMarkdown()
        }
        source = text
        if let boundary = advanceBoundary(in: text), boundary > stableEnd {
            let chunk = Self.slice(text, from: stableEnd, to: boundary)
            stableBlocks += MarkdownRenderer.blocks(from: chunk)
            stableEnd = boundary
        }
        let tail = Self.slice(text, from: stableEnd, to: text.utf8.count)
        return stableBlocks + MarkdownRenderer.blocks(from: tail)
    }

    /// Scans complete lines after stableEnd; returns the offset just after the last blank
    /// line that is outside a code fence.
    private mutating func advanceBoundary(in text: String) -> Int? {
        let bytes = Array(text.utf8)
        var lineStart = stableEnd
        var fence = fenceOpen
        var boundary: Int?
        var fenceAtBoundary = fence
        var i = stableEnd
        while i < bytes.count {
            if bytes[i] == UInt8(ascii: "\n") {
                let line = bytes[lineStart..<i]
                let trimmed = line.drop { $0 == UInt8(ascii: " ") }
                if trimmed.starts(with: Array("```".utf8)) || trimmed.starts(with: Array("~~~".utf8)) {
                    fence.toggle()
                } else if !fence && line.allSatisfy({ $0 == UInt8(ascii: " ") || $0 == UInt8(ascii: "\r") }) {
                    boundary = i + 1
                    fenceAtBoundary = fence
                }
                lineStart = i + 1
            }
            i += 1
        }
        if boundary != nil { fenceOpen = fenceAtBoundary }
        return boundary
    }

    private static func slice(_ text: String, from: Int, to: Int) -> String {
        let u = text.utf8
        let a = u.index(u.startIndex, offsetBy: from)
        let b = u.index(u.startIndex, offsetBy: to)
        return String(decoding: u[a..<b], as: UTF8.self)
    }
}
