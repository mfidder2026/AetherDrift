import AppKit
import CoreImage.CIFilterBuiltins
import SwiftUI
import SystemConfiguration

/// "Verbind een apparaat" (start-prompt-v2 §8.8): one switch, then URL, key, QR code and
/// examples. Sharing never reloads the model.
struct ConnectSheet: View {
    @Environment(AppState.self) private var state
    @Environment(\.dismiss) private var dismiss
    @State private var showKey = false
    @State private var confirmRegenerate = false
    @State private var example = Example.curl
    @State private var clients: [NetworkClient] = []
    @State private var busy = false

    enum Example: String, CaseIterable, Identifiable {
        case curl, python, other
        var id: String { rawValue }
        var title: LocalizedStringKey {
            switch self {
            case .curl: "curl"
            case .python: "Python"
            case .other: "Andere apps"
            }
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    if state.isSharing, let address = NetworkAddress.current(port: port) {
                        sharing(address)
                    } else if state.isSharing {
                        Label("Geen netwerkverbinding gevonden.", systemImage: "wifi.slash")
                            .foregroundStyle(.secondary)
                    } else {
                        Text("Zet delen aan om je Mac te gebruiken vanaf je telefoon, een andere computer of een app zoals Open WebUI. macOS kan één keer vragen of AetherDrift inkomende verbindingen mag accepteren.")
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .padding(20)
            }
        }
        .frame(width: 560, height: state.isSharing ? 640 : 200)
        .task(id: state.isSharing) { await pollClients() }
        .confirmationDialog("Nieuwe sleutel maken?", isPresented: $confirmRegenerate) {
            Button("Maak nieuwe sleutel", role: .destructive) {
                Task { await state.regenerateAPIKey() }
            }
        } message: {
            Text("Apparaten die de huidige sleutel gebruiken, moeten opnieuw verbinden.")
        }
    }

    private var port: Int {
        if case .ready(let port, _) = state.supervisor.status { return port }
        return EngineSupervisor.preferredPort
    }

    private var header: some View {
        HStack {
            Text("Verbind een apparaat").font(.headline)
            Spacer()
            Toggle("Delen op netwerk", isOn: Binding(
                get: { state.isSharing },
                set: { on in
                    busy = true
                    Task {
                        await state.setShare(on)
                        busy = false
                    }
                }
            ))
            .toggleStyle(.switch)
            .disabled(busy || !state.supervisor.isReady)
            Button("Gereed") { dismiss() }
                .keyboardShortcut(.defaultAction)
        }
        .padding(16)
    }

    @ViewBuilder
    private func sharing(_ address: NetworkAddress) -> some View {
        let key = state.apiKey ?? ""
        HStack(alignment: .top, spacing: 20) {
            if let qr = QRCode.image(for: address.webchatURL(key: key)) {
                Image(nsImage: qr)
                    .interpolation(.none)
                    .resizable()
                    .frame(width: 168, height: 168)
                    .accessibilityLabel("QR-code naar de webchat")
            }
            VStack(alignment: .leading, spacing: 8) {
                Text("Scan met je telefoon").font(.subheadline.bold())
                Text("De camera opent de webchat en verbindt meteen. Je telefoon moet op hetzelfde wifi-netwerk zitten.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }

        GroupBox("Adres") {
            VStack(alignment: .leading, spacing: 6) {
                CopyRow(label: "Base URL", value: address.baseURL)
                if let ip = address.ipBaseURL {
                    CopyRow(label: "Via IP-adres", value: ip)
                }
            }
            .padding(4)
        }

        GroupBox("API-sleutel") {
            HStack {
                Text(showKey ? key : String(repeating: "•", count: 24))
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Button(showKey ? "Verberg" : "Toon") { showKey.toggle() }
                Button("Kopieer") { Pasteboard.copy(key) }
                Button("Nieuw…") { confirmRegenerate = true }
            }
            .padding(4)
        }

        GroupBox("Voorbeeld") {
            VStack(alignment: .leading, spacing: 8) {
                Picker("", selection: $example) {
                    ForEach(Example.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                Text(snippet(address: address, key: showKey ? key : String(localized: "<sleutel>")))
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 6))
                HStack {
                    Spacer()
                    Button("Kopieer voorbeeld") { Pasteboard.copy(snippet(address: address, key: key)) }
                }
            }
            .padding(4)
        }

        GroupBox("Recente apparaten") {
            if clients.isEmpty {
                Text("Nog geen verbindingen.")
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(4)
            } else {
                VStack(spacing: 4) {
                    ForEach(clients) { c in
                        HStack {
                            Text(c.ip).font(.system(.body, design: .monospaced))
                            Spacer()
                            Text("\(c.requests) verzoeken · \(c.tokens ?? 0) tokens")
                                .foregroundStyle(.secondary)
                            if let t = c.lastRequest {
                                Text(Date(timeIntervalSince1970: t), style: .relative)
                                    .foregroundStyle(.secondary)
                                    .frame(width: 90, alignment: .trailing)
                            }
                        }
                        .font(.callout)
                    }
                }
                .padding(4)
            }
        }
    }

    private func snippet(address: NetworkAddress, key: String) -> String {
        switch example {
        case .curl:
            return """
            curl \(address.baseURL)/chat/completions \\
              -H "Authorization: Bearer \(key)" \\
              -H "Content-Type: application/json" \\
              -d '{"model": "default", "messages": [{"role": "user", "content": "Hallo!"}]}'
            """
        case .python:
            return """
            from openai import OpenAI

            client = OpenAI(base_url="\(address.baseURL)", api_key="\(key)")
            reply = client.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": "Hallo!"}],
            )
            print(reply.choices[0].message.content)
            """
        case .other:
            return """
            Base URL:   \(address.baseURL)
            API key:    \(key)
            Model:      default

            \(String(localized: "Werkt met elke OpenAI-compatibele app (Open WebUI, LibreChat, Continue, …)."))
            \(String(localized: "Ollama-clients gebruiken \(address.root) met dezelfde sleutel."))
            """
        }
    }

    private func pollClients() async {
        guard state.isSharing else { return }
        while !Task.isCancelled {
            if let list = try? await state.supervisor.client?.clients() { clients = list }
            try? await Task.sleep(for: .seconds(2))
        }
    }
}

private struct CopyRow: View {
    let label: LocalizedStringKey
    let value: String

    var body: some View {
        HStack {
            Text(label).foregroundStyle(.secondary).frame(width: 90, alignment: .leading)
            Text(value).font(.system(.body, design: .monospaced)).textSelection(.enabled)
            Spacer()
            Button("Kopieer") { Pasteboard.copy(value) }
        }
    }
}

enum Pasteboard {
    @MainActor static func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

/// How other devices reach this Mac.
struct NetworkAddress: Equatable {
    let hostName: String  // Bonjour name, e.g. "studio.local"
    let ip: String?
    let port: Int

    var root: String { "http://\(hostName):\(port)" }
    var baseURL: String { "\(root)/v1" }
    var ipBaseURL: String? { ip.map { "http://\($0):\(port)/v1" } }

    /// The QR code uses the IP address when there is one: not every phone resolves ".local"
    /// names (Android in particular). The key sits in the fragment, so it is never sent to
    /// the server or written to a log.
    func webchatURL(key: String) -> String {
        let host = ip ?? hostName
        let fragment = key.addingPercentEncoding(withAllowedCharacters: .alphanumerics.union(.init(charactersIn: "-_"))) ?? key
        return "http://\(host):\(port)/#key=\(fragment)"
    }

    static func current(port: Int) -> NetworkAddress? {
        let local = SCDynamicStoreCopyLocalHostName(nil) as String?
        let ip = primaryIPv4()
        guard local != nil || ip != nil else { return nil }
        return NetworkAddress(hostName: local.map { "\($0).local" } ?? (ip ?? "localhost"), ip: ip, port: port)
    }

    /// First IPv4 address on an active Wi-Fi or Ethernet interface (en0, en1, …).
    static func primaryIPv4() -> String? {
        var head: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&head) == 0, let first = head else { return nil }
        defer { freeifaddrs(head) }
        var candidates: [(name: String, ip: String)] = []
        for ptr in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let ifa = ptr.pointee
            guard let addr = ifa.ifa_addr, addr.pointee.sa_family == UInt8(AF_INET),
                  ifa.ifa_flags & UInt32(IFF_UP) != 0, ifa.ifa_flags & UInt32(IFF_LOOPBACK) == 0 else { continue }
            let name = String(cString: ifa.ifa_name)
            guard name.hasPrefix("en") else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(addr, socklen_t(addr.pointee.sa_len), &host, socklen_t(host.count),
                           nil, 0, NI_NUMERICHOST) == 0 {
                let ip = String(decoding: host.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
                if !ip.hasPrefix("169.254.") { candidates.append((name, ip)) }
            }
        }
        return candidates.sorted { $0.name < $1.name }.first?.ip
    }
}

enum QRCode {
    static func image(for text: String) -> NSImage? {
        let filter = CIFilter.qrCodeGenerator()
        filter.message = Data(text.utf8)
        filter.correctionLevel = "M"
        guard let output = filter.outputImage?.transformed(by: CGAffineTransform(scaleX: 8, y: 8)) else { return nil }
        let rep = NSCIImageRep(ciImage: output)
        let image = NSImage(size: rep.size)
        image.addRepresentation(rep)
        return image
    }
}
