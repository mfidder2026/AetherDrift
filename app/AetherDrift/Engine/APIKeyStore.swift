import Foundation
import Security

/// The network API key, kept in the login Keychain (start-prompt-v2 §8.8). The engine gets it
/// over loopback after every start; it never appears on a command line or in a file.
enum APIKeyStore {
    private static let service = "app.fremen.AetherDrift"
    private static let account = "network-api-key"

    /// The stored key, created on first use.
    static func key() -> String {
        if let existing = read() { return existing }
        let fresh = generate()
        write(fresh)
        return fresh
    }

    /// Replaces the key; devices that used the old one must reconnect. Returns nil (and keeps
    /// the old key) when the Keychain refuses the new one: a key that is not stored would
    /// silently change back at the next launch.
    static func regenerate() -> String? {
        let fresh = generate()
        return write(fresh) ? fresh : nil
    }

    static func generate() -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        let status = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        precondition(status == errSecSuccess, "no secure random bytes")
        return Data(bytes).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    private static var baseQuery: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service,
         kSecAttrAccount as String: account]
    }

    private static func read() -> String? {
        var query = baseQuery
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    @discardableResult
    private static func write(_ key: String) -> Bool {
        let data = Data(key.utf8)
        let update = [kSecValueData as String: data]
        var status = SecItemUpdate(baseQuery as CFDictionary, update as CFDictionary)
        if status == errSecItemNotFound {
            var add = baseQuery
            add[kSecValueData as String] = data
            add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
            status = SecItemAdd(add as CFDictionary, nil)
        }
        return status == errSecSuccess
    }
}
