import Carbon.HIToolbox
import Foundation

/// A system-wide shortcut via Carbon's RegisterEventHotKey: works while other apps are in
/// front and needs no accessibility permission.
@MainActor
final class GlobalHotKey {
    private var ref: EventHotKeyRef?
    private var handler: EventHandlerRef?
    private let action: @MainActor () -> Void
    private static var registry: [UInt32: GlobalHotKey] = [:]
    private static var nextID: UInt32 = 1

    /// ⌥Space by default (start-prompt-v2 §8.9).
    init?(keyCode: UInt32 = UInt32(kVK_Space), modifiers: UInt32 = UInt32(optionKey),
          action: @escaping @MainActor () -> Void) {
        self.action = action
        let id = Self.nextID
        Self.nextID += 1

        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        let installed = InstallEventHandler(GetApplicationEventTarget(), { _, event, _ in
            var hk = EventHotKeyID()
            GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                              nil, MemoryLayout<EventHotKeyID>.size, nil, &hk)
            let id = hk.id
            MainActor.assumeIsolated { GlobalHotKey.registry[id]?.action() }
            return noErr
        }, 1, &spec, nil, &handler)
        guard installed == noErr else { return nil }

        let hotKeyID = EventHotKeyID(signature: OSType(0x4145_5448), id: id)  // "AETH"
        guard RegisterEventHotKey(keyCode, modifiers, hotKeyID, GetApplicationEventTarget(), 0, &ref) == noErr else {
            if let handler { RemoveEventHandler(handler) }
            return nil
        }
        Self.registry[id] = self
    }

    func unregister() {
        if let ref { UnregisterEventHotKey(ref) }
        if let handler { RemoveEventHandler(handler) }
        ref = nil
        handler = nil
        Self.registry = Self.registry.filter { $0.value !== self }
    }
}
