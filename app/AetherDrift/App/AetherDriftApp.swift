import AppKit
import SwiftUI

@main
struct AetherDriftApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var state: AppState

    init() {
        let db: AppDatabase
        do {
            db = try AppDatabase.onDisk()
        } catch {
            // Never block startup on history: fall back to a session-only store.
            db = try! AppDatabase.inMemory()
        }
        let state = AppState(db: db)
        _state = State(initialValue: state)
        AppDelegate.state = state
        // Unit tests run inside the app; they must not start a real engine.
        if ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] == nil {
            state.start()
        }
    }

    var body: some Scene {
        Window("AetherDrift", id: "main") {
            RootView()
                .environment(state)
                .frame(minWidth: 760, minHeight: 480)
        }
        .defaultSize(width: 1180, height: 760)
        .commands { AppCommands(state: state) }

        Settings {
            SettingsView()
                .environment(state)
        }

        MenuBarExtra {
            MenuBarContent()
                .environment(state)
        } label: {
            MenuBarLabel()
                .environment(state)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    @MainActor static var state: AppState?

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated { Self.state?.supervisor.stop() }
    }

    /// Keeps serving other devices from the menu bar when the window closes (§8.9).
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        MainActor.assumeIsolated { !(Self.state?.keepRunning ?? false) }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        #if DEBUG
        if let dir = ProcessInfo.processInfo.environment["AETHER_SNAPSHOT_DIR"] {
            WindowSnapshotter.start(directory: URL(filePath: dir))
        }
        #endif
    }
}

#if DEBUG
/// Development aid: the app draws its own window to PNG files every second, so UI can be
/// checked without screen-recording permission.
@MainActor
enum WindowSnapshotter {
    static func start(directory: URL) {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        Task { @MainActor in
            for _ in 0..<600 {
                try? await Task.sleep(for: .seconds(1))
                // latest.png is the main window; sheets and panels get latest-1.png, latest-2.png, …
                let windows = NSApp.windows.filter { $0.isVisible && $0.contentView != nil }
                for (i, window) in windows.enumerated() {
                    guard let view = window.contentView?.superview,
                          let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { continue }
                    view.cacheDisplay(in: view.bounds, to: rep)
                    let data = rep.representation(using: .png, properties: [:])
                    try? data?.write(to: directory.appending(path: i == 0 ? "latest.png" : "latest-\(i).png"))
                }
            }
        }
    }
}
#endif

struct AppCommands: Commands {
    let state: AppState

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("Nieuw gesprek") { state.chat.newConversation() }
                .keyboardShortcut("n")
        }
        CommandMenu("Model") {
            Button("Kies model…") { state.showModelPalette = true }
                .keyboardShortcut("k")
            Button("Stop generatie") { state.chat.stop() }
                .keyboardShortcut(".")
                .disabled(!state.chat.isGenerating)
        }
        CommandGroup(after: .sidebar) {
            Button(state.showInspector ? "Verberg Live-paneel" : "Toon Live-paneel") {
                state.showInspector.toggle()
            }
            .keyboardShortcut("i")
            Button(state.showLog ? "Verberg log" : "Toon log") {
                state.showLog.toggle()
            }
            .keyboardShortcut("l")
            Button("Verbind een apparaat…") { state.showConnect = true }
                .keyboardShortcut("c", modifiers: [.command, .shift])
        }
    }
}
