import os
import SwiftUI

struct RootView: View {
    @Environment(AppState.self) private var state

    var body: some View {
        Group {
            if state.onboardingDone {
                MainView()
            } else {
                OnboardingView()
            }
        }
        .overlay(alignment: .top) { ToastStack() }
        .onAppear { StartupTiming.firstFrame() }
    }
}

/// Cold-start measurement (target < 1 s, start-prompt-v2 §9): process start → first frame.
enum StartupTiming {
    @MainActor private static var reported = false

    @MainActor static func firstFrame() {
        guard !reported else { return }
        reported = true
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.stride
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
        guard sysctl(&mib, 4, &info, &size, nil, 0) == 0 else { return }
        let start = info.kp_proc.p_starttime
        let started = Double(start.tv_sec) + Double(start.tv_usec) / 1e6
        let ms = (Date().timeIntervalSince1970 - started) * 1000
        Logger(subsystem: "app.fremen.AetherDrift", category: "startup")
            .info("first frame after \(Int(ms)) ms")
        #if DEBUG
        FileHandle.standardError.write(Data("startup: first frame after \(Int(ms)) ms\n".utf8))
        #endif
    }
}

struct MainView: View {
    @Environment(AppState.self) private var state

    var body: some View {
        @Bindable var state = state
        NavigationSplitView {
            ConversationList()
                .navigationSplitViewColumnWidth(min: 180, ideal: 230, max: 320)
        } detail: {
            VStack(spacing: 0) {
                EngineStatusBanner()
                ChatView()
                if state.showLog {
                    Divider()
                    LogConsole()
                        .transition(.move(edge: .bottom))
                }
            }
        }
        .inspector(isPresented: $state.showInspector) {
            LiveInspector()
                .inspectorColumnWidth(min: 220, ideal: 250, max: 320)
        }
        .toolbar {
            ToolbarItemGroup(placement: .principal) {
                ModelChip()
                ModePicker()
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    state.showInspector.toggle()
                } label: {
                    Label("Live", systemImage: "waveform.path.ecg")
                }
                .help("Live-paneel (⌘I)")
            }
        }
        .sheet(isPresented: $state.showModelPalette) {
            ModelPalette()
        }
    }
}

/// Engine installing / restarting / failed. Hidden when everything is fine.
struct EngineStatusBanner: View {
    @Environment(AppState.self) private var state

    var body: some View {
        switch state.supervisor.status {
        case .ready, .idle:
            EmptyView()
        case .starting:
            banner("Engine start…", icon: "bolt.horizontal", tint: .secondary, spinning: true)
        case .installing(let line):
            banner("Engine installeren (eenmalig) — \(line)", icon: "arrow.down.circle", tint: .blue, spinning: true)
        case .restarting(let attempt):
            banner("De engine wordt herstart (poging \(attempt)). Je gesprek is bewaard.",
                   icon: "arrow.clockwise", tint: .orange, spinning: true)
        case .failed(let message):
            banner("De engine kon niet starten. Nieuwe poging volgt automatisch.\n\(message.prefix(300))",
                   icon: "exclamationmark.triangle", tint: .red, spinning: false)
        }
    }

    private func banner(_ text: String, icon: String, tint: Color, spinning: Bool) -> some View {
        HStack(spacing: 8) {
            if spinning { ProgressView().controlSize(.small) } else { Image(systemName: icon) }
            Text(text).font(.callout).lineLimit(3)
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(tint.opacity(0.12))
        .foregroundStyle(tint == .secondary ? Color.primary : tint)
        .transition(.move(edge: .top).combined(with: .opacity))
    }
}

struct ToastStack: View {
    @Environment(AppState.self) private var state

    var body: some View {
        VStack(spacing: 8) {
            ForEach(state.toasts) { toast in
                HStack(alignment: .top, spacing: 10) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(toast.message).font(.callout.weight(.medium))
                        if let hint = toast.hint, !hint.isEmpty {
                            Text(hint).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    if let action = toast.action, let label = toast.actionLabel {
                        Button(label) {
                            state.perform(action)
                            state.dismiss(toast.id)
                        }
                        .controlSize(.small)
                    }
                    Button {
                        state.dismiss(toast.id)
                    } label: {
                        Image(systemName: "xmark").font(.caption)
                    }
                    .buttonStyle(.borderless)
                    .accessibilityLabel("Sluit melding")
                }
                .padding(12)
                .frame(maxWidth: 460, alignment: .leading)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
                .shadow(radius: 6, y: 2)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .padding(.top, 10)
        .animation(.snappy, value: state.toasts)
    }
}
