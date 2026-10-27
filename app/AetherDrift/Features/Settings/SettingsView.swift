import SwiftUI

struct SettingsView: View {
    @Environment(AppState.self) private var state
    @State private var autoReserve = true
    @State private var launchAtLogin = LoginItem.isEnabled
    @State private var loginError: String?
    @State private var osReserve = 16.0
    @State private var capWeights = false
    @State private var weightsCap = 8.0

    var body: some View {
        Form {
            Section("Modus") {
                Picker("Modus", selection: Binding(
                    get: { state.settings?.mode ?? "balans" },
                    set: { mode in Task { await state.setMode(mode) } }
                )) {
                    Text("Snel").tag("snel")
                    Text("Balans").tag("balans")
                    Text("Lange context").tag("lange_context")
                }
                .pickerStyle(.radioGroup)
                Text("Snel geeft het model zoveel mogelijk geheugen. Lange context ruimt plaats in voor lange gesprekken en documenten.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Geavanceerd") {
                Toggle("Reserve voor macOS automatisch", isOn: $autoReserve)
                    .onChange(of: autoReserve) { _, on in
                        Task { await state.updateOSReserve(on ? nil : osReserve) }
                    }
                HStack {
                    Slider(value: $osReserve, in: 4...48, step: 1) {
                        Text("Reserve voor macOS")
                    } onEditingChanged: { editing in
                        if !editing { Task { await state.updateOSReserve(osReserve) } }
                    }
                    .disabled(autoReserve)
                    Text("\(Int(osReserve)) GB").monospacedDigit().frame(width: 48, alignment: .trailing)
                }
                Text("Geheugen dat altijd vrij blijft voor macOS en je andere apps. Automatisch is een kwart van het RAM, tussen 4 en 16 GB. Geldt na opnieuw laden van het model.")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("Begrens geheugen voor modelgewichten", isOn: $capWeights)
                    .onChange(of: capWeights) { _, on in
                        Task { await state.updateWeightsBudget(on ? weightsCap : nil) }
                    }
                if capWeights {
                    Stepper(value: $weightsCap, in: 1...512, step: 1) {
                        Text("Maximaal \(Int(weightsCap)) GB")
                    } onEditingChanged: { editing in
                        if !editing { Task { await state.updateWeightsBudget(weightsCap) } }
                    }
                }
                Text("Past een model niet binnen deze grens, dan draait het via Drift: een deel blijft in RAM, de rest stroomt van de SSD.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Algemeen") {
                Toggle("Snelchat met ⌥Spatie", isOn: Binding(
                    get: { state.quickChatEnabled }, set: { state.quickChatEnabled = $0 }
                ))
                Toggle("Blijf actief in de menubalk na sluiten van het venster", isOn: Binding(
                    get: { state.keepRunning }, set: { state.keepRunningSetting = $0 }
                ))
                Text("Standaard aan zolang delen aan staat, zodat andere apparaten het model kunnen blijven gebruiken.")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("Start bij inloggen", isOn: $launchAtLogin)
                    .onChange(of: launchAtLogin) { _, on in
                        do {
                            try LoginItem.set(on)
                        } catch {
                            loginError = error.localizedDescription
                            launchAtLogin = LoginItem.isEnabled
                        }
                    }
                if let loginError {
                    Text(loginError).font(.caption).foregroundStyle(.red)
                }
            }
            Section("Engine") {
                LabeledContent("Status", value: statusText)
                Button("Herstart engine") { state.supervisor.restart() }
                Button("Toon onboarding opnieuw") { state.onboardingDone = false }
            }
        }
        .formStyle(.grouped)
        .frame(width: 480)
        .onAppear {
            if let gb = state.settings?.osReserveGb {
                autoReserve = false
                osReserve = gb
            } else {
                osReserve = state.system?.osReserveAutoGb ?? 16
            }
            if let cap = state.settings?.weightsBudgetGb {
                capWeights = true
                weightsCap = cap
            }
        }
    }

    private var statusText: String {
        switch state.supervisor.status {
        case .ready(let port, let external):
            external ? String(localized: "Actief op poort \(port) (extern)") : String(localized: "Actief op poort \(port)")
        case .installing: String(localized: "Installeren…")
        case .starting: String(localized: "Starten…")
        case .restarting(let n): String(localized: "Herstarten (poging \(n))")
        case .failed: String(localized: "Fout")
        case .idle: String(localized: "Uit")
        }
    }
}
