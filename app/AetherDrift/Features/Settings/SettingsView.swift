import SwiftUI

struct SettingsView: View {
    @Environment(AppState.self) private var state
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
                HStack {
                    Slider(value: $osReserve, in: 8...48, step: 2) {
                        Text("Reserve voor macOS")
                    } onEditingChanged: { editing in
                        if !editing { Task { await state.updateOSReserve(osReserve) } }
                    }
                    Text("\(Int(osReserve)) GB").monospacedDigit().frame(width: 48, alignment: .trailing)
                }
                Text("Geheugen dat altijd vrij blijft voor macOS en je andere apps. Geldt na opnieuw laden van het model.")
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
            Section("Engine") {
                LabeledContent("Status", value: statusText)
                Button("Herstart engine") { state.supervisor.restart() }
                Button("Toon onboarding opnieuw") { state.onboardingDone = false }
            }
        }
        .formStyle(.grouped)
        .frame(width: 480)
        .onAppear {
            osReserve = state.settings?.osReserveGb ?? 16
            if let cap = state.settings?.weightsBudgetGb {
                capWeights = true
                weightsCap = cap
            }
        }
    }

    private var statusText: String {
        switch state.supervisor.status {
        case .ready(let port, let external): "Actief op poort \(port)\(external ? " (extern)" : "")"
        case .installing: "Installeren…"
        case .starting: "Starten…"
        case .restarting(let n): "Herstarten (poging \(n))"
        case .failed: "Fout"
        case .idle: "Uit"
        }
    }
}
