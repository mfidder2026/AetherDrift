# AetherDrift v2 — Ontwerp & startopdracht

**Repository:** `https://github.com/mfidder2026/AetherDrift.git`
**Platform:** macOS 14 of nieuwer, Apple Silicon. Referentiemachine: MacBook Pro M3/M4 Max met 128 GB unified memory.
**Referentiemodel:** GLM 5.3, MLX-formaat, 4-bit. De app zelf is model-agnostisch.
**Motto:** *Infinite Intelligence. Local Hardware.*
**Gaat boven alles:** snelheid en gebruikersgemak.

---

## 0. Voor de AI developer: zo gebruik je dit document

- Lees het hele document voordat je code schrijft.
- De beslissingen in §4 liggen vast. Wil je ervan afwijken, schrijf dan eerst een ADR in `doc/adr/` met reden en meetresultaten, en vraag akkoord.
- Werk de mijlpalen uit §12 in volgorde af. Begin pas aan de volgende als alle acceptatiecriteria van de huidige groen zijn.
- Lever per mijlpaal: werkende code, tests, een benchmarkrapport in `bench/reports/` en een korte status (wat werkt, gemeten cijfers, open punten).
- Prioriteiten: **P0** = nodig voor de mijlpaal, **P1** = hoort in v1, **P2** = na v1 (niet bouwen tenzij gevraagd).
- Stop en vraag bij: een wijziging van het API-contract (§7), een nieuwe externe dependency, of een prestatiedoel (§9) dat onhaalbaar blijkt.

---

## 1. De opdracht in het kort

AetherDrift is een native macOS-app waarmee je grote taalmodellen lokaal draait, ook als het model groter is dan het werkgeheugen.

- Past het model in het geheugen, dan draait het op volle snelheid, zonder enige extra overhead.
- Past het niet, dan neemt de **Drift Engine** het over: de gewichten blijven op de SSD, alleen wat nodig is staat in RAM, en het volgende stuk wordt al ingelezen terwijl het huidige wordt berekend.
- De Mac blijft altijd bruikbaar. Liever een antwoord netjes afbreken dan een Mac die vastloopt.
- Andere apparaten in het netwerk (laptop, telefoon) kunnen het model gebruiken via een OpenAI-compatibele API of een ingebouwde webchat.
- De gebruiker heeft nooit een terminal nodig. Installeren, model kiezen, chatten en delen gaan allemaal vanuit de app.

---

## 2. Ontwerpprincipes

1. **Snelste pad eerst.** Drift schakelt alleen in als het model niet in het geheugenbudget past. Anders draait het standaard mlx-lm-pad.
2. **Nooit bevriezen.** Er is een hard geheugenbudget en een geheugenbewaker die ingrijpt voordat macOS gaat swappen.
3. **Direct feedback.** Elke actie geeft binnen 100 ms zichtbare reactie. Alles wat langer duurt toont voortgang en is te annuleren.
4. **Nul terminal.** De app installeert, start en bewaakt de engine zelf.
5. **Slimme standaarden, verborgen knoppen.** De gebruiker kiest een modus (Snel, Balans, Lange context). Technische instellingen staan onder "Geavanceerd".
6. **Eén API.** De app praat met de engine via precies dezelfde API als externe clients. Er is geen apart intern protocol.
7. **Eerlijke cijfers.** Elk getal in de UI komt uit een echte meting. Geen animaties die activiteit suggereren die er niet is. Onbekend toont als "—".

---

## 3. Wat er verandert ten opzichte van v1

| v1 | v2 | Waarom |
|---|---|---|
| Oude lagen worden "teruggeschreven naar de SSD" | Gewichten zijn read-only. Evicten is alleen geheugen vrijgeven; er wordt nooit geschreven. | Halveert de I/O en voorkomt onnodige SSD-slijtage. |
| "Swap Rate: hoeveel data naar SSD wordt geschreven" | SSD-**lees**snelheid (GB/s) plus cache-hitrate | Lezen is de echte bottleneck; schrijven hoort 0 te zijn. |
| `MLXPagedKVCache` / "PagedMemory" voor lagen | Eigen weight-streaming-laag (Drift Engine). De KV-cache wordt los daarvan beheerd. | Een paged KV-cache gaat over de context, niet over modelgewichten. Voor zover bekend biedt mlx-lm geen kant-en-klare weight-swapping (verifiëren in M0). |
| "1500 GB op 128 GB met minimaal snelheidsverlies" | Realistisch rekenmodel per modeltype (§6.2), vooraf getoond in de app | Een dense model dat veel groter is dan RAM kost seconden per token. Bij MoE-modellen komt Drift pas echt tot zijn recht. |
| WebSocket + REST | OpenAI-compatibele REST + SSE-streaming, ook voor statistieken | Werkt direct met bestaande clients. Eén codepad voor app en buitenwereld. |
| Server open op het netwerk, zonder beveiliging | Standaard alleen `127.0.0.1`. Delen is één schakelaar, met API-sleutel en QR-code. | Veilig, en makkelijker verbinden. |
| `pip install` + `uvicorn` met de hand starten | App installeert en beheert de engine zelf | Gebruikersgemak. |
| Model met de hand in `models/` zetten | Downloaden vanuit de app, of een bestaand model kiezen of erin slepen | Gebruikersgemak. |
| Een logregel per swap-event | Geaggregeerde events in de UI (max ~20/s), volledig log naar bestand | Bij veel swaps per seconde loopt de UI anders vast. |
| Vaste 128 GB | Budget op basis van gemeten hardware | Werkt op elke Apple Silicon Mac. |
| Repo-structuur: "maak zelf" | Vastgelegd in §10 | Voorspelbaar voor iedereen die meewerkt. |

---

## 4. Vaste architectuurbeslissingen

| # | Beslissing | Reden |
|---|---|---|
| B1 | Twee processen: SwiftUI-app + Python-engine als sidecar | mlx-lm heeft de breedste modelondersteuning en laat snel itereren op de engine. Een crash of geheugenprobleem in de engine neemt de UI nooit mee. |
| B2 | Engine: Python 3.12, MLX + mlx-lm, FastAPI + uvicorn (één worker), beheerd met `uv` | Volwassen ecosysteem, reproduceerbare installatie via `uv.lock`. |
| B3 | Transport: HTTP + Server-Sent Events (SSE), ook voor statistieken en events | OpenAI-clients gebruiken SSE. Eenrichtingsstreams volstaan; annuleren gaat via een apart endpoint of door de stream te sluiten. |
| B4 | API: OpenAI-compatibel (P0), Ollama-compatibel (P1), eigen `/aether/v1/*` voor beheer en telemetrie | Elke bestaande chat-app werkt direct. |
| B5 | Standaard alleen op `127.0.0.1`. Netwerkdelen is opt-in en vereist een API-sleutel. | Veilig standaardgedrag. De macOS-firewallvraag verschijnt pas als de gebruiker zelf gaat delen. |
| B6 | Gewichten zijn read-only. Drift leest alleen. | Zie §3. De enige schrijfactie is de optionele, eenmalige repack (§6.5). |
| B7 | Eén generatie tegelijk, met een FIFO-wachtrij | Voorspelbaar geheugengebruik. Batching is P2. |
| B8 | Alle MLX-aanroepen gebeuren vanuit één generatorthread | MLX is niet ontworpen voor gelijktijdig gebruik vanuit meerdere threads. |
| B9 | App: Swift 6, SwiftUI, Observation (`@Observable`), async/await. Opslag in SQLite via GRDB. Markdown via `swift-markdown`. | Native, snel en voorspelbaar. Minimale dependencies. |
| B10 | Geen hardcoded hardware. RAM, SSD-snelheid en Metal-limieten worden gemeten. | Werkt op elke Mac, en verwachtingen kloppen. |

*Overwogen alternatief:* alles in Swift met `mlx-swift` in één proces. Dat is niet voor v1: de modelondersteuning loopt achter en de Drift Engine vraagt om snel experimenteren. Omdat de API de grens tussen app en engine is, kan de engine later zonder UI-wijzigingen worden vervangen.

---

## 5. Architectuuroverzicht

```
┌─────────────────────────── AetherDrift.app (Swift / SwiftUI) ───────────────────────────┐
│  UI: Chat · Modellen · Live · Log · Verbinden · Instellingen                            │
│  AppState (@Observable)     ConversationStore (GRDB)     EngineClient (HTTP + SSE)      │
│  EngineSupervisor: starten · handshake · health checks · herstarten                     │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │ HTTP / SSE via 127.0.0.1
┌────────────────────────────────────────────┴───────────── aether-engine (Python) ───────┐
│  API-laag (FastAPI): /v1/* (OpenAI) · /api/* (Ollama) · /aether/v1/* · / (webchat)      │
│  Scheduler (wachtrij, annuleren) ──▶ Generatorthread (enige eigenaar van MLX)           │
│  Drift Engine: WeightIndex · ResidencyManager · CachePolicy · Prefetcher · IOBackend    │
│  MemoryGuard                         Telemetry (tellers · ringbuffer · aggregatie 4 Hz) │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │ alleen lezen (pread / mmap)
                                SSD: modelbestanden (safetensors of repacked)

Andere apparaten ── HTTP, alleen als "Delen" aan staat, met Bearer-sleutel ──▶ API-laag
```

### 5.1 Threads in de engine

- **Asyncio-loop** (uvicorn): alleen HTTP en SSE. Nooit blokkerend werk.
- **Generatorthread**: verwerkt de wachtrij één verzoek tegelijk, doet alle MLX-aanroepen en zet tokens via `loop.call_soon_threadsafe` terug op de loop.
- **I/O-pool** (8–16 threads): leest gewichten van de SSD. `os.pread`/`os.preadv` geven de GIL vrij, dus dit is echt parallel.
- **Telemetrie-tik** (4 Hz): bundelt tellers tot een statistieken-snapshot.

### 5.2 Opstartvolgorde (geoptimaliseerd voor snelheid)

1. De app opent direct het venster met het laatste gesprek uit de lokale database (doel < 1 s). Er wordt niet op de engine gewacht.
2. Tegelijk start de `EngineSupervisor` de engine. Zodra die zijn `ready`-regel print (§7.5), abonneert de app zich op de stats- en eventstreams.
3. De app laadt het laatst gebruikte model op de achtergrond. De modelchip in de toolbar toont voortgang en een ETA.
4. De gebruiker kan meteen typen en versturen. Een bericht dat vertrekt voordat het model klaar is, krijgt de status "Wacht op model…" en gaat automatisch door zodra het model geladen is.
5. (P1) **Progressief laden:** ook bij een model dat in RAM past, start de engine in Drift-modus zodra de altijd-residente delen geladen zijn. Zodra alles binnen is, promoveert de engine naar Resident. Zo kan er gechat worden voordat het model volledig geladen is.

---

## 6. De Drift Engine (de kern)

### 6.1 Modi

De engine kiest de modus automatisch bij het laden. Onder "Geavanceerd" kan de gebruiker die overschrijven.

| Modus | Wanneer | Werking |
|---|---|---|
| **Resident** | Model + KV-budget past binnen het geheugenbudget | Standaard mlx-lm-pad met alle gewichten in RAM. Nul Drift-overhead. |
| **Drift-Dense** | Dense model dat niet past | Een vaste set lagen staat permanent in RAM; de rest stroomt elke token door een ringbuffer met prefetch. |
| **Drift-MoE** | MoE-model dat niet past | Gedeelde delen (embeddings, attention, norms, router, shared experts, lm_head) staan altijd in RAM. Routed experts zitten in een expert-cache die op aanvraag laadt. |

### 6.2 Realistische verwachtingen (rekenmodel)

Drift wordt begrensd door de SSD. Per token moet alles wat niet in RAM staat van de SSD komen:

```
tijd_per_token ≈ max( rekentijd , bytes_van_SSD_per_token / SSD_leessnelheid )
```

- **Dense:** bytes per token = modelgrootte − gepind deel.
  Rekenvoorbeeld: model van 150 GB, 100 GB gepind, SSD 6 GB/s → 50 GB / 6 GB/s ≈ **8 s per token**.
- **MoE:** bytes per token = (1 − hitrate) × expertbytes per token.
  Rekenvoorbeeld: 15 GB expertgewichten per token, hitrate 90 % → 1,5 GB → ≈ 0,25 s → **~4 tok/s**. Bij een hitrate van 70 % wordt dat ~1,3 tok/s.

Wat dit betekent voor het ontwerp:
- Een dense model dat flink groter is dan RAM wordt nooit snel. De app toont dat eerlijk vóór het laden (§8.4).
- Bij MoE bepaalt de **cache-hitrate** de snelheid. Dat is de belangrijkste metriek om te optimaliseren en te tonen.
- Het doel voor de engine is geen absoluut tok/s-getal maar **efficiëntie**: tijdens Drift wordt ≥ 80 % van de gemeten SSD-bandbreedte benut en overlapt het rekenwerk volledig met de I/O.
- De getallen hierboven zijn illustraties. De app rekent altijd met de gemeten SSD-snelheid en de gemeten hitrate.

### 6.3 Gewichtenindex en units

- Bij het eerste laden leest de engine alleen de **headers** van alle safetensors-bestanden (8 bytes lengte + JSON) en bouwt een index: per tensor het bestand, offset, lengte, dtype, shape, laagnummer en (bij MoE) expert-ID.
- Tensors worden gegroepeerd in **units**, de kleinste eenheid die geladen of vrijgegeven wordt:
  - Dense: één decoderlaag = één unit.
  - MoE: per laag één *kern-unit* (altijd resident) plus één unit per expert.
- In MLX-formaat staan experts meestal gestapeld in één tensor per laag (`[n_experts, …]`), inclusief de bijbehorende quantization-scales en -biases. Eén expert is dan een aaneengesloten slice op as 0: `offset = start + expert_id × stride`.
- De index wordt bewaard als `<modelmap>/.aether/index.json` en is ongeldig zodra bestandsgrootte of mtime verandert.

### 6.4 I/O-pad

Er zijn twee kandidaten, allebei achter één interface `IOBackend`. M0 bepaalt de keuze (ADR-002).

| | A. mmap + `madvise` | B. `pread` + `F_NOCACHE` (verwachte winnaar) |
|---|---|---|
| Werking | Bestanden mappen; `WILLNEED` om vooruit te laden, `DONTNEED` om vrij te geven | Lezen in vooraf gealloceerde, 16 KB-aligned buffers via de I/O-pool, daarna kopiëren naar een MLX-array |
| Voordeel | Simpel | Voorspelbaar geheugengebruik, volledige controle over timing |
| Nadeel | De page cache van het OS telt niet mee in ons budget en gedraagt zich onvoorspelbaar onder druk | Meer code |

Regels voor beide:
- Lees in grote blokken (≥ 4 MB, tenzij een unit kleiner is).
- Houd meerdere verzoeken tegelijk open (queue depth 8–32). Een NVMe-SSD haalt pas volle bandbreedte bij parallelle verzoeken.
- De kopie van buffer naar MLX-array is verwaarloosbaar: unified memory is ordes van grootte sneller dan de SSD.

### 6.5 Repack (P1, alleen als M0 winst laat zien)

Eenmalig de gewichten herschrijven naar een Drift-geoptimaliseerd bestand: per unit aaneengesloten, 16 KB-aligned, in gebruiksvolgorde. Eén unit is dan één leesactie.
- De UI toont dit als "Model optimaliseren voor Drift (eenmalig, ±X min)", met voortgang en een annuleerknop.
- Controleer vooraf de vrije schijfruimte. Het origineel blijft onaangetast.

### 6.6 Cachebeleid

**Drift-Dense: statisch pinnen, géén LRU.**
Bij een dense model worden de lagen elke token in dezelfde volgorde 0…N−1 doorlopen. LRU is dan het slechtst denkbare beleid: zodra de cache kleiner is dan het model, is de laag die je straks nodig hebt altijd net verwijderd (hitrate 0 %). Daarom:
- Lagen 0…P−1 blijven permanent in RAM (P zo groot als het budget toelaat).
- Lagen P…N−1 stromen door een ringbuffer van K slots met prefetch-diepte D (K ≥ D + 1).
- Terwijl de gepinde lagen rekenen, worden de eerste D gestreamde lagen al ingelezen. De SSD staat zo nooit stil.

**Drift-MoE: expert-cache met LFU en veroudering.**
- Score per expert = toegangsfrequentie die exponentieel vervalt (instelbare halfwaardetijd, startwaarde ~256 tokens).
- Evict de expert met de laagste score. Experts die in de huidige laagstap gebruikt worden, zijn gepind.
- Laad alle ontbrekende experts van een laag **tegelijk** (parallelle reads).
- **Prefill telt niet mee** voor de scores. Tijdens prefill wordt bijna elke expert geraakt, en anders verdringt prefill de werkset van het decoden.
- Verwerk prefill in grote chunks (standaard 2048 tokens) zodat elke expert per chunk maar één keer geladen hoeft te worden.
- Suggestie voor de implementatie (te valideren in M0/M4): per laag een vaste slot-tensor in RAM plus een tabel expert-ID → slot. De router-indices worden vóór `gather_qmm` naar slot-indices omgezet. Meet of slot-updates in MLX zonder volledige kopie gebeuren; zo niet, gebruik losse arrays per expert.
- P2: speculatieve prefetch. Voorspel de experts van laag L+1 door de router van L+1 toe te passen op de hidden state vóór laag L+1 berekend is.

### 6.7 Uitvoerlus

- **Resident:** gebruik `stream_generate` van mlx-lm direct.
- **Drift:** een eigen lus over de lagen. Let op de lazy evaluation van MLX: berekeningen gebeuren pas bij `mx.eval`. Bouw je de graaf over alle lagen op zonder tussentijdse eval, dan moeten alle gewichten tegelijk in het geheugen staan. Daarom per laag (of groep lagen):
  1. `ensure_resident(unit)`, wachtend op de prefetch indien nodig;
  2. de laag berekenen;
  3. `mx.eval(hidden)`;
  4. referenties naar ge-evicte gewichten loslaten.
- Houd nergens extra referenties naar gewichtarrays vast, anders komt het geheugen niet vrij. Zet in Drift-modus de MLX-bufferlimiet laag (`set_cache_limit`; de namespace verschilt per MLX-versie).
- Hergebruik de modeldefinities van mlx-lm voor de architectuur, maar beheer de gewichten zelf: zet per laag vóór gebruik de parameters (`module.update(...)`) en vervang ze na eviction door lege placeholders.
- Annuleren wordt tussen lagen gecontroleerd, dus stoppen gebeurt binnen één laagstap.

### 6.8 KV-cache en prompt-cache

- Het KV-budget volgt uit de gekozen modus (§8.5). Het aantal KV-bytes per token wordt bij het laden uit de modelconfig gehaald of gemeten (bij afwijkende attention-varianten klopt de standaardformule niet).
- Optioneel 8-bit KV-cache (mlx-lm ondersteunt gekwantiseerde KV): standaard aan in "Lange context", uit in de andere modi.
- **Prompt-cache via prefix-matching op tokens:** bewaar de KV van de laatste 3 sequenties (LRU, binnen het KV-budget) en hergebruik de langste gemeenschappelijke prefix. Er is geen sessie-ID nodig, dus dit werkt ook voor externe, stateless OpenAI-clients.
  Effect: bij een vervolgbericht worden alleen de nieuwe tokens verwerkt, en de wachttijd tot het eerste token groeit niet mee met de lengte van het gesprek.

### 6.9 Geheugenbudget en geheugenbewaker

```
gewichtenbudget = min(RAM_totaal − OS_reserve, Metal_werkset_limiet) − KV_budget − engine_overhead
```

- Standaardwaarden: OS-reserve 16 GB, engine-overhead 2 GB. Beide instelbaar onder "Geavanceerd".
- `Metal_werkset_limiet` = `recommendedMaxWorkingSetSize` van het Metal-device (in MLX via `device_info()`). Op macOS is dat standaard een deel van het totale RAM, niet alles.
- **Controle vóór het laden:** als het minimum (altijd-residente delen + 2 streaming-slots + minimaal KV) niet in het budget past, laadt de engine niet en geeft `model_too_large` terug met de benodigde en beschikbare GB.
- **Bewaking tijdens gebruik:** de engine leest elke 500 ms `kern.memorystatus_vm_pressure_level` en de swapgroei (`psutil.swap_memory()`). De app luistert daarnaast naar `DispatchSource.makeMemoryPressureSource` om het in de UI te tonen.

| Niveau | Actie van de engine | Wat de gebruiker ziet |
|---|---|---|
| Normaal | — | — |
| Waarschuwing (of swap groeit > 1 GB in 10 s) | Prefetch-diepte naar 1, cache in stappen van 10 % verkleinen tot het weer normaal is | Toast: "Geheugen krap — Drift-cache verkleind" |
| Kritiek | Huidige generatie stoppen (gedeeltelijk antwoord, `finish_reason: "memory"`), cache naar minimum | Melding met actie (§8.11) |

- P2: een begeleide optie om de Metal-wired-limiet te verhogen (`iogpu.wired_limit_mb`, vraagt om admin-rechten). Nooit automatisch.

### 6.10 Telemetrie

- Tellers in de engine worden direct bijgewerkt door de I/O-laag en de generator: exact, niet geschat.
- De aggregator maakt 4× per seconde een snapshot (schema in §7.4).
- Events gaan naar een ringbuffer (10.000 stuks) en een roterend logbestand. De SSE-stream naar de UI wordt begrensd: boven 20 events/s stuurt de engine samenvattingen (`summary`).
- "Swappen" betekent hier **lezen**. De schrijfsnelheid wordt getoond en hoort 0 te zijn; dat dient als controle.

---

## 7. API-contract

### 7.1 Algemeen

- Basis-URL: `http://127.0.0.1:8080`. De poort is instelbaar; is die bezet, dan pakt de engine de eerstvolgende vrije poort en meldt die in de handshake.
- JSON in UTF-8. Fouten hebben altijd deze vorm, met een machineleesbare `code` zodat de app er een vriendelijke melding van kan maken:
  ```json
  { "error": { "code": "model_too_large", "message": "…", "hint": "…" } }
  ```
- Authenticatie: verzoeken via loopback hebben geen sleutel nodig. Verzoeken via het netwerk vereisen `Authorization: Bearer <sleutel>`; zonder geldige sleutel volgt `401`.
- Geen wildcard-CORS.

### 7.2 OpenAI-compatibel (P0)

| Methode | Pad | Opmerking |
|---|---|---|
| GET | `/v1/models` | Geladen en lokaal beschikbare modellen. `default` is een alias voor het geladen model. |
| POST | `/v1/chat/completions` | `stream: true` geeft SSE (`data: {…}` per chunk, afgesloten met `data: [DONE]`). Parameters: `temperature`, `top_p`, `max_tokens`, `stop`, `seed`. Antwoord bevat `usage`. |
| POST | `/v1/completions` | P1 |

Vraagt een client om een ander model dan het geladen model, dan volgt `409 model_not_loaded`, tenzij de instelling "Clients mogen van model wisselen" aan staat (standaard uit, om de lokale gebruiker te beschermen).

### 7.3 Ollama-compatibel (P1)

`POST /api/chat`, `POST /api/generate` (het endpoint uit v1) en `GET /api/tags`, met NDJSON-streaming zoals Ollama.

### 7.4 AetherDrift-beheer en telemetrie (P0 tenzij anders vermeld)

| Methode | Pad | Doel |
|---|---|---|
| GET | `/health` | `{status, version, model_state}` |
| GET | `/aether/v1/system` | Chip, RAM, gemeten SSD-snelheid, Metal-werksetlimiet |
| POST | `/aether/v1/system/ssd-test` | Snelle SSD-test (≤ 5 s, tijdelijk bestand met `F_NOCACHE`, daarna verwijderd) |
| GET | `/aether/v1/models` | Lokale modellen + catalogus, elk met fit-inschatting (modus, verwachte tok/s) |
| POST | `/aether/v1/models/load` | `{id, mode?}`. Voortgang loopt via de eventstream. |
| POST | `/aether/v1/models/unload` | |
| POST | `/aether/v1/models/download` | `{repo}`. Hervatbaar; voortgang via de eventstream. |
| DELETE | `/aether/v1/models/download/{id}` | Download annuleren |
| POST | `/aether/v1/generations/{id}/cancel` | Generatie stoppen |
| GET | `/aether/v1/stats` | Eén snapshot |
| GET | `/aether/v1/stats/stream` | SSE, 4 Hz |
| GET | `/aether/v1/events/stream` | SSE, begrensd. `?since=<seq>` om na een herverbinding verder te gaan. |
| GET / PUT | `/aether/v1/config` | Instellingen (modus, budgetten, delen, poort) |
| GET | `/` | Ingebouwde webchat (P1, alleen als delen aan staat of via loopback) |

**Stats-snapshot:**
```json
{
  "ts": 1761492121.25,
  "model": { "id": "glm-5.3-q4", "state": "ready", "mode": "drift-moe" },
  "memory": {
    "budget_bytes": 103079215104, "weights_bytes": 96636764160, "kv_bytes": 3221225472,
    "system_used_bytes": 118111600640, "system_total_bytes": 137438953472,
    "swap_used_bytes": 0, "pressure": "normal"
  },
  "drift": {
    "layers_total": 92, "current_layer": 45,
    "layer_residency": [1.0, 1.0, 0.62, 0.58],
    "hit_rate": 0.91, "read_bytes_per_s": 5800000000, "read_p95_ms": 21,
    "write_bytes_per_s": 0
  },
  "generation": {
    "active": true, "decode_tps": 4.1, "prefill_tps": 210, "ttft_ms": 950,
    "context_tokens": 3120, "context_max": 32768, "queue_depth": 0
  }
}
```
`layer_residency` heeft één waarde per laag (ingekort in het voorbeeld): 0–1, bij dense 0 of 1, bij MoE het aandeel experts van die laag dat in de cache staat.

**Events:**
```json
{"seq": 1201, "ts": 1761492121.233, "type": "load",    "unit": "L11", "bytes": 1210000000, "ms": 188}
{"seq": 1202, "ts": 1761492121.240, "type": "evict",   "unit": "L10", "bytes": 1210000000}
{"seq": 1203, "ts": 1761492122.000, "type": "summary", "loads": 312, "evicts": 298, "bytes": 2100000000, "window_ms": 1000}
{"seq": 1204, "ts": 1761492125.512, "type": "pressure", "level": "warning", "action": "cache verkleind met 8 GB"}
```
Verder: `model_progress`, `download_progress`, `model_loaded`, `error`.

### 7.5 Handshake engine → app

De app start: `aether-engine --host 127.0.0.1 --port 8080 --parent-pid <pid> --data-dir <pad>`.
- De API-sleutel gaat via stdin naar de engine, **nooit** als command-line-argument (die zijn zichtbaar in `ps`).
- Zodra de engine klaar is, print die precies één JSON-regel op stdout:
  `{"event":"ready","port":8080,"version":"0.1.0","pid":12345}`
- Alle logging gaat naar stderr en het logbestand, nooit naar stdout.
- De engine bewaakt `--parent-pid` en stopt zelf als de app verdwenen is. Zo blijft er nooit een weesproces achter dat 100 GB vasthoudt.

---

## 8. App- en UX-ontwerp

### 8.1 Hoofdvenster

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ [● GLM 5.3 · Drift-MoE ▾]  [Balans ▾]                  [Verbind]  [Live ⌘I]  │
├────────────────┬──────────────────────────────────────┬──────────────────────┤
│ + Nieuw    ⌘N  │                                      │ RAM      94 / 128 GB │
│                │  Jij                                 │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░  │
│ Vandaag        │  Leg dynamic layer swapping uit.     │ Tokens/s   4,1 ▁▃▅▆▅ │
│ ▸ Drift uitleg │                                      │ SSD-lezen  5,8 GB/s  │
│ ▸ Swift-vraag  │  AetherDrift                         │ Cache-hit  91 %      │
│                │  Dynamic layer swapping betekent…▍   │ Laag       45 / 92   │
│ Gisteren       │  ▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▯▯▯▯▯▯▯▯▯▯▯▯      │ Context  3,1k / 32k  │
│ ▸ …            │                                      │                      │
│                │ ┌──────────────────────────────────┐ │ Drift-kaart          │
│                │ │ Stel een vraag…              [↑] │ │ ▮▮▮▮▮▮▮▮▮▮▮▯▯▯▯▯▯▯▯  │
│                │ └──────────────────────────────────┘ │ ▸ Log  ⌘L            │
└────────────────┴──────────────────────────────────────┴──────────────────────┘
```

- `NavigationSplitView`: gesprekken (links), chat (midden), Live-paneel als inspector (rechts, ⌘I).
- Toolbar: modelchip met statuspunt (groen = klaar, oranje = laden, grijs = geen model), moduskiezer, knop Verbind.
- Logconsole: inklapbaar paneel onderin (⌘L).

### 8.2 Eerste start (onboarding, max. 3 stappen)

1. **Welkom & check** (automatisch): chip, RAM, vrije schijfruimte en een snelle SSD-test (≤ 5 s). De engine-runtime installeert intussen op de achtergrond, met voortgang.
2. **Kies een model**: drie aanbevelingen voor déze Mac, elk met grootte, badge ("Past in RAM" / "Drift") en verwachte snelheid. Plus: "Ik heb al een model" (map kiezen, erin slepen, of automatisch gevonden in de Hugging Face-cache).
3. **Downloaden & starten**: hervatbare download met voortgang en ETA. (P1) Tijdens het downloaden van een groot model kan de gebruiker alvast chatten met een klein startmodel.

Doel: van eerste start tot eerste antwoord zonder één technische vraag.

### 8.3 Chat

- Enter = versturen, ⇧Enter = nieuwe regel, ⌘. = stoppen, ↑ in een leeg invoerveld = laatste bericht bewerken.
- Streaming met Markdown, codeblokken met kopieerknop en (P1) syntax highlighting.
- Onder elk antwoord een subtiele regel: `4,1 tok/s · 312 tokens · Drift-MoE`.
- Acties per bericht: kopiëren, opnieuw genereren, bewerken en opnieuw versturen.
- Gesprekken worden lokaal opgeslagen (SQLite). De titel bestaat uit de eerste woorden van het eerste bericht; geen extra modelaanroep. Zoeken in gesprekken (P1), exporteren als Markdown (P1).

### 8.4 Modelkiezer (⌘K)

- Een command palette met zoekveld. Per model: naam, grootte, status (geladen / gedownload / niet gedownload), badge en verwachte snelheid.
- De verwachte snelheid komt uit het rekenmodel (§6.2) met de gemeten SSD-snelheid. Na gebruik wordt die vervangen door het gemeten gemiddelde.
- Wisselen kost één klik. Het oude model wordt eerst ontladen, zodat er nooit twee tegelijk in het geheugen staan.
- Een modelmap op het venster slepen voegt het model toe.
- Catalogus (`engine/aether/models/catalog.json`): een gecureerde lijst met per model het HF-repo, de grootte, het type (dense/MoE) en het minimale RAM. Begin met een klein startmodel, een middelgroot model dat in 64 GB past, en GLM 5.3 Q4.

### 8.5 Modi

| Modus | Standaardcontext | KV | Bedoeld voor |
|---|---|---|---|
| **Snel** | 8k tokens | 16-bit | Zoveel mogelijk RAM voor gewichten → hoogste tok/s |
| **Balans** (standaard) | 32k tokens | 16-bit | Normaal gebruik |
| **Lange context** | Maximum dat past bij ≥ 50 % RAM voor gewichten | 8-bit | Lange documenten en gesprekken |

Onder "Geavanceerd": OS-reserve, gewichtenbudget, KV-budget, prefetch-diepte, aantal I/O-threads, prefill-chunkgrootte, modus forceren. Elke instelling heeft een uitleg van één zin.

### 8.6 Live-paneel en Drift-kaart

- Velden: RAM (engine / systeem / totaal), tokens/s (decode) met sparkline over 60 s, prefill-tokens/s, tijd tot eerste token, SSD-lezen (GB/s) met sparkline, cache-hitrate, huidige laag / totaal, context gebruikt / max, geheugendruk, wachtrij.
- **Drift-kaart:** één cel per laag (bij meer dan 128 lagen gegroepeerd). De vulling van een cel is `layer_residency`, de huidige laag krijgt een omlijning, en cellen die nu laden pulseren zacht (uit bij "Beperk beweging").
- **Onder het bericht dat nog binnenkomt** staat in Drift-modus een compacte strip van één regel met de echte voortgang van de forward pass. Als één token seconden duurt, ziet de gebruiker zo dat er gewerkt wordt. In Resident-modus is er alleen een cursor, zonder nep-animatie.
- Het paneel ververst maximaal 4× per seconde.

### 8.7 Logconsole

- Kolommen: tijd, type, bericht. Filters (Laden / Vrijgeven / Waarschuwing / Fout), pauzeren, zoeken, kopiëren, "Open logbestand".
- De UI toont maximaal 2.000 regels; het volledige log staat op schijf.
- Voorbeeldregels:
  ```
  14:02:01.233  LOAD    laag 11 · 1,21 GB · 188 ms (6,4 GB/s)
  14:02:01.240  EVICT   laag 10 · 1,21 GB vrijgegeven
  14:02:02.000  SUMMARY 312 expert-loads · 2,1 GB in 1 s · hitrate 91 %
  14:02:05.512  WARN    Geheugendruk: Drift-cache verkleind met 8 GB
  ```

### 8.8 Verbind een apparaat

- Een sheet met één schakelaar: **Delen op netwerk**. Vóór de eerste keer legt één zin uit dat macOS om toestemming voor netwerkverbindingen zal vragen.
- Als delen aan staat:
  - URL: `http://<computernaam>.local:8080/v1`, met het IP-adres als alternatief.
  - API-sleutel: tonen, kopiëren, opnieuw genereren. Opgeslagen in de Keychain.
  - **QR-code** naar de webchat: `http://<computernaam>.local:8080/#key=<sleutel>`. De sleutel staat in het fragment en komt dus nooit in serverlogs. Telefoon scannen → meteen chatten, zonder app te installeren.
  - Codevoorbeelden in tabbladen: curl, Python (`openai`-pakket), en "Andere apps" (base URL + sleutel voor elke OpenAI-compatibele app).
  - Lijst met recente clients: IP, laatste verzoek, aantal tokens.
- Delen aan- of uitzetten mag het geladen model nooit herladen.
- P2: Bonjour-advertentie (`_aetherdrift._tcp`) zodat clients de Mac automatisch vinden.

### 8.9 Menubalk en snelle chat (P1)

- `MenuBarExtra` met status (model, tok/s, delen aan/uit) en de acties Nieuwe chat, Stop generatie, Delen aan/uit en Open AetherDrift.
- "Blijf actief na sluiten van het venster" (standaard aan als delen aan staat), zodat de server blijft draaien.
- Globale sneltoets ⌥Spatie opent een zwevend snelchatvenster (`NSPanel`, à la Spotlight). Enter verstuurt, Esc sluit.
- Starten bij inloggen via `SMAppService`.

### 8.10 Sneltoetsen

| Toets | Actie |
|---|---|
| ⌘N | Nieuw gesprek |
| ⌘K | Modelkiezer |
| ⌘. | Generatie stoppen |
| ⌘I | Live-paneel tonen/verbergen |
| ⌘L | Logconsole tonen/verbergen |
| ⌘⇧C | Verbind een apparaat |
| ⌘, | Instellingen |
| ⌥Spatie | Snelchat (globaal, P1) |

### 8.11 Foutmeldingen: begrijpelijke taal en altijd een actie

| Code | Melding | Actie |
|---|---|---|
| `model_too_large` | "Dit model heeft minimaal 140 GB werkgeheugen nodig. Op deze Mac is 112 GB beschikbaar." | Kies een kleiner model |
| `memory` | "Je Mac had het geheugen nodig, dus het antwoord is ingekort." | Opnieuw in modus Snel |
| `engine_restarted` | "De engine is herstart. Je gesprek is bewaard." | (automatisch) |
| `port_in_use` | "Poort 8080 is bezet. AetherDrift gebruikt nu poort 8081." | Kopieer nieuwe URL |
| `download_interrupted` | "Download onderbroken. Die gaat verder zodra je weer online bent." | (automatisch) |
| `disk_full` | "Niet genoeg schijfruimte: X GB nodig, Y GB vrij." | Open opslagbeheer |
| `model_not_loaded` | "Er is nog geen model geladen." | Kies een model |

### 8.12 Toegankelijkheid en taal

- VoiceOver-labels voor alle statistieken (bijvoorbeeld "RAM-gebruik 94 van 128 gigabyte"), volledige toetsenbordnavigatie, respect voor "Beperk beweging" en voldoende contrast in licht en donker.
- UI-teksten via een String Catalog in het Nederlands en Engels. Getallen worden opgemaakt volgens de systeemtaal ("1,2 GB" in het Nederlands).

### 8.13 Engine-beheer door de app (EngineSupervisor)

- Start de engine volgens §7.5 en wacht op de `ready`-regel (time-out 15 s, met duidelijke melding).
- Health check elke 2 s. Na 3 gemiste checks volgt een herstart met backoff (1, 2, 5, 10, 30 s). Tijdens een herstart blijft de UI bruikbaar en worden berichten in de wachtrij gezet.
- Ontwikkelmodus: draait er al een engine op de ingestelde poort (bijvoorbeeld via `scripts/dev-engine.sh`), dan gebruikt de app die in plaats van een eigen engine te starten.
- **Runtime-installatie:**
  - M2: de app-bundle bevat het `uv`-binary, de engine-broncode en `uv.lock`. Bij de eerste start draait `uv sync --frozen` in `~/Library/Application Support/AetherDrift/runtime/` (eenmalig internet nodig; voortgang zichtbaar in de onboarding).
  - M6: een volledig zelfstandige runtime in de bundle (standalone Python + wheels), werkt offline. Alle binaries en `.so`-bestanden worden gesigneerd voor notarisatie.

### 8.14 Prestatieregels voor de SwiftUI-code

- Tokens komen binnen via een `AsyncStream` en worden gebufferd. De UI wordt hooguit één keer per frame bijgewerkt, niet per token.
- Markdown: afgeronde blokken worden één keer geparsed en gecachet. Alleen het laatste, nog open blok wordt opnieuw geparsed.
- `LazyVStack` met stabiele ID's. Automatisch scrollen alleen als de gebruiker al onderaan staat.
- Statistieken staan in een eigen `@Observable`-object. Alleen de views van het Live-paneel observeren dat, zodat updates op 4 Hz de chat niet opnieuw laten renderen.
- Geen werk op de main thread dat langer duurt dan 2 ms. Database en netwerk draaien altijd async.

---

## 9. Prestatiedoelen

| Meting | Doel | Hoe meten |
|---|---|---|
| Koude start app → bruikbaar venster | < 1 s | `os_signpost` + Instruments |
| Engine-start → `ready` (zonder model) | < 3 s | Tijdstempel van de handshake |
| Toetsaanslag → zichtbaar | < 16 ms | Instruments |
| Streaming-weergave | Geen haperingen bij 100 tok/s | Instruments (Hitches) |
| Resident-overhead t.o.v. `mlx_lm.generate` | ≤ 5 % minder tok/s | `bench/generate.py` |
| Drift: benutting van de SSD-bandbreedte | ≥ 80 % van het gemeten maximum | `bench/generate.py --drift` |
| Model laden (Resident) | ≥ 80 % van de SSD-bandbreedte | Load-events |
| Vervolgbericht | Alleen nieuwe tokens worden geprefilld | Test op prompt-cache-hits |
| Stoppen (⌘.) | < 250 ms (Resident), ≤ 1 laagstap (Drift) | Integratietest |
| Vertraging van statistieken | ≤ 250 ms | `ts` in de snapshot |
| CPU-gebruik app: in rust / tijdens streaming | < 1 % / < 15 % | Instruments |
| Geheugen engine | Nooit boven budget + 5 %, geen swapgroei | Test van de geheugenbewaker |

---

## 10. Repositorystructuur

```
AetherDrift/
├── README.md                    # Voor gebruikers: installeren en gebruiken
├── start-prompt-v2.md           # Dit document
├── doc/
│   ├── ARCHITECTURE.md          # Actuele architectuur (bijhouden bij elke afwijking)
│   ├── adr/                     # Architecture Decision Records (001-…)
│   └── api.md                   # API-referentie, gegenereerd uit OpenAPI
├── app/                         # macOS-app
│   ├── AetherDrift.xcodeproj
│   ├── AetherDrift/
│   │   ├── App/                 # Entry point, AppState, Commands (sneltoetsen)
│   │   ├── Engine/              # EngineSupervisor, EngineClient, SSE-parser, API-types
│   │   ├── Features/
│   │   │   ├── Chat/            # ChatView, MessageView, Composer, StreamingText
│   │   │   ├── Models/          # ModelPalette, ModelCard, Downloads
│   │   │   ├── Live/            # LiveInspector, DriftMap, Sparkline
│   │   │   ├── Log/             # LogConsole
│   │   │   ├── Connect/         # ConnectSheet, QR-code
│   │   │   ├── Onboarding/
│   │   │   └── Settings/
│   │   ├── Persistence/         # GRDB-database, ConversationStore
│   │   └── Resources/           # Localizable.xcstrings, Assets
│   ├── AetherDriftTests/
│   └── AetherDriftUITests/
├── engine/                      # Python-engine
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── aether/
│   │   ├── __main__.py          # CLI: aether-engine
│   │   ├── config.py
│   │   ├── api/                 # app.py, openai.py, ollama.py, aether.py, auth.py, sse.py
│   │   ├── runtime/             # scheduler.py, generator.py, sampling.py, prompt_cache.py
│   │   ├── drift/               # index.py, units.py, residency.py, policy.py, prefetch.py, io_backend.py, repack.py
│   │   ├── memory/              # budget.py, guard.py
│   │   ├── models/              # registry.py, loader.py, download.py, catalog.json
│   │   ├── telemetry/           # counters.py, events.py, stats.py
│   │   └── web/                 # index.html: webchat zonder build-stap
│   └── tests/
├── bench/
│   ├── ssd.py                   # Sequentieel + random lezen, blokgroottes, queue depths
│   ├── generate.py              # tok/s, TTFT, I/O-benutting; Resident vs Drift
│   └── reports/
└── scripts/
    ├── bootstrap.sh             # Installeert uv indien nodig, uv sync
    ├── dev-engine.sh            # Engine met auto-reload op 127.0.0.1:8080
    ├── build-app.sh
    └── package.sh               # Signeren + notariseren (M6)
```

**Gegevenslocaties op de Mac van de gebruiker:**
- Modellen: `~/Library/Application Support/AetherDrift/Models/`, plus alleen-lezen verwijzingen naar de Hugging Face-cache (geen dubbele downloads).
- Drift-index en repack: `<modelmap>/.aether/`
- Gesprekken: `~/Library/Application Support/AetherDrift/history.sqlite`
- Logs: `~/Library/Logs/AetherDrift/engine.log` (roterend)
- Runtime: `~/Library/Application Support/AetherDrift/runtime/`

---

## 11. Ontwikkelomgeving

```bash
git clone https://github.com/mfidder2026/AetherDrift.git
cd AetherDrift
./scripts/bootstrap.sh            # installeert uv indien nodig en zet de engine-omgeving op
./scripts/dev-engine.sh           # engine op 127.0.0.1:8080 met auto-reload
open app/AetherDrift.xcodeproj    # ⌘R; de app gebruikt de draaiende dev-engine
```

Snelle test vanaf de terminal:
```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"Hallo!"}],"stream":true}'
```

**Testmodellen**, zodat Drift op elke Mac te ontwikkelen is zonder een model van 150 GB+:
- Klein dense model, bijvoorbeeld `mlx-community/Qwen2.5-0.5B-Instruct-4bit`.
- Klein MoE-model, bijvoorbeeld `mlx-community/Qwen3-30B-A3B-4bit`.
- Forceer Drift met een kunstmatig laag budget: `aether-engine --weights-budget 2GB --mode drift`.

---

## 12. Mijlpalen en acceptatiecriteria

### M0 — Haalbaarheid en metingen
Bouw: `bench/ssd.py`, een Drift-spike op een klein model, en verificatie van het referentiemodel (zie §15).
- [ ] Rapport `bench/reports/M0.md` met SSD-snelheden: sequentieel en random, blokgroottes 1–64 MB, queue depth 1–32, met `F_NOCACHE`.
- [ ] Vergelijking mmap vs. pread vastgelegd in ADR-002.
- [ ] Besluit over repack (ADR-003).
- [ ] GLM 5.3 Q4 geverifieerd: grootte, architectuur, mlx-lm-ondersteuning. Afwijkingen gemeld.

### M1 — Engine, snelle pad
Bouw: FastAPI-app, OpenAI-endpoints met SSE, modelregister (laden/ontladen), Resident-modus, prompt-cache, stats- en eventstreams, handshake, annuleren.
- [ ] `curl`-streaming werkt; de officiële `openai`-Python-client werkt zonder aanpassingen.
- [ ] Resident-overhead ≤ 5 % ten opzichte van `mlx_lm.generate`.
- [ ] Bij een vervolgbericht worden alleen de nieuwe tokens geprefilld (test).
- [ ] Annuleren < 250 ms.

### M2 — App-MVP
Bouw: EngineSupervisor, runtime-installatie via uv, onboarding met download, chat met streaming, gesprekken opslaan, modelkiezer, Live-paneel (zonder Drift-kaart), foutmeldingen.
- [ ] Van installatie tot eerste antwoord zonder terminal.
- [ ] Koude start < 1 s; geen haperingen bij het streamen.
- [ ] Engine-crash (kill -9) → automatische herstart met banner. De UI bevriest nooit en het gesprek blijft bewaard.

### M3 — Drift-Dense
Bouw: gewichtenindex, IOBackend, pinnen + ringbuffer + prefetch, eval per laag, geheugenbewaker, Drift-telemetrie, Drift-kaart, logconsole.
- [ ] Klein dense model met geforceerd laag budget: uitvoer identiek aan Resident bij greedy decoding (temperature 0) over de testprompts. Eventuele afwijkingen zijn onderzocht en verklaard.
- [ ] I/O-benutting ≥ 80 % van de gemeten SSD-bandbreedte.
- [ ] Engine-geheugen blijft binnen budget + 5 % (elke 50 ms gemeten); geen swapgroei.
- [ ] Logconsole en Drift-kaart tonen echte events.

### M4 — Drift-MoE
Bouw: expert-units, expert-cache (LFU met veroudering), parallel laden, prefill in chunks, hitrate-telemetrie.
- [ ] Klein MoE-model met geforceerd laag budget: uitvoer identiek aan Resident (greedy).
- [ ] **GLM 5.3 Q4 draait op de 128 GB-referentiemachine.**
- [ ] Soaktest van 1 uur continu genereren terwijl de gebruiker andere apps gebruikt (browser, Xcode): de Mac blijft responsief, zonder beachball, en het geheugen groeit niet.
- [ ] Live-paneel toont aantoonbaar laden en vrijgeven, plus de hitrate.

### M5 — Netwerk en integraties
Bouw: delen-schakelaar, API-sleutel in de Keychain, verbind-sheet met QR-code, webchat, Ollama-endpoints, wachtrij voor meerdere clients.
- [ ] Een telefoon op hetzelfde wifi-netwerk chat binnen 30 s na het scannen van de QR-code.
- [ ] Een OpenAI-compatibele app (bijvoorbeeld Open WebUI) verbindt met alleen base URL + sleutel.
- [ ] Een verzoek via het netwerk zonder sleutel krijgt `401`.
- [ ] Delen aan/uit herlaadt het model niet.

### M6 — Afwerking en distributie
Bouw: menubalk, snelchat met globale sneltoets, progressief laden, startmodel tijdens download, zelfstandige runtime, signeren + notariseren, NL/EN, toegankelijkheidsronde.
- [ ] Een gesigneerde en genotariseerde `.app` werkt op een schone Mac zonder internet (behalve voor het downloaden van modellen).
- [ ] Alle prestatiedoelen uit §9 gehaald en gerapporteerd.

### Definitie van klaar (v1)
1. De Mac loopt niet vast tijdens het draaien van GLM 5.3 Q4, ook niet bij langdurig gebruik.
2. Het Live-paneel toont aantoonbaar dat er tussen SSD en RAM gewisseld wordt, met echte cijfers.
3. Een ander apparaat kan het model via de API en via de webchat gebruiken.
4. Een nieuwe gebruiker krijgt zonder terminal of technische kennis een eerste antwoord.
5. Een model dat in RAM past, draait even snel als kale mlx-lm.

---

## 13. Testen en benchmarks

- **Engine-unittests (pytest):**
  - Cachebeleid tegen gesimuleerde toegangspatronen. Een cyclisch dense patroon moet laten zien dat pinnen beter is dan LRU; een scheef MoE-patroon dat LFU met veroudering minstens zo goed is als LRU.
  - Indexbouwer tegen fixture-safetensors, budgetberekening, SSE-formaat, authenticatie.
- **Integratietests:** klein dense en klein MoE-model in geforceerde Drift-modus. Gecontroleerd worden: greedy-gelijkheid met Resident, het geheugenplafond, annuleren en het hergebruik van de prompt-cache.
- **Compatibiliteit:** de `openai`-Python-client, curl, en (P1) een Ollama-client.
- **App (XCTest):** SSE-parser, EngineClient, herstarten door de supervisor (met een nep-engine-script), database. UI-test voor het gelukkige pad van de onboarding met een gemockte engine.
- **Benchmarks:** `bench/ssd.py` en `bench/generate.py` schrijven JSON plus een Markdown-samenvatting naar `bench/reports/<datum>-<mijlpaal>.md`.

---

## 14. Buiten scope voor v1

Fine-tunen of trainen · meerdere modellen tegelijk geladen · batching van meerdere gebruikers (in v1 is er een wachtrij) · Windows, Linux en Intel-Macs · multimodaal (beeld/audio) · RAG of documenten uploaden · tool/function calling (P2) · accounts en cloudsync · een iOS-app (de webchat dekt de telefoon).

---

## 15. Te verifiëren in M0

1. **GLM 5.3 Q4 in MLX:** beschikbaarheid, exacte grootte op schijf, architectuur (verwacht: MoE; aantal experts, experts per token, grootte per expert) en ondersteuning door mlx-lm. Wordt de architectuur niet ondersteund, dan is het porten van de modeldefinitie een aparte taak: meld dat vóór M1.
2. **SSD-snelheid op de referentiemachine:** sequentieel en random, verschillende blokgroottes en queue depths.
3. **mmap vs. pread** (ADR-002) en of een repack loont (ADR-003).
4. **MLX-geheugen-API's** in de geïnstalleerde versie (`set_memory_limit`, `set_cache_limit`, `set_wired_limit`, `get_active_memory`; de namespace verschilt per versie), en of geheugen na eviction echt vrijkomt.
5. **Metal-werksetlimiet** op 128 GB (`recommendedMaxWorkingSetSize`) en wat dat betekent voor het budget.
6. **Expert-slots:** gebeuren slot-updates in MLX zonder dat de hele tensor gekopieerd wordt (§6.6)?

---

## 16. Werkafspraken voor de code

- **Python:** 3.12, `uv`, `ruff` (lint + format), `mypy` op `aether/`, `pytest`. Type hints overal; Pydantic v2 voor API-modellen. Geen blokkerend werk op de asyncio-loop.
- **Swift:** Swift 6 met strict concurrency, SwiftUI, `@Observable`, async/await. Geen Combine in nieuwe code. Dependencies: alleen GRDB en `swift-markdown`; al het andere vraagt een ADR.
- Code en commentaar in het Engels. UI-teksten via de String Catalog (NL + EN).
- Elk getal in de UI is terug te voeren op een teller in de code. Een geschatte waarde krijgt altijd het label "geschat".
- Kleine, reviewbare commits per feature, met conventional commit messages.
- Houd `doc/ARCHITECTURE.md` bij zodra je van dit document afwijkt.

---

**Tot slot, voor de AI developer:** de "magie" van AetherDrift zit in twee dingen. Ten eerste een Drift Engine die de SSD continu op volle snelheid laat lezen terwijl de GPU rekent, met het juiste cachebeleid per modeltype. Ten tweede een app die dat onzichtbaar maakt: snel openen, direct kunnen typen, eerlijke cijfers, en nooit een vastgelopen Mac. Begin bij M0. Meten gaat vóór bouwen.
