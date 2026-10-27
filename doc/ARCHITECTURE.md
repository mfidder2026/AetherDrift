# Architectuur — actuele stand

Het ontwerp staat in [start-prompt-v2.md](start-prompt-v2.md). Dit document houdt bij wat er gebouwd is en waar de praktijk afwijkt van dat ontwerp.

## Stand per mijlpaal (2026-10-27)

| Mijlpaal | Status |
|---|---|
| M0 Metingen | Klaar. De Drift-spike is aan het begin van M3 gedaan (zie ADR-004). Zie [bench/reports/M0.md](../bench/reports/M0.md). |
| M1 Engine, snelle pad | Klaar. Alle acceptatiecriteria gehaald (zie hieronder). |
| M2 App-MVP | Klaar. Zie M2-acceptatie hieronder. |
| M3 Drift-Dense | Klaar. Zie M3-acceptatie hieronder. |
| M4 Drift-MoE | Klaar. GLM 5.3 draait; de soak-test is gehaald (zie M4-acceptatie). |
| M5 Netwerk en integraties | Gebouwd en getest via het LAN-adres van deze Mac. Nog te doen: één keer met een echte telefoon en met Open WebUI (zie M5-acceptatie). |
| M6 Afwerking en distributie | Grotendeels klaar: menubalk, snelchat, NL/EN, zelfstandige runtime, startmodel tijdens download, wakker blijven. Open: signeren en notariseren (vraagt een Developer ID), toegankelijkheidsronde, prestatierapport §9. |

### M1-acceptatie
| Criterium | Resultaat | Bewijs |
|---|---|---|
| curl-streaming en de officiële `openai`-client werken | ✓ | `tests/test_engine.py::test_official_openai_client` |
| Resident-overhead ≤ 5 % | −0,2 % (Qwen2.5 0.5B), −0,3 % (Qwen3 30B-A3B) | `bench/reports/2026-10-26-M1-generate-*.md` |
| Bij een vervolgbericht worden alleen nieuwe tokens geprefilld | ✓ | `test_follow_up_only_prefills_new_tokens` |
| Annuleren < 250 ms | ✓ | `test_cancel_endpoint_stops_quickly`, en ook bij het wegvallen van de client |
| Engine-start → `ready` < 3 s | 0,23 s | generate-benchmark |
| Greedy-uitvoer identiek aan kale mlx-lm | ✓ | `test_greedy_output_matches_plain_mlx_lm` |

### M2-acceptatie
| Criterium | Resultaat | Bewijs |
|---|---|---|
| Van installatie tot eerste antwoord zonder terminal | ✓ De runtime installeert zich bij de eerste start (~7 s met snel internet), daarna onboarding → model → chat | Handmatige run met lege Application Support |
| Koude start < 1 s | 195–296 ms tot het eerste frame | `StartupTiming` (log `startup`) |
| Geen haperingen bij streaming | Tokens worden gebufferd tot één UI-update per frame; Markdown is incrementeel; berichten zijn losse `@Observable`-objecten | `ChatStore.TokenBuffer`, `StreamingMarkdown` |
| Engine-crash → automatische herstart met banner, gesprek bewaard | ✓ Na `kill -9` binnen 3 s terug, model automatisch opnieuw geladen | Snapshots + `SupervisorTests` |

Niet geautomatiseerd getest: echte muis- en toetsenbordinteractie (geen toegankelijkheidsrechten in de ontwikkelomgeving). Debug-builds hebben daarom twee hulpmiddelen:
- `AETHER_SNAPSHOT_DIR=<map>`: het venster tekent zichzelf elke seconde naar `latest.png`.
- `AETHER_AUTOPILOT_ONBOARD=<model>` en `AETHER_AUTOPILOT_CHAT=<tekst>`: doorlopen de onboarding en versturen een bericht via dezelfde functies als de knoppen.

Beperking: de doorschijnende zijbalk wordt niet mee getekend in deze snapshots en ziet er daarin zwart uit.

### M3-acceptatie
| Criterium | Resultaat | Bewijs |
|---|---|---|
| Uitvoer in Drift identiek aan Resident (greedy) | ✓ 4/4 prompts op Qwen2.5-0.5B en 2/2 op Qwen2.5-7B (scheve tensors) | `test_drift_engine_matches_resident`, handmatige 7B-vergelijking |
| I/O-benutting ≥ 80 % | 102 % van de I/O-grens: 2,48 tok/s bij 2,58 GB per token; SSD 6,2–6,6 GB/s (`iostat`: 5,9 GB/s) | `bench/reports/2026-10-26-M3-drift-Qwen2.5-7B-Instruct-4bit.md` |
| Geheugen binnen budget + 5 % | MLX-piek 2,04 GB bij een grens van 2,11 GB; proces-footprint 2,92 GB (incl. Python en tokenizer) | idem, 50 ms-samples |
| Geen swapgroei | 0 MB | idem |
| Logconsole en Drift-kaart tonen echte events | ✓ | App-snapshot met load/evict-samenvattingen van 1,6 GB per 250 ms |

De I/O-grens is gerekend met de beste gemeten SSD-snelheid (6,77 GB/s). Iets boven 100 % komt doordat de werkelijke regio's per laag soms iets kleiner zijn dan het gemiddelde.

**Drift-Dense in het kort:** de lagen 0…P−1 blijven vast in RAM. De rest stroomt door een ring van 3 slots met prefetch-diepte 2. Elke gestreamde laag wordt direct na het rekenen geëvalueerd (`mx.eval`), waarna zijn slot vrijkomt. Bij geheugendruk geeft de bewaker in stappen van 10 % vastgezette lagen terug (met 3 s rust ertussen).

### M4-acceptatie
| Criterium | Resultaat | Bewijs |
|---|---|---|
| Klein MoE-model, laag budget: uitvoer identiek aan Resident (greedy) | ✓ 5/5 prompts, Qwen3-30B-A3B met 20 van 128 experts per laag in RAM | `tests/test_moe.py`, `bench/reports/2026-10-26-M4-drift-Qwen3-30B-A3B-4bit.md` |
| **GLM 5.3 Q4 draait op de 128 GB-machine** | ✓ 418 GB-model, 56 van 256 experts per laag in RAM (~98 GB). Decode 0,9–1,3 tok/s, eerste token 17–28 s | ADR-005 |
| Soak-test 1 uur: Mac responsief, geheugen groeit niet | ✓ Herhaling (60 min, met builds en tests op de achtergrond): footprint +0,08 GB, actief MLX-geheugen vlak op 98,09 GB, geen swapgroei, geen geheugendruk, 0 fouten. Proces starten p99 3,8 ms | `bench/reports/2026-10-27-M4-soak-GLM-5.3-4bit.md` |
| Live-paneel toont laden, vrijgeven en hitrate | Engine levert expert-events en hitrate; de app toont ze in het Live-paneel en de Drift-strip | `ExpertCache.publish`, `Features/Live` |

Over de soak-tests:
- **Eerste run (26-10, 60 min):** +1,24 GB, gestaag lineair. Die run mat het MLX-geheugen nog niet apart.
- **Diagnose (30 min) en herhaling (60 min), beide mét MLX-meting:** geheugen vlak. Het actieve geheugen groeit alleen tot de prompt-cache vol is (3 items, 2× 23 MB).
- **Conclusie:** de groei uit de eerste run is niet gereproduceerd. Er zit geen groeiende structuur in de engine (alle buffers zijn begrensd). Mocht het terugkomen, dan laat `bench/soak.py` nu zien of het binnen of buiten MLX zit.
- **Slaapstand:** in de herhaling ging de Mac na ~53 minuten zonder gebruik in slaap, midden in een antwoord. De engine houdt daarom tijdens het genereren een IOKit-assertion vast (`power.py`, getest met `pmset`). De app voorkomt slaap zolang delen aan staat.

### M5-acceptatie
| Criterium | Resultaat | Bewijs |
|---|---|---|
| Een telefoon op hetzelfde wifi-netwerk chat binnen 30 s na het scannen van de QR-code | Deels: de QR-code is door macOS gedecodeerd naar `http://192.168.1.111:8080/#key=…`; de webchat-pagina en de API werken via het LAN-adres. **Nog niet met een echte telefoon geprobeerd** | Snapshot Verbind-scherm, LAN-test |
| Een OpenAI-compatibele app verbindt met alleen base URL + sleutel | ✓ Officiële `openai`-client via `http://192.168.1.111:<poort>/v1`; Ollama-stream via `/api/chat`. Open WebUI zelf nog niet geprobeerd | LAN-test, `tests/test_api.py` |
| Netwerkverzoek zonder sleutel → `401` | ✓ Beheer via het netwerk → `403 local_only` | LAN-test, `test_network_surface` |
| Delen aan/uit herlaadt het model niet | ✓ `requires_reload: false` | `test_share_toggle_does_not_reload`, LAN-test |

Wachtrij voor meerdere clients: FIFO zoals ontworpen. Een client die in de wachtrij of tijdens de prefill afhaakt, wordt eruit gehaald zonder dat zijn verzoek nog draait (getest met een echte engine). Beveiliging: zie ADR-006.

### M6-stand
| Onderdeel | Stand |
|---|---|
| Menubalk (status, nieuwe chat, snelchat, stop, delen, open) | ✓ `MenuBarExtra` |
| Blijf actief na sluiten venster | ✓ standaard aan zolang delen aan staat; instelbaar |
| Snelchat ⌥Spatie | ✓ Spotlight-achtig paneel (Carbon-hotkey, geen toegankelijkheidsrechten nodig) |
| Starten bij inloggen | ✓ `SMAppService` (bij een ad-hoc-gesigneerde app kan macOS dit weigeren; de fout wordt getoond) |
| Startmodel tijdens download | ✓ bij een groot model eerst het startmodel; daarna automatisch wisselen |
| Zelfstandige runtime | ✓ Release-build, 487 MB, engine start zonder internet (ADR-007) |
| NL/EN | ✓ String Catalog met 201 teksten; Engels gecontroleerd met `-AppleLanguages (en)` |
| Signeren + notariseren | ✗ vraagt een Developer ID-certificaat |
| Toegankelijkheidsronde | Deels: VoiceOver-labels op statistieken, QR-code en menubalkpictogram. Geen volledige ronde gedaan |
| Prestatierapport §9 | Losse metingen in `bench/reports/`; nog geen samenvattend rapport |

## App-modules (`app/AetherDrift/`)

| Map | Rol |
|---|---|
| `App/` | Startpunt, `AppState` (modellen, downloads, meldingen, events, delen), commando's/sneltoetsen, opmaak, menubalk (`MenuBar.swift`, starten bij inloggen), globale sneltoets (`GlobalHotKey`, Carbon) |
| `Engine/` | `EngineRuntime` (meegebundelde runtime in Release, anders eenmalige installatie via uv; ADR-007), `EngineSupervisor` (start, handshake, health checks, backoff, SIGTERM → SIGKILL), `EngineClient` (HTTP/SSE), `SSEParser`, API-types, `APIKeyStore` (Keychain) |
| `Persistence/` | SQLite via GRDB: gesprekken en berichten |
| `Features/Chat` | `ChatStore`, berichten, invoerveld, Markdown-weergave (swift-markdown), denkproces |
| `Features/Models` | Modelchip, moduskiezer, modelpalet (⌘K) met fit-badges en downloads |
| `Features/Live` | Live-paneel met sparklines en Drift-kaart; compacte Drift-strip onder een binnenkomend antwoord |
| `Features/Log` | Logconsole (⌘L): filters, zoeken, pauzeren, kopiëren, logbestand openen |
| `Features/Onboarding` | Eerste start in 3 stappen |
| `Features/Settings` | Modus, OS-reserve (automatisch of vast), gewichtenlimiet, snelchat, actief blijven, starten bij inloggen, engine-status |
| `Features/Connect` | Verbind een apparaat (⌘⇧C): delen aan/uit, adres, sleutel, QR-code, voorbeelden, recente apparaten (ADR-006) |
| `Features/QuickChat` | Zwevend snelchatvenster (⌥Spatie); een vraag wordt een gewoon gesprek |
| `Resources/Localizable.xcstrings` | String Catalog: Nederlands (bron) en Engels |

Het Xcode-project wordt gegenereerd uit `app/project.yml` (XcodeGen). De bouwstap `app/scripts/bundle-engine.sh` kopieert de engine-code en `uv` in de bundle. Bij de eerste start draait `uv sync` naar `~/Library/Application Support/AetherDrift/runtime/`. De engine draait daarna direct met de Python uit die venv, met `PYTHONDONTWRITEBYTECODE=1` zodat er nooit in de gesigneerde bundle wordt geschreven.

## Engine-modules (`engine/aether/`)

| Module | Rol |
|---|---|
| `__main__.py` | CLI, logging, poortkeuze (met fallback), ready-handshake op stdout, parent-pid-bewaking, netwerk-listener aan/uit |
| `config.py` | `EngineConfig` (startopties) en `Settings` (via `/aether/v1/config` aan te passen) |
| `system.py` | Chip, RAM, geheugendruk (sysctl), snelle SSD-test |
| `api/` | FastAPI-app; `auth.py` is pure ASGI-middleware (loopback vrij maar geen vreemde Host/Origin, netwerk alleen chat-API's met sleutel; ADR-006), `openai.py` (chat + completions), `ollama.py`, `aether.py`, `sse.py`, webchat op `/` |
| `web/index.html` | Webchat voor telefoon en browser: één bestand, geen build-stap, sleutel uit het URL-fragment |
| `runtime/worker.py` | Generatorthread: enige eigenaar van MLX. Laden met voortgang, prompt-cache (mlx-lm `LRUPromptCache`, prefix-matching), stopsequenties, annuleren |
| `memory/budget.py` | Budget en fit-inschatting (resident / drift-dense / drift-moe / too_large) |
| `memory/guard.py` | Geheugendruk en swapgroei → events; bij kritiek wordt de generatie gestopt |
| `models/registry.py` | Modellen vinden (eigen map + HF-cache), catalogus, hervatbare download |
| `drift/index.py` | Gewichtenindex uit safetensors-headers (units, gestapelde experts, offsets) |
| `drift/io_backend.py` | `PreadBackend`: parallelle pread met F_NOCACHE (ADR-002); `invalidate_page_cache` voor benchmarks |
| `drift/units.py` | Lagen → units met page-aligned leesregio's (ADR-004); keuze van het aantal vaste lagen |
| `drift/residency.py` | Vaste lagen + ring van slots, prefetch, eviction, hitrate, verkleinen bij geheugendruk |
| `drift/model.py` | Omwikkelt de decoderlagen van een mlx-lm-model: gewichten ophalen → rekenen → eval → vrijgeven. `DriftSwitch` vervangt de expert-laag van MoE-blokken (ADR-005) |
| `drift/experts.py` | Expert-slots per laag, LFU met veroudering, speculatieve prefetch, volledige buffer voor prefill (ADR-005) |
| `telemetry/` | Eventbus (ringbuffer, throttling naar samenvattingen) en statistieken-snapshot |

## Afwijkingen van start-prompt-v2

1. **GLM 5.3 Q4 is 418 GB, niet ~150 GB.** Het is een MoE-model (256 experts, 8 per token); routed experts beslaan 407,7 GB. Op 128 GB past ~22 % van de experts in de cache. De verwachting is 0,5–2 tok/s, afhankelijk van de hitrate (M0-rapport §3).
2. **De gewichtenindex staat in `<data-dir>/indexes/`**, niet in `<modelmap>/.aether/`. We schrijven nooit in modelmappen of in de Hugging Face-cache.
3. **Geen repack** (ADR-003) **en geen mmap-backend** (ADR-002).
4. Draait er al een engine op poort 8080 (bijvoorbeeld via `dev-engine.sh`), dan gebruikt de app die in plaats van een eigen engine te starten (§8.13). Met `AETHER_NO_EXTERNAL=1` gaat dat uit.
5. `/aether/v1/models` geeft per model ook `fit.loadable`: of deze engine-versie de benodigde modus al kan draaien. De catalogus bevat daarvoor vormgegevens per model.
6. De app bewaart de engine-instellingen zelf en stuurt ze na elke (her)start van de engine opnieuw mee.
7. Drift-MoE gebruikt een vast aantal expert-slots per laag in plaats van één globale cache (ADR-005).
8. mlx-lm is gepind op een commit van `main` voor GLM 5.3-ondersteuning (ADR-005); terug naar PyPI zodra er een release is.
9. De OS-reserve is standaard automatisch: een kwart van het RAM, tussen 4 en 16 GB (`auto_os_reserve_gb`). Een vaste 16 GB liet op een Mac met 16 GB niets over.
10. De QR-code gebruikt het IP-adres in plaats van `<naam>.local` (ADR-006).
11. Beheer-endpoints zijn via het netwerk niet bereikbaar, en loopback weigert vreemde `Host`/`Origin` (ADR-006).

## Onafhankelijke review (2026-10-26)

Een aparte agent zonder voorkennis heeft de code doorgelopen en 15 bevindingen gemeld. Ze zijn allemaal gecontroleerd en opgelost, de meeste met een regressietest:
- **Hoog:**
  - Een mislukte read liet een expert of laag als "aanwezig" staan, zodat er stil met halve gewichten gerekend kon worden.
  - De CLI-default zette de automatische OS-reserve uit.
  - De prompt-cache kon het KV-geheugen verdubbelen.
- **Midden:**
  - CSRF en DNS rebinding op loopback.
  - De geheugenwachttijd bij laden keek naar te weinig geheugen.
  - Verkleiningsverzoeken stapelden zich op zonder actieve generatie.
  - De supervisor wachtte niet op de oude engine (nu SIGTERM → SIGKILL).
  - De SSD-test sloot een file descriptor twee keer en reserveerde 2 GB.
  - Stoppen werkte niet tijdens de wachtrij of de prefill.
- **Laag:**
  - Tokenizerfout tijdens het laden.
  - Uitlijning na het einde van een bestand.
  - Modelpaden via de chat-API.
  - Delen bleef "aan" na een mislukte start.
  - Keychain-fouten werden genegeerd.

## Bekende beperkingen (M1–M6)

- Drift-MoE kan zijn expert-slots niet verkleinen bij geheugendruk; de bewaker stopt bij "kritiek" alleen de generatie.
- GLM 5.3 haalt ~1 tok/s; de SSD is de grens (ADR-005). Het eerste antwoord duurt 17–28 s.

- Eén model tegelijk en één generatie tegelijk (FIFO-wachtrij), zoals ontworpen.
- Een wijziging van de modus geldt pas na het opnieuw laden van het model (`requires_reload: true`). De app herlaadt dan automatisch.
- Debug-builds installeren de runtime bij de eerste start (eenmalig internet); Release-builds hebben hem aan boord.
- De app is ad-hoc gesigneerd. Signeren met een Developer ID en notariseren kan pas met een certificaat.
- Bonjour-advertentie (P2) is niet gebouwd.
