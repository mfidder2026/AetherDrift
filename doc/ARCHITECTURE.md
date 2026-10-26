# Architectuur — actuele stand

Het ontwerp staat in [start-prompt-v2.md](start-prompt-v2.md). Dit document houdt bij wat er gebouwd is en waar de praktijk afwijkt van dat ontwerp.

## Stand per mijlpaal (2026-10-26)

| Mijlpaal | Status |
|---|---|
| M0 Metingen | Klaar. De Drift-spike is aan het begin van M3 gedaan (zie ADR-004). Zie [bench/reports/M0.md](../bench/reports/M0.md). |
| M1 Engine, snelle pad | Klaar. Alle acceptatiecriteria gehaald (zie hieronder). |
| M2 App-MVP | Klaar. Zie M2-acceptatie hieronder. |
| M3 Drift-Dense | Klaar. Zie M3-acceptatie hieronder. |
| M4–M6 | Niet gestart |

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

## App-modules (`app/AetherDrift/`)

| Map | Rol |
|---|---|
| `App/` | Startpunt, `AppState` (modellen, downloads, meldingen, events), commando's/sneltoetsen, opmaak |
| `Engine/` | `EngineRuntime` (gebundelde engine + uv, eenmalige installatie), `EngineSupervisor` (start, handshake, health checks, backoff), `EngineClient` (HTTP/SSE), `SSEParser`, API-types |
| `Persistence/` | SQLite via GRDB: gesprekken en berichten |
| `Features/Chat` | `ChatStore`, berichten, invoerveld, Markdown-weergave (swift-markdown), denkproces |
| `Features/Models` | Modelchip, moduskiezer, modelpalet (⌘K) met fit-badges en downloads |
| `Features/Live` | Live-paneel met sparklines en Drift-kaart; compacte Drift-strip onder een binnenkomend antwoord |
| `Features/Log` | Logconsole (⌘L): filters, zoeken, pauzeren, kopiëren, logbestand openen |
| `Features/Onboarding` | Eerste start in 3 stappen |
| `Features/Settings` | Modus, OS-reserve, engine-status |

Het Xcode-project wordt gegenereerd uit `app/project.yml` (XcodeGen). De bouwstap `app/scripts/bundle-engine.sh` kopieert de engine-code en `uv` in de bundle. Bij de eerste start draait `uv sync` naar `~/Library/Application Support/AetherDrift/runtime/`. De engine draait daarna direct met de Python uit die venv, met `PYTHONDONTWRITEBYTECODE=1` zodat er nooit in de gesigneerde bundle wordt geschreven.

## Engine-modules (`engine/aether/`)

| Module | Rol |
|---|---|
| `__main__.py` | CLI, logging, poortkeuze (met fallback), ready-handshake op stdout, parent-pid-bewaking, netwerk-listener aan/uit |
| `config.py` | `EngineConfig` (startopties) en `Settings` (via `/aether/v1/config` aan te passen) |
| `system.py` | Chip, RAM, geheugendruk (sysctl), snelle SSD-test |
| `api/` | FastAPI-app; `auth.py` is pure ASGI-middleware (loopback vrij, netwerk alleen met sleutel), `openai.py`, `aether.py`, `sse.py` |
| `runtime/worker.py` | Generatorthread: enige eigenaar van MLX. Laden met voortgang, prompt-cache (mlx-lm `LRUPromptCache`, prefix-matching), stopsequenties, annuleren |
| `memory/budget.py` | Budget en fit-inschatting (resident / drift-dense / drift-moe / too_large) |
| `memory/guard.py` | Geheugendruk en swapgroei → events; bij kritiek wordt de generatie gestopt |
| `models/registry.py` | Modellen vinden (eigen map + HF-cache), catalogus, hervatbare download |
| `drift/index.py` | Gewichtenindex uit safetensors-headers (units, gestapelde experts, offsets) |
| `drift/io_backend.py` | `PreadBackend`: parallelle pread met F_NOCACHE (ADR-002); `invalidate_page_cache` voor benchmarks |
| `drift/units.py` | Lagen → units met page-aligned leesregio's (ADR-004); keuze van het aantal vaste lagen |
| `drift/residency.py` | Vaste lagen + ring van slots, prefetch, eviction, hitrate, verkleinen bij geheugendruk |
| `drift/model.py` | Omwikkelt de decoderlagen van een mlx-lm-model: gewichten ophalen → rekenen → eval → vrijgeven |
| `telemetry/` | Eventbus (ringbuffer, throttling naar samenvattingen) en statistieken-snapshot |

## Afwijkingen van start-prompt-v2

1. **GLM 5.3 Q4 is 418 GB, niet ~150 GB.** Het is een MoE-model (256 experts, 8 per token); routed experts beslaan 407,7 GB. Op 128 GB past ~22 % van de experts in de cache. De verwachting is 0,5–2 tok/s, afhankelijk van de hitrate (M0-rapport §3).
2. **De gewichtenindex staat in `<data-dir>/indexes/`**, niet in `<modelmap>/.aether/`. We schrijven nooit in modelmappen of in de Hugging Face-cache.
3. **Geen repack** (ADR-003) **en geen mmap-backend** (ADR-002).
4. Draait er al een engine op poort 8080 (bijvoorbeeld via `dev-engine.sh`), dan gebruikt de app die in plaats van een eigen engine te starten (§8.13). Met `AETHER_NO_EXTERNAL=1` gaat dat uit.
5. `/aether/v1/models` geeft per model ook `fit.loadable`: of deze engine-versie de benodigde modus al kan draaien. De catalogus bevat daarvoor vormgegevens per model.
6. De app bewaart de engine-instellingen zelf en stuurt ze na elke (her)start van de engine opnieuw mee.
7. MoE-modellen die niet in RAM passen, geven nog `model_too_large` met de hint dat Drift-MoE in M4 volgt.

## Bekende beperkingen (M1/M2)

- Eén model tegelijk en één generatie tegelijk (FIFO-wachtrij), zoals ontworpen.
- De webchat (`/`), de Ollama-endpoints en `/v1/completions` zijn P1 en nog niet gebouwd.
- Een wijziging van de modus geldt pas na het opnieuw laden van het model (`requires_reload: true`). De app herlaadt dan automatisch.
- De runtime-installatie heeft één keer internet nodig. Een volledig zelfstandige runtime volgt in M6.
- De app is ad-hoc gesigneerd. Signeren met een Developer ID en notariseren volgt in M6.
- Lokalisatie (String Catalog NL/EN), menubalk en snelchat volgen in M6.
