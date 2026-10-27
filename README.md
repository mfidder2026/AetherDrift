# AetherDrift

*Infinite Intelligence. Local Hardware.*

AetherDrift draait grote taalmodellen lokaal op een Mac met Apple Silicon. Modellen die in het geheugen passen, draaien op volle snelheid. Grotere modellen worden via de Drift Engine vanaf de SSD gestreamd, zelfs GLM 5.3 (418 GB) op een Mac met 128 GB. Andere apparaten gebruiken het model via de webchat (QR-code), een OpenAI-compatibele API of de Ollama-API.

> Status: M0–M5 zijn gebouwd. Van M6 zijn de menubalk, de snelchat (⌥Spatie), Nederlands/Engels en een zelfstandige runtime klaar; signeren en notariseren wachten op een Developer ID. Zie [doc/ARCHITECTURE.md](doc/ARCHITECTURE.md).

## De app bouwen

Nodig: Xcode, `uv` en XcodeGen (`brew install uv xcodegen`).

```bash
./scripts/build-app.sh            # bouwt en test; de app staat in app/build/Build/Products/Debug/
open app/build/Build/Products/Debug/AetherDrift.app
./scripts/build-app.sh Release    # met eigen Python-runtime: werkt zonder internet (ADR-007)
```

De app start de engine zelf. Er is geen terminal nodig. Een Debug-build installeert de runtime bij de eerste start (eenmalig internet), een Release-build heeft hem al aan boord.

## Andere apparaten verbinden

Kies in de app **Verbind een apparaat** (⌘⇧C) en zet **Delen op netwerk** aan. Scan de QR-code met je telefoon om de webchat te openen, of gebruik de base URL plus de API-sleutel in elke OpenAI-compatibele app (Open WebUI, LibreChat, Continue). Ollama-clients gebruiken dezelfde poort zonder `/v1`. Via het netwerk zijn alleen de chat-API's bereikbaar; beheer kan alleen op de Mac zelf (ADR-006).

## Alleen de engine (ontwikkelaars)

```bash
./scripts/bootstrap.sh
./scripts/dev-engine.sh --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"Hallo!"}],"stream":true}'
```

Elke OpenAI-compatibele client werkt met base URL `http://127.0.0.1:8080/v1`. Live statistieken: `curl -N http://127.0.0.1:8080/aether/v1/stats/stream`. API-documentatie: <http://127.0.0.1:8080/docs>.

## Testen en meten

```bash
./scripts/test.sh                                   # lint + tests
uv run --project engine python bench/ssd.py         # SSD-snelheid
uv run --project engine python bench/generate.py --model mlx-community/Qwen3-30B-A3B-4bit
uv run --project engine python bench/drift.py --model mlx-community/Qwen2.5-7B-Instruct-4bit --weights-budget 2GB
```

## Documenten

- [start-prompt-v2.md](doc/start-prompt-v2.md): het ontwerp en de mijlpalen
- [doc/ARCHITECTURE.md](doc/ARCHITECTURE.md): wat er gebouwd is en waar het afwijkt
- [doc/adr/](doc/adr/): architectuurbeslissingen
- [bench/reports/](bench/reports/): meetresultaten
