# AetherDrift

*Infinite Intelligence. Local Hardware.*

AetherDrift draait grote taalmodellen lokaal op een Mac met Apple Silicon. Modellen die in het geheugen passen, draaien op volle snelheid. Grotere modellen worden via de Drift Engine vanaf de SSD gestreamd (in ontwikkeling). Andere apparaten kunnen het model gebruiken via een OpenAI-compatibele API.

> Status: de engine en de macOS-app werken. Dense modellen die niet in het geheugen passen, draaien via Drift (M3). Drift voor MoE-modellen zoals GLM 5.3 volgt in M4. Zie [doc/ARCHITECTURE.md](doc/ARCHITECTURE.md).

## De app bouwen

Nodig: Xcode, `uv` en XcodeGen (`brew install uv xcodegen`).

```bash
./scripts/build-app.sh            # bouwt en test; de app staat in app/build/Build/Products/Debug/
open app/build/Build/Products/Debug/AetherDrift.app
```

De app installeert en start de engine zelf. Er is geen terminal nodig.

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
