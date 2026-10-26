# Generatie-benchmark: mlx-community/Qwen2.5-0.5B-Instruct-4bit (2026-10-26)

| Meting | Waarde |
|---|---|
| Engine start → `ready` | 0.23 s |
| Engine start → model geladen | 1.05 s |
| TTFT (client, mediaan) | 15 ms |
| Decode via engine (client-side, HTTP+SSE, mediaan) | 486.1 tok/s |
| Decode kale mlx-lm (mediaan) | 485.2 tok/s |
| **Overhead** | **-0.2 %** (doel ≤ 5 %: OK) |

3 runs van 256 tokens, temperature 0, na één opwarm-run.
