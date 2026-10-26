# Generatie-benchmark: mlx-community/Qwen3-30B-A3B-4bit (2026-10-26)

| Meting | Waarde |
|---|---|
| Engine start → `ready` | 0.23 s |
| Engine start → model geladen | 1.64 s |
| TTFT (client, mediaan) | 50 ms |
| Decode via engine (client-side, HTTP+SSE, mediaan) | 113.3 tok/s |
| Decode kale mlx-lm (mediaan) | 113.0 tok/s |
| **Overhead** | **-0.3 %** (doel ≤ 5 %: OK) |

3 runs van 256 tokens, temperature 0, na één opwarm-run.
