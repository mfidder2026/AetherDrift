# ADR-001: Twee processen — SwiftUI-app + Python-engine

Status: aangenomen (2026-10-26)

Zie start-prompt-v2 §4 (B1–B10) voor de vaste beslissingen. Kort samengevat: een native SwiftUI-app beheert een Python-engine (MLX + mlx-lm + FastAPI) als sidecar. Ze praten via de publieke HTTP/SSE-API op 127.0.0.1. Alle MLX-aanroepen gebeuren in één generatorthread.
