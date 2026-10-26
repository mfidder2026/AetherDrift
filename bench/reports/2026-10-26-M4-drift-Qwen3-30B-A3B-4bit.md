# Drift-benchmark (drift-moe): mlx-community/Qwen3-30B-A3B-4bit (2026-10-26)

Budget voor gewichten: 4GB · 48 lagen · 20 van 128 experts per laag in RAM · page cache geleegd: ja

| Meting | Waarde |
|---|---|
| Decode (mediaan) | 10.51 tok/s |
| **SSD-benutting tijdens decode** | **61 %** van 6.77 GB/s |
| GB van SSD per token (gem.) | 0.39 |
| SSD-lezen tijdens decode (gem.) | 4.13 GB/s |
| Cache-hitrate / totale wachttijd op I/O | 0.934 / 12.616 s |
| MLX-piekgeheugen | 3.58 GB (budget + KV + 5 % = 4.22 GB) |
| Proces-footprint: basis → piek (50 ms-samples) | 0.06 → 4.05 GB |
| Swapgroei | 0 MB |
