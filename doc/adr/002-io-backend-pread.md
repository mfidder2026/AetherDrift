# ADR-002: I/O-backend is pread + F_NOCACHE

Status: aangenomen (2026-10-26)

## Context
De Drift Engine moet gewichten zo snel mogelijk van de SSD in RAM krijgen. Er waren twee kandidaten: mmap + madvise, of pread met F_NOCACHE in eigen buffers (start-prompt-v2 §6.4).

## Meting (M3 Max, `bench/ssd.py`)
- pread + F_NOCACHE: 6,6–6,8 GB/s vanaf queue depth 4 bij blokken van 1 MB, voor zowel sequentieel als random lezen.
- mmap + madvise(WILLNEED/DONTNEED), 16 MB per keer met de data eruit gekopieerd: 3,2 GB/s.

## Besluit
`PreadBackend` (`engine/aether/drift/io_backend.py`): een threadpool (standaard 16 threads), units opgeknipt in chunks van 8 MB en F_NOCACHE op elke file descriptor. Er komt geen mmap-backend.

## Gevolgen
- Het geheugengebruik is voorspelbaar: de page cache van het OS speelt geen rol.
- Elke unit wordt één keer gekopieerd van de buffer naar een MLX-array. Dat is verwaarloosbaar ten opzichte van de SSD-snelheid.
