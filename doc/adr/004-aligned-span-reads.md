# ADR-004: Drift leest page-aligned regio's per laag

Status: aangenomen (2026-10-26). Vult ADR-002 aan.

## Context
ADR-002 koos `pread` + `F_NOCACHE`, zodat Drift-reads niet in de page cache van macOS terechtkomen. Tijdens M3 bleek dat dit maar half klopte.

## Metingen (M3 Max, macOS 27)
- `F_NOCACHE` omzeilt de page cache **alleen** als de bestandsoffset, de lengte én het doeladres in het geheugen 4 KB-uitgelijnd zijn. Bij reads per tensor (willekeurige offsets) kwam de tweede cyclus met 23 GB/s uit RAM. Bij een doeladres dat 16 of 512 bytes verschoven was gebeurde hetzelfde (49 GB/s); bij 4096 bytes niet.
- Wat al in de page cache staat (bijvoorbeeld vlak na een download), wordt ook met `F_NOCACHE` uit RAM gelezen. Met `msync(MS_INVALIDATE)` op een mmap van het bestand kun je de cache legen zonder sudo. Dat doen de benchmarks vóór elke run.
- MLX-bestanden vullen hun safetensors-header niet op. Tensors liggen daardoor vaak niet uitgelijnd op hun elementgrootte (bij Qwen2.5-7B 4-bit alle 735). MLX handelt een `view` op zo'n offset correct af, maar maakt dan een kopie op de GPU.

## Besluit
- Per laag wordt de aaneengesloten bestandsregio gelezen, afgerond op 16 KB, en direct in een page-aligned MLX-buffer gezet. Tensors zijn views op de juiste offset binnen die buffer.
- Uitgelijnde tensors zijn zero-copy. Bij scheve tensors laten we MLX op de GPU herschikken (~400 GB/s, verwaarloosbaar per token). Het budget reserveert daarvoor één laag tijdelijk geheugen.
- Gepinde lagen worden bij het laden direct gematerialiseerd, waarna de ruwe buffer vrijkomt. Zonder die stap kostte een gepinde laag het dubbele.

## Gevolgen
- Drift-reads komen niet meer in de page cache. Het geheugengebruik is voorspelbaar en de benchmarks meten de SSD (gecontroleerd met `iostat`).
- Minder, grotere reads: één tot enkele per laag in plaats van ~20 per laag.
- Het M0-rapport blijft geldig: `bench/ssd.py` gebruikte al uitgelijnde reads.
