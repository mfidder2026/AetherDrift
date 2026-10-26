# ADR-005: Drift-MoE — expert-slots per laag, volledige buffer voor prefill, speculatieve prefetch

Status: aangenomen (2026-10-26)

## Context
Bij MoE-modellen zoals GLM 5.3 (418 GB, 256 experts per laag, 8 per token) is vrijwel alles routed experts. Per token is maar ~3 % daarvan nodig, maar welke experts dat zijn weet je pas als de router van die laag gerekend heeft.

## Besluit

**1. Vaste expert-slots per MoE-laag.** Elke gestapelde expert-tensor (`[E, …]` in het bestand) krijgt in RAM een slot-tensor `[S, …]`. Een tabel vertaalt expert-ID naar slot. De router-indices worden vóór `gather_qmm` omgezet naar slotnummers, zodat de mlx-lm-kernels ongewijzigd blijven. S is gelijk voor alle lagen en volgt uit het budget. Eviction gebeurt per laag met LFU en exponentiële veroudering (halfwaardetijd 256 tokens).
- *Afwijking van §6.6:* geen globale cache maar een vast quotum per laag, omdat slot-tensors een vaste vorm hebben. Bij scheef verdeelde routering kan dat in de toekomst adaptief per laag.

**2. Prefill via een tijdelijke volledige buffer.** Prefill raakt per laag meestal meer experts dan er slots zijn. Rekenen in groepen bleek **niet** bit-identiek: MLX kiest bij een andere rijverdeling andere `gather_qmm`-kernels, en de afrondingsverschillen (≤ 0,008 per laag) gaven na 48 lagen andere tokens. Daarom rekent prefill met een tijdelijke `[E, …]`-buffer per laag en de originele indices:
- Experts die al in een slot staan, worden in RAM gekopieerd.
- De rest wordt één keer van de SSD gelezen.
- Prefill verandert de slots niet, dus de decode-werkset blijft staan.
- Kosten: één laag aan experts extra geheugen (GLM 5.3: 5,4 GB; Qwen3-30B: 0,34 GB), meegerekend in het budget.

**3. Speculatieve prefetch (decode).** Terwijl laag L zijn ontbrekende experts laadt, wordt de router van de volgende MoE-laag toegepast op de invoer van laag L. De voorspelde experts worden alvast gelezen. Bij Qwen3-30B (20 van 128 experts in RAM, lege page cache):

| | Zonder prefetch | Met prefetch |
|---|---|---|
| tok/s | 7,80 | 10,51 |
| SSD-benutting | 42 % | 61 % |
| Hitrate | 65 % | 93 % |

**4. Uitgelijnd lezen met staging.** Expert-slices beginnen op scheve offsets. Ze worden 4 KB-uitgelijnd gelezen in een thread-lokale staging-buffer en daarna gekopieerd (ADR-004), zodat de page cache buiten spel blijft (6,7 GB/s, ook bij herhaling).

## Correctheid
- Uitvoer bit-identiek aan mlx-lm bij greedy decoding (5/5 prompts, Qwen3-30B met 16 % van de experts in RAM).
- Ontbreekt een benodigde expert na het laden, dan volgt een harde fout in plaats van stil verkeerd rekenen. Dit vangnet is toegevoegd na een gevonden bug: voorspelde maar ongebruikte experts hielden hun slot voor altijd bezet.

## Neveneffect: prompt-cache
Tijdens dit onderzoek bleek dat ook de Resident-modus afweek van mlx-lm zodra de prompt-cache een korte prefix (3 tokens) hergebruikte. Een prefill in twee stukken rondt anders af. Hergebruik gebeurt nu pas vanaf 32 gedeelde tokens: korte prompts zijn reproduceerbaar, lange gesprekken houden de snelheidswinst.

## Nog open
- Expert-slots verkleinen bij geheugendruk wordt niet ondersteund (vaste tensors). Bij MoE stopt de bewaker bij "kritiek" alleen de generatie.
- Bij kleine experts (Qwen3: 2,65 MB in 9 tensors) is de I/O latentie-gebonden. Bij GLM 5.3 (21 MB per expert) verwachten we een hogere SSD-benutting.
