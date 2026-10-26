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

**Resultaat op GLM 5.3 (4-bit, 418 GB) op 128 GB, page cache geleegd:**
- 56 van de 256 experts per laag in RAM; ~98 GB MLX-geheugen.
- Decode: **1,15–1,29 tok/s**. Eerste token: **17–28 s** (prefill van een korte vraag leest ~100–150 GB aan experts).
- Echte cache-hitrate 62–64 %. Per token ~4,7 GB aan ontbrekende experts; de SSD leest op volle snelheid (~6,2 GB/s). Dit is de natuurkundige grens bij deze cachegrootte.
- Prefetch heeft een trefzekerheid van 68–70 %, maar levert bij GLM geen snelheidswinst op: de SSD is al verzadigd. Het staat standaard aan (+35 % bij Qwen3) en is te schakelen met de instelling `prefetch_experts`.

## Correctheid
- Uitvoer bit-identiek aan mlx-lm bij greedy decoding (5/5 prompts, Qwen3-30B met 16 % van de experts in RAM).
- Ontbreekt een benodigde expert na het laden, dan volgt een harde fout in plaats van stil verkeerd rekenen. Dit vangnet is toegevoegd na een gevonden bug: voorspelde maar ongebruikte experts hielden hun slot voor altijd bezet.

## Neveneffect: prompt-cache
Tijdens dit onderzoek bleek dat ook de Resident-modus afweek van mlx-lm zodra de prompt-cache een korte prefix (3 tokens) hergebruikte. Een prefill in twee stukken rondt anders af. Hergebruik gebeurt nu pas vanaf 32 gedeelde tokens: korte prompts zijn reproduceerbaar, lange gesprekken houden de snelheidswinst.

## Ondersteuning van GLM 5.3
GLM 5.3 gebruikt een sparse-attention-indexer die maar in elke 4e laag eigen gewichten heeft ("shared"). mlx-lm 0.32.2 ondersteunt dat niet. De engine is daarom gepind op mlx-lm-commit `4cfc647` van `main` (`[tool.uv.sources]` in `engine/pyproject.toml`). Terug naar PyPI zodra een release dit bevat.

## Gevonden en opgelost tijdens de GLM-test
- De Drift-laadpaden lieten de stop-tokens uit `config.json` weg, waardoor GLM na zijn antwoord doorpraatte met verzonnen `<|user|>`-beurten. Nu worden ze geladen zoals `mlx_lm.load` dat doet.
- GLM opent `<think>` al in de prompt. De engine stuurt die tag nu als eerste token mee, zodat clients een compleet denkblok zien.
- Prefetch draaide door een niet-toegepaste wijziging als verplichte load (kon wachten, of in theorie falen). Hersteld; met test en meting van de trefzekerheid.
- Een mislukte laadpoging bij het opstarten liet de status op "loading" staan. Nu: status "error" plus een error-event.
- Een nieuwe engine kon ~100 GB reserveren terwijl een vorig engine-proces dat geheugen nog vrijgaf (swappiek van 21 GB tijdens een handmatige herstart). De engine wacht nu tot 60 s tot er genoeg vrij is, en geeft anders `not_enough_memory`.

## Nog open
- Expert-slots verkleinen bij geheugendruk wordt niet ondersteund (vaste tensors). Bij MoE stopt de bewaker bij "kritiek" alleen de generatie.
- Bij kleine experts (Qwen3: 2,65 MB in 9 tensors) is de I/O latentie-gebonden. Bij GLM 5.3 (21 MB per expert) verwachten we een hogere SSD-benutting.
