# ADR-003: Geen repack van modelbestanden in v1

Status: aangenomen (2026-10-26)

## Context
Een repack zou de gewichten per unit aaneengesloten op schijf zetten, zodat één unit in één leesactie binnenkomt (start-prompt-v2 §6.5).

## Meting
Random lezen is even snel als sequentieel (6,7–6,8 GB/s) zodra er ≥ 4 verzoeken van ≥ 1 MB tegelijk lopen. Een GLM 5.3-expert (21 MB) bestaat uit 3 aaneengesloten blokken van 6,3 MB plus 6 kleine blokken van 0,39 MB. Die worden parallel gelezen.

## Besluit
Geen repack in v1. Het model wordt rechtstreeks uit de originele safetensors gelezen. Dat scheelt de gebruiker een eenmalige wachttijd en 2× schijfruimte (418 GB extra voor GLM 5.3).

## Herzien als
In M4 blijkt dat de kleine scale/bias-reads de I/O-benutting onder 80 % drukken.
