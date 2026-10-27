# ADR-007: Zelfstandige Python-runtime in Release-builds

Status: aangenomen (2026-10-26)

## Context
Tot M5 installeerde de app bij de eerste start de Python-runtime via `uv sync` in `~/Library/Application Support/AetherDrift/runtime`. Dat vraagt eenmalig internet en ~1 minuut. M6 vereist dat een `.app` op een schone Mac werkt zonder internet, behalve voor het downloaden van modellen.

## Besluit
- `app/scripts/bundle-runtime.sh` (bouwstap; alleen actief met `AETHER_BUNDLE_RUNTIME=1`, wat `build-app.sh Release` zet) doet het volgende:
  - Het zet een relocatable CPython 3.12 (python-build-standalone, via `uv python install`) in `Contents/Resources/runtime/python`.
  - Het installeert alle afhankelijkheden uit `uv.lock` met `uv pip install --target` in `runtime/site-packages`. Bewust zonder venv: een venv bevat absolute paden.
  - Het resultaat wordt per `uv.lock`-hash gecachet in `app/build/runtime-cache/`.
- Elk Mach-O-bestand wordt gesigneerd met hardened runtime. Het Python-programma krijgt `com.apple.security.cs.disable-library-validation`. Zonder die entitlement weigert macOS de extensies (MLX, numpy) te laden. Bij ad-hoc-signering hebben ze geen Team ID; met een Developer ID dekt de entitlement ook wheels die niet opnieuw gesigneerd zijn.
- `EngineRuntime` gebruikt de meegebundelde runtime als die bestaat: `PYTHONPATH=engine:site-packages` en `PYTHONNOUSERSITE=1`. Er wordt dan niets geïnstalleerd. Debug-builds behouden de installatie bij de eerste start (snellere builds).

## Gevolgen
- De app wordt ~490 MB, waarvan 436 MB runtime (MLX, transformers, numpy, Python).
- De eerste start heeft geen internet nodig; de engine is na ~0,3 s klaar.
- Notariseren vraagt een Developer ID-certificaat (`DEVELOPMENT_TEAM` en `CODE_SIGN_IDENTITY` in `project.yml`). Daarna: `xcrun notarytool submit` + `stapler`. Dat kan pas zodra er een certificaat is.
