# ADR-006: Delen op het netwerk — wat bereikbaar is en hoe het beveiligd is

Status: aangenomen (2026-10-26)

## Context
M5 maakt de engine bereikbaar voor andere apparaten: een telefoon via de webchat, en apps zoals Open WebUI via de OpenAI- of Ollama-API. Loopback-verzoeken hebben geen sleutel nodig (§7.1), want daar is de app zelf de client. Een review liet zien dat "loopback" niet automatisch "de gebruiker" betekent.

## Besluit
**1. Twee listeners.** De engine luistert altijd op `127.0.0.1`. Met delen aan komt er een tweede listener op `0.0.0.0` bij, op dezelfde poort. Het model wordt daarbij niet opnieuw geladen. Lukt het starten niet (poort bezet), dan blijft delen uit en volgt de fout `share_failed`.

**2. Wat bereikbaar is via het netwerk** (altijd met `Authorization: Bearer <sleutel>`):
- `/v1/*`: OpenAI (chat, completions, models)
- `/api/*`: Ollama (`chat`, `generate`, `tags`, `ps`, `version`)
- `/health`

Zonder sleutel bereikbaar zijn alleen de webchat-pagina (`/`) en het manifest. De pagina bevat geen gegevens. Een browser kan bij het openen van een URL geen header meesturen; de pagina stuurt de sleutel mee bij haar eigen API-aanroepen. Al het beheer (`/aether/v1/*`: modellen laden, instellingen, de sleutel zelf) is **alleen lokaal** bereikbaar en geeft via het netwerk `403 local_only`. Zo kan een apparaat dat de Mac gebruikt hem niet herconfigureren.

**3. Loopback is alleen de Mac zelf, niet een webpagina.** Een website in de browser van de gebruiker kan verzoeken naar `127.0.0.1` sturen (CSRF), of 127.0.0.1 bereiken onder een eigen domeinnaam (DNS rebinding). Lokale verzoeken worden daarom geweigerd (`403 foreign_origin`) als:
- de `Host` geen loopback-naam is (`127.0.0.1`, `localhost`, `[::1]`), of
- er een `Origin` meekomt die geen loopback-pagina is.

De app zelf en curl sturen geen `Origin`; de webchat op loopback heeft een loopback-`Origin`.

**4. De sleutel.**
- 32 willekeurige bytes (base64url), bewaard in de Keychain van de gebruiker.
- De app geeft de sleutel na elke engine-start door via `PUT /aether/v1/api-key` (loopback) en pas daarna de opgeslagen instellingen. Een opgeslagen "delen aan" opent het netwerk dus nooit zonder sleutel.
- Er is bewust geen `GET` voor de sleutel.
- De QR-code bevat `http://<ip>:<poort>/#key=<sleutel>`. De sleutel staat in het fragment en komt dus nooit in een serverlog. De webchat bewaart hem in `localStorage` en haalt hem uit de adresbalk.

**5. Chat-API's laden geen paden.** Ook met "Clients mogen van model wisselen" aan accepteren `/v1` en `/api` alleen model-id's, geen mappen op schijf.

**6. De webchat** is één HTML-bestand zonder externe bronnen. Strikte CSP: `connect-src 'self'`, `frame-ancestors 'none'`, geen referrer. Markdown wordt eerst volledig ge-escaped; daarna worden alleen enkele tags zonder attributen toegevoegd.

## Afwijking van §8.8
De QR-code gebruikt het IP-adres in plaats van `<naam>.local`. Niet elke telefoon lost `.local`-namen op (vooral Android). De `.local`-URL staat als tekst in het Verbind-scherm.

## Gevolgen
- Wie de sleutel heeft, kan chatten en de lijst met modellen zien, maar niets beheren.
- Een nieuwe sleutel maken verbreekt alle gekoppelde apparaten. Ze moeten de QR-code opnieuw scannen.
