# AetherDrift

*Infinite Intelligence. Local Hardware.*

**English** · [Nederlands](README.nl.md)

AetherDrift runs large language models locally on a Mac with Apple Silicon. Models that fit in memory run at full speed. Larger ones are streamed from the SSD by the Drift Engine, even GLM 5.3 (418 GB) on a Mac with 128 GB of RAM. Other devices can use the model through a web chat (scan a QR code), an OpenAI-compatible API or the Ollama API.

![The AetherDrift app (Dutch interface) with Qwen3-30B-A3B in RAM and the live panel on the right](doc/images/aetherdrift.png)

> Status: milestones M0–M5 are built. Of M6, the menu bar, quick chat (⌥Space), Dutch/English and a self-contained runtime are done; signing and notarization wait for a Developer ID. See [doc/ARCHITECTURE.md](doc/ARCHITECTURE.md) (in Dutch).

## Features

- **Full speed for models that fit in memory**, on the GPU with MLX. Qwen3-30B-A3B runs at about 80 tokens/s on an M3 Max.
- **The Drift Engine for models larger than RAM.** Dense models stream layer by layer. Mixture-of-experts models keep the experts they use most in RAM and read the rest from the SSD when they are needed.
- **Thinking control** for reasoning models such as GLM and Qwen3: off, brief, thorough or maximum. *Automatic* skips thinking for models that stream via Drift.
- **Turbo** for large mixture-of-experts models: skips experts that are not in memory yet and contribute little. About twice as fast, slightly less accurate.
- **Use it from other devices**: a web chat via QR code, an OpenAI-compatible API (`/v1`) and the Ollama API. Over the network only the chat APIs are reachable, and only with the API key.
- **A native macOS app**: a live panel with memory, speed and a map of what is in RAM, a menu bar item and a quick chat. In English and Dutch.

## Requirements

- A Mac with Apple Silicon (M1 or later) and macOS 14 or later.
- Free SSD space for the models you download.
- To build from source: Xcode, [uv](https://github.com/astral-sh/uv) and XcodeGen (`brew install uv xcodegen`).

## Install

Download the zip from [Releases](https://github.com/mfidder2026/AetherDrift/releases), unzip it and move AetherDrift to Applications. It carries its own Python runtime and works offline.

The app is not signed with a Developer ID yet, so macOS blocks the first launch. Open System Settings → Privacy & Security and click **Open Anyway**, or run:

```bash
xattr -dr com.apple.quarantine /Applications/AetherDrift.app
```

## Build from source

```bash
./scripts/build-app.sh            # builds and tests; the app is in app/build/Build/Products/Debug/
open app/build/Build/Products/Debug/AetherDrift.app
./scripts/build-app.sh Release    # with its own Python runtime: works offline (ADR-007)
```

The app starts the engine itself; no terminal needed. A Debug build installs the runtime on first launch, which needs internet once. A Release build has the runtime on board.

## Speed of large models

When a model does not fit in memory, the Drift Engine reads it from the SSD while it computes, and the SSD sets the pace. GLM 5.3 (418 GB) on a Mac with 128 GB runs at 1.2–1.4 tokens per second, or about 3 with Turbo. After a new question the first word takes 10–30 seconds, 6–15 with Turbo ([measurements](bench/reports/2026-10-27-GLM-speed.md), in Dutch).

- **Thinking** (light bulb next to the input field): models such as GLM and Qwen3 can think before they answer. Via Drift that can take many minutes, so *Automatic* skips it for models that stream. Apps choose for themselves with `reasoning_effort` (OpenAI) or `think` (Ollama).
- **Turbo** (mode menu in the toolbar) skips experts that are not in memory yet and contribute little. GLM 5.3 becomes about twice as fast. Answers differ slightly, comparable to a coarser quantization. Turbo is off by default.
- **A model that fits in RAM** is tens of times faster. GLM 5.3 fits (almost) entirely in a Mac with 512 GB.

## Connect other devices

In the app, choose **Connect a device** (⌘⇧C) and turn on **Share on network**. Scan the QR code with your phone to open the web chat, or use the base URL and the API key in any OpenAI-compatible app (Open WebUI, LibreChat, Continue). Ollama clients use the same port without `/v1`. Over the network only the chat APIs are reachable; management works only on the Mac itself (ADR-006).

## Engine only (developers)

```bash
./scripts/bootstrap.sh
./scripts/dev-engine.sh --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

Every OpenAI-compatible client works with base URL `http://127.0.0.1:8080/v1`. Live statistics: `curl -N http://127.0.0.1:8080/aether/v1/stats/stream`. API documentation: <http://127.0.0.1:8080/docs>.

## Tests and benchmarks

```bash
./scripts/test.sh                                   # lint + tests
uv run --project engine python bench/ssd.py         # SSD speed
uv run --project engine python bench/generate.py --model mlx-community/Qwen3-30B-A3B-4bit
uv run --project engine python bench/drift.py --model mlx-community/Qwen2.5-7B-Instruct-4bit --weights-budget 2GB
```

## Documents

The design documents are in Dutch.

- [doc/start-prompt-v2.md](doc/start-prompt-v2.md): the design and the milestones
- [doc/ARCHITECTURE.md](doc/ARCHITECTURE.md): what has been built and where it deviates from the design
- [doc/adr/](doc/adr/): architecture decisions
- [bench/reports/](bench/reports/): measurements
