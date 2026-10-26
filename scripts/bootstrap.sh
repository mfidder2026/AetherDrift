#!/usr/bin/env bash
# Sets up the engine environment. Installs uv if it is missing.
set -euo pipefail
cd "$(dirname "$0")/../engine"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv niet gevonden; installeren…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync
echo "Klaar. Start de engine met: ./scripts/dev-engine.sh"
