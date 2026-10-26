#!/usr/bin/env bash
# Lint + tests for the engine (integration tests need the small test model in the HF cache).
set -euo pipefail
cd "$(dirname "$0")/../engine"
uv run ruff check aether tests ../bench
uv run ruff format --check aether tests ../bench
uv run pytest -q "$@"
