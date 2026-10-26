#!/usr/bin/env bash
# Engine on 127.0.0.1:8080 that restarts when code in engine/aether changes.
# Extra arguments go to aether-engine, e.g.: ./scripts/dev-engine.sh --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
set -euo pipefail
cd "$(dirname "$0")/../engine"
exec uv run --with watchfiles watchfiles --filter python "aether-engine --port 8080 $*" aether
