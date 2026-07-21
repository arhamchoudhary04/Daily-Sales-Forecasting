#!/usr/bin/env bash
# Build and run both services with compose. Frontend on :3000, backend on :8000.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

if docker compose version >/dev/null 2>&1; then
  docker compose up --build
else
  docker-compose up --build
fi
