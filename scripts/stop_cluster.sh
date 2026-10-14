#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "Required command not found: docker" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose v2 is required (the 'docker compose' command)." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "The active Docker daemon is not available. Start the runtime used for this project and retry." >&2
  exit 1
fi

docker compose down

echo "Spark services stopped. The host Docker runtime was left running."
