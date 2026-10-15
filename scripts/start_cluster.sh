#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI not found. Install Docker Desktop, Docker Engine with Compose v2, Rancher Desktop, or Colima." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose v2 is required (the 'docker compose' command)." >&2
  exit 1
fi

if docker info >/dev/null 2>&1; then
  active_context="$(docker context show 2>/dev/null || printf 'current')"
  echo "Using the active Docker daemon (context: ${active_context})."
else
  if ! command -v colima >/dev/null 2>&1; then
    echo "No healthy Docker daemon is available. Start Docker Desktop, Docker Engine, Rancher Desktop, or install/start Colima, then retry." >&2
    exit 1
  fi

  if colima status >/dev/null 2>&1; then
    echo "Colima is already running; selecting its Docker context."
  else
    available_kib="$(df -Pk "${PROJECT_ROOT}" | awk 'NR == 2 {print $4}')"
    required_kib=$((25 * 1024 * 1024))
    if (( available_kib < required_kib )); then
      available_gib=$((available_kib / 1024 / 1024))
      echo "At least 25 GiB of free host storage is required to create the Colima VM and build the Spark image." >&2
      echo "Currently available: approximately ${available_gib} GiB." >&2
      exit 1
    fi

    echo "Starting Colima with 2 CPUs, 6 GiB memory, and a 40 GiB disk..."
    colima start --cpu 2 --memory 6 --disk 40
  fi

  docker context use colima >/dev/null

  if [[ "$(docker context show)" != "colima" ]]; then
    echo "Docker did not switch to the Colima context." >&2
    exit 1
  fi

  if ! docker info >/dev/null 2>&1; then
    echo "Colima started, but its Docker daemon is not healthy. Run 'colima status' and retry." >&2
    exit 1
  fi
fi

cd "${PROJECT_ROOT}"
docker compose up -d --build

service_url() {
  local address
  address="$(docker compose port "$1" "$2" 2>/dev/null)" || address="127.0.0.1:$2"
  printf 'http://%s' "${address/127.0.0.1/localhost}"
}

echo
echo "Airline Spark cluster is starting:"
echo "  JupyterLab:          $(service_url spark-notebook 8888)"
echo "  Jupyter login URL:   docker compose exec spark-notebook jupyter server list"
echo "  Spark Master UI:     $(service_url spark-master 8080)"
echo "  Spark Application UI $(service_url spark-notebook 4040) (while a Spark app is active)"
