#!/usr/bin/env bash
# AI-generated test code: generated with OpenAI Codex.
# Check cluster startup behavior with controlled command stubs.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_TMP="$(mktemp -d "${TMPDIR:-/tmp}/airline-cluster-environment.XXXXXX")"
trap 'rm -rf "${TEST_TMP}"' EXIT

# Create one test case directory.
make_case() {
  local name="$1"
  local case_root="${TEST_TMP}/${name}"
  mkdir -p "${case_root}/bin"
  printf '%s\n' "${case_root}"
}

# Create a Docker stub for the test.
write_fake_docker() {
  local case_root="$1"
  cat >"${case_root}/bin/docker" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'docker %s\n' "$*" >>"${TRACE_FILE}"

case "$*" in
  "compose version"|"compose up -d --build"|"compose down") exit 0 ;;
  "compose port spark-notebook 8888") printf '127.0.0.1:8889\n'; exit 0 ;;
  "compose port spark-master 8080") printf '127.0.0.1:8082\n'; exit 0 ;;
  "compose port spark-notebook 4040") printf '127.0.0.1:4041\n'; exit 0 ;;
  "context show")
    if [[ -f "${CONTEXT_MARKER}" ]]; then printf 'colima\n'; else printf 'desktop-linux\n'; fi
    exit 0
    ;;
  "context use colima")
    : >"${CONTEXT_MARKER}"
    exit 0
    ;;
  "info")
    if [[ "${DOCKER_MODE}" == "healthy" || -f "${CONTEXT_MARKER}" ]]; then exit 0; fi
    exit 1
    ;;
esac

printf 'Unexpected docker command: %s\n' "$*" >&2
exit 90
EOF
  chmod +x "${case_root}/bin/docker"
}

# Create a Colima stub for the test.
write_fake_colima() {
  local case_root="$1"
  cat >"${case_root}/bin/colima" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'colima %s\n' "$*" >>"${TRACE_FILE}"
case "${1:-}" in
  status) exit 1 ;;
  start) exit 0 ;;
esac
exit 90
EOF
  chmod +x "${case_root}/bin/colima"

  cat >"${case_root}/bin/df" <<'EOF'
#!/usr/bin/env bash
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n'
printf 'fake 100000000 1 99999999 1%% /\n'
EOF
  chmod +x "${case_root}/bin/df"
}

# Run the cluster startup script with the test stubs.
run_start() {
  local case_root="$1"
  local mode="$2"
  TRACE_FILE="${case_root}/trace.log" \
  CONTEXT_MARKER="${case_root}/colima-context" \
  DOCKER_MODE="${mode}" \
  PATH="${case_root}/bin:/usr/bin:/bin" \
    "${PROJECT_ROOT}/scripts/start_cluster.sh"
}

healthy_root="$(make_case healthy)"
write_fake_docker "${healthy_root}"
run_start "${healthy_root}" healthy >"${healthy_root}/stdout.log"
grep -Fq 'http://localhost:8889' "${healthy_root}/stdout.log"
grep -Fq 'http://localhost:8082' "${healthy_root}/stdout.log"
grep -Fq 'http://localhost:4041' "${healthy_root}/stdout.log"
grep -Fxq 'docker info' "${healthy_root}/trace.log"
grep -Fxq 'docker compose up -d --build' "${healthy_root}/trace.log"
if grep -q 'context use\|colima ' "${healthy_root}/trace.log"; then
  echo "A healthy active daemon must not switch Docker contexts or invoke Colima." >&2
  exit 1
fi

fallback_root="$(make_case fallback)"
write_fake_docker "${fallback_root}"
write_fake_colima "${fallback_root}"
run_start "${fallback_root}" unavailable >/dev/null
grep -Fxq 'colima status' "${fallback_root}/trace.log"
grep -Fxq 'colima start --cpu 2 --memory 6 --disk 40' "${fallback_root}/trace.log"
grep -Fxq 'docker context use colima' "${fallback_root}/trace.log"
grep -Fxq 'docker compose up -d --build' "${fallback_root}/trace.log"

missing_root="$(make_case missing)"
write_fake_docker "${missing_root}"
set +e
run_start "${missing_root}" unavailable >"${missing_root}/stdout.log" 2>"${missing_root}/stderr.log"
missing_status=$?
set -e
if [[ "${missing_status}" -eq 0 ]]; then
  echo "A missing runtime unexpectedly passed." >&2
  exit 1
fi
grep -q 'No healthy Docker daemon is available' "${missing_root}/stderr.log"

stop_root="$(make_case stop)"
write_fake_docker "${stop_root}"
TRACE_FILE="${stop_root}/trace.log" \
CONTEXT_MARKER="${stop_root}/colima-context" \
DOCKER_MODE=healthy \
PATH="${stop_root}/bin:/usr/bin:/bin" \
  "${PROJECT_ROOT}/scripts/stop_cluster.sh" >/dev/null
grep -Fxq 'docker compose down' "${stop_root}/trace.log"
if grep -q -- '--context\|colima ' "${stop_root}/trace.log"; then
  echo "Shutdown must use the active context and leave the host runtime alone." >&2
  exit 1
fi

python3 "${PROJECT_ROOT}/tests/pipeline/test_runtime_contract.py"

printf 'Docker daemon discovery, Colima fallback, failure, and shutdown checks passed.\n'
