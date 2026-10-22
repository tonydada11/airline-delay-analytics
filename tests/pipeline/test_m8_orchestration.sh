#!/usr/bin/env bash
# AI-generated test code: generated with OpenAI Codex.
# Check production pipeline orchestration behavior.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

TEST_TMP="$(mktemp -d "${TMPDIR:-/tmp}/airline-m8-orchestration.XXXXXX")"
trap 'rm -rf "${TEST_TMP}"' EXIT

FAKE_BIN="${TEST_TMP}/bin"
TRACE_FILE="${TEST_TMP}/docker-trace.log"
mkdir -p "${FAKE_BIN}"

cat >"${FAKE_BIN}/docker" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "$*" == "compose version" ]]; then
  exit 0
fi

if [[ "$*" == "compose ps --status running --services" ]]; then
  printf '%s\n' spark-master spark-worker-1 spark-worker-2 spark-notebook
  exit 0
fi

if [[ "${1:-}" == "compose" && "${2:-}" == "exec" ]]; then
  printf '%s\n' "$*" >>"${TRACE_FILE}"
  if [[ -n "${TEST_FAILURE_PATTERN:-}" && "$*" == *"${TEST_FAILURE_PATTERN}"* ]]; then
    exit 29
  fi
  exit 0
fi

printf 'Unexpected docker command: %s\n' "$*" >&2
exit 90
EOF
chmod +x "${FAKE_BIN}/docker"

export TRACE_FILE
PATH="${FAKE_BIN}:${PATH}" scripts/run_pipeline.sh >/dev/null

expected_order=(
  "airline.common.cluster_health"
  "airline.common.raw_snapshot"
  "airline.ods.build_flights"
  "test_ods_quality.py"
  "airline.dwd.build_flights"
  "test_dwd_quality.py"
  "airline.dim.build_date"
  "airline.dim.build_carrier"
  "airline.dim.build_airport"
  "airline.dim.build_route"
  "test_dim_quality.py"
  "airline.dws.build_carrier_daily"
  "airline.dws.build_origin_airport_daily"
  "airline.dws.build_route_daily"
  "test_dws_quality.py"
  "airline.dws.build_ml_flight_features --phase selection"
  "test_ml_feature_quality.py"
  "airline.ads.build_monthly_operations"
  "airline.ads.build_carrier_scorecard"
  "airline.ads.build_airport_scorecard"
  "airline.ads.build_route_ranking"
  "test_ads_quality.py"
  "airline.ml.train"
  "test_training_bundle"
  "airline.ml.evaluate --split validation"
  "airline.dws.build_ml_flight_features --phase test"
  "airline.ml.evaluate --split test"
  "test_ml_quality.py$"
)

previous_line=0
for pattern in "${expected_order[@]}"; do
  current_line="$(grep -n -m1 "${pattern}" "${TRACE_FILE}" | cut -d: -f1)"
  if [[ -z "${current_line}" || "${current_line}" -le "${previous_line}" ]]; then
    printf 'Pipeline order check failed at: %s\n' "${pattern}" >&2
    exit 1
  fi
  previous_line="${current_line}"
done

: >"${TRACE_FILE}"
export TEST_FAILURE_PATTERN="test_dwd_quality.py"
set +e
PATH="${FAKE_BIN}:${PATH}" scripts/run_pipeline.sh >/dev/null 2>"${TEST_TMP}/failure.log"
failure_status=$?
set -e
unset TEST_FAILURE_PATTERN

if [[ "${failure_status}" -eq 0 ]]; then
  printf 'Controlled quality-gate failure unexpectedly passed.\n' >&2
  exit 1
fi

grep -q 'airline.ods.build_flights' "${TRACE_FILE}"
grep -q 'test_ods_quality.py' "${TRACE_FILE}"
grep -q 'airline.dwd.build_flights' "${TRACE_FILE}"
grep -q 'test_dwd_quality.py' "${TRACE_FILE}"

if grep -Eq 'airline\.dim|airline\.dws|airline\.ads|test_m8_pipeline_quality' "${TRACE_FILE}"; then
  printf 'Downstream command executed after the controlled failure.\n' >&2
  exit 1
fi

for required_suite in tests/unit tests/integration test_ods_quality.py \
  test_ads_quality.py test_ml_quality.py test_ml_feature_quality.py \
  test_cluster_environment.sh; do
  grep -q "${required_suite}" scripts/run_tests.sh
done

printf 'Production orchestration order and fail-fast checks passed.\n'

: >"${TRACE_FILE}"
export TEST_FAILURE_PATTERN="airline.common.raw_snapshot"
if PATH="${FAKE_BIN}:${PATH}" scripts/run_pipeline.sh >/dev/null 2>&1; then
  echo "Snapshot failure unexpectedly passed" >&2; exit 1
fi
if grep -q 'airline.ods' "${TRACE_FILE}"; then
  echo "ODS ran after snapshot failure" >&2; exit 1
fi

for failing in airline.ml.train test_training_bundle; do
  : >"${TRACE_FILE}"
  export TEST_FAILURE_PATTERN="${failing}"
  if PATH="${FAKE_BIN}:${PATH}" scripts/run_pipeline.sh >/dev/null 2>&1; then
    echo "Selection failure unexpectedly passed" >&2; exit 1
  fi
  if grep -Eq -- '--phase test|--split test' "${TRACE_FILE}"; then
    echo "Test stage ran before completed selection" >&2; exit 1
  fi
done
unset TEST_FAILURE_PATTERN
