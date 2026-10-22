#!/usr/bin/env bash
# Run unit, integration, and configured quality checks.

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CURRENT_STEP="startup"

# Print a concise status message.
log() {
  printf '[M8-TEST][%s][%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$1" "$2"
}

# Report the command that failed.
on_error() {
  local exit_code=$?
  log "FAIL" "${CURRENT_STEP} (exit ${exit_code})" >&2
  exit "${exit_code}"
}

trap on_error ERR

# Run one named command and record its result.
run_step() {
  CURRENT_STEP="$1"
  shift
  log "START" "${CURRENT_STEP}"
  "$@"
  log "PASS" "${CURRENT_STEP}"
}

# Check that the required runtime is available.
preflight() {
  local running_services

  command -v docker >/dev/null 2>&1
  docker compose version >/dev/null
  running_services="$(docker compose ps --status running --services)"

  for service_name in spark-master spark-worker-1 spark-worker-2 spark-notebook; do
    if ! grep -Fxq "${service_name}" <<<"${running_services}"; then
      printf 'Required service is not running: %s\n' "${service_name}" >&2
      return 1
    fi
  done
}

run_step "Shell syntax" bash -n scripts/scripts_download_bts.sh \
  scripts/start_cluster.sh scripts/stop_cluster.sh scripts/run_pipeline.sh \
  scripts/run_tests.sh tests/pipeline/test_cluster_environment.sh \
  tests/pipeline/test_m8_orchestration.sh
run_step "Docker runtime discovery and context-neutral shutdown tests" \
  bash tests/pipeline/test_cluster_environment.sh
run_step "Pipeline orchestration and controlled fail-fast tests" \
  bash tests/pipeline/test_m8_orchestration.sh
run_step "Preflight: running Spark services" preflight
# Run Python inside the Spark container.
runtime_python() {
  docker compose exec -T -w "${AIRLINE_RUNTIME_ROOT:-/workspace}" -e "AIRLINE_TEST_TMP_ROOT=${AIRLINE_RUNTIME_ROOT:-/workspace}/data/.test-tmp" \
    -e "PYTHONPATH=${AIRLINE_RUNTIME_ROOT:-/workspace}/src:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip" \
    -e AIRLINE_RUN_FULL_DATA_QUALITY=1 -e AIRLINE_RUN_FULL_ML_QUALITY=1 \
    -e "AIRLINE_RUN_REBUILD_ACCEPTANCE=${AIRLINE_RUN_REBUILD_ACCEPTANCE:-0}" \
    spark-notebook python3 "$@"
}
run_step "Cluster health" runtime_python -m airline.common.cluster_health
run_step "Pinned raw snapshot" runtime_python -m airline.common.raw_snapshot
run_step "Unit and integration tests" runtime_python -m pytest -q tests/unit tests/integration
# A separate process per suite releases the Spark driver JVM between full-data jobs.
for quality_suite in \
  tests/data_quality/test_ods_quality.py tests/data_quality/test_dwd_quality.py \
  tests/data_quality/test_dim_quality.py tests/data_quality/test_dws_quality.py \
  tests/data_quality/test_ads_quality.py tests/data_quality/test_ml_feature_quality.py \
  tests/data_quality/test_ml_quality.py; do
  AIRLINE_RUN_REBUILD_ACCEPTANCE=0 run_step "Full-data quality: ${quality_suite}" \
    runtime_python -m pytest -q -s "${quality_suite}"
done
if [[ "${AIRLINE_RUN_REBUILD_ACCEPTANCE:-0}" == "1" ]]; then
  run_step "Comprehensive feature rebuild equality" runtime_python -m pytest -q -s \
    tests/data_quality/test_ml_feature_quality.py::test_comprehensive_rebuild_multiset_equality
fi
CURRENT_STEP="complete"
log "PASS" "Production tests passed (rebuild acceptance=${AIRLINE_RUN_REBUILD_ACCEPTANCE:-0})"
