#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CURRENT_STEP="startup"

log() {
  printf '[PIPELINE][%s][%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$1" "$2"
}

on_error() {
  local exit_code=$?
  log "FAIL" "${CURRENT_STEP} (exit ${exit_code})" >&2
  exit "${exit_code}"
}

trap on_error ERR

run_step() {
  CURRENT_STEP="$1"
  shift
  log "START" "${CURRENT_STEP}"
  "$@"
  log "PASS" "${CURRENT_STEP}"
}

spark_python() {
  docker compose exec -T -w "${AIRLINE_RUNTIME_ROOT:-/workspace}" -e "AIRLINE_TEST_TMP_ROOT=${AIRLINE_RUNTIME_ROOT:-/workspace}/data/.test-tmp" -e "PYTHONPATH=${AIRLINE_RUNTIME_ROOT:-/workspace}/src:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip" spark-notebook python3 "$@"
}

spark_quality() {
  local environment_name="$1"
  shift
  docker compose exec -T -w "${AIRLINE_RUNTIME_ROOT:-/workspace}" -e "AIRLINE_TEST_TMP_ROOT=${AIRLINE_RUNTIME_ROOT:-/workspace}/data/.test-tmp" -e "PYTHONPATH=${AIRLINE_RUNTIME_ROOT:-/workspace}/src:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip" -e "${environment_name}=1" \
    spark-notebook python3 -m pytest -q -s "$@"
}

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

run_step "Preflight: running Spark services" preflight
run_step "Preflight: registered workers and Spark execution" spark_python -m airline.common.cluster_health
run_step "Preflight: pinned raw snapshot" spark_python -m airline.common.raw_snapshot

run_step "ODS: build flights" \
  spark_python -m airline.ods.build_flights
run_step "ODS quality gate" \
  spark_quality AIRLINE_RUN_FULL_DATA_QUALITY \
  tests/data_quality/test_ods_quality.py

run_step "DWD: build trusted flights" \
  spark_python -m airline.dwd.build_flights
run_step "DWD quality gate" \
  spark_quality AIRLINE_RUN_FULL_DATA_QUALITY \
  tests/data_quality/test_dwd_quality.py

run_step "DIM: build date" \
  spark_python -m airline.dim.build_date
run_step "DIM: build carrier" \
  spark_python -m airline.dim.build_carrier
run_step "DIM: build airport" \
  spark_python -m airline.dim.build_airport
run_step "DIM: build route" \
  spark_python -m airline.dim.build_route
run_step "DIM quality gate" \
  spark_quality AIRLINE_RUN_FULL_DATA_QUALITY \
  tests/data_quality/test_dim_quality.py

run_step "DWS: build carrier daily" \
  spark_python -m airline.dws.build_carrier_daily
run_step "DWS: build origin-airport daily" \
  spark_python -m airline.dws.build_origin_airport_daily
run_step "DWS: build route daily" \
  spark_python -m airline.dws.build_route_daily
run_step "DWS quality gate" \
  spark_quality AIRLINE_RUN_FULL_DATA_QUALITY \
  tests/data_quality/test_dws_quality.py

run_step "ML features: build canonical rolling histories" \
  spark_python -m airline.dws.build_ml_flight_features
run_step "ML feature quality and chronology gate" \
  spark_quality AIRLINE_RUN_FULL_ML_QUALITY \
  tests/data_quality/test_ml_feature_quality.py

run_step "ADS: build monthly operations" \
  spark_python -m airline.ads.build_monthly_operations
run_step "ADS: build carrier scorecard" \
  spark_python -m airline.ads.build_carrier_scorecard
run_step "ADS: build airport scorecard" \
  spark_python -m airline.ads.build_airport_scorecard
run_step "ADS: build route ranking" \
  spark_python -m airline.ads.build_route_ranking
run_step "ADS quality gate" \
  spark_quality AIRLINE_RUN_FULL_DATA_QUALITY \
  tests/data_quality/test_ads_quality.py

run_step "Training: fit and publish frozen Logistic Regression" \
  spark_python -m airline.ml.train
run_step "Training quality gate" \
  spark_quality AIRLINE_RUN_FULL_ML_QUALITY \
  tests/data_quality/test_ml_quality.py::test_training_bundle
run_step "Validation evaluation" \
  spark_python -m airline.ml.evaluate --split validation
run_step "Historical test evaluation" \
  spark_python -m airline.ml.evaluate --split test
run_step "Final artifact quality gate" \
  spark_quality AIRLINE_RUN_FULL_ML_QUALITY \
  tests/data_quality/test_ml_quality.py

CURRENT_STEP="complete"
log "PASS" "Raw-to-artifacts production pipeline completed"
