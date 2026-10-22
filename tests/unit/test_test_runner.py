# AI-generated test code: generated with OpenAI Codex.
"""Verify full-data jobs use independent drivers and rebuild is opt-in."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("rebuild", ["0", "1"])
def test_quality_process_isolation(tmp_path, rebuild):
    """Exercise the quality process isolation case."""
    root = Path(__file__).resolve().parents[2]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(root / "scripts/run_tests.sh", scripts / "run_tests.sh")
    for name in ("scripts_download_bts.sh", "start_cluster.sh", "stop_cluster.sh", "run_pipeline.sh"):
        (scripts / name).write_text("#!/bin/bash\nexit 0\n")
    pipeline = tmp_path / "tests/pipeline"
    pipeline.mkdir(parents=True)
    for name in ("test_cluster_environment.sh", "test_m8_orchestration.sh"):
        (pipeline / name).write_text("#!/bin/bash\nexit 0\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text('''#!/bin/bash
if [[ "$2" == "ps" ]]; then
  printf '%s\\n' spark-master spark-worker-1 spark-worker-2 spark-notebook
elif [[ "$2" == "exec" ]]; then
  printf '%s\\n' "$*" >> "$TRACE_FILE"
fi
''')
    docker.chmod(0o755)
    trace = tmp_path / "trace"
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               TRACE_FILE=str(trace), AIRLINE_RUN_REBUILD_ACCEPTANCE=rebuild)
    subprocess.run(["bash", str(scripts / "run_tests.sh")], env=env,
                   check=True, capture_output=True, text=True)
    calls = trace.read_text().splitlines()
    quality = [call for call in calls if "tests/data_quality/" in call]
    ordinary = [call for call in quality if "::" not in call]
    assert len(ordinary) == 7
    assert all(call.count("tests/data_quality/") == 1 for call in ordinary)
    assert all("AIRLINE_RUN_REBUILD_ACCEPTANCE=0" in call for call in ordinary)
    comprehensive = [call for call in quality if "::" in call]
    assert len(comprehensive) == int(rebuild)
    if comprehensive:
        assert "AIRLINE_RUN_REBUILD_ACCEPTANCE=1" in comprehensive[0]
        assert comprehensive[0].endswith("::test_comprehensive_rebuild_multiset_equality")
