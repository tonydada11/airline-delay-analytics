#!/usr/bin/env python3
"""Verify the locked additions and base runtime; emit machine-readable provenance."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import pyspark

root = Path(__file__).resolve().parents[1]
lock = root / 'requirements.lock'
packages = {}
for line in lock.read_text().splitlines():
    if not line or line.startswith('#'):
        continue
    name, pinned = line.split()[0].split('==')
    actual = version(name)
    if actual != pinned:
        raise SystemExit(f'Version mismatch: {name}: {actual} != {pinned}')
    packages[name] = actual
assert 'pyspark' not in packages
assert platform.python_version() == '3.10.12'
assert pyspark.__version__ == '3.5.9'
assert platform.machine() in ('aarch64', 'x86_64')
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True, stdout=sys.stderr)
print(json.dumps({'python': platform.python_version(), 'pyspark': pyspark.__version__,
                  'architecture': platform.machine(), 'packages': packages,
                  'requirements_lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest()}, indent=2, sort_keys=True))
