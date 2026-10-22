# AI-generated test code: generated with OpenAI Codex.
"""Tests runtime contract behavior."""

from pathlib import Path


def test_supported_architecture_and_pinned_runtime_contract():
    """Exercise the supported architecture and pinned runtime contract case."""
    root = Path(__file__).resolve().parents[2]
    compose = (root / 'docker-compose.yml').read_text()
    dockerfile = (root / 'docker/Dockerfile').read_text()
    verifier = (root / 'scripts/verify_dependencies.py').read_text()
    assert 'platform:' not in compose  # Docker selects the host's native architecture.
    assert '3.5.9-python3@sha256:' in dockerfile
    assert 'COPY requirements.lock /tmp/requirements.lock' in dockerfile
    assert 'python3 -m pip install --no-cache-dir -r /tmp/requirements.lock' in dockerfile
    assert 'python3 -m pip check' in dockerfile
    assert "platform.machine() in ('aarch64', 'x86_64')" in verifier
    assert "platform.python_version() == '3.10.12'" in verifier
    assert "pyspark.__version__ == '3.5.9'" in verifier
    assert 'pyspark==' not in (root / 'requirements.lock').read_text()


if __name__ == '__main__':
    test_supported_architecture_and_pinned_runtime_contract()
    print('Native ARM64/x86-64 architecture, digest pin, and dependency runtime contracts passed.')
