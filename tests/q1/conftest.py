"""Q1-only import path and clearly labelled software-verification fixtures."""

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))


def pytest_addoption(parser):
    parser.addoption("--reuse-q1-artifacts",action="store_true",default=False,
                     help="M8 audit: validate/reuse completed M4/M5 CLI artifacts instead of repeating full experiments; unit tests still execute")


def existing_run(relative):
    """Strict original manifest validation, not a fixture pretending to train."""
    import hashlib
    directory = ROOT/relative
    manifest = json.loads((directory/"artifact_manifest.json").read_bytes())
    for item in manifest["files"]:
        path = directory/item["name"]
        assert path.resolve().is_relative_to(directory.resolve())
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    return directory,json.loads((directory/"metrics.json").read_text(encoding="utf-8"))


@pytest.fixture
def synthetic_cases():
    path = Path(__file__).parent / "fixtures/synthetic/m2a_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    assert cases["provenance"] == "SYNTHETIC_SOFTWARE_VERIFICATION_ONLY"
    return cases
