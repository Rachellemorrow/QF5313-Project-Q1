"""Q1-only import path and clearly labelled software-verification fixtures."""

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))


@pytest.fixture
def synthetic_cases():
    path = Path(__file__).parent / "fixtures/synthetic/m2a_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    assert cases["provenance"] == "SYNTHETIC_SOFTWARE_VERIFICATION_ONLY"
    return cases
