"""Approach A development/CLI verification with a strict real-data gate."""

from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from q1.approach_a import calibrate_quotes, empirical_gate, make_metrics, prepare_quotes
from q1.classical_svi import CalibrationConfig
from q1.market_data import load_registered_q1
from q1.verification_fixtures import development_quotes


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT/"code/approach1_q1_svi.py"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("No network in M4")))


def test_synthetic_full_workflow_and_required_metrics_schema():
    frame, fixture = development_quotes()
    result = calibrate_quotes(frame, mode="development")
    metrics = make_metrics(result, mode="development", dataset="q1_synthetic_svi_verification", seed=42,
                           config=CalibrationConfig(), input_rows=len(frame), fixture=fixture)
    assert set(metrics) == {"params", "metrics", "fit_seconds", "history"}
    assert metrics["metrics"]["successful_slices"] == 4
    assert metrics["metrics"]["max_known_parameter_absolute_error"] < 1e-6
    assert metrics["metrics"]["cross_date_p95_of_date_maxima"] < 1e-8
    assert not metrics["params"]["empirical_comparison_eligible"]
    assert len(result["quote_ledger"]) == 328 and len(result["parity"]) == 164
    assert max(abs(row["dividend_adjusted_residual"]) for row in result["parity"]) < 5e-12
    assert max(abs(row["examination_residual"]) for row in result["parity"]) > 0.5
    json.dumps(metrics, allow_nan=False)


def test_missing_pair_invalid_inversion_and_failed_slices_are_not_dropped():
    frame, _ = development_quotes()
    ledger, pairs = prepare_quotes(frame.drop(index=1), mode="development")
    assert any(row["status"] == "MISSING_OR_AMBIGUOUS_CALL_PUT_PAIR" for row in pairs)
    assert any(not row["eligible_for_fit"] for row in ledger)
    bad = frame.copy()
    bad.loc[0, "price"] = -1
    result = calibrate_quotes(bad, mode="development")
    assert len(result["slices"]) == 4
    assert result["quote_ledger"][0]["iv"]["reason"] == "INVALID_PRICE"
    assert result["slices"][0]["quote_preparation_failures"] == 2
    assert result["repricing_summary"]["cross_date_median_of_date_maxima"] is None
    insufficient = calibrate_quotes(frame.iloc[:8], mode="development")
    assert len(insufficient["slices"]) == 1 and not insufficient["slices"][0]["fit"]["converged"]


def test_inconsistent_slice_conventions_fail_and_origins_cannot_mix():
    frame, _ = development_quotes()
    frame.loc[0, "forward"] *= 1.01
    result = calibrate_quotes(frame, mode="development")
    assert not result["slices"][0]["fit"]["converged"]
    assert any("INCONSISTENT_SLICE_MARKET_INPUTS" in row["reasons"] for row in result["quote_ledger"])
    frame.loc[0, "data_origin"] = "REAL_ARCHIVED_OBSERVATION"
    with pytest.raises(ValueError, match="origin"):
        prepare_quotes(frame, mode="development")


def test_empirical_gate_uses_strict_flags_not_partial_preview():
    loaded = load_registered_q1()
    accepted, report = empirical_gate(loaded["quotes"])
    assert not len(accepted) and report["input_rows"] == report["rejected_rows"] == 606
    assert not report["strict_pipeline_eligibility_relaxed"]
    forged = loaded["quotes"].copy()
    forged["eligible_for_empirical"] = True
    accepted, _ = empirical_gate(forged)
    assert not len(accepted)


def test_cli_help_and_dataset_mode_boundary(tmp_path):
    help_result = subprocess.run([sys.executable, str(CLI), "--help"], cwd=ROOT, capture_output=True, text=True)
    assert help_result.returncode == 0
    assert all(flag in help_result.stdout for flag in ["--dataset", "--seed", "--output-dir", "--mode"])
    invalid = subprocess.run([sys.executable, str(CLI), "--mode", "development", "--dataset", "q1_510050", "--output-dir", str(tmp_path)], cwd=ROOT, capture_output=True)
    assert invalid.returncode == 2


def test_cli_development_figures_are_inline_and_isolated(tmp_path):
    command = [sys.executable, str(CLI), "--mode", "development", "--dataset", "q1_synthetic_svi_verification",
               "--seed", "42", "--output-dir", str(tmp_path)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    output = tmp_path/"verification/approach1/q1_synthetic_svi_verification"
    metrics = json.loads((output/"metrics.json").read_text(encoding="utf-8"))
    assert metrics["history"]["SOFTWARE_VERIFICATION"] == "PASS"
    assert metrics["history"]["EMPIRICAL_REQUIREMENTS"] == "BLOCKED"
    assert len(metrics["history"]["figures"]) == 7
    assert not list((tmp_path/"approach1").glob("*/metrics.json"))

    class Scripts(HTMLParser):
        def __init__(self):
            super().__init__()
            self.external = []
        def handle_starttag(self, tag, attrs):
            if tag == "script":
                self.external.extend(value for name, value in attrs if name == "src")

    for name in metrics["history"]["figures"]:
        html = (output/name).read_text(encoding="utf-8")
        assert "SYNTHETIC SOFTWARE VERIFICATION" in html and "plotly.js" in html
        parser = Scripts()
        parser.feed(html)
        assert not parser.external
    manifest = json.loads((output/"artifact_manifest.json").read_text())
    for entry in manifest["files"]:
        assert hashlib.sha256((output/entry["name"]).read_bytes()).hexdigest() == entry["sha256"]
    again = subprocess.run(command, cwd=ROOT, capture_output=True)
    assert again.returncode == 2  # Existing run is preserved, not overwritten.


def test_cli_current_empirical_dataset_is_blocked_without_figures(tmp_path):
    result = subprocess.run([sys.executable, str(CLI), "--mode", "empirical", "--dataset", "q1_510050",
                             "--output-dir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "BLOCKED" in result.stdout
    output = tmp_path/"approach1/q1_510050"
    metrics = json.loads((output/"metrics.json").read_text())
    assert metrics["history"]["status"] == "BLOCKED"
    assert metrics["history"]["eligibility"]["eligible_rows"] == 0
    assert metrics["metrics"]["attempted_slices"] == 0 and metrics["fit_seconds"] == 0
    assert metrics["metrics"]["cross_date_p95_of_date_maxima"] is None
    assert not list(output.glob("*.html"))
