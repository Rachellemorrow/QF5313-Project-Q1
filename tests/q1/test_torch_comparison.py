"""Matched M4 fixture, unchanged baseline, shared verification and strict CLI gates."""

from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pytest

from q1 import approach_a, classical_svi, svi, torch_comparison
from q1.approach_a import calibrate_quotes, plain
from q1.classical_svi import reprice_quote, repricing_summary, verify_surface
from q1.torch_comparison import compare_quotes, input_fingerprint, scipy_timing_trace
from q1.verification_fixtures import development_quotes


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT/"code/approach2_q1_torch.py"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("M5 must run offline")))


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("m5_comparison")
    command = [sys.executable, str(CLI), "--mode", "development", "--dataset", "q1_synthetic_svi_optimizer_comparison",
               "--seed", "42", "--output-dir", str(base)]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
    # The default L-BFGS budget is genuinely insufficient for one M4 slice.
    # Preserve the failed acceptance status rather than relaxing its tolerance.
    assert process.returncode == 1, process.stdout+process.stderr
    output = base/"verification/approach2/q1_synthetic_svi_optimizer_comparison"
    metrics = json.loads((output/"metrics.json").read_text(encoding="utf-8"))
    return metrics, output, command


def test_matched_cohort_objective_weights_and_initial_parameters(completed_run):
    metrics, _, _ = completed_run
    result = metrics["history"]["comparison"]
    assert len(result["quote_ledger"]) == 328 and len(result["parity"]) == 164
    for index, matched in enumerate(result["matched_inputs"]):
        ledger = [row for row in result["quote_ledger"] if row["eligible_for_fit"] and
                  f"{row['input']['observation_date']}|{row['input']['expiry_date']}" == matched["slice_id"]]
        k, target = [np.array([row[key] for row in ledger]) for key in ["k", "total_variance"]]
        weights = np.array(matched["weights"])
        assert len(k) == 82 and np.all(weights == .5)
        assert input_fingerprint(k, target, weights) == matched["input_sha256"]
        for name, algorithm in result["algorithms"].items():
            fit = algorithm["slices"][index]["fit"]
            assert fit["initial_parameters"] == matched["raw_initial_parameters"]
            assert fit["weights"] == matched["weights"]
            assert fit["comparison_input_sha256"] == matched["input_sha256"]
            p = svi.SVIParameters(**fit["fitted_parameters"])
            residual = classical_svi.weighted_residual(classical_svi.encode_parameters(p), k, target, weights)
            # Squaring tiny residuals makes relative RSS error ill-conditioned.
            # Compare sqrt(RSS) in residual units with a float64 operation-count
            # roundoff budget scaled to the common weighted targets. This is
            # objective-parity verification, NOT an optimizer stopping criterion.
            roundoff = 64*np.finfo(float).eps*max(1., np.linalg.norm(np.sqrt(weights)*target))
            assert abs(np.sqrt(fit["objective_value"])-np.linalg.norm(residual)) <= roundoff
            if name != "scipy":
                assert fit["initial_reconstruction_max_abs_error"] < 1e-15


def test_synthetic_recovery_and_actual_optimizer_counts(completed_run):
    metrics, _, _ = completed_run
    assert set(metrics) == {"params", "metrics", "fit_seconds", "history"}
    assert metrics["history"]["SOFTWARE_VERIFICATION"] == "PARTIAL"
    assert metrics["history"]["status"] == "COMPLETED_WITH_FAILURES"
    assert metrics["history"]["EMPIRICAL_REQUIREMENTS"] == "BLOCKED"
    assert not metrics["params"]["empirical_comparison_eligible"]
    assert not metrics["params"]["perfectly_controlled_optimization_comparison"]
    for name, summary in metrics["metrics"].items():
        assert summary["attempted_slices"] == 4
        if name != "lbfgs":
            assert summary["successful_slices"] == 4 and summary["convergence_success_rate"] == 1
            assert summary["maximum_known_variance_grid_error"] <= 1e-6
            assert summary["maximum_known_parameter_error"] <= 1e-4
        else:
            assert summary["successful_slices"] == 3 and summary["convergence_success_rate"] == .75
            assert summary["failed_slice_ids"] == ["2000-01-03|2000-04-02"]
            assert summary["maximum_known_variance_grid_error"] > 1e-6
            assert summary["maximum_known_parameter_error"] > 1e-4
            assert summary["maximum_final_objective"] > 1e-14
            assert summary["repricing"]["cross_date_median_of_date_maxima"] is None
            assert summary["arbitrage"]["failed_slices_without_diagnostics"] == summary["failed_slice_ids"]
        for item in metrics["history"]["comparison"]["algorithms"][name]["slices"]:
            fit = item["fit"]
            if name == "scipy":
                assert fit["native_iterations"] is None
                assert fit["objective_evaluations"] == fit["nfev"] == len(fit["comparison_history"])
                assert fit["total_residual_evaluations_including_postcheck"] == fit["nfev"]+1
                assert fit["post_fit_residual_evaluations"] == 1
            else:
                assert fit["device"] == "cpu" and fit["dtype"] == "float64"
                assert fit["objective_evaluations"] == fit["gradient_evaluations"] == len(fit["evaluation_history"])
                assert fit["fit_seconds"] >= fit["optimization_seconds"] > 0
                if name == "adam":
                    assert fit["iterations"]+1 == fit["objective_evaluations"]
                else:
                    assert fit["native_func_evals"] == fit["lbfgs_closure_calls"] == fit["objective_evaluations"]-1
            row = next(row for row in metrics["history"]["comparison"]["comparison_rows"]
                       if row["algorithm"] == name and row["slice_id"] == item["slice_id"])
            if fit["converged"]:
                assert row["known_variance_grid_max_abs_error"] <= 1e-6
                assert row["known_parameter_max_abs_error"] <= 1e-4
            else:
                assert fit["iterations"] == fit["optimizer_config"]["lbfgs_max_iter"]
                assert fit["failure_reason"]


def test_instrumented_scipy_is_identical_to_plain_m4(completed_run):
    frame, _ = development_quotes()
    baseline = calibrate_quotes(frame, mode="development")
    traced = completed_run[0]["history"]["comparison"]["algorithms"]["scipy"]
    assert plain(baseline["arbitrage"]) == traced["arbitrage"]
    assert plain(baseline["repricing_summary"]) == traced["repricing_summary"]
    for before, after in zip(baseline["slices"], traced["slices"]):
        assert before["repricing"] == after["repricing"]
        for key, value in before["fit"].items():
            if key != "fit_seconds":
                assert plain(value) == after["fit"][key]


def test_scipy_instrumentation_restores_on_exception_and_separates_identical_inputs():
    original_fit, original_residual = approach_a.fit_slice, classical_svi.weighted_residual
    k = np.linspace(-.4,.4,21)
    target = svi.svi_total_variance(k, svi.SVIParameters(.02,.15,-.3,.04,.2))
    with pytest.raises(RuntimeError):
        with scipy_timing_trace():
            first = approach_a.fit_slice(k,target)
            second = approach_a.fit_slice(k,target)
            for fit in [first, second]:
                assert fit["total_residual_evaluations_including_postcheck"] == fit["nfev"]+1
                assert fit["comparison_history"][0]["elapsed_since_first_evaluation_seconds"] == 0
            raise RuntimeError("Test restoration")
    assert approach_a.fit_slice is original_fit and classical_svi.weighted_residual is original_residual


def test_shared_arbitrage_and_repricing_are_reused_exactly(completed_run):
    result = completed_run[0]["history"]["comparison"]
    for name, algorithm in result["algorithms"].items():
        assert plain(verify_surface(algorithm["slices"])) == algorithm["arbitrage"]
        assert plain(repricing_summary(algorithm["slices"])) == algorithm["repricing_summary"]
        for item in algorithm["slices"]:
            if not item["fit"]["converged"]:
                assert not item["repricing"]
                continue
            parameters = svi.SVIParameters(**item["fit"]["fitted_parameters"])
            rows = [row for row in result["quote_ledger"] if row["eligible_for_fit"] and
                    f"{row['input']['observation_date']}|{row['input']['expiry_date']}" == item["slice_id"]]
            assert [reprice_quote(row["input"], parameters) for row in rows] == item["repricing"]
        assert result["summary"][name]["arbitrage"]["examination_butterfly_violations"] == 0
        assert result["summary"][name]["arbitrage"]["complete_butterfly_violations"] == 0
        assert result["summary"][name]["arbitrage"]["calendar_violations"] == 0


def test_failed_fits_and_rejected_quotes_remain_in_comparison():
    frame, _ = development_quotes()
    small = frame.iloc[:8].copy()
    result = compare_quotes(small, mode="development")
    assert len(result["quote_ledger"]) == 8
    for algorithm in result["algorithms"].values():
        assert len(algorithm["slices"]) == 1
        assert not algorithm["slices"][0]["fit"]["converged"]
        assert len(algorithm["arbitrage"]["failed_slice_ids"]) == 1
        assert algorithm["repricing_summary"]["cross_date_median_of_date_maxima"] is None
    small.loc[0, "forward"] *= 1.01
    rejected = compare_quotes(small, mode="development")
    assert all(not row["eligible_for_fit"] for row in rejected["quote_ledger"])
    assert all(info["successful_slices"] == 0 for info in rejected["summary"].values())


def test_orchestration_does_not_suppress_arbitrage_violations(monkeypatch):
    # Deliberately bad synthetic candidate injected ONLY in this unit test.
    # No optimizer run or empirical result is claimed by this stub.
    def bad_candidate(k, target, *, weights, initial, **kwargs):
        scale = .01/float(np.median(target))
        p = svi.SVIParameters(.001*scale, 1.5*scale, -.7, 0, .01)
        residual = svi.svi_total_variance(k, p)-target
        return {"converged": True, "failure_reason": None, "fitted_parameters": asdict(p),
                "initial_parameters": asdict(initial), "objective_value": float(np.sum(weights*residual**2)),
                "fit_seconds": 0., "objective_evaluations": 0, "test_stub_not_optimizer": True}
    monkeypatch.setattr(torch_comparison, "fit_torch_slice", bad_candidate)
    frame, _ = development_quotes()
    result = compare_quotes(frame, mode="development")
    for name in ["adam", "lbfgs"]:
        summary = result["summary"][name]["arbitrage"]
        assert summary["examination_butterfly_violations"] > 0
        assert summary["calendar_violations"] > 0 and summary["negative_forward_variance_locations"] > 0
        diagnostics = result["algorithms"][name]["arbitrage"]
        assert any(row["examination"]["locations"] for row in diagnostics["slices"])
        assert all(row["magnitude"] > 0 for row in diagnostics["negative_forward_variance"])


def test_cli_inline_figures_artifacts_and_no_overwrite(completed_run):
    metrics, output, command = completed_run
    assert len(metrics["history"]["figures"]) == 6

    class Scripts(HTMLParser):
        def __init__(self):
            super().__init__(); self.external = []
        def handle_starttag(self, tag, attrs):
            if tag == "script":
                self.external.extend(value for key, value in attrs if key == "src")

    for name in metrics["history"]["figures"]:
        html = (output/name).read_text(encoding="utf-8")
        assert "SYNTHETIC SOFTWARE VERIFICATION" in html and "plotly.js" in html
        parser = Scripts(); parser.feed(html)
        assert not parser.external
    manifest = json.loads((output/"artifact_manifest.json").read_text())
    for entry in manifest["files"]:
        contents = (output/entry["name"]).read_bytes()
        assert len(contents) == entry["bytes"] and hashlib.sha256(contents).hexdigest() == entry["sha256"]
    assert (output/"optimizer_comparison.csv").is_file() and (output/"parameter_comparison.json").is_file()
    assert not list(output.parents[2].glob("approach2/*/metrics.json"))
    again = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
    assert again.returncode == 2


def test_cli_help_and_rejected_arguments(tmp_path):
    result = subprocess.run([sys.executable, str(CLI), "--help"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert all(flag in result.stdout for flag in ["--component", "--mode", "--seed", "--dataset", "--adam-lr", "--lbfgs-max-iter"])
    for args in [["--mode", "development"], ["--seed", "-1"], ["--component", "neural"], ["--adam-lr", "0"]]:
        invalid = subprocess.run([sys.executable, str(CLI), "--output-dir", str(tmp_path), *args], cwd=ROOT, capture_output=True)
        assert invalid.returncode == 2


def test_cli_empirical_gate_remains_blocked(tmp_path):
    result = subprocess.run([sys.executable, str(CLI), "--mode", "empirical", "--dataset", "q1_510050", "--output-dir", str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 2 and "BLOCKED" in result.stdout
    output = tmp_path/"approach2/q1_510050"
    metrics = json.loads((output/"metrics.json").read_text(encoding="utf-8"))
    assert metrics["history"]["status"] == metrics["history"]["EMPIRICAL_REQUIREMENTS"] == "BLOCKED"
    assert metrics["history"]["eligibility"]["eligible_rows"] == 0
    assert metrics["history"]["eligibility"]["input_rows"] == 606
    assert not metrics["history"]["eligibility"]["strict_pipeline_eligibility_relaxed"]
    assert metrics["fit_seconds"] == 0 and metrics["metrics"] == {}
    assert metrics["history"]["comparison"] is None and not list(output.glob("*.html"))
