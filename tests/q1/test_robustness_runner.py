"""Small M7 integration runs only; complete three-seed experiment runs once separately."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from q1 import svi
from q1.classical_svi import CalibrationConfig, verify_surface
from q1.neural_surface import SurfaceConfig
from q1.neural_training import TrainingConfig
from q1.robustness import FitCache, prediction, prepare_universe, run_mask
from q1.robustness_cli import execute_study, validate_run_leakage, verify_artifacts, write_outputs
from q1.robustness_protocol import METHODS, digest, quote_drop_manifest
from q1.torch_svi import TorchOptimizerConfig
from q1.verification_fixtures import development_quotes


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT/"code/approach2_q1_torch.py"


@pytest.fixture(scope="module")
def tiny_study(tmp_path_factory):
    output = tmp_path_factory.mktemp("m7_tiny")
    frame,fixture = development_quotes()
    rows,ledger,parity = prepare_universe(frame,mode="development")
    cache = FitCache(output/"fits")
    kwargs = dict(mode="development",dataset="q1_synthetic_missing_quotes_smoke",seeds=(42,),config=CalibrationConfig(),
                  optimizer_config=TorchOptimizerConfig(adam_max_steps=2,lbfgs_max_iter=2),surface_config=SurfaceConfig(hidden=(4,)),
                  training_config=TrainingConfig(epochs=3,patience=3,collocation_k_points=3,collocation_t_points=2),fixture=fixture,smoke=True)
    metrics = execute_study(rows,ledger,parity,cache=cache,output=output,**kwargs)
    return output,metrics,rows,ledger,parity,kwargs


def test_all_four_methods_share_ids_with_retained_only_fit_and_validation(tiny_study):
    _,metrics,rows,_,_,_ = tiny_study
    history = metrics["history"]
    assert set(metrics) == {"params","metrics","fit_seconds","history"}
    manifest = history["manifests"][0]
    result = {name:history[name] for name in ["fits","predictions"]}
    validate_run_leakage(rows,manifest,result)
    assert len(history["predictions"]) == 400 and len(history["fits"]) == 16
    for method in METHODS:
        fits = [row for row in history["fits"] if row["method"] == method]
        assert sum(row["held_out_count"] for row in fits) == 100
        assert sum(row["retained_count"] for row in fits) == 228
        assert all(not row["parity_derived_inputs_used"] for row in fits)
        if method == "neural":
            assert sum(row["training_count"]+row["validation_count"] for row in fits) == 228
            assert all(row["converged"] is None for row in fits)
        else:
            assert sum(row["training_count"] for row in fits) == 228 and sum(row["validation_count"] for row in fits) == 0
    assert all(row["training"]["lambda_butterfly"] == row["training"]["lambda_calendar"] == 1 for row in history["neural_training"])
    changed = deepcopy(result)
    changed["fits"][0]["training_ids"].append(manifest["slices"][0]["dropped_ids"][0])
    with pytest.raises(ValueError,match="leaked"):
        validate_run_leakage(rows,manifest,changed)


def test_failures_and_arbitrage_match_shared_diagnostics(tiny_study):
    _,metrics,_,_,_,_ = tiny_study
    history = metrics["history"]
    assert metrics["history"]["SOFTWARE_VERIFICATION"] == "PARTIAL"
    assert metrics["history"]["EMPIRICAL_REQUIREMENTS"] == "BLOCKED"
    for method in ["adam","lbfgs"]:
        info = metrics["metrics"]["seed_aggregates"][method]["pooled_with_coverage"]
        assert info["prediction_coverage"] == 0 and info["failed_or_uncovered"] == 100
        assert info["available_prediction_errors"]["iv_mae"] is None
        diag = next(row for row in history["diagnostics"] if row["method"] == method)
        assert diag["diagnosed_expiries"] == 0 and len(diag["missing_expiries"]) == 4
        assert diag["counts"]["examination_butterfly"] is None
    for method in ["scipy","adam","lbfgs"]:
        fits = [row for row in history["fits"] if row["method"] == method]
        expected = verify_surface(fits)
        actual = next(row["report"] for row in history["diagnostics"] if row["method"] == method)
        assert expected == actual


def test_resume_reuses_successes_and_failures_without_any_retraining(tiny_study,monkeypatch):
    from q1 import robustness
    output,original,rows,ledger,parity,kwargs = tiny_study
    def forbidden(*args,**kwargs):
        raise AssertionError("Completed fits must not be recomputed on resume")
    monkeypatch.setattr(robustness,"fit_slice",forbidden)
    monkeypatch.setattr(robustness,"fit_torch_slice",forbidden)
    monkeypatch.setattr(robustness,"train_surface",forbidden)
    cache = FitCache(output/"fits")
    repeated = execute_study(rows,ledger,parity,cache=cache,output=output,**kwargs)
    assert repeated["history"]["predictions"] == original["history"]["predictions"]
    assert repeated["history"]["fits"] == original["history"]["fits"]
    assert cache.new_jobs == 0 and cache.hits == 14  # reference + 12 SVI + one dated MLP
    assert repeated["metrics"] == original["metrics"]


def test_insufficient_support_is_retained_for_every_method(tiny_study):
    rows = tiny_study[2][:8]
    manifest = quote_drop_manifest(rows,42,configuration_sha256="insufficient")
    assert manifest["slices"][0]["status"] == "INSUFFICIENT_SUPPORT"
    result = run_mask(rows,manifest,optimizer_config=TorchOptimizerConfig(adam_max_steps=1,lbfgs_max_iter=1),training_config=TrainingConfig(epochs=1))
    assert len(result["predictions"]) == 8
    assert all(row["status"] == "INSUFFICIENT_SUPPORT" for row in result["predictions"])
    assert len(result["fits"]) == 4 and all(not row["usable"] for row in result["fits"])


def test_near_zero_relative_error_is_null_not_denominator_floored(tiny_study):
    row = deepcopy(tiny_study[2][0]); row["price"] = 1e-12
    output = prediction(row,"scipy",42,tiny_study[2],parameters=svi.SVIParameters(.02,.15,-.3,.04,.2))
    assert output["status"] == "PREDICTED" and output["relative_price_error"] is None
    assert output["absolute_price_error"] > 0 and output["relative_error_status"] == "NEAR_ZERO_DENOMINATOR_EXCLUDED"


def test_eight_figure_groups_manifests_and_artifacts_without_duplicate_html(tiny_study,monkeypatch):
    from plotly.basedatatypes import BaseFigure
    output,metrics,_,_,_,_ = tiny_study
    metrics = deepcopy(metrics)
    captured = []
    def capture(self,**kwargs):
        assert kwargs["include_plotlyjs"] is True and kwargs["full_html"] is True
        assert "NOT EMPIRICAL MARKET RESULTS" in self.layout.title.text
        captured.append(kwargs["div_id"])
        return "<html>"+metrics["params"]["configuration_sha256"]+"</html>"
    monkeypatch.setattr(BaseFigure,"to_html",capture)
    write_outputs(output,metrics)
    assert len(captured) == len(metrics["history"]["figures"]) == 8
    assert json.loads((output/"split_manifests/seed_42.json").read_text()) == metrics["history"]["manifests"][0]
    saved = verify_artifacts(output)
    assert saved["metrics"] == metrics["metrics"]
    # Existing figures are retained, not regenerated.
    from q1.robustness_figures import write_robustness_figures
    write_robustness_figures(output,metrics)
    assert len(captured) == 8


def test_cli_blocked_empirical_and_mandatory_seed_contract(tmp_path):
    command = [sys.executable,str(CLI),"--component","robustness","--mode","empirical","--dataset","q1_510050_missing_quotes", "--output-dir",str(tmp_path)]
    result = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode == 2 and "BLOCKED" in result.stdout
    output = tmp_path/"approach2/q1_510050_missing_quotes"
    metrics = verify_artifacts(output)
    assert metrics["history"]["eligibility"]["input_rows"] == 606 and metrics["history"]["eligibility"]["eligible_rows"] == 0
    assert metrics["fit_seconds"] == 0 and not metrics["history"]["predictions"]
    assert not list(output.glob("*.html")) and not list(output.rglob("*.pt"))
    again = subprocess.run(command+["--resume"],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert again.returncode == 2
    invalid = subprocess.run(command+["--seeds","42"],cwd=ROOT,capture_output=True,text=True)
    assert invalid.returncode == 2 and "42 43 44" in invalid.stderr
