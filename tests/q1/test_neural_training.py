"""Small deterministic M6 tests; never repeat the complete verification run."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from q1.approach_a import prepare_quotes
from q1.neural_cli import save_neural_run
from q1.neural_experiment import ablation_configs, neural_metrics, run_ablations
from q1.neural_surface import SurfaceConfig, arbitrage_penalties
from q1.neural_training import (TrainingConfig, load_checkpoint, paired_split, quote_tensors,
                                save_checkpoint, state_fingerprint, train_surface, validate_split)
from q1.verification_fixtures import development_quotes


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT/"code/approach2_q1_torch.py"


@pytest.fixture(scope="module")
def rows():
    frame,_ = development_quotes()
    ledger,_ = prepare_quotes(frame,mode="development")
    return [record["input"] for record in ledger if record["eligible_for_fit"]]


@pytest.fixture(scope="module")
def small_fit(rows):
    config = TrainingConfig(epochs=4,batch_size=64,patience=4,collocation_k_points=5,collocation_t_points=3)
    split = paired_split(rows,config=config)
    model, record = train_surface(rows,split,lambda_butterfly=1.,lambda_calendar=1.,training_config=config,
                                  surface_config=SurfaceConfig(hidden=(4,4)))
    return model,record,split,config


def test_splits_are_deterministic_paired_disjoint_and_complete(rows):
    split = paired_split(rows)
    assert split == paired_split(rows)
    validate_split(rows,split)
    assert {name:len(indices) for name,indices in split.items()} == {"train":200,"validation":64,"test":64}
    broken = deepcopy(split); broken["test"][0] = broken["train"][0]
    with pytest.raises(ValueError,match="partition"):
        validate_split(rows,broken)
    swapped = deepcopy(split)
    swapped["train"][0],swapped["test"][0] = swapped["test"][0],swapped["train"][0]
    with pytest.raises(ValueError,match="leaked"):
        validate_split(rows,swapped)


def test_small_training_uses_price_loss_and_actual_counts(rows,small_fit):
    model,record,split,config = small_fit
    assert record["success"],record["failure_reason"]
    assert record["optimizer_updates"] == record["backward_calls"] == 16
    assert record["training_objective_evaluations"] == record["optimizer_updates"]+2
    assert record["validation_price_evaluations"] == config.epochs+2
    assert len(record["history"]) == config.epochs+1
    assert not record["mathematical_convergence_claimed"]
    training = quote_tensors([rows[i] for i in split["train"]])
    x = torch.stack((training["k"],training["time_to_expiry"]),dim=1)
    torch.testing.assert_close(model.input_mean,x.mean(0),atol=0,rtol=0)
    torch.testing.assert_close(model.input_scale,x.std(0,correction=0),atol=0,rtol=0)
    for row in record["history"]:
        assert row["training_total"] == pytest.approx(row["training_price_mse"]+row["training_butterfly_penalty"]+row["training_calendar_penalty"],abs=1e-12)
    grid = record["penalty_grid"]
    penalties = arbitrage_penalties(model,torch.tensor(grid["k"],dtype=torch.float64),torch.tensor(grid["maturity"],dtype=torch.float64))
    assert all(torch.isfinite(value) for value in penalties.values())


def test_checkpoint_restores_exact_predictions_and_metadata(tmp_path,small_fit):
    model,record,_,_ = small_fit
    path = tmp_path/"model.pt"
    metadata = save_checkpoint(path,model,record)
    restored, loaded = load_checkpoint(path)
    assert loaded == metadata
    assert state_fingerprint(model) == state_fingerprint(restored)
    k,t = torch.tensor([-.13,.24],dtype=torch.float64),torch.tensor([.2,.7],dtype=torch.float64)
    torch.testing.assert_close(model(k,t),restored(k,t),atol=0,rtol=0)
    with pytest.raises(ValueError,match="new repository-local"):
        save_checkpoint(path,model,record)


def test_held_out_quotes_never_change_fit_or_normalization(rows,small_fit):
    model,record,split,config = small_fit
    altered = deepcopy(rows)
    for i in split["test"]:
        altered[i]["price"] *= 100
        altered[i]["forward"] *= 1.1
    second, other = train_surface(altered,split,lambda_butterfly=1.,lambda_calendar=1.,training_config=config,
                                  surface_config=SurfaceConfig(hidden=(4,4)))
    assert other["success"]
    assert record["history"] == other["history"]
    assert state_fingerprint(model) == state_fingerprint(second)
    # Validation is used for stopping/selection, but NOT input statistics.
    changed_val = deepcopy(rows)
    for i in split["validation"]:
        changed_val[i]["forward"] *= 1.2
    _, val_record = train_surface(changed_val,split,lambda_butterfly=1.,lambda_calendar=1.,training_config=config,
                                  surface_config=SurfaceConfig(hidden=(4,4)))
    assert val_record["input_mean"] == record["input_mean"] and val_record["input_scale"] == record["input_scale"]
    assert val_record["initial_state_sha256"] == record["initial_state_sha256"]


def test_failures_are_not_checkpoints_or_successes(rows,tmp_path):
    split = paired_split(rows)
    broken = deepcopy(rows)
    broken[split["train"][0]]["price"] = float("nan")
    model,record = train_surface(broken,split,training_config=TrainingConfig(epochs=1))
    assert not record["success"] and record["failure_reason"]
    assert not record["optimizer_updates"]
    with pytest.raises(ValueError,match="failed training"):
        save_checkpoint(tmp_path/"bad.pt",model,record)
    with pytest.raises(ValueError):
        TrainingConfig(epochs=0)
    with pytest.raises(ValueError):
        ablation_configs(0,1)


def test_nonfinite_gradients_fail_explicitly(rows,monkeypatch):
    # Test-only nonfinite backward hook, never an experiment outcome.
    from q1 import neural_training
    original_model = neural_training.NeuralSurface
    def poisoned_model(*args,**kwargs):
        model = original_model(*args,**kwargs)
        for parameter in model.parameters():
            parameter.register_hook(lambda grad:grad*float("nan"))
        return model
    monkeypatch.setattr(neural_training,"NeuralSurface",poisoned_model)
    _,record = train_surface(rows,paired_split(rows),training_config=TrainingConfig(epochs=1))
    assert not record["success"] and record["failure_reason"] == "NONFINITE_OR_MISSING_GRADIENT"
    assert record["backward_calls"] == 1 and record["optimizer_updates"] == 0


def test_early_stopping_records_actual_epoch_and_restores_best(rows):
    config = TrainingConfig(epochs=10,patience=2,min_delta=1e6)
    _,record = train_surface(rows,paired_split(rows,config=config),training_config=config,surface_config=SurfaceConfig(hidden=(4,)))
    assert record["success"] and record["epochs_completed"] == 2
    assert record["stop_reason"] == "EARLY_STOPPING_VALIDATION_PRICE_PATIENCE"
    assert record["restored_validation_price_mse"] == record["best_validation_price_mse"]


@pytest.fixture(scope="module")
def small_ablation():
    frame,fixture = development_quotes()
    config = TrainingConfig(epochs=3,patience=3,collocation_k_points=5,collocation_t_points=3)
    surface = SurfaceConfig(hidden=(4,))
    result,models = run_ablations(frame,mode="development",surface_config=surface,training_config=config)
    metrics = neural_metrics(result,mode="development",dataset="q1_synthetic_neural_verification",seed=42,
                            surface_config=surface,training_config=config,fixture=fixture)
    return result,models,metrics


def test_all_four_ablations_same_split_start_grid_and_schema(small_ablation):
    result,models,metrics = small_ablation
    date = result["dates"][0]
    assert set(date["ablations"]) == {"A","B","C","D"}
    assert len(models) == 4 and date["status"] == "COMPLETED"
    initial = [run["training"]["initial_state_sha256"] for run in date["ablations"].values()]
    assert len(set(initial)) == 1
    for case,(lb,lc) in ablation_configs().items():
        run = date["ablations"][case]
        assert run["training"]["lambda_butterfly"] == lb and run["training"]["lambda_calendar"] == lc
        assert run["training"]["penalty_grid"] == date["ablations"]["A"]["training"]["penalty_grid"]
        assert run["diagnostics"]["k"] == date["ablations"]["A"]["diagnostics"]["k"]
        assert len(run["evaluation"]["predictions"]) == 328
        assert run["evaluation"]["fold_metrics"]["test"]["quote_count"] == 64
        assert len(run["diagnostics"]["per_expiry"]) == 4
        assert run["diagnostics"]["grid_origin"] != run["training"]["penalty_grid"]["origin"]
    chosen = min(date["ablations"],key=lambda case:(date["ablations"][case]["training"]["best_validation_price_mse"],case))
    assert date["selected_ablation"] == chosen
    assert set(metrics) == {"params","metrics","fit_seconds","history"}
    assert not metrics["params"]["empirical_comparison_eligible"]
    assert metrics["history"]["EMPIRICAL_REQUIREMENTS"] == "BLOCKED"
    json.dumps(metrics,allow_nan=False)


def test_writer_checkpoints_and_figure_contract_without_repeated_html(small_ablation,tmp_path,monkeypatch):
    from plotly.basedatatypes import BaseFigure
    captured = []
    def capture(self,path,**kwargs):
        assert kwargs["include_plotlyjs"] is True and kwargs["full_html"] is True
        assert "SYNTHETIC SOFTWARE VERIFICATION" in self.layout.title.text
        captured.append(Path(path).name)
    monkeypatch.setattr(BaseFigure,"write_html",capture)
    _,models,metrics = small_ablation
    metrics = deepcopy(metrics)
    output = tmp_path/"saved"
    save_neural_run(output,metrics,models)
    assert len(captured) == len(metrics["history"]["figures"]) == 6
    assert len(metrics["history"]["checkpoints"]) == 4
    assert all(item["restoration_verified"] for item in metrics["history"]["checkpoints"])
    saved = json.loads((output/"metrics.json").read_text(encoding="utf-8"))
    assert saved["params"]["component"] == "neural"
    assert (output/"artifact_manifest.json").is_file()
    with pytest.raises(FileExistsError):
        save_neural_run(output,metrics,models)


def test_cli_empirical_gate_and_arguments(tmp_path):
    result = subprocess.run([sys.executable,str(CLI),"--component","neural","--mode","empirical","--dataset","q1_510050_neural",
                             "--output-dir",str(tmp_path)],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode == 2 and "BLOCKED" in result.stdout
    output = tmp_path/"approach2/q1_510050_neural"
    metrics = json.loads((output/"metrics.json").read_text(encoding="utf-8"))
    assert metrics["history"]["eligibility"]["eligible_rows"] == 0 and metrics["history"]["eligibility"]["input_rows"] == 606
    assert metrics["history"]["result"] is None and metrics["fit_seconds"] == 0
    assert not list(output.glob("*.html")) and not list(output.glob("*.pt"))
    invalid = subprocess.run([sys.executable,str(CLI),"--component","neural","--mode","development","--dataset","q1_synthetic_neural_verification", "--epochs","0"],cwd=ROOT,capture_output=True)
    assert invalid.returncode == 2
