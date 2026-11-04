"""M7 masks, isolated retained-only inputs, coverage and aggregation arithmetic."""

from copy import deepcopy

import numpy as np
import pytest

from q1.neural_training import TrainingConfig, train_surface, validate_split
from q1.neural_surface import SurfaceConfig
from q1.robustness import prepare_universe
from q1.robustness_protocol import (METHODS, digest, error_metrics, quote_drop_manifest,
                                    retained_neural_split, summarize_predictions, support_class, validate_manifest)
from q1.verification_fixtures import development_quotes


@pytest.fixture(scope="module")
def universe():
    frame,_ = development_quotes()
    return prepare_universe(frame,mode="development")[0]


def test_quote_deletion_count_ids_order_seed_and_saved_masks(universe):
    for seed in [42,43,44]:
        manifest = quote_drop_manifest(universe,seed,configuration_sha256="test")
        validate_manifest(universe,manifest,"test")
        assert manifest == quote_drop_manifest(list(reversed(universe)),seed,configuration_sha256="test")
        assert sum(row["dropped_count"] for row in manifest["slices"]) == 100
        assert sum(row["retained_count"] for row in manifest["slices"]) == 228
        for row in manifest["slices"]:
            assert row["dropped_count"] == 25 and row["universe_count"] == 82
            assert set(row["dropped_ids"]).isdisjoint(row["retained_ids"])
            assert [value for value,mask in zip(row["quote_ids"],row["dropped_mask"]) if mask] == row["dropped_ids"]
        # Mandatory QUOTE-level deletion actually splits some call/put groups.
        dropped = {value for row in manifest["slices"] for value in row["dropped_ids"]}
        assert any((row["quote_id"] in dropped) != (row["quote_id"][:-1]+("P" if row["option_type"] == "C" else "C") in dropped) for row in universe)
    a = quote_drop_manifest(universe,42,configuration_sha256="test")
    b = quote_drop_manifest(universe,43,configuration_sha256="test")
    assert a["slices"] != b["slices"]
    altered = deepcopy(a); altered["slices"][0]["dropped_mask"][0] = not altered["slices"][0]["dropped_mask"][0]
    with pytest.raises(ValueError):
        validate_manifest(universe,altered,"test")


def test_internal_retained_split_and_no_heldout_parameter_or_normalization_leakage(universe):
    manifest = quote_drop_manifest(universe,42,configuration_sha256="test")
    retained_ids = {value for row in manifest["slices"] for value in row["retained_ids"]}
    retained = [row for row in universe if row["quote_id"] in retained_ids]
    split = retained_neural_split(retained,42)
    validate_split(retained,split,require_test=False)
    assert split == retained_neural_split(retained,42)
    assert len(split["train"])+len(split["validation"]) == 228
    assert all(retained[i]["quote_id"] in retained_ids for indices in split.values() for i in indices)
    cfg = TrainingConfig(epochs=2,patience=2,collocation_k_points=3,collocation_t_points=2)
    kwargs = dict(seed=42,surface_config=SurfaceConfig(hidden=(3,)),training_config=cfg,retained_only=True,lambda_calendar=1.)
    _,a = train_surface(retained,split,**kwargs)
    changed = deepcopy(universe)
    for row in changed:
        if row["quote_id"] not in retained_ids:
            row["price"] *= 10000
            row["_iv"] *= 10
            row["forward"] *= 2
    other = [row for row in changed if row["quote_id"] in retained_ids]
    _,b = train_surface(other,retained_neural_split(other,42),**kwargs)
    assert a["success"] and b["success"]
    for key in ["input_mean","input_scale","history","final_state_sha256","initial_state_sha256"]:
        assert a[key] == b[key]
    # M6's default three-way interface has NOT been relaxed.
    with pytest.raises(ValueError,match="three"):
        validate_split(retained,split)


def test_support_class_and_insufficient_support(universe):
    group = [row for row in universe if row["expiry_date"] == universe[0]["expiry_date"]]
    strikes = sorted({row["strike"] for row in group})
    retained = [row for row in group if strikes[1] <= row["strike"] <= strikes[-2]]
    assert support_class(min(group,key=lambda row:row["strike"]),retained) == "EXTRAPOLATION"
    assert support_class(retained[0],retained) == "INTERPOLATION"
    assert support_class(group[0],retained[:2]) == "INSUFFICIENT_SUPPORT"


def sample_prediction(quote_id, method="scipy", seed=42, *, failed=False, relative=.1):
    return {"quote_id":quote_id,"method":method,"seed":seed,"observation_date":"2000-01-03","expiry_date":"2000-02-02",
            "support_class":"INTERPOLATION","status":"FIT_FAILED" if failed else "PREDICTED",
            "predicted_iv":.3,"market_iv":.2,"predicted_price":2.,"market_price":1.,"relative_price_error":relative}


def test_metrics_do_not_hide_failed_predictions_or_near_zero_denominators():
    rows = [sample_prediction("a"),sample_prediction("b",failed=True),sample_prediction("c",relative=None)]
    summary = error_metrics(rows)
    assert summary["held_out_expected"] == 3 and summary["predicted"] == 2
    assert summary["prediction_coverage"] == pytest.approx(2/3)
    assert summary["available_prediction_errors"]["iv_mae"] == pytest.approx(.1)
    assert summary["available_prediction_errors"]["price_mae"] == 1
    assert summary["near_zero_relative_exclusions"] == 1
    assert all(value is None for value in summary["complete_cohort_errors"].values())
    empty = error_metrics([])
    assert empty["prediction_coverage"] is None and empty["available_prediction_errors"]["iv_mae"] is None


def test_seed_aggregation_and_matched_success_intersection():
    predictions = [sample_prediction(q,method,seed,failed=(method == "lbfgs" and q == "b"))
                   for seed in [42,43,44] for method in METHODS for q in ["a","b"]]
    fits = [{"seed":seed,"method":method,"usable":method != "lbfgs","fit_seconds_attributed":float(seed-40)} for seed in [42,43,44] for method in METHODS]
    summary = summarize_predictions(predictions,fits,[],[42,43,44])
    assert summary["seed_aggregates"]["scipy"]["fit_seconds"]["mean"] == 3
    assert summary["seed_aggregates"]["scipy"]["fit_seconds"]["sample_std"] == 1
    assert summary["seed_aggregates"]["lbfgs"]["prediction_coverage"]["mean"] == .5
    assert all(row["common_quote_ids"] == ["a"] for row in summary["common_success_intersection"])
    assert summary["seed_aggregates"]["lbfgs"]["pooled_with_coverage"]["complete_cohort_errors"]["price_mae"] is None
