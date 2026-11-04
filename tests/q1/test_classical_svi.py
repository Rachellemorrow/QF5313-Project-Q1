"""Synthetic classical-SVI verification, never real-market calibration results."""

from dataclasses import asdict

import numpy as np
import pytest

from q1 import svi
from q1.classical_svi import (CalibrationConfig, decode_parameters, encode_parameters, fit_slice,
                              reprice_quote, repricing_summary, residual_jacobian, verify_surface, weighted_residual)
from q1.verification_fixtures import development_quotes


@pytest.mark.parametrize("scale", [0.4, 0.75, 1.25, 2.0])
def test_known_svi_recovery_and_constraints(synthetic_cases, scale):
    source = synthetic_cases["svi_smooth"]
    truth = svi.SVIParameters(**(source | {"a": source["a"]*scale, "b": source["b"]*scale}))
    grid = np.linspace(-0.4, 0.4, 61)
    target = svi.svi_total_variance(grid, truth)
    result = fit_slice(grid, target)
    assert result["converged"], result
    for name, number in asdict(truth).items():
        assert result["fitted_parameters"][name] == pytest.approx(number, abs=1e-6)
    p = svi.SVIParameters(**result["fitted_parameters"])
    assert svi.svi_global_minimum(p) > 0 and p.b >= 0 and abs(p.rho) < 1 and p.eta > 0
    assert result["objective_value"] < 1e-18
    assert result["nfev"] == result["objective_evaluations"] == len(result["evaluation_history"])
    assert result["initial_parameters"] is not None and result["fit_seconds"] >= 0


def test_weighted_objective_and_transformed_jacobian(synthetic_cases):
    p = svi.SVIParameters(**synthetic_cases["svi_smooth"])
    theta = encode_parameters(p)
    grid = np.linspace(-0.3, 0.3, 11)
    observed = svi.svi_total_variance(grid, p)+np.linspace(-0.001, 0.001, len(grid))
    weights = np.linspace(0.5, 2, len(grid))
    residual = weighted_residual(theta, grid, observed, weights)
    assert residual@residual == pytest.approx(np.sum(weights*(svi.svi_total_variance(grid, p)-observed)**2), abs=1e-18)
    analytic = residual_jacobian(theta, grid, observed, weights)
    numerical = np.empty_like(analytic)
    for index in range(5):
        direction = np.zeros(5)
        direction[index] = 1e-6
        numerical[:, index] = (weighted_residual(theta+direction, grid, observed, weights)-weighted_residual(theta-direction, grid, observed, weights))/(2e-6)
    np.testing.assert_allclose(analytic, numerical, atol=1e-8, rtol=0)
    fitted = fit_slice(grid, observed, weights=weights)
    assert fitted["converged"]
    assert fitted["objective_value"] == pytest.approx(2*fitted["scipy_cost"], abs=1e-15)


@pytest.mark.parametrize("k,w,weights,reason", [
    ([0, 0.1], [0.02, 0.03], None, "INSUFFICIENT_DISTINCT_STRIKES"),
    ([0]*10, [0.02]*10, None, "INSUFFICIENT_DISTINCT_STRIKES"),
    ([-.2,-.1,0,.1,.2], [0.02]*4, None, "INVALID_INPUT_OR_NUMERICAL_FAILURE"),
    ([-.2,-.1,0,.1,.2], [0.02,0.02,np.nan,0.02,0.02], None, "INVALID_INPUT_OR_NUMERICAL_FAILURE"),
    ([-.2,-.1,0,.1,.2], [0.02]*5, [1,1,0,1,1], "INVALID_INPUT_OR_NUMERICAL_FAILURE"),
    ([-.2,-.1,0,.1,.2], [-0.02]*5, None, "INVALID_INPUT_OR_NUMERICAL_FAILURE"),
])
def test_invalid_and_insufficient_slices_are_retained(k, w, weights, reason):
    result = fit_slice(k, w, weights=weights)
    assert not result["converged"] and result["failure_reason"] == reason
    assert result["message"] and result["fit_seconds"] >= 0


def test_bad_initialization_and_iteration_exhaustion_are_not_successes(synthetic_cases):
    grid = np.linspace(-.4, .4, 41)
    truth = svi.SVIParameters(**synthetic_cases["svi_smooth"])
    observed = svi.svi_total_variance(grid, truth)
    invalid = svi.SVIParameters(-1, 0.1, 0, 0, .2)
    failed = fit_slice(grid, observed, initial=invalid)
    assert not failed["converged"] and "Initial parameters" in failed["message"]
    failed = fit_slice(grid, observed, config=CalibrationConfig(max_nfev=1))
    assert not failed["converged"] and failed["failure_reason"] == "SOLVER_NONCONVERGENCE"
    assert failed["fitted_parameters"] is not None
    with pytest.raises(ValueError):
        CalibrationConfig(max_nfev=0)
    with pytest.raises(ValueError):
        CalibrationConfig(rho_margin=0)


def test_parameter_decode_globally_positive_and_reproducible(synthetic_cases):
    theta = np.array([0.01, .3, -.8, .1, .15])
    p = decode_parameters(theta)
    assert svi.svi_global_minimum(p) == pytest.approx(.01)
    grid = np.linspace(-.4, .4, 41)
    observed = svi.svi_total_variance(grid, svi.SVIParameters(**synthetic_cases["svi_smooth"]))
    a, b = fit_slice(grid, observed), fit_slice(grid, observed)
    for name in ["fitted_parameters", "objective_value", "nfev", "evaluation_history"]:
        assert a[name] == b[name]


def make_slice(name, maturity, p, *, converged=True):
    return {"slice_id": name, "observation_date": "SYNTHETIC_DATE", "expiry_date": name,
            "time_to_expiry": maturity, "fit": {"converged": converged, "fitted_parameters": asdict(p)}}


def test_both_butterfly_calendar_and_forward_variance_violations_are_saved(synthetic_cases):
    base = synthetic_cases["svi_both_violation"]
    first = svi.SVIParameters(**(base | {"a": base["a"]*2, "b": base["b"]*2}))
    second = svi.SVIParameters(**base)
    result = verify_surface([make_slice("early", .5, first), make_slice("late", 1, second),
                             make_slice("failed", 2, second, converged=False)])
    assert result["failed_slice_ids"] == ["failed"]
    assert result["slices"][-1]["status"] == "NO_VALID_FIT"
    assert result["slices"][0]["examination"]["count"] > 0
    assert result["slices"][0]["complete"]["count"] > 0
    assert result["calendar"][0]["count"] == 201
    assert len(result["negative_forward_variance"]) == 201
    assert all(item["magnitude"] > 0 for item in result["negative_forward_variance"])
    assert result["interpolation"][0]["endpoint_max_abs_error"] == 0


def test_repricing_and_near_zero_denominator_policy():
    frame, truth = development_quotes()
    quote = frame.iloc[0].to_dict()
    p = svi.SVIParameters(**truth["truth_by_expiry"][quote["expiry_date"]])
    result = reprice_quote(quote, p)
    assert result["status"] == "REPRICED" and result["relative_error"] < 1e-12
    near = reprice_quote(quote | {"price": 1e-12}, p)
    assert near["absolute_error"] > 0 and near["relative_error"] is None
    assert near["relative_error_status"] == "NEAR_ZERO_DENOMINATOR_EXCLUDED"
    for value in [0, -1, float("nan")]:
        failed = reprice_quote(quote | {"price": value}, p)
        assert failed["status"] == "FAILED" and failed["failure_reason"]


def test_cross_date_statistics_are_not_pooled_quotes_and_failures_remain_visible():
    def item(date, expiry, error, ok=True):
        return {"slice_id": date+expiry, "observation_date": date, "expiry_date": expiry,
                "fit": {"converged": ok}, "repricing": [] if not ok else [{"status": "REPRICED", "relative_error": error}]}
    rows = [item("A", "X", .1), item("A", "Y", .2), item("B", "X", .3), item("C", "X", None, False)]
    result = repricing_summary(rows)
    assert result["cross_date_median_of_date_maxima"] == pytest.approx(.25)
    assert result["cross_date_p95_of_date_maxima"] == pytest.approx(.295)
    assert result["pooled_quote_relative_error_median_separately_labelled"] == .2
    assert result["incomplete_dates_excluded_from_primary_statistics"] == 1
    partial = repricing_summary(rows+[item("A", "Z", None, False)])
    assert partial["cross_date_median_of_date_maxima"] == .3
    assert len(partial["per_slice"]) == 5
