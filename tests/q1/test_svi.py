"""Synthetic SVI calculus/arbitrage checks, not fitted market surfaces."""

import numpy as np
import pytest

from q1 import svi


@pytest.mark.parametrize("changes", [
    {"b": -0.1}, {"rho": -1}, {"rho": 1}, {"eta": 0}, {"eta": -1},
    {"a": np.nan}, {"m": np.inf}, {"b": np.inf},
])
def test_parameter_constraints(synthetic_cases, changes):
    parameters = synthetic_cases["svi_smooth"] | changes
    with pytest.raises(ValueError):
        svi.SVIParameters(**parameters)


def test_formula_and_analytical_derivatives(synthetic_cases):
    p = svi.SVIParameters(**synthetic_cases["svi_smooth"])
    k = np.linspace(-0.8, 0.8, 81)
    x = k - p.m
    expected = p.a + p.b * (p.rho * x + np.sqrt(x*x + p.eta*p.eta))
    np.testing.assert_allclose(svi.svi_total_variance(k, p), expected, atol=1e-15)
    h1, h2 = 1e-5, 1e-4
    first_fd = (svi.svi_total_variance(k+h1, p)-svi.svi_total_variance(k-h1, p))/(2*h1)
    second_fd = (svi.svi_total_variance(k+h2, p)-2*svi.svi_total_variance(k, p)+svi.svi_total_variance(k-h2, p))/(h2*h2)
    np.testing.assert_allclose(svi.svi_first_derivative(k, p), first_fd, atol=1e-9, rtol=0)
    np.testing.assert_allclose(svi.svi_second_derivative(k, p), second_fd, atol=1e-6, rtol=0)


def test_valid_negative_a_and_nonpositive_variance_reporting():
    valid = svi.SVIParameters(a=-0.01, b=0.3, rho=0, m=0, eta=0.2)
    assert svi.svi_global_minimum(valid) == pytest.approx(0.05)
    invalid = svi.SVIParameters(a=-0.1, b=0.1, rho=0, m=0, eta=0.1)
    report = svi.variance_positivity([-2, 0, 2], invalid)
    assert not report["positive_on_grid"] and not report["positive_globally"]
    np.testing.assert_array_equal(report["nonpositive_locations"], [0])
    assert report["nonpositive_values"][0] < 0
    with pytest.raises(ValueError, match="positive"):
        svi.svi_total_variance([0], invalid)
    with pytest.raises(ValueError):
        svi.examination_butterfly([0], invalid)


def test_flat_svi_has_zero_derivatives_and_unit_butterfly():
    p = svi.SVIParameters(0.04, 0, 0.99, 0, 0.1)
    k = np.linspace(-1, 1, 5)
    np.testing.assert_array_equal(svi.svi_total_variance(k, p), np.full(5, 0.04))
    np.testing.assert_array_equal(svi.svi_first_derivative(k, p), np.zeros(5))
    np.testing.assert_array_equal(svi.svi_second_derivative(k, p), np.zeros(5))
    np.testing.assert_array_equal(svi.examination_butterfly(k, p), np.ones(5))


def test_exam_and_complete_butterfly_must_not_be_conflated(synthetic_cases):
    p = svi.SVIParameters(**synthetic_cases["svi_exam_only_violation"])
    diagnostic = svi.butterfly_diagnostics(np.array([0.0]), p)
    exam, complete = diagnostic["examination"], diagnostic["complete"]
    assert exam.count == 1 and complete.count == 0
    assert exam.values[0] < 0 < complete.values[0]
    np.testing.assert_array_equal(exam.locations, [0.0])
    np.testing.assert_allclose(exam.magnitudes, -exam.values)
    np.testing.assert_allclose(complete.values-exam.values, svi.svi_second_derivative([0], p)/2)


def test_deliberate_complete_butterfly_violation_is_located(synthetic_cases):
    p = svi.SVIParameters(**synthetic_cases["svi_both_violation"])
    grid = np.array([-10.0, 0.0, 10.0])
    result = svi.butterfly_diagnostics(grid, p)
    for report in result.values():
        assert report.count == 2
        np.testing.assert_array_equal(report.locations, [-10, 10])
        assert (report.magnitudes > 0).all()


def test_calendar_violation_and_tolerance_keep_raw_values():
    k = [-0.1, 0, 0.1]
    report = svi.calendar_diagnostics(k, [0.02, 0.04, 0.06], [0.03, 0.03, 0.06], 0.25, 0.5)
    assert report.count == 1
    np.testing.assert_allclose(report.locations, [0])
    np.testing.assert_allclose(report.magnitudes, [0.01])
    tolerant = svi.calendar_diagnostics(k, [0.02, 0.04, 0.06], [0.03, 0.03, 0.06], 0.25, 0.5, tolerance=0.02)
    assert tolerant.count == 0 and tolerant.values[1] < 0
    with pytest.raises(ValueError):
        svi.calendar_total_variance_difference([0.04], [0.06], 1, 0.5)


def test_term_interpolation_and_negative_forward_variance(synthetic_cases):
    case = synthetic_cases["term_structure"]
    actual = svi.interpolate_total_variance(case["maturities"], case["total_variance"], case["query"])
    np.testing.assert_allclose(actual, case["expected"], atol=1e-15)
    # Nonmonotone total variance is retained, not isotonic-repaired or clipped.
    diagnostic = svi.forward_variance_diagnostics(case["maturities"], case["total_variance"])
    np.testing.assert_array_equal(diagnostic.indices, [[1, 0]])
    np.testing.assert_allclose(diagnostic.magnitudes, [0.02])
    assert diagnostic.interval_start[1] == 0.5 and diagnostic.interval_end[1] == 1
    endpoints = svi.interpolate_total_variance(case["maturities"], case["total_variance"], case["maturities"])
    np.testing.assert_array_equal(endpoints, case["total_variance"])
    assert svi.interpolate_total_variance([0.5, 1], [0.02, 0.08], 0.75) == pytest.approx(0.05)


def test_explicit_zero_anchor_and_multidimensional_query_shape():
    np.testing.assert_allclose(svi.interpolate_total_variance([0, 1], [0, 0.04], [0, 0.5, 1]), [0, 0.02, 0.04])
    w = np.arange(12).reshape(3, 2, 2) / 100 + 0.01
    query = np.array([[0.25, 0.5], [0.75, 1.0]])
    result = svi.interpolate_total_variance([0.25, 0.5, 1.0], w, query)
    assert result.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(result[0, 0], w[0])
    np.testing.assert_array_equal(result[-1, -1], w[-1])


@pytest.mark.parametrize("times,w,query", [
    ([1, 1], [0.1, 0.2], 1), ([1, 0.5], [0.1, 0.2], 0.8),
    ([0.5, 1], [0.1, -0.2], 0.8), ([0.5, 1], [0.1], 0.8),
    ([0, 1], [0.01, 0.04], 0.5), ([0.5, 1], [0.01, 0.04], 0.25),
    ([0.5, 1], [0.01, 0.04], 1.1), ([0.5, 1], [0.01, 0.04], np.nan),
    ([-1, 1], [0.01, 0.04], 0),
])
def test_invalid_term_structures_or_extrapolation_raise(times, w, query):
    with pytest.raises(ValueError):
        svi.interpolate_total_variance(times, w, query)


def test_explicit_fixed_grid_and_numerical_validation(synthetic_cases):
    grid = svi.fixed_log_moneyness_grid(-0.5, 0.5, 101)
    np.testing.assert_array_equal(grid, svi.fixed_log_moneyness_grid(-0.5, 0.5, 101))
    assert len(grid) == 101 and grid[50] == 0
    p = svi.SVIParameters(**synthetic_cases["svi_smooth"])
    assert svi.variance_positivity(grid, p)["positive_globally"]
    with pytest.raises(ValueError):
        svi.fixed_log_moneyness_grid(1, 0, 5)
    with pytest.raises(ValueError):
        svi.fixed_log_moneyness_grid(0, 1, 1)
    with pytest.raises(ValueError):
        svi.butterfly_diagnostics(grid, p, tolerance=-1)
    with pytest.raises(ValueError):
        svi.svi_total_variance([np.nan], p)
    with pytest.raises(ValueError):
        svi.variance_positivity([], p)
