"""Synthetic mathematical verification, never empirical option experiments."""

import math
import socket

import numpy as np
import pytest

from q1 import pricing


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Mathematical verification is data independent and offline")
    monkeypatch.setattr(socket.socket, "connect", denied)


def test_textbook_call_and_put(synthetic_cases):
    case = synthetic_cases["textbook_case"]
    f = pricing.forward_from_spot(case["spot"], case["rate"], case["dividend_yield"], case["time_to_expiry"])
    d = pricing.discount_factor(case["rate"], case["time_to_expiry"])
    inputs = (f, case["strike"], case["time_to_expiry"], case["volatility"], d)
    assert pricing.call_price_scalar(*inputs) == pytest.approx(case["call_reference"], abs=1e-12)
    assert pricing.put_price_scalar(*inputs) == pytest.approx(case["put_reference"], abs=1e-12)


@pytest.mark.parametrize("kind", ["C", "P"])
@pytest.mark.parametrize("sigma", [0.01, 0.2, 0.7, 2.5])
def test_recover_known_iv(kind, sigma):
    # Near-ATM prices retain enough sensitivity; deep tail identifiability is
    # tested separately rather than asserted for an unidentifiable premium.
    premium = pricing.black_scholes(100, 100, 0.75, sigma, 0.97, kind)
    result = pricing.implied_volatility(premium, 100, 100, 0.75, 0.97, kind)
    assert result.converged, result
    assert result.volatility == pytest.approx(sigma, abs=1e-10)
    assert abs(result.price_residual) <= 1e-10
    assert result.function_calls >= 5
    assert result.solver_flag == "converged"


@pytest.mark.parametrize("kind", ["C", "P"])
@pytest.mark.parametrize("sigma", [1e-4, 5.0])
def test_exact_iv_bracket_endpoints(kind, sigma):
    premium = pricing.black_scholes(100, 100, 1, sigma, 1, kind)
    result = pricing.implied_volatility(premium, 100, 100, 1, 1, kind)
    assert result.converged and result.reason == "BRACKET_ENDPOINT"
    assert result.volatility == sigma
    assert result.iterations == 0 and result.function_calls == 2


@pytest.mark.parametrize("kind", ["C", "P"])
@pytest.mark.parametrize("sigma", [5e-5, 6.0])
def test_economically_bounded_but_outside_brent_bracket(kind, sigma):
    premium = pricing.black_scholes(100, 100, 1, sigma, 1, kind)
    result = pricing.implied_volatility(premium, 100, 100, 1, 1, kind)
    assert result.reason == "BRACKET_FAILURE"
    assert not result.converged and result.volatility is None
    assert result.bracket == (1e-4, 5)
    assert result.bracket_prices is not None


@pytest.mark.parametrize("price,kind,reason", [
    (-1, "C", "INVALID_PRICE"), (151, "C", "OUT_OF_BOUNDS"),
    (49, "C", "OUT_OF_BOUNDS"), (101, "P", "OUT_OF_BOUNDS"),
    (0, "P", "LOWER_BOUND_UNIDENTIFIABLE"), (50, "C", "LOWER_BOUND_UNIDENTIFIABLE"),
    (150, "C", "UPPER_BOUND_UNIDENTIFIABLE"), (100, "P", "UPPER_BOUND_UNIDENTIFIABLE"),
])
def test_price_rejections(price, kind, reason):
    result = pricing.implied_volatility(price, 150, 100, 1, 1, kind)
    assert result.reason == reason
    assert result.volatility is None and not result.converged


def test_no_false_iv_at_expiry_flat_or_saturated_price():
    result = pricing.implied_volatility(0, 100, 100, 0, 1, "C")
    assert result.reason == "EXPIRED_UNIDENTIFIABLE"
    result = pricing.implied_volatility(np.nextafter(50.0, 100.0), 150, 100, 1e-10, 1, "C")
    assert result.reason == "FLAT_BRACKET_UNIDENTIFIABLE"
    assert not result.converged


@pytest.mark.parametrize("settings", [{"xtol": 0}, {"rtol": 0}, {"price_atol": -1},
                                      {"maxiter": 0}, {"maxiter": 1.2}, {"maxiter": True}])
def test_invalid_solver_settings_have_diagnostics(settings):
    result = pricing.implied_volatility(10, 100, 100, 1, 1, "C", **settings)
    assert result.reason == "INVALID_INPUT_OR_NUMERICS"
    assert result.message and not result.converged


def test_actual_brent_iteration_failure_is_not_a_success():
    premium = pricing.call_price(102, 100, 0.8, 0.37, 0.98)
    result = pricing.implied_volatility(premium, 102, 100, 0.8, 0.98, "C", maxiter=1)
    assert result.reason == "NON_CONVERGENCE"
    assert result.volatility is None and result.iterations == 1
    assert result.price_residual is not None


def test_price_residual_postcheck_is_explicit():
    premium = pricing.call_price(102, 100, 0.8, 0.371234567, 0.98)
    result = pricing.implied_volatility(premium, 102, 100, 0.8, 0.98, "C", xtol=0.1, price_atol=1e-14)
    assert result.reason == "PRICE_RESIDUAL_FAILURE"
    assert not result.converged and result.volatility is None


@pytest.mark.parametrize("arguments", [
    (0, 100, 1, 0.2, 1), (100, -1, 1, 0.2, 1), (100, 100, -1, 0.2, 1),
    (100, 100, 1, -0.1, 1), (100, 100, 1, 0.2, 0),
    (np.nan, 100, 1, 0.2, 1), (100, np.inf, 1, 0.2, 1),
    (100, 100, 0, 0.2, 0.99),
])
def test_invalid_model_inputs_raise(arguments):
    with pytest.raises(ValueError):
        pricing.call_price(*arguments)


@pytest.mark.parametrize("price", [np.nan, np.inf, -np.inf, [1, 2], "bad"])
def test_invalid_iv_inputs_are_not_silently_dropped(price):
    result = pricing.implied_volatility(price, 100, 100, 1, 1, "C")
    assert result.reason == "INVALID_INPUT_OR_NUMERICS"
    assert result.volatility is None


def test_broadcasting_and_scalar_contract():
    forward = np.array([90, 100, 110])[:, None]
    sigma = np.array([0.1, 0.3])[None, :]
    actual = pricing.call_price(forward, 100, 1, sigma, 0.98)
    assert actual.shape == (3, 2)
    for i in range(3):
        for j in range(2):
            assert actual[i, j] == pricing.call_price_scalar(forward[i, 0], 100, 1, sigma[0, j], 0.98)
    assert isinstance(pricing.put_price_scalar(100, 100, 1, 0.2, 1), float)
    with pytest.raises(ValueError):
        pricing.call_price_scalar([100, 110], 100, 1, 0.2, 1)
    with pytest.raises(ValueError, match="shapes"):
        pricing.call_price([100, 110], [90, 100, 110], 1, 0.2, 1)
    assert pricing.call_price([], [], [], [], []).shape == (0,)


def test_iv_batch_preserves_failures_and_order():
    premiums = [pricing.call_price(100, 100, 1, 0.2, 1), -1, np.nan]
    results = pricing.implied_volatility_batch(premiums, 100, 100, 1, 1, ["C", "P", "C"])
    assert results.shape == (3,)
    assert results[0].converged
    assert results[1].reason == "INVALID_PRICE"
    assert results[2].reason == "INVALID_INPUT_OR_NUMERICS"


@pytest.mark.parametrize("kind", ["C", "P"])
def test_zero_volatility_and_expiry_limits(kind):
    f = np.array([80.0, 100.0, 120.0])
    expected = np.maximum(f - 100 if kind == "C" else 100 - f, 0)
    np.testing.assert_array_equal(pricing.black_scholes(f, 100, 0, 0.2, 1, kind), expected)
    np.testing.assert_array_equal(pricing.black_scholes(f, 100, 2, 0, 0.95, kind), 0.95 * expected)


def test_tail_extremes_small_atm_and_float_failures():
    assert pricing.call_price(1e-200, 1e200, 1, 0.2, 1) == 0
    assert pricing.put_price(1e200, 1e-200, 1, 0.2, 1) == 0
    expected = 100 * 1e-14 / math.sqrt(2 * math.pi)
    assert pricing.call_price(100, 100, 1, 1e-14, 1) == pytest.approx(expected, rel=1e-14)
    with pytest.raises(ValueError, match="underflow"):
        pricing.call_price(100, 100, 1e-300, 1e-300, 1)
    with pytest.raises(ValueError, match="overflow"):
        pricing.call_price(100, 100, 1e308, 1e308, 1)
    with pytest.raises(ValueError):
        pricing.discount_factor(-1000, 1)
    with pytest.raises(ValueError):
        pricing.discount_factor(1000, 1)


def test_dividend_conventions_and_negative_rates_are_separate():
    spot, strike, r, q, t, sigma = 100, 102, -0.01, 0.025, 1.3, 0.25
    forward = pricing.forward_from_spot(spot, r, q, t)
    discount = pricing.discount_factor(r, t)
    assert discount > 1
    call = pricing.call_price(forward, strike, t, sigma, discount)
    put = pricing.put_price(forward, strike, t, sigma, discount)
    adjusted = pricing.dividend_adjusted_parity_residual(call, put, spot, strike, r, t, q)
    exam = pricing.examination_parity_residual(call, put, spot, strike, r, t)
    assert adjusted == pytest.approx(0, abs=5e-13)
    assert exam == pytest.approx(spot * math.expm1(-q * t), abs=5e-13)
    assert abs(exam) > 1


def test_no_dividend_parity_and_bounds_on_vector_grid():
    strikes = np.linspace(70, 130, 61)
    spot, rate, t = 100, 0.03, 0.8
    f = pricing.forward_from_spot(spot, rate, 0.0, t)
    d = pricing.discount_factor(rate, t)
    call = pricing.call_price(f, strikes, t, 0.25, d)
    put = pricing.put_price(f, strikes, t, 0.25, d)
    residual = pricing.examination_parity_residual(call, put, spot, strikes, rate, t)
    np.testing.assert_allclose(residual, 0, atol=5e-13, rtol=0)
    for kind, prices in [("C", call), ("P", put)]:
        lower, upper = pricing.price_bounds(f, strikes, t, d, kind)
        assert np.all(prices >= lower) and np.all(prices <= upper)


def test_adapter_requires_explicit_pipeline_quantities_and_no_unit_guessing():
    record = {"price": pricing.call_price(3, 3.1, 0.5, 0.2, 0.99), "strike": 3.1,
              "option_type": "C", "contract_id": "SYNTHETIC_TEST_ONLY", "contract_unit": 10180}
    with pytest.raises(ValueError, match="Missing explicit"):
        pricing.ForwardQuote.from_record(record)
    record.update(forward=3, time_to_expiry=0.5, discount_factor=0.99)
    quote = pricing.ForwardQuote.from_record(record)
    assert quote.price == record["price"]
    assert quote.invert().volatility == pytest.approx(0.2, abs=1e-10)
    assert quote.contract_id == "SYNTHETIC_TEST_ONLY"


def test_iv_reproducibility_and_bad_option_type():
    premium = pricing.call_price(102, 100, 0.8, 0.31, 0.97)
    assert pricing.implied_volatility(premium, 102, 100, 0.8, 0.97, "C") == pricing.implied_volatility(premium, 102, 100, 0.8, 0.97, "C")
    with pytest.raises(ValueError):
        pricing.black_scholes(100, 100, 1, 0.2, 1, "call")
    assert not pricing.implied_volatility(10, 100, 100, 1, 1, "call").converged


def test_spot_path_zero_time_and_complex_rejection():
    assert pricing.forward_from_spot(100, 0.03, 0.03, 1) == 100
    assert pricing.black_scholes_from_spot(100, 100, 0, 0.2, 0.03, 0.01, "C") == 0
    direct = pricing.black_scholes_from_spot(100, 105, 0.7, 0.2, 0.03, 0.01, "P")
    assert direct == pricing.put_price(pricing.forward_from_spot(100, 0.03, 0.01, 0.7),
                                       105, 0.7, 0.2, pricing.discount_factor(0.03, 0.7))
    with pytest.raises(ValueError, match="complex"):
        pricing.call_price(np.array([100+1j]), 100, 1, 0.2, 1)
    with pytest.raises(ValueError):
        pricing.implied_volatility_batch([1+1j], 100, 100, 1, 1, "C")


def test_numerical_functions_do_not_mutate_input_arrays():
    forwards = np.array([90., 100., 110.])
    original = forwards.copy()
    pricing.call_price(forwards, 100, 1, 0.2, 0.98)
    np.testing.assert_array_equal(forwards, original)


def test_log_grid_scaling_high_volatility_and_expired_price_bounds():
    assert pricing.log_moneyness(1e-200, 1e200) == pytest.approx(-400*math.log(10))
    with pytest.raises(ValueError):
        pricing.log_moneyness(0, 100)
    assert pricing.call_price(101, 100, 1, 50, 1) <= 101
    assert pricing.put_price(101, 100, 1, 50, 1) <= 100
    lower, upper = pricing.price_bounds(110, 100, 0, 1, "C")
    assert lower == upper == 10
    assert pricing.implied_volatility(11, 110, 100, 0, 1, "C").reason == "OUT_OF_BOUNDS"
    base = pricing.call_price(105, 100, 1, 0.2, 0.98)
    assert pricing.call_price(1.05, 1, 1, 0.2, 0.98) == pytest.approx(base/100, abs=1e-14)


@pytest.mark.parametrize("kind", ["C", "P"])
def test_wide_log_moneyness_prices_are_nonnegative_and_bounded(kind):
    forward = np.exp(np.linspace(-100, 100, 201))
    sigma = np.array([1e-4, 0.2, 5., 50.])[:, None]
    price = pricing.black_scholes(forward, 1, 1, sigma, 1.02, kind)
    lower, upper = pricing.price_bounds(forward, 1, 1, 1.02, kind)
    assert np.isfinite(price).all()
    assert (price >= lower).all() and (price <= upper).all()
