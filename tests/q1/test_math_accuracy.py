"""Measured errors on SYNTHETIC verification cases; never empirical fit metrics."""

import json
import math
import socket

import numpy as np

from q1 import pricing, svi


def test_synthetic_accuracy_evidence(synthetic_cases, tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Accuracy verification must not acquire market data")
    monkeypatch.setattr(socket.socket, "connect", denied)
    rng = np.random.default_rng(synthetic_cases["seed"])
    count = 256
    forward = 100 * np.exp(rng.uniform(-0.15, 0.15, count))
    strike = np.full(count, 100.0)
    maturity = rng.uniform(0.3, 2.0, count)
    sigma = rng.uniform(0.15, 0.9, count)
    rate = rng.uniform(-0.02, 0.08, count)
    discount = pricing.discount_factor(rate, maturity)
    price_errors, iv_errors, residuals = [], [], []
    for kind in ["C", "P"]:
        prices = pricing.black_scholes(forward, strike, maturity, sigma, discount, kind)
        for i in range(count):
            v = float(sigma[i] * math.sqrt(maturity[i]))
            d1 = math.log(forward[i] / strike[i]) / v + v/2
            d2 = d1-v
            cdf = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))
            # Independent scalar evaluation of the specified equations, not the
            # production log-CDF/parity kernel. Cases deliberately avoid extreme tails.
            reference = discount[i] * (forward[i]*cdf(d1)-strike[i]*cdf(d2)) if kind == "C" else discount[i] * (strike[i]*cdf(-d2)-forward[i]*cdf(-d1))
            price_errors.append(abs(float(prices[i])-reference))
            result = pricing.implied_volatility(prices[i], forward[i], strike[i], maturity[i], discount[i], kind)
            assert result.converged, result
            iv_errors.append(abs(result.volatility - sigma[i]))
            residuals.append(abs(result.price_residual))

    call = pricing.call_price(forward, strike, maturity, sigma, discount)
    put = pricing.put_price(forward, strike, maturity, sigma, discount)
    parity_error = np.max(np.abs(call-put-discount*(forward-strike)))
    parameters = svi.SVIParameters(**synthetic_cases["svi_smooth"])
    grid = svi.fixed_log_moneyness_grid(-0.8, 0.8, 81)
    h1, h2 = 1e-5, 1e-4
    first_fd = (svi.svi_total_variance(grid+h1, parameters)-svi.svi_total_variance(grid-h1, parameters))/(2*h1)
    second_fd = (svi.svi_total_variance(grid+h2, parameters)-2*svi.svi_total_variance(grid, parameters)+svi.svi_total_variance(grid-h2, parameters))/(h2*h2)
    evidence = {
        "provenance": "SYNTHETIC_SOFTWARE_VERIFICATION_ONLY",
        "historical_market_experiment": False,
        "seed": synthetic_cases["seed"], "option_cases": 2*count,
        "max_reference_price_absolute_error": float(max(price_errors)),
        "max_recovered_iv_absolute_error": float(max(iv_errors)),
        "max_iv_repricing_absolute_residual": float(max(residuals)),
        "max_forward_parity_absolute_error": float(parity_error),
        "max_svi_first_derivative_fd_absolute_error": float(np.max(np.abs(svi.svi_first_derivative(grid, parameters)-first_fd))),
        "max_svi_second_derivative_fd_absolute_error": float(np.max(np.abs(svi.svi_second_derivative(grid, parameters)-second_fd))),
        "finite_difference_steps": {"first": h1, "second": h2},
    }
    assert evidence["max_reference_price_absolute_error"] < 2e-12
    assert evidence["max_recovered_iv_absolute_error"] < 5e-10
    assert evidence["max_iv_repricing_absolute_residual"] <= 1e-10
    assert evidence["max_forward_parity_absolute_error"] < 5e-13
    assert evidence["max_svi_first_derivative_fd_absolute_error"] < 1e-9
    assert evidence["max_svi_second_derivative_fd_absolute_error"] < 1e-6
    (tmp_path / "SYNTHETIC_math_accuracy.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print("\nSYNTHETIC SOFTWARE VERIFICATION (not historical results):")
    print(json.dumps(evidence, indent=2))
