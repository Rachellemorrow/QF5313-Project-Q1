"""Forward Black–Scholes, Brent IV diagnostics and distinctly labelled parity.

All premiums are per underlying unit. T is caller-supplied years; r and q are
continuous annual decimal rates. An exogenous positive discount factor is accepted
directly (including D > 1 for negative rates). No calendar, curve, dividend or
contract-multiplier assumption is inferred from a market record.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, log_ndtr

from ._validation import finite_arrays, scalar, value


IV_BRACKET = (1e-4, 5.0)


def _option_type(option_type):
    if not isinstance(option_type, str) or option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'")
    return option_type


def _forward_inputs(forward, strike, time_to_expiry, discount_factor, **extra):
    arrays = finite_arrays(forward=forward, strike=strike, time_to_expiry=time_to_expiry,
                           discount_factor=discount_factor, **extra)
    f, k, t, d = arrays[:4]
    if np.any(f <= 0) or np.any(k <= 0):
        raise ValueError("forward and strike must be positive")
    if np.any(t < 0) or np.any(d <= 0):
        raise ValueError("time_to_expiry must be nonnegative and discount_factor positive")
    if np.any((t == 0) & (d != 1)):
        raise ValueError("discount_factor must equal one at expiry")
    return arrays


def discount_factor(rate, time_to_expiry):
    """exp(-r*T), with no implicit conversion from percent or choice of day count."""
    r, t = finite_arrays(rate=rate, time_to_expiry=time_to_expiry)
    if np.any(t < 0):
        raise ValueError("time_to_expiry must be nonnegative")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.exp(-r * t)
    if np.any(result <= 0):
        raise ValueError("Discount factor underflowed to zero")
    return value(result)


def forward_from_spot(spot, rate, dividend_yield, time_to_expiry):
    """S*exp((r-q)*T); q is explicit, not a silently assumed empirical dividend."""
    s, r, q, t = finite_arrays(spot=spot, rate=rate, dividend_yield=dividend_yield,
                               time_to_expiry=time_to_expiry)
    if np.any(s <= 0) or np.any(t < 0):
        raise ValueError("spot must be positive and time_to_expiry nonnegative")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.exp(np.log(s) + (r - q) * t)
    result = np.where((t == 0) | (r == q), s, result)
    if np.any(result <= 0):
        raise ValueError("Forward underflowed to zero")
    return value(result)


def price_bounds(forward, strike, time_to_expiry, discount_factor, option_type):
    """Intrinsic lower / infinite-volatility upper limit; unique payoff at T=0."""
    kind = _option_type(option_type)
    f, k, t, d = _forward_inputs(forward, strike, time_to_expiry, discount_factor)
    with np.errstate(over="ignore", invalid="ignore"):
        lower = d * np.maximum(f - k if kind == "C" else k - f, 0)
        upper = d * (f if kind == "C" else k)
        upper = np.where(t == 0, lower, upper)
    return value(lower), value(upper)


def log_moneyness(strike, forward):
    """ln(K/F), using log differences so the ratio need not be representable."""
    k, f = finite_arrays(strike=strike, forward=forward)
    if np.any(k <= 0) or np.any(f <= 0):
        raise ValueError("strike and forward must be positive")
    return value(np.log(k) - np.log(f))


def black_scholes(forward, strike, time_to_expiry, volatility, discount_factor, option_type):
    """Broadcasting float64 forward-price formula; scalar input returns float.

    d1 = ln(F/K)/(sigma*sqrt(T)) + sigma*sqrt(T)/2, d2=d1-sigma*sqrt(T).
    Evaluate the OTM side in log-CDF space and obtain the ITM side by parity to
    avoid subtracting large nearly equal premiums. T=0 or sigma=0 gives the
    discounted intrinsic limit. Empty broadcast arrays are permitted.
    """
    kind = _option_type(option_type)
    f, k, t, d, sigma = _forward_inputs(forward, strike, time_to_expiry,
                                       discount_factor, volatility=volatility)
    if np.any(sigma < 0):
        raise ValueError("volatility must be nonnegative")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        stddev = sigma * np.sqrt(t)
    if not np.all(np.isfinite(stddev)):
        raise ValueError("sigma*sqrt(T) overflowed; rescale inputs")
    if np.any((sigma > 0) & (t > 0) & (stddev == 0)):
        raise ValueError("Positive sigma*sqrt(T) underflowed; rescale inputs")
    active = stddev > 0
    safe_stddev = np.where(active, stddev, 1.0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        log_fk = np.log(f) - np.log(k)
        d1 = log_fk / safe_stddev + 0.5 * safe_stddev
        d2 = log_fk / safe_stddev - 0.5 * safe_stddev
        # F<=K: OTM call. F>K: OTM put. Work in undiscounted units first.
        call_otm = f <= k
        log_a = np.where(call_otm, np.log(f) + log_ndtr(d1), np.log(k) + log_ndtr(-d2))
        log_b = np.where(call_otm, np.log(k) + log_ndtr(d2), np.log(f) + log_ndtr(-d1))
        log_ratio = np.minimum(log_b - log_a, 0.0)
        small = np.exp(log_a) * (-np.expm1(log_ratio))
        # At a saturated tail both log probabilities can be -inf: the float64
        # premium has underflowed to zero, not a negative price or a usable IV.
        small = np.where(np.isneginf(log_a), 0.0, small)
        # At exact ATM use the erf identity, including tiny positive stddevs.
        small = np.where(f == k, f * erf(stddev / (2 * np.sqrt(2.0))), small)
        small = np.where(active, small, 0.0)
        # Guard only floating-point roundoff of the analytic OTM expression.
        # Market prices and arbitrage diagnostics are never clipped or repaired.
        small = np.clip(small, 0.0, np.minimum(f, k))
        call = small + np.maximum(f - k, 0)
        put = small + np.maximum(k - f, 0)
        result = d * (call if kind == "C" else put)
        intrinsic = d * np.maximum(f-k if kind == "C" else k-f, 0.0)
        upper = d * (f if kind == "C" else k)
        result = np.clip(result, intrinsic, upper)
    return value(result)


def call_price(forward, strike, time_to_expiry, volatility, discount_factor):
    return black_scholes(forward, strike, time_to_expiry, volatility, discount_factor, "C")


def put_price(forward, strike, time_to_expiry, volatility, discount_factor):
    return black_scholes(forward, strike, time_to_expiry, volatility, discount_factor, "P")


def black_scholes_from_spot(spot, strike, time_to_expiry, volatility, rate, dividend_yield, option_type):
    """Convenience path deriving BOTH F and D from the same explicit r, q and T."""
    forward = forward_from_spot(spot, rate, dividend_yield, time_to_expiry)
    discount = discount_factor(rate, time_to_expiry)
    return black_scholes(forward, strike, time_to_expiry, volatility, discount, option_type)


def call_price_scalar(forward, strike, time_to_expiry, volatility, discount_factor):
    inputs = [scalar(name, item) for name, item in zip(
        ["forward", "strike", "time_to_expiry", "volatility", "discount_factor"],
        [forward, strike, time_to_expiry, volatility, discount_factor])]
    return call_price(*inputs)


def put_price_scalar(forward, strike, time_to_expiry, volatility, discount_factor):
    inputs = [scalar(name, item) for name, item in zip(
        ["forward", "strike", "time_to_expiry", "volatility", "discount_factor"],
        [forward, strike, time_to_expiry, volatility, discount_factor])]
    return put_price(*inputs)


@dataclass(frozen=True)
class IVResult:
    volatility: float | None
    converged: bool
    reason: str
    message: str
    iterations: int = 0
    function_calls: int = 0
    bracket: tuple[float, float] = IV_BRACKET
    bracket_prices: tuple[float, float] | None = None
    price_residual: float | None = None
    solver_flag: str | None = None


def implied_volatility(price, forward, strike, time_to_expiry, discount_factor,
                       option_type, *, xtol=1e-12, rtol=1e-12, maxiter=100,
                       price_atol=1e-10):
    """Scalar Brent inversion on the examination bracket, never silently clipped.

    Failure volatility is None. Counts include the two bracket-price evaluations.
    A boundary/intrinsic or saturated quote can fail identification even if it is
    mathematically arbitrage-bounded. Positive IV is not inferred from such quotes.
    """
    calls = 0
    bracket_prices = None

    def result(reason, message, **kwargs):
        return IVResult(volatility=None, converged=False, reason=reason, message=message,
                        function_calls=calls, bracket_prices=bracket_prices, **kwargs)

    try:
        p, f, k, t, d = [scalar(name, item) for name, item in zip(
            ["price", "forward", "strike", "time_to_expiry", "discount_factor"],
            [price, forward, strike, time_to_expiry, discount_factor])]
        _option_type(option_type)
        _forward_inputs(f, k, t, d)
        xtol = scalar("xtol", xtol)
        rtol = scalar("rtol", rtol)
        price_atol = scalar("price_atol", price_atol)
        if xtol <= 0 or rtol < 4 * np.finfo(float).eps or price_atol < 0:
            raise ValueError("Require xtol>0, rtol>=4*float64 eps, price_atol>=0")
        if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        if p < 0:
            return result("INVALID_PRICE", "Option price must be nonnegative")
        lower, upper = price_bounds(f, k, t, d, option_type)
        if p < lower or p > upper:
            return result("OUT_OF_BOUNDS", f"Price is outside [{lower}, {upper}]")
        if t == 0:
            return result("EXPIRED_UNIDENTIFIABLE", "Volatility cannot be identified at expiry")
        if p == upper:
            return result("UPPER_BOUND_UNIDENTIFIABLE", "Upper limit has no identifiable finite positive IV")
        if p == lower:
            return result("LOWER_BOUND_UNIDENTIFIABLE", "Intrinsic-limit/rounded price does not identify positive IV")

        def objective(sigma):
            nonlocal calls
            calls += 1
            return black_scholes(f, k, t, sigma, d, option_type) - p

        left = objective(IV_BRACKET[0])
        right = objective(IV_BRACKET[1])
        bracket_prices = (left + p, right + p)
        if bracket_prices[0] == bracket_prices[1]:
            return result("FLAT_BRACKET_UNIDENTIFIABLE", "Bracket prices coincide at float64 precision")
        for sigma, residual in zip(IV_BRACKET, (left, right)):
            if residual == 0:
                return IVResult(sigma, True, "BRACKET_ENDPOINT", "Exact bracket endpoint price", 0,
                                calls, IV_BRACKET, bracket_prices, 0.0, "endpoint")
        if left > 0 or right < 0:
            return result("BRACKET_FAILURE", "Arbitrage-bounded price is outside the [1e-4, 5] price bracket")
        root, diagnostic = brentq(objective, *IV_BRACKET, xtol=xtol, rtol=rtol,
                                  maxiter=maxiter, full_output=True, disp=False)
        residual = objective(root)
        if not diagnostic.converged:
            return result("NON_CONVERGENCE", "Brent iteration limit reached", iterations=diagnostic.iterations,
                          price_residual=residual, solver_flag=str(diagnostic.flag))
        if abs(residual) > price_atol:
            return result("PRICE_RESIDUAL_FAILURE", "Root did not meet the requested price tolerance",
                          iterations=diagnostic.iterations, price_residual=residual, solver_flag=str(diagnostic.flag))
        return IVResult(root, True, "CONVERGED", "Brent root and price residual verified",
                        diagnostic.iterations, calls, IV_BRACKET, bracket_prices, residual, str(diagnostic.flag))
    except (ValueError, TypeError, OverflowError, FloatingPointError) as error:
        return result("INVALID_INPUT_OR_NUMERICS", str(error))


def implied_volatility_batch(price, forward, strike, time_to_expiry, discount_factor, option_type, **settings):
    """Broadcast numeric inputs; return an object array of per-quote IVResult records."""
    try:
        if any(np.iscomplexobj(item) for item in [price, forward, strike, time_to_expiry, discount_factor]):
            raise ValueError("Batch numerical inputs must be real")
        arrays = np.broadcast_arrays(*(np.asarray(item, dtype=float) for item in
                                       [price, forward, strike, time_to_expiry, discount_factor]),
                                     np.asarray(option_type, dtype=str))
    except (ValueError, TypeError) as error:
        raise ValueError("Batch inputs must have numeric, broadcast-compatible shapes") from error
    output = np.empty(arrays[0].shape, dtype=object)
    for index in np.ndindex(output.shape):
        output[index] = implied_volatility(*(item[index] for item in arrays), **settings)
    return output


def _parity_inputs(call, put, spot, strike, rate, time_to_expiry, **extra):
    arrays = finite_arrays(call=call, put=put, spot=spot, strike=strike, rate=rate,
                           time_to_expiry=time_to_expiry, **extra)
    c, p, s, k, _, t = arrays[:6]
    if np.any(c < 0) or np.any(p < 0) or np.any(s <= 0) or np.any(k <= 0) or np.any(t < 0):
        raise ValueError("Require nonnegative premiums/time and positive spot/strike")
    return arrays


def examination_parity_residual(call, put, spot, strike, rate, time_to_expiry):
    """Preserve exactly C-P-(S-K*exp(-r*T)); no dividend adjustment."""
    c, p, s, k, r, t = _parity_inputs(call, put, spot, strike, rate, time_to_expiry)
    return value((c - p) - (s - k * discount_factor(r, t)))


def dividend_adjusted_parity_residual(call, put, spot, strike, rate, time_to_expiry, dividend_yield):
    """Separately labelled C-P-(S*exp(-q*T)-K*exp(-r*T))."""
    c, p, s, k, r, t, q = _parity_inputs(call, put, spot, strike, rate, time_to_expiry,
                                        dividend_yield=dividend_yield)
    return value((c - p) - (s * discount_factor(q, t) - k * discount_factor(r, t)))


@dataclass(frozen=True)
class ForwardQuote:
    """Minimal numerical record adapter, NOT a historical-data validation gate.

    Existing audit columns price/strike/option_type/contract_id are preserved;
    forward, time_to_expiry and discount_factor must be supplied explicitly by a
    future verified pipeline. Contract units must already match premium units.
    """
    price: float
    forward: float
    strike: float
    time_to_expiry: float
    discount_factor: float
    option_type: str
    contract_id: str | None = None

    @classmethod
    def from_record(cls, record: Mapping):
        fields = ["price", "forward", "strike", "time_to_expiry", "discount_factor"]
        missing = [name for name in [*fields, "option_type"] if name not in record]
        if missing:
            raise ValueError(f"Missing explicit numerical fields: {', '.join(missing)}")
        inputs = {name: scalar(name, record[name]) for name in fields}
        _forward_inputs(inputs["forward"], inputs["strike"], inputs["time_to_expiry"], inputs["discount_factor"])
        return cls(**inputs, option_type=_option_type(record["option_type"]), contract_id=record.get("contract_id"))

    def invert(self, **settings):
        return implied_volatility(self.price, self.forward, self.strike, self.time_to_expiry,
                                  self.discount_factor, self.option_type, **settings)
