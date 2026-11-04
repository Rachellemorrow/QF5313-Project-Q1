"""Classical Q1 SVI fitting and evaluation; reuses verified M2A primitives."""

from dataclasses import asdict, dataclass
import math
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from . import pricing, svi


@dataclass(frozen=True)
class CalibrationConfig:
    grid_lower: float = -0.5
    grid_upper: float = 0.5
    grid_points: int = 201
    positive_floor: float = 1e-10
    rho_margin: float = 1e-8
    eta_floor: float = 1e-8
    max_nfev: int = 2000
    min_distinct_strikes: int = 5
    ftol: float = 1e-12
    xtol: float = 1e-12
    gtol: float = 1e-12
    near_zero_price_floor: float = 1e-8

    def __post_init__(self):
        self.grid()
        if not 0 < self.rho_margin < 1 or self.positive_floor <= 0 or self.eta_floor <= 0:
            raise ValueError("Invalid constraint margins")
        for number in (self.positive_floor, self.eta_floor, self.ftol, self.xtol, self.gtol, self.near_zero_price_floor):
            if not math.isfinite(number) or number <= 0:
                raise ValueError("Solver/price tolerances must be finite and positive")
        if (isinstance(self.max_nfev, bool) or not isinstance(self.max_nfev, int) or self.max_nfev < 1
                or not isinstance(self.min_distinct_strikes, int) or self.min_distinct_strikes < 5):
            raise ValueError("Require positive integer max_nfev and >=5 distinct strikes")

    def grid(self):
        return svi.fixed_log_moneyness_grid(self.grid_lower, self.grid_upper, self.grid_points)


def decode_parameters(theta):
    """Fit v_min rather than a: a=v_min-b*eta*sqrt(1-rho^2)."""
    v_min, b, rho, m, eta = np.asarray(theta, dtype=float)
    a = v_min - b*eta*np.sqrt((1-rho)*(1+rho))
    return svi.SVIParameters(a, b, rho, m, eta)


def encode_parameters(parameters):
    return np.array([svi.svi_global_minimum(parameters), parameters.b, parameters.rho,
                     parameters.m, parameters.eta], dtype=float)


def weighted_residual(theta, k, observed_w, weights):
    return np.sqrt(weights)*(np.asarray(svi.svi_total_variance(k, decode_parameters(theta)))-observed_w)


def residual_jacobian(theta, k, observed_w, weights):
    """Analytic derivative of the residual in (v_min,b,rho,m,eta) coordinates."""
    p = decode_parameters(theta)
    k = np.asarray(k, dtype=float)
    x, radius = k-p.m, np.hypot(k-p.m, p.eta)
    s = np.sqrt((1-p.rho)*(1+p.rho))
    jac = np.column_stack([
        np.ones_like(k), p.rho*x+radius-p.eta*s,
        p.b*(x+p.eta*p.rho/s), -np.asarray(svi.svi_first_derivative(k, p)),
        p.b*(p.eta/radius-s),
    ])
    return np.sqrt(weights)[:, None]*jac


def fit_slice(k, observed_w, *, weights=None, initial=None, config=CalibrationConfig()):
    """Constrained total-variance LS; every failure returns an explicit fit record.

    Reported objective = sum_i weight_i*(w_model-w_observed)^2; SciPy cost is half
    that value. No butterfly/calendar penalty or repair is hidden in this objective.
    Constraint bounds enforce positive total variance globally, not just at quotes.
    """
    start = perf_counter()
    record = {
        "converged": False, "failure_reason": None, "message": None, "fitted_parameters": None,
        "initial_parameters": None, "initial_transformed_parameters": None,
        "parameter_order": ["a", "b", "rho", "m", "eta"],
        "transformed_order": ["v_min", "b", "rho", "m", "eta"],
        "objective": "sum(weights*(SVI_total_variance-observed_total_variance)**2)",
        "objective_value": None, "scipy_cost": None, "nfev": 0, "njev": 0,
        "objective_evaluations": 0, "evaluation_history": [], "weights": None,
        "solver": {"method": "trf", "jac": "analytic", "x_scale": "jac", "loss": "linear",
                   "max_nfev": config.max_nfev, "ftol": config.ftol, "xtol": config.xtol, "gtol": config.gtol},
        "constraints": {"b_min": 0.0, "rho_bounds": [-1+config.rho_margin, 1-config.rho_margin],
                        "eta_min": config.eta_floor, "v_min_min": config.positive_floor,
                        "a_definition": "v_min-b*eta*sqrt(1-rho**2)", "m_bounds": "unbounded"},
        "scipy_status": None, "optimality": None, "global_minimum_variance": None,
        "quote_count": 0, "distinct_k_count": 0, "fit_seconds": 0.0,
    }
    try:
        if any(np.iscomplexobj(item) for item in (k, observed_w, weights)):
            raise ValueError("Inputs must be real")
        k, target = np.asarray(k, dtype=float), np.asarray(observed_w, dtype=float)
        if k.ndim != 1 or target.shape != k.shape or not np.isfinite(k).all() or not np.isfinite(target).all():
            raise ValueError("Require finite matching one-dimensional k/variance arrays")
        record["quote_count"], record["distinct_k_count"] = len(k), len(np.unique(k))
        if len(np.unique(k)) < config.min_distinct_strikes:
            record["failure_reason"] = "INSUFFICIENT_DISTINCT_STRIKES"
            record["message"] = f"Need >= {config.min_distinct_strikes} distinct liquid strikes"
            return record
        if np.any(target <= 0):
            raise ValueError("Observed total variance must be positive")
        weights = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)
        if weights.shape != k.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Weights must be finite, positive and match quotes")
        record["weights"] = weights.tolist()
        if initial is None:
            theta = np.array([max(float(np.median(target))*0.5, config.positive_floor*10),
                              0.1, 0., float(np.median(k)), 0.2])
            initial = decode_parameters(theta)
            record["initialization"] = "v_min=0.5*median(w), b=0.1, rho=0, m=median(k), eta=0.2; positive floor guard"
        else:
            theta = encode_parameters(initial)
            record["initialization"] = "Explicit caller-supplied raw SVI parameters; not silently projected"
        record["initial_parameters"] = asdict(initial)
        record["initial_transformed_parameters"] = theta.tolist()
        lower = np.array([config.positive_floor, 0., -1+config.rho_margin, -np.inf, config.eta_floor])
        upper = np.array([np.inf, np.inf, 1-config.rho_margin, np.inf, np.inf])
        if np.any(theta < lower) or np.any(theta > upper):
            raise ValueError("Initial parameters violate the declared transformed bounds")

        def objective(candidate, observed_k, observed_target, observed_weights):
            residual = weighted_residual(candidate, observed_k, observed_target, observed_weights)
            record["objective_evaluations"] += 1
            record["evaluation_history"].append({"evaluation": record["objective_evaluations"],
                                                 "weighted_rss": float(residual@residual)})
            return residual

        fitted = least_squares(objective, theta, jac=residual_jacobian, args=(k, target, weights),
                               bounds=(lower, upper), method="trf", x_scale="jac", loss="linear",
                               ftol=config.ftol, xtol=config.xtol, gtol=config.gtol, max_nfev=config.max_nfev)
        p = decode_parameters(fitted.x)
        residual = weighted_residual(fitted.x, k, target, weights)
        record.update(fitted_parameters=asdict(p), objective_value=float(residual@residual),
                      scipy_cost=float(fitted.cost), nfev=int(fitted.nfev), njev=int(fitted.njev),
                      scipy_status=int(fitted.status), optimality=float(fitted.optimality), message=str(fitted.message),
                      global_minimum_variance=float(svi.svi_global_minimum(p)))
        positive = svi.variance_positivity(np.r_[config.grid(), k], p)
        if not fitted.success:
            record["failure_reason"] = "SOLVER_NONCONVERGENCE"
        elif not positive["positive_on_grid"] or not positive["positive_globally"]:
            record["failure_reason"] = "TOTAL_VARIANCE_POSITIVITY_FAILURE"
        else:
            record["converged"] = True
    except (ValueError, TypeError, OverflowError, FloatingPointError) as error:
        record["failure_reason"] = "INVALID_INPUT_OR_NUMERICAL_FAILURE"
        record["message"] = str(error)
    finally:
        record["fit_seconds"] = perf_counter()-start
    return record


def reprice_quote(quote, parameters, *, price_floor=1e-8):
    """Absolute errors for positive prices; relative denominator never clipped."""
    output = {"quote_id": quote.get("quote_id"), "status": "FAILED", "failure_reason": None,
              "market_price": None, "model_price": None, "fitted_iv": None,
              "absolute_error": None, "relative_error": None, "relative_error_status": None}
    try:
        price = float(quote["price"])
        if not math.isfinite(price) or price <= 0 or not math.isfinite(price_floor) or price_floor <= 0:
            raise ValueError("Require positive finite market price and denominator threshold")
        k = pricing.log_moneyness(quote["strike"], quote["forward"])
        w = svi.svi_total_variance(k, parameters)
        t = float(quote["time_to_expiry"])
        if t <= 0:
            raise ValueError("Nonpositive maturity")
        sigma = math.sqrt(w/t)
        model = pricing.black_scholes(quote["forward"], quote["strike"], t, sigma,
                                      quote["discount_factor"], quote["option_type"])
        error = abs(model-price)
        output.update(status="REPRICED", market_price=price, model_price=model,
                      fitted_iv=sigma, absolute_error=error,
                      relative_error=error/price if price >= price_floor else None,
                      relative_error_status="DEFINED" if price >= price_floor else "NEAR_ZERO_DENOMINATOR_EXCLUDED")
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        output["failure_reason"] = str(error)
    return output


def verify_surface(slices, config=CalibrationConfig()):
    """Report every grid violation; retain failed slices and missing diagnostics."""
    grid = config.grid()
    diagnostics = {"grid": grid.tolist(), "slices": [], "calendar": [], "negative_forward_variance": [],
                   "interpolation": [], "failed_slice_ids": []}
    groups = {}
    for item in slices:
        key = item["slice_id"]
        if not item["fit"]["converged"]:
            diagnostics["failed_slice_ids"].append(key)
            diagnostics["slices"].append({"slice_id": key, "status": "NO_VALID_FIT", "examination": None, "complete": None})
            continue
        p = svi.SVIParameters(**item["fit"]["fitted_parameters"])
        positivity = svi.variance_positivity(grid, p)
        butterfly = svi.butterfly_diagnostics(grid, p)
        record = {"slice_id": key, "status": "VERIFIED", "total_variance": positivity["total_variance"].tolist(),
                  "positive_on_grid": positivity["positive_on_grid"], "global_minimum": positivity["global_minimum"]}
        for label, report in butterfly.items():
            record[label] = {"values": report.values.tolist(), "count": report.count,
                             "locations": report.locations.tolist(), "magnitudes": report.magnitudes.tolist()}
        diagnostics["slices"].append(record)
        groups.setdefault(item["observation_date"], []).append(item)
    for date, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: row["time_to_expiry"])
        if len(ordered) < 2:
            diagnostics["interpolation"].append({"observation_date": date, "status": "INSUFFICIENT_FITTED_MATURITIES"})
            continue
        times = np.array([item["time_to_expiry"] for item in ordered])
        if np.any(np.diff(times) <= 0):
            diagnostics["interpolation"].append({"observation_date": date, "status": "DUPLICATE_MATURITIES_NO_TERM_STRUCTURE"})
            continue
        w = np.array([svi.svi_total_variance(grid, svi.SVIParameters(**item["fit"]["fitted_parameters"])) for item in ordered])
        # Compare every maturity pair, not just adjacent expiries.
        for i in range(len(ordered)):
            for j in range(i+1, len(ordered)):
                report = svi.calendar_diagnostics(grid, w[i], w[j], times[i], times[j])
                diagnostics["calendar"].append({"observation_date": date, "earlier_slice": ordered[i]["slice_id"],
                    "later_slice": ordered[j]["slice_id"], "difference": report.values.tolist(),
                    "count": report.count, "locations": report.locations.tolist(), "magnitudes": report.magnitudes.tolist()})
        forward = svi.forward_variance_diagnostics(times, w)
        for interval, column in forward.indices:
            diagnostics["negative_forward_variance"].append({"observation_date": date,
                "earlier_slice": ordered[interval]["slice_id"], "later_slice": ordered[interval+1]["slice_id"],
                "k": float(grid[column]), "magnitude": float(-forward.forward_variance[interval, column])})
        query = np.unique(np.r_[times, (times[:-1]+times[1:])/2])
        interpolated = svi.interpolate_total_variance(times, w, query)
        endpoints = svi.interpolate_total_variance(times, w, times)
        diagnostics["interpolation"].append({"observation_date": date, "status": "INTERPOLATED_WITHOUT_EXTRAPOLATION",
            "maturities": times.tolist(), "query_maturities": query.tolist(), "total_variance": interpolated.tolist(),
            "endpoint_max_abs_error": float(np.max(np.abs(endpoints-w))),
            "missing_failed_expiries_not_filled": True})
    return diagnostics


def repricing_summary(slices):
    """Slice maxima -> date maxima -> median/p95 across completely evaluated dates.

    Partial-date available maxima remain visible, but are not silently treated as
    complete maxima when a slice, inversion, repricing or denominator is missing.
    """
    by_date, by_expiry = {}, {}
    per_slice = []
    relative = []
    for item in slices:
        values = [row["relative_error"] for row in item.get("repricing", []) if row["relative_error"] is not None]
        relative.extend(values)
        maximum = max(values) if values else None
        failed_repricing = sum(row["status"] != "REPRICED" for row in item.get("repricing", []))
        excluded_denominators = sum(row["status"] == "REPRICED" and row["relative_error"] is None for row in item.get("repricing", []))
        complete = bool(item["fit"]["converged"] and not item.get("quote_preparation_failures", 0)
                        and not failed_repricing and not excluded_denominators and maximum is not None)
        record = {"slice_id": item["slice_id"], "observation_date": item["observation_date"],
                          "expiry_date": item.get("expiry_date"),
                          "max_relative_error": maximum, "defined_relative_errors": len(values),
                          "fit_converged": item["fit"]["converged"], "complete_relative_error_coverage": complete,
                          "failed_repricing": failed_repricing, "near_zero_denominators": excluded_denominators}
        per_slice.append(record)
        by_date.setdefault(item["observation_date"], []).append(record)
        if item.get("expiry_date"):
            by_expiry.setdefault(item["expiry_date"], []).append(record)
    dates = []
    for date, records in sorted(by_date.items()):
        available = [row["max_relative_error"] for row in records if row["max_relative_error"] is not None]
        dates.append({"observation_date": date, "maximum_across_available_expiry_slices": max(available) if available else None,
                      "complete_relative_error_coverage": all(row["complete_relative_error_coverage"] for row in records),
                      "incomplete_slices": sum(not row["complete_relative_error_coverage"] for row in records)})
    maxima = [row["maximum_across_available_expiry_slices"] for row in dates if row["complete_relative_error_coverage"]]
    expiry_summary = []
    for expiry, records in sorted(by_expiry.items()):
        available = [row["max_relative_error"] for row in records if row["complete_relative_error_coverage"]]
        expiry_summary.append({"expiry_date": expiry, "complete_dates": len(available),
            "incomplete_dates": sum(not row["complete_relative_error_coverage"] for row in records),
            "cross_date_median_of_slice_maxima": float(np.median(available)) if available else None,
            "cross_date_p95_of_slice_maxima": float(np.percentile(available, 95)) if available else None})
    return {"per_slice": per_slice, "per_date": dates,
            "per_expiry_cross_date_summary": expiry_summary,
            "cross_date_median_of_date_maxima": float(np.median(maxima)) if maxima else None,
            "cross_date_p95_of_date_maxima": float(np.percentile(maxima, 95)) if maxima else None,
            "pooled_quote_relative_error_median_separately_labelled": float(np.median(relative)) if relative else None,
            "pooled_quote_relative_error_p95_separately_labelled": float(np.percentile(relative, 95)) if relative else None,
            "dates_with_defined_errors": sum(row["maximum_across_available_expiry_slices"] is not None for row in dates),
            "complete_dates_used_for_primary_cross_date_statistics": len(maxima),
            "incomplete_dates_excluded_from_primary_statistics": sum(not row["complete_relative_error_coverage"] for row in dates),
            "slices_without_relative_errors": sum(row["max_relative_error"] is None for row in per_slice)}
