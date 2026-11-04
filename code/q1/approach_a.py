"""Q1 Approach A orchestration: eligibility -> pairing/IV -> slices -> fit -> verify.

Development and empirical modes have distinct provenance gates. No module here
changes M2 eligibility or fills the missing historical-data/pricing conventions.
"""

from collections import Counter
from dataclasses import asdict
import json
import math
import re

import numpy as np
import pandas as pd

from . import pricing, svi
from .classical_svi import CalibrationConfig, fit_slice, reprice_quote, repricing_summary, verify_surface
from .market_data import VERIFIED_EMPIRICAL
from .rates import date_value
from .verification_fixtures import SYNTHETIC_ORIGIN


def plain(value):
    """Strict JSON-compatible diagnostics: unavailable values are null, never NaN."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(item) for item in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def true(value):
    return value is True or isinstance(value, np.bool_) and bool(value)


def empirical_gate(frame):
    """Recheck required evidence flags and prepared values; never use preview rows."""
    accepted, rows = [], []
    counts = Counter()
    required_flags = ["eligible_for_empirical", "provenance_verified", "price_unit_verified",
                      "metadata_effective_valid", "metadata_complete", "conventions_approved"]
    numeric = ["price", "strike", "forward", "discount_factor", "time_to_expiry", "spot", "contract_unit", "zero_rate", "dividend_yield"]
    for index, raw in enumerate(frame.to_dict(orient="records")):
        reasons = []
        for name in required_flags:
            if not true(raw.get(name)):
                reasons.append("PIPELINE_FLAG_"+name.upper())
        if raw.get("row_readiness") != VERIFIED_EMPIRICAL or raw.get("data_origin") != "REAL_ARCHIVED_OBSERVATION":
            reasons.append("NOT_VERIFIED_EMPIRICAL_ORIGIN")
        if raw.get("instrument_family") != "ETF" or raw.get("underlying") != "510050":
            reasons.append("NOT_50ETF_OPTIONS")
        if raw.get("price_kind") != "daily_close":
            reasons.append("NOT_DAILY_CLOSE")
        date = date_value(raw.get("observation_date"))
        expiry, listing = date_value(raw.get("expiry_date")), date_value(raw.get("listing_date"))
        first, last = date_value(raw.get("metadata_valid_from")), date_value(raw.get("metadata_valid_to"))
        if not (date and expiry and listing and first and last and listing <= date < expiry and first <= date <= last):
            reasons.append("DATE_EFFECTIVE_METADATA_NOT_ESTABLISHED")
        if raw.get("metadata_basis") != "VERIFIED_DATE_EFFECTIVE":
            reasons.append("NO_VERIFIED_HISTORICAL_METADATA_BASIS")
        for name in ["source_sha256", "metadata_source_sha256"]:
            if not isinstance(raw.get(name), str) or not re.fullmatch(r"[0-9a-f]{64}", raw[name]):
                reasons.append("MISSING_PROVENANCE_DIGEST")
        try:
            values = {name: float(raw[name]) for name in numeric}
            if not all(math.isfinite(number) for number in values.values()):
                raise ValueError("nonfinite")
            if any(values[name] <= 0 for name in numeric if name not in {"zero_rate", "dividend_yield"}):
                raise ValueError("nonpositive")
            if raw.get("option_type") not in {"C", "P"} or not raw.get("observation_date") or not raw.get("expiry_date"):
                raise ValueError("missing contract fields")
        except (ValueError, TypeError, KeyError):
            reasons.append("INVALID_PREPARED_NUMERICAL_RECORD")
        try:
            if json.loads(raw.get("rejection_reasons", "[]")):
                reasons.append("PIPELINE_REJECTIONS_PRESENT")
        except (TypeError, ValueError):
            reasons.append("INVALID_PIPELINE_REJECTION_LEDGER")
        reasons = sorted(set(reasons))
        counts.update(reasons)
        rows.append({"quote_id": raw.get("quote_id"), "eligible": not reasons, "reasons": reasons})
        if not reasons:
            accepted.append(index)
    return frame.iloc[accepted].copy(), {"input_rows": len(frame), "eligible_rows": len(accepted),
        "rejected_rows": len(frame)-len(accepted), "reason_counts": dict(sorted(counts.items())), "rows": rows,
        "strict_pipeline_eligibility_relaxed": False}


def prepare_quotes(frame, *, mode, config=CalibrationConfig()):
    if mode not in {"development", "empirical"}:
        raise ValueError("Unknown execution mode")
    origin = SYNTHETIC_ORIGIN if mode == "development" else "REAL_ARCHIVED_OBSERVATION"
    if not (frame["data_origin"] == origin).all():
        raise ValueError("Development/empirical origin boundary violated")
    ledger = []
    for raw in frame.to_dict(orient="records"):
        record = {"quote_id": raw.get("quote_id"), "input": plain(raw), "iv": None,
                  "k": None, "total_variance": None, "eligible_for_fit": False, "reasons": [], "warnings": []}
        try:
            for name in ["price", "forward", "strike", "time_to_expiry", "discount_factor", "spot", "zero_rate", "dividend_yield", "contract_unit"]:
                raw[name] = float(raw[name])
                if not math.isfinite(raw[name]):
                    raise ValueError("Nonfinite "+name)
            expected_d = pricing.discount_factor(raw["zero_rate"], raw["time_to_expiry"])
            expected_f = pricing.forward_from_spot(raw["spot"], raw["zero_rate"], raw["dividend_yield"], raw["time_to_expiry"])
            if not math.isclose(raw["discount_factor"], expected_d, rel_tol=1e-10, abs_tol=1e-12) or not math.isclose(raw["forward"], expected_f, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Forward/discount/spot/rate/dividend conventions are inconsistent")
            inversion = pricing.implied_volatility(raw["price"], raw["forward"], raw["strike"], raw["time_to_expiry"], raw["discount_factor"], raw["option_type"])
            record["iv"] = asdict(inversion)
            record["k"] = pricing.log_moneyness(raw["strike"], raw["forward"])
            if inversion.converged:
                record["total_variance"] = inversion.volatility**2*raw["time_to_expiry"]
            else:
                record["reasons"].append("IV_"+inversion.reason)
            if not config.grid_lower <= record["k"] <= config.grid_upper:
                record["reasons"].append("OUTSIDE_DECLARED_LOG_MONEYNESS_BAND")
            volume = float(raw.get("volume", float("nan")))
            if not math.isfinite(volume) or volume < 1:
                record["reasons"].append("MISSING_OR_LOW_VOLUME")
            bid, ask = raw.get("bid"), raw.get("ask")
            if bid is None or ask is None or pd.isna(bid) or pd.isna(ask):
                record["warnings"].append("BID_ASK_UNAVAILABLE_NOT_IMPUTED")
            else:
                bid, ask = float(bid), float(ask)
                if not math.isfinite(bid+ask) or bid < 0 or ask <= 0 or bid > ask:
                    record["reasons"].append("INVALID_BID_ASK")
                elif (ask-bid)/((ask+bid)/2) > 0.25:
                    record["reasons"].append("WIDE_RELATIVE_SPREAD")
            if raw["price"] < config.near_zero_price_floor:
                record["warnings"].append("NEAR_ZERO_PRICE_RELATIVE_ERROR_WILL_BE_UNDEFINED")
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            record["reasons"].append("INVALID_QUOTE_INPUT: "+str(error))
        ledger.append(record)
    # An expiry slice cannot silently mix different forwards, discount factors or maturities.
    slices = {}
    for record in ledger:
        raw = record["input"]
        slices.setdefault((raw.get("observation_date"), raw.get("expiry_date")), []).append(record)
    for group in slices.values():
        for name in ["forward", "discount_factor", "time_to_expiry", "spot", "zero_rate", "dividend_yield"]:
            values = [r["input"].get(name) for r in group]
            try:
                numbers = np.asarray(values, dtype=float)
                if not np.isfinite(numbers).all() or not np.allclose(numbers, numbers[0], atol=1e-12, rtol=1e-12):
                    raise ValueError("inconsistent")
            except (ValueError, TypeError):
                for record in group:
                    record["reasons"].append("INCONSISTENT_SLICE_MARKET_INPUTS")
    pairs = {}
    for record in ledger:
        raw = record["input"]
        key = tuple(raw.get(name) for name in ["observation_date", "expiry_date", "strike", "contract_unit"])
        pairs.setdefault(key, []).append(record)
    parity = []
    for key, group in pairs.items():
        calls = [r for r in group if r["input"].get("option_type") == "C"]
        puts = [r for r in group if r["input"].get("option_type") == "P"]
        pair = {"observation_date": key[0], "expiry_date": key[1], "strike": key[2], "contract_unit": key[3],
                "call_count": len(calls), "put_count": len(puts), "status": None,
                "examination_residual": None, "dividend_adjusted_residual": None, "failure_reason": None}
        if len(calls) != 1 or len(puts) != 1:
            pair["status"] = "MISSING_OR_AMBIGUOUS_CALL_PUT_PAIR"
        else:
            call, put = calls[0]["input"], puts[0]["input"]
            try:
                inputs = (call["price"], put["price"], call["spot"], call["strike"], call["zero_rate"], call["time_to_expiry"])
                pair["examination_residual"] = pricing.examination_parity_residual(*inputs)
                pair["dividend_adjusted_residual"] = pricing.dividend_adjusted_parity_residual(*inputs, call["dividend_yield"])
                pair["status"] = "PAIRED" if all(not r["reasons"] for r in group) else "PAIR_HAS_QUOTE_FAILURE"
            except (ValueError, TypeError, KeyError) as error:
                pair.update(status="INVALID_PARITY_INPUTS", failure_reason=str(error))
        if pair["status"] != "PAIRED":
            for record in group:
                record["reasons"].append(pair["status"])
        parity.append(pair)
    for record in ledger:
        record["reasons"] = sorted(set(record["reasons"]))
        record["eligible_for_fit"] = not record["reasons"]
    return ledger, parity


def calibrate_quotes(frame, *, mode, config=CalibrationConfig()):
    ledger, parity = prepare_quotes(frame, mode=mode, config=config)
    grouped = {}
    for record in ledger:
        raw = record["input"]
        grouped.setdefault((raw.get("observation_date"), raw.get("expiry_date")), []).append(record)
    slices = []
    for (date, expiry), records in sorted(grouped.items(), key=lambda item: str(item[0])):
        valid = [row for row in records if row["eligible_for_fit"]]
        counts = Counter(row["input"]["strike"] for row in valid)
        weights = [1/counts[row["input"]["strike"]] for row in valid]
        fit = fit_slice([row["k"] for row in valid], [row["total_variance"] for row in valid], weights=weights, config=config)
        maturity = records[0]["input"].get("time_to_expiry")
        item = {"slice_id": f"{date}|{expiry}", "observation_date": date, "expiry_date": expiry,
                "time_to_expiry": maturity, "input_quote_count": len(records), "fit_quote_count": len(valid),
                "quote_preparation_failures": len(records)-len(valid), "weight_policy": "Each strike has total weight 1, split equally over its call/put observations",
                "fit": fit, "repricing": []}
        if fit["converged"]:
            parameters = svi.SVIParameters(**fit["fitted_parameters"])
            item["repricing"] = [reprice_quote(row["input"], parameters, price_floor=config.near_zero_price_floor) for row in valid]
        slices.append(item)
    return {"quote_ledger": ledger, "parity": parity, "slices": slices,
            "arbitrage": verify_surface(slices, config), "repricing_summary": repricing_summary(slices)}


def make_metrics(result, *, mode, dataset, seed, config, input_rows, eligibility=None, fixture=None, coverage=None):
    slices = result.get("slices", [])
    valid = [item for item in slices if item["fit"]["converged"]]
    summary = result.get("repricing_summary", {})
    metrics = {
        "params": {"dataset": dataset, "mode": mode, "seed": seed, "config": asdict(config),
                   "data_origin": SYNTHETIC_ORIGIN if mode == "development" else "REAL_ARCHIVED_OBSERVATION",
                   "empirical_comparison_eligible": mode == "empirical" and bool(valid),
                   "split_protocol": "Exact valuation-date/expiry slices; no random holdout; reported repricing is in-sample",
                   "fixed_grid": config.grid().tolist(), "input_rows": input_rows,
                   "fixture": fixture, "pipeline_coverage": coverage},
        "metrics": {"attempted_slices": len(slices), "successful_slices": len(valid), "failed_slices": len(slices)-len(valid),
                    "iv_failures": sum(r["iv"] is not None and not r["iv"]["converged"] for r in result.get("quote_ledger", [])),
                    "quote_preparation_rejections": sum(not r["eligible_for_fit"] for r in result.get("quote_ledger", [])),
                    "cross_date_median_of_date_maxima": summary.get("cross_date_median_of_date_maxima"),
                    "cross_date_p95_of_date_maxima": summary.get("cross_date_p95_of_date_maxima"),
                    "butterfly_examination_violations": sum(row["examination"]["count"] for row in result.get("arbitrage", {}).get("slices", []) if row["examination"]),
                    "butterfly_complete_violations": sum(row["complete"]["count"] for row in result.get("arbitrage", {}).get("slices", []) if row["complete"]),
                    "calendar_violations": sum(row["count"] for row in result.get("arbitrage", {}).get("calendar", [])),
                    "negative_forward_variance_locations": len(result.get("arbitrage", {}).get("negative_forward_variance", []))},
        "fit_seconds": sum(item["fit"]["fit_seconds"] for item in slices),
        "history": {"status": "BLOCKED" if mode == "empirical" and eligibility and not eligibility["eligible_rows"] else "COMPLETED_WITH_FAILURES" if len(valid) != len(slices) else "COMPLETED",
                    "slices": slices, "repricing_summary": summary, "eligibility": eligibility,
                    "all_failed_slices_retained": True, "figures": [],
                    "historical_requirements_satisfied": False if mode == "development" else bool(coverage and coverage.get("examination_historical_requirement_satisfied"))},
    }
    if mode == "development" and fixture:
        errors, variance_errors = [], []
        for item in valid:
            truth = fixture["truth_by_expiry"][item["expiry_date"]]
            errors.extend(abs(item["fit"]["fitted_parameters"][name]-truth[name]) for name in truth)
            recovered = svi.SVIParameters(**item["fit"]["fitted_parameters"])
            expected = svi.SVIParameters(**truth)
            variance_errors.append(float(np.max(np.abs(svi.svi_total_variance(config.grid(), recovered)-svi.svi_total_variance(config.grid(), expected)))))
        metrics["metrics"].update(max_known_parameter_absolute_error=max(errors) if errors else None,
                                  max_known_variance_absolute_error=max(variance_errors) if variance_errors else None)
    return plain(metrics)
