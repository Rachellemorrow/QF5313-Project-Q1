"""Read-only Q1 adapters. No fitting, model selection, repairs or result promotion."""

import numpy as np


def errors(predictions, *, iv_targets=None):
    valid = [p for p in predictions if p.get("status") in {"PREDICTED","REPRICED"}]
    relative = [p.get("relative_price_error",p.get("relative_error")) for p in valid]
    relative = [v for v in relative if v is not None]
    absolute, iv = [],[]
    for p in valid:
        predicted = p.get("predicted_price",p.get("model_price"))
        if predicted is not None:
            absolute.append(predicted-p["market_price"])
        target = p.get("market_iv",(iv_targets or {}).get(p["quote_id"]))
        fitted = p.get("predicted_iv",p.get("fitted_iv"))
        if fitted is not None and target is not None:
            iv.append(fitted-target)
    return {"predicted":len(valid),"relative_defined":len(relative),
            "relative_median":float(np.median(relative)) if relative else None,
            "relative_p95":float(np.percentile(relative,95)) if relative else None,
            "relative_max":max(relative) if relative else None,
            "price_mae":float(np.mean(np.abs(absolute))) if absolute else None,
            "price_rmse":float(np.sqrt(np.mean(np.square(absolute)))) if absolute else None,
            "iv_mae":float(np.mean(np.abs(iv))) if iv else None,
            "iv_rmse":float(np.sqrt(np.mean(np.square(iv)))) if iv else None}


def targets(ledger):
    return {p["quote_id"]:p["iv"]["volatility"] for p in ledger if p.get("iv") and p["iv"]["converged"]}


def diagnostic_counts(report, slice_id):
    item = next((s for s in report.get("slices",[]) if s.get("slice_id") == slice_id),None)
    if not item or item.get("status") != "VERIFIED":
        return {"butterfly_exam":None,"butterfly_complete":None,"calendar":None,"negative_forward":None,"diagnostics_available":False}
    pairs = [p for p in report.get("calendar",[]) if p.get("later_slice") == slice_id]
    return {"butterfly_exam":item["examination"]["count"],"butterfly_complete":item["complete"]["count"],
            "calendar":sum(p["count"] for p in pairs) if pairs else None,
            "negative_forward":sum(p.get("later_slice") == slice_id for p in report.get("negative_forward_variance",[])) if pairs else None,
            "diagnostics_available":True}


def q1_rows(payload, *, auxiliary=None):
    params,history = payload["params"],payload["history"]
    if history.get("status") == "BLOCKED":
        gate = history.get("eligibility") or {}
        return [{"model":params.get("component","svi"),"cohort":"EMPIRICAL_GATE_ONLY","observation_date":None,"expiry_date":None,
                 "observed_quotes":gate.get("input_rows",0),"eligible_quotes":gate.get("eligible_rows",0),"expected":0,"predicted":0,
                 "coverage":None,"converged":None,"failed_fits":0,"fit_seconds":0.,"status":"BLOCKED"}]
    if history.get("comparison"):
        comparison = history["comparison"]
        result = []
        for name,algorithm in comparison["algorithms"].items():
            result.extend(svi_rows(algorithm,name,params,comparison.get("quote_ledger",[])))
        return result
    if history.get("result") and "dates" in history["result"]:
        result = history["result"]
        iv = targets(result["quote_ledger"])
        rows = []
        for date in result["dates"]:
            for case,run in date["ablations"].items():
                training, evaluation, diag = run["training"],run.get("evaluation"),run.get("diagnostics")
                predictions = evaluation["predictions"] if evaluation else []
                expiries = sorted({p["input"]["expiry_date"] for p in result["quote_ledger"] if p["input"]["observation_date"] == date["observation_date"]})
                for expiry in expiries:
                    item = next((p for p in (diag or {}).get("per_expiry",[]) if p["expiry_date"] == expiry),{})
                    maturity = item.get("maturity")
                    for fold in ["train","validation","test"]:
                        selected = [p for p in predictions if p["expiry_date"] == expiry and p["fold"] == fold]
                        expected = sum(p["expiry_date"] == expiry and p["fold"] == fold for p in predictions)
                        rows.append({"model":"neural_"+case,"observation_date":date["observation_date"],"expiry_date":expiry,"maturity":maturity,
                            "seed":params.get("seed"),"cohort":"M6_"+fold,"observed_quotes":sum(p["input"]["expiry_date"] == expiry for p in result["quote_ledger"]),
                            "eligible_quotes":sum(p["input"]["expiry_date"] == expiry and p["eligible_for_fit"] for p in result["quote_ledger"]),
                            "expected":expected,"coverage":1. if selected and all(p["status"] == "REPRICED" for p in selected) else None,
                            "converged":None,"usable":training["success"],"failed_fits":int(not training["success"]),"status":training["status"],
                            "termination":training.get("stop_reason"),"fit_seconds":training["fit_seconds"]/max(1,len(expiries)),
                            "shared_fit_group":date["observation_date"]+"|neural_"+case,"timing_repeated_across_folds":True,
                            "butterfly_exam":item.get("examination",{}).get("count"),"butterfly_complete":item.get("complete",{}).get("count"),
                            "calendar":sum(p["count"] for p in (diag or {}).get("calendar",[]) if p["later_maturity"] == maturity) if maturity else None,
                            "negative_forward":sum(p["later_maturity"] == maturity for p in (diag or {}).get("negative_forward_variance",[])) if maturity else None,
                            "diagnostic_domain":"M6_DENSE_15_MATURITY_GRID_LATER_NODE_ATTRIBUTION",**errors(selected,iv_targets=iv)})
        return rows
    if "predictions" in history and "fits" in history:
        rows = []
        for fit in history["fits"]:
            selected = [p for p in history["predictions"] if p["method"] == fit["method"] and p["seed"] == fit["seed"] and p["observation_date"] == fit["observation_date"] and p["expiry_date"] == fit["expiry_date"]]
            report = next((d for d in history["diagnostics"] if d["method"] == fit["method"] and d["seed"] == fit["seed"] and d["observation_date"] == fit["observation_date"]),{})
            rows.append({"model":fit["method"],"observation_date":fit["observation_date"],"expiry_date":fit["expiry_date"],"maturity":fit["time_to_expiry"],
                         "seed":fit["seed"],"cohort":"M7_DROPPED_30_PERCENT","observed_quotes":fit["retained_count"]+fit["held_out_count"],
                         "eligible_quotes":fit["retained_count"]+fit["held_out_count"],"training_quotes":fit["training_count"],"validation_quotes":fit["validation_count"],
                         "expected":fit["held_out_count"],"coverage":fit["prediction_coverage"],"converged":fit["converged"],"usable":fit["usable"],
                         "failed_fits":int(not fit["usable"]),"failure_reason":fit["failure_reason"],"status":"USABLE" if fit["usable"] else "FAILED",
                         "termination":fit["fit"].get("stop_reason",fit["fit"].get("message")),"fit_seconds":fit["fit_seconds_attributed"],
                         "objective":fit["fit"].get("objective_value"),"held_out_ids_sha256":fit["held_out_ids_sha256"],
                         "diagnostic_domain":"COMMON_ORIGINAL_EXPIRIES_201_K",**diagnostic_counts(report.get("report",{}),fit["slice_id"]),**errors(selected)})
        for report in history["diagnostics"]:
            dense = report.get("dense_neural_supplement")
            if dense:
                rows.append({"model":"neural","seed":report["seed"],"observation_date":report["observation_date"],"expiry_date":None,
                    "cohort":"M7_DENSE_ARBITRAGE_SUPPLEMENT","diagnostic_domain":"DENSE_15_MATURITIES_201_K_NOT_COMMON_EXPIRY_GRID",
                    "butterfly_exam":dense["violations"]["examination"]["count"],"butterfly_complete":dense["violations"]["complete"]["count"],
                    "calendar_derivative":dense["violations"]["w_T"]["count"],"calendar":dense["calendar_violation_count"],
                    "negative_forward":dense["negative_forward_variance_count"],"status":"VIOLATIONS" if dense["calendar_violation_count"] else "CHECKED"})
        return rows
    return svi_rows({**history,"arbitrage":(auxiliary or {}).get("arbitrage",history.get("arbitrage",{}))},"scipy",params,(auxiliary or {}).get("quote_ledger",[]))


def svi_rows(algorithm, model, params, ledger):
    rows = []
    iv = targets(ledger)
    report = algorithm.get("arbitrage",{})
    for item in algorithm.get("slices",[]):
        fit = item["fit"]
        selected = item.get("repricing",[])
        expected = item.get("fit_quote_count",fit.get("quote_count",0))
        usable = fit["converged"]
        rows.append({"model":model,"observation_date":item["observation_date"],"expiry_date":item["expiry_date"],"maturity":item["time_to_expiry"],
                     "seed":params.get("seed"),"cohort":"IN_SAMPLE_CALIBRATION","observed_quotes":item.get("input_quote_count",expected),"eligible_quotes":expected,
                     "training_quotes":expected,"validation_quotes":0,"expected":expected,"coverage":len(selected)/expected if expected else None,
                     "converged":usable,"usable":usable,"failed_fits":int(not usable),"failure_reason":fit["failure_reason"],
                     "status":"CONVERGED" if usable else "FAILED","termination":fit.get("message"),"objective":fit.get("objective_value"),
                     "iterations":fit.get("iterations"),"objective_evaluations":fit.get("objective_evaluations"),"fit_seconds":fit["fit_seconds"],
                     "diagnostic_domain":"ORIGINAL_EXPIRIES_201_K",**diagnostic_counts(report,item["slice_id"]),**errors(selected,iv_targets=iv)})
    return rows
