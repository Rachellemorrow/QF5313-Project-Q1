"""M7 quote-level masks and failure/coverage-aware predictive metric aggregation."""

from collections import defaultdict
import hashlib
import json
import math

import numpy as np

from .approach_a import plain


METHODS = ("scipy", "adam", "lbfgs", "neural")
SEEDS = (42,43,44)


def digest(value):
    return hashlib.sha256(json.dumps(plain(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def slice_key(row):
    return row["observation_date"]+"|"+row["expiry_date"]


def quote_drop_manifest(rows, seed, *, configuration_sha256):
    """Uniform WITHOUT replacement within each date/expiry, at quote-ID level.

    Drop nearest integer to 0.30*N (half up). Exactly 30% is impossible for many
    finite slices; realized count/fraction is explicit. Not paired deletion.
    """
    ids = [row["quote_id"] for row in rows]
    if len(set(ids)) != len(ids) or any(not isinstance(value,str) for value in ids):
        raise ValueError("Require unique string quote IDs")
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0,2**32)")
    groups = defaultdict(list)
    for row in rows:
        groups[slice_key(row)].append(row)
    rng = np.random.default_rng(seed)
    slices = []
    for key,group in sorted(groups.items()):
        universe = sorted(row["quote_id"] for row in group)
        count = int(math.floor(.3*len(universe)+.5))
        dropped = set(rng.choice(universe,size=count,replace=False).tolist())
        retained = [row for row in group if row["quote_id"] not in dropped]
        strikes = sorted({float(row["strike"]) for row in retained})
        slices.append({"slice_id":key,"observation_date":group[0]["observation_date"],"expiry_date":group[0]["expiry_date"],
                       "quote_ids":universe,"dropped_mask":[value in dropped for value in universe],
                       "retained_ids":[value for value in universe if value not in dropped],"dropped_ids":sorted(dropped),
                       "universe_count":len(universe),"retained_count":len(retained),"dropped_count":count,
                       "realized_drop_fraction":count/len(universe),"retained_distinct_strikes":len(strikes),
                       "retained_strike_bounds":[strikes[0],strikes[-1]] if strikes else None,
                       "status":"SUFFICIENT_SUPPORT" if len(strikes) >= 5 else "INSUFFICIENT_SUPPORT"})
    manifest = {"seed":seed,"deletion_unit":"INDIVIDUAL_QUOTE","requested_fraction":.3,"integer_rule":"floor(0.30*N+0.5) per date/expiry",
                "configuration_sha256":configuration_sha256,"universe_ids_sha256":digest(sorted(ids)),"slices":slices}
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


def validate_manifest(rows, manifest, configuration_sha256):
    expected = quote_drop_manifest(rows,manifest["seed"],configuration_sha256=configuration_sha256)
    if manifest != expected:
        raise ValueError("Split manifest does not match deterministic quote-level protocol/configuration")


def retained_neural_split(rows, seed, *, fraction=.2):
    """Only retained quotes enter here. Keep available same-strike legs together.

    Incomplete pairs are legitimate after QUOTE-level deletion; never synthesize
    a missing leg or reject its retained counterpart. Validation uses 20% of the
    retained strike groups per expiry, at least one; remaining groups train.
    """
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be in (0,1)")
    groups = defaultdict(lambda:defaultdict(list))
    for index,row in enumerate(rows):
        groups[row["expiry_date"]][(row["strike"],row["contract_unit"])].append(index)
    split = {"train":[],"validation":[]}
    rng = np.random.default_rng(seed)
    for expiry,strikes in sorted(groups.items()):
        keys = sorted(strikes)
        count = max(1,int(len(keys)*fraction))
        if len(keys)-count < 5:
            raise ValueError("INSUFFICIENT_SUPPORT: neural train/validation split needs >=5 training strike groups per expiry")
        order = rng.permutation(len(keys))
        for position,index in enumerate(order):
            split["validation" if position < count else "train"].extend(strikes[keys[index]])
    return {name:sorted(indices) for name,indices in split.items()}


def support_class(row, retained):
    strikes = sorted({float(value["strike"]) for value in retained if slice_key(value) == slice_key(row)})
    if len(strikes) < 5:
        return "INSUFFICIENT_SUPPORT"
    return "INTERPOLATION" if strikes[0] <= row["strike"] <= strikes[-1] else "EXTRAPOLATION"


def error_metrics(predictions):
    """Available prediction errors are NEVER reported without their denominator.

    Primary complete-cohort fields are null if any target lacks a prediction.
    Near-zero market premiums (<1e-8) retain absolute errors, not clipped relative
    denominators. Null empty categories do not turn into zero predictive errors.
    """
    valid = [row for row in predictions if row["status"] == "PREDICTED"]
    relative = [row["relative_price_error"] for row in valid if row["relative_price_error"] is not None]
    iv = np.array([row["predicted_iv"]-row["market_iv"] for row in valid])
    price = np.array([row["predicted_price"]-row["market_price"] for row in valid])
    n, expected = len(valid),len(predictions)
    available = {"iv_mae":float(np.mean(np.abs(iv))) if n else None,"iv_rmse":float(np.sqrt(np.mean(iv**2))) if n else None,
                 "price_mae":float(np.mean(np.abs(price))) if n else None,"price_rmse":float(np.sqrt(np.mean(price**2))) if n else None,
                 "relative_price_mae":float(np.mean(relative)) if relative else None,
                 "relative_price_median":float(np.median(relative)) if relative else None,
                 "relative_price_p95":float(np.percentile(relative,95)) if relative else None,
                 "relative_price_max":max(relative) if relative else None}
    complete = {key:value if n == expected and expected > 0 and (not key.startswith("relative") or len(relative) == n) else None for key,value in available.items()}
    return {"held_out_expected":expected,"predicted":n,"failed_or_uncovered":expected-n,"prediction_coverage":n/expected if expected else None,
            "relative_errors_defined":len(relative),"near_zero_relative_exclusions":n-len(relative),
            "available_prediction_errors":available,"complete_cohort_errors":complete}


def summarize_predictions(predictions, fits, diagnostics, seeds):
    rows = []
    for seed in seeds:
        for method in METHODS:
            group = [row for row in predictions if row["seed"] == seed and row["method"] == method]
            levels = [("seed",None,None,group)]
            dates = sorted({row["observation_date"] for row in group})
            for date in dates:
                subset = [row for row in group if row["observation_date"] == date]
                levels.append(("date",date,None,subset))
                for expiry in sorted({row["expiry_date"] for row in subset}):
                    levels.append(("expiry",date,expiry,[row for row in subset if row["expiry_date"] == expiry]))
            for level,date,expiry,subset in levels:
                scoped_fits = [row for row in fits if row["seed"] == seed and row["method"] == method
                               and (date is None or row.get("observation_date") == date) and (expiry is None or row.get("expiry_date") == expiry)]
                scoped_ids = {row.get("slice_id") for row in scoped_fits}
                reports = [row for row in diagnostics if row["seed"] == seed and row["method"] == method
                           and (date is None or row["observation_date"] == date)]
                slice_reports = [item for row in reports for item in row.get("report",{}).get("slices",[]) if item["slice_id"] in scoped_ids and item["status"] == "VERIFIED"]
                calendar = [item for row in reports for item in row.get("report",{}).get("calendar",[]) if item["later_slice"] in scoped_ids]
                forward = [item for row in reports for item in row.get("report",{}).get("negative_forward_variance",[]) if item["later_slice"] in scoped_ids]
                fit_summary = {"attempted":len(scoped_fits),"successful":sum(row["usable"] for row in scoped_fits),
                               "failed":sum(not row["usable"] for row in scoped_fits),"fit_seconds":sum(row["fit_seconds_attributed"] for row in scoped_fits),
                               "diagnosed_expiries":len(slice_reports),"missing_diagnostic_expiries":len(scoped_fits)-len(slice_reports),
                               "examination_butterfly":sum(row["examination"]["count"] for row in slice_reports) if slice_reports else None,
                               "complete_butterfly":sum(row["complete"]["count"] for row in slice_reports) if slice_reports else None,
                               "calendar":sum(row["count"] for row in calendar) if calendar else None,
                               "negative_forward_variance":len(forward) if calendar else None,
                               "calendar_attribution":"Each pair/forward interval is attributed to its later expiry"}
                for category in ["ALL","INTERPOLATION","EXTRAPOLATION","INSUFFICIENT_SUPPORT"]:
                    selected = subset if category == "ALL" else [row for row in subset if row["support_class"] == category]
                    rows.append({"seed":seed,"method":method,"level":level,"observation_date":date,"expiry_date":expiry,
                                 "support_class":category,"fit_summary_all_support_categories":fit_summary,**error_metrics(selected)})
    per_seed = []
    common_metrics = []
    for seed in seeds:
        groups = {method:[row for row in predictions if row["seed"] == seed and row["method"] == method] for method in METHODS}
        successful = [{row["quote_id"] for row in group if row["status"] == "PREDICTED"} for group in groups.values()]
        intersection = set.intersection(*successful)
        for method in METHODS:
            selected_fits = [row for row in fits if row["seed"] == seed and row["method"] == method]
            reports = [row for row in diagnostics if row["seed"] == seed and row["method"] == method]
            per_seed.append({"seed":seed,"method":method,**error_metrics(groups[method]),
                             "attempted_expiry_fits":len(selected_fits),"successful_expiry_fits":sum(row["usable"] for row in selected_fits),
                             "failed_expiry_fits":sum(not row["usable"] for row in selected_fits),
                             "fit_success_rate":sum(row["usable"] for row in selected_fits)/len(selected_fits) if selected_fits else None,
                             "fit_seconds":sum(row["fit_seconds_attributed"] for row in selected_fits),
                             "diagnosed_expiries":sum(row["diagnosed_expiries"] for row in reports),
                             "missing_diagnostic_expiries":sum(len(row["missing_expiries"]) for row in reports),
                             "arbitrage_available_counts":{key:sum(row["counts"][key] or 0 for row in reports) if any(row["counts"][key] is not None for row in reports) else None for key in
                                 ["examination_butterfly","complete_butterfly","calendar","negative_forward_variance"]}})
            common_metrics.append({"seed":seed,"method":method,"cohort":"SECONDARY_INTERSECTION_OF_SUCCESSFUL_PREDICTIONS_ALL_FOUR_METHODS",
                                   "common_quote_ids":sorted(intersection),**error_metrics([row for row in groups[method] if row["quote_id"] in intersection])})
    aggregates = {}
    for method in METHODS:
        cohort = [row for row in per_seed if row["method"] == method]
        numeric = {key:[row[key] for row in cohort] for key in ["prediction_coverage","fit_success_rate","fit_seconds","failed_expiry_fits","missing_diagnostic_expiries","diagnosed_expiries"]}
        numeric.update({"arbitrage_"+key:[row["arbitrage_available_counts"][key] for row in cohort] for key in
                        ["examination_butterfly","complete_butterfly","calendar","negative_forward_variance"]})
        numeric.update({key:[row["available_prediction_errors"][key] for row in cohort] for key in
                        ["iv_mae","iv_rmse","price_mae","price_rmse","relative_price_mae"]})
        aggregate = {}
        for key,values in numeric.items():
            finite = [value for value in values if value is not None]
            aggregate[key] = {"mean":float(np.mean(finite)) if finite else None,"sample_std":float(np.std(finite,ddof=1)) if len(finite) > 1 else None,
                              "available_seeds":len(finite),"expected_seeds":len(seeds)}
        aggregate["metric_errors_are_conditional_on_available_predictions"] = True
        aggregate["pooled_with_coverage"] = error_metrics([row for row in predictions if row["method"] == method])
        aggregates[method] = aggregate
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"],row["level"],row["observation_date"],row["expiry_date"],row["support_class"])].append(row)
    group_aggregates = []
    for (method,level,date,expiry,support),group in grouped.items():
        values = {key:[row[key] for row in group] for key in ["held_out_expected","predicted","prediction_coverage","failed_or_uncovered"]}
        values.update({key:[row["available_prediction_errors"][key] for row in group] for key in
                       ["iv_mae","iv_rmse","price_mae","price_rmse","relative_price_mae","relative_price_p95","relative_price_max"]})
        values.update({"fit_"+key:[row["fit_summary_all_support_categories"][key] for row in group] for key in
                       ["attempted","successful","failed","fit_seconds","diagnosed_expiries","missing_diagnostic_expiries",
                        "examination_butterfly","complete_butterfly","calendar","negative_forward_variance"]})
        statistics = {}
        for key,items in values.items():
            finite = [value for value in items if value is not None]
            statistics[key] = {"mean":float(np.mean(finite)) if finite else None,"sample_std":float(np.std(finite,ddof=1)) if len(finite) > 1 else None,
                               "available_seeds":len(finite),"expected_seeds":len(seeds)}
        group_aggregates.append({"method":method,"level":level,"observation_date":date,"expiry_date":expiry,"support_class":support,
                                 "statistics_conditional_with_coverage":statistics})
    return {"group_metrics":rows,"per_seed":per_seed,"seed_aggregates":aggregates,"group_seed_aggregates":group_aggregates,
            "common_success_intersection":common_metrics,
            "pooled_interpolation_extrapolation":{method:{category:error_metrics([row for row in predictions if row["method"] == method and row["support_class"] == category])
                for category in ["INTERPOLATION","EXTRAPOLATION","INSUFFICIENT_SUPPORT"]} for method in METHODS}}
