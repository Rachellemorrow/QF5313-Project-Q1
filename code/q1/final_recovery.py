"""Bounded offline diagnosis of existing Q1 evidence; no downloads or model fits."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.market_data import load_registered_q1
from q1.provenance import ROOT, sha256


def diagnose(frame):
    """Primary reasons are an explicit reporting precedence, not new filter rules."""
    remaining = list(frame.to_dict(orient="records"))
    waterfall = []
    precedence = [
        ("NON_TARGET_INSTRUMENT_FAMILY", lambda r: r["instrument_family"] != "ETF"),
        ("MISSING_OR_NON_TARGET_UNDERLYING", lambda r: str(r["underlying"]) != "510050"),
        ("INVALID_OR_NONPOSITIVE_PRICE", lambda r: "INVALID_OR_NONPOSITIVE_PRICE" in json.loads(r["rejection_reasons"])),
        ("HISTORICAL_METADATA_NOT_VERIFIED", lambda r: not r["metadata_effective_valid"]),
        ("PRICE_UNIT_UNVERIFIED", lambda r: not r["price_unit_verified"]),
        ("CONVENTIONS_UNAPPROVED", lambda r: not r["conventions_approved"]),
        ("OTHER_PIPELINE_REJECTION", lambda r: not r["eligible_for_empirical"]),
    ]
    assignments = []
    for reason, predicate in precedence:
        rejected = [r for r in remaining if predicate(r)]
        ids = {r["quote_id"] for r in rejected}
        remaining = [r for r in remaining if r["quote_id"] not in ids]
        assignments.extend({"quote_id": r["quote_id"], "primary_reason": reason} for r in rejected)
        waterfall.append({"primary_reason": reason, "rejected": len(rejected), "remaining": len(remaining)})
    reasons = Counter(reason for r in frame.to_dict(orient="records") for reason in json.loads(r["rejection_reasons"]))
    patterns = Counter(frame["rejection_reasons"])
    target = frame[(frame["instrument_family"] == "ETF") & (frame["underlying"] == "510050")]
    slices = []
    for (date, expiry), group in target.groupby(["observation_date", "expiry_date"], sort=True):
        daily = group[group["price_kind"] == "daily_close"]
        paired = sum(set(g["option_type"]) == {"C", "P"} for _, g in daily.groupby("strike"))
        slices.append({"date": date, "candidate_expiry": expiry, "daily_quotes": len(daily),
                       "candidate_strikes": int(daily["strike"].nunique()), "paired_strikes": paired})
    return {"input_rows": len(frame), "eligible_before": int(frame["eligible_for_empirical"].sum()),
            "eligible_after": int(frame["eligible_for_empirical"].sum()),
            "recovered_dates": [], "recovered_expiries": [], "waterfall": waterfall,
            "primary_assignments": assignments, "eligible_after_waterfall": len(remaining),
            "overlapping_failures": dict(sorted(reasons.items())),
            "mutually_exclusive_failure_patterns": [{"reasons": json.loads(p), "rows": n} for p, n in sorted(patterns.items())],
            "target_candidate_rows": len(target), "target_candidate_dates": int(target["observation_date"].nunique()),
            "candidate_slices_NOT_verified": slices,
            "max_candidate_daily_strikes_per_slice": max((s["candidate_strikes"] for s in slices), default=0),
            "max_candidate_paired_strikes_per_slice": max((s["paired_strikes"] for s in slices), default=0),
            "network_checks": 0, "model_fits": 0, "eligibility_rules_changed": False,
            "stop_reason": "No verified historical metadata/units; even candidate ETF slices have fewer than five distinct strikes required by existing SVI. Existing index histories lack matched spot and complete metadata. No bounded metadata-only check can repair strike support."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New repository-local JSON evidence path")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        parser.error("Output must be new and repository-local")
    bundle = load_registered_q1()
    report = diagnose(bundle["quotes"])
    report["source_manifest_sha256"] = sha256((ROOT/"data/clean/q1/m2_pipeline/manifest.json").read_bytes())
    report["scope"] = "Existing hash-verified archive only; primary precedence is presentation, overlapping failures remain unchanged"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in ["input_rows", "eligible_before", "eligible_after", "max_candidate_daily_strikes_per_slice", "max_candidate_paired_strikes_per_slice", "network_checks", "model_fits"]}))


if __name__ == "__main__":
    main()
