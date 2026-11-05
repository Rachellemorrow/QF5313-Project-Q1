"""Final recovery diagnosis only: no fitting, source fetching or eligibility edits."""

import json

from q1.final_recovery import diagnose
from q1.market_data import load_registered_q1
from q1.provenance import ROOT


def test_exact_archive_waterfall_and_support():
    frame = load_registered_q1()["quotes"]
    original = frame.copy(deep=True)
    report = diagnose(frame)
    assert report["input_rows"] == 606
    assert [s["rejected"] for s in report["waterfall"]] == [94, 23, 0, 489, 0, 0, 0]
    assert report["eligible_before"] == report["eligible_after"] == report["eligible_after_waterfall"] == 0
    ids = [r["quote_id"] for r in report["primary_assignments"]]
    assert len(ids) == len(set(ids)) == 606
    assert set(ids) == set(frame["quote_id"])
    assert sum(p["rows"] for p in report["mutually_exclusive_failure_patterns"]) == 606
    assert report["overlapping_failures"]["UNDATED_CURRENT_METADATA"] == 529
    assert report["overlapping_failures"]["MISSING_EFFECTIVE_METADATA"] == 77
    assert report["overlapping_failures"]["INVALID_OR_NONPOSITIVE_PRICE"] == 1
    assert report["target_candidate_rows"] == 489
    assert report["target_candidate_dates"] == 130
    assert report["max_candidate_daily_strikes_per_slice"] == 3
    assert report["max_candidate_paired_strikes_per_slice"] == 1
    assert frame.equals(original)


def test_saved_evidence_matches_read_only_diagnosis():
    saved = json.loads((ROOT/"docs/q1_final_recovery_evidence.json").read_text(encoding="utf-8"))
    report = diagnose(load_registered_q1()["quotes"])
    assert all(saved[key] == value for key, value in report.items())
    assert report["recovered_dates"] == report["recovered_expiries"] == []
    assert report["network_checks"] == report["model_fits"] == 0
