"""M8 comparison, archive, session fidelity and targeted scientific audit tests."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from q1.validation_bundle import DEFAULT, restore
from q1.final_audit import dense_check


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("group_compare",ROOT/"code/compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
export_spec = importlib.util.spec_from_file_location("session_export",ROOT/"scripts/q1_session_export.py")
session_export = importlib.util.module_from_spec(export_spec)
export_spec.loader.exec_module(session_export)


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    output = tmp_path_factory.mktemp("m8_replay")/"restored"
    manifest = restore(DEFAULT,output)
    report = compare.load_comparison(output/"results")
    return output,manifest,report


def test_bundle_lossless_replay_and_all_required_model_rows(replay):
    output,manifest,report = replay
    assert len(manifest["files"]) == 24
    assert len(report["runs"]) == 8 and not report["issues"]
    assert {r["mode"] for r in report["runs"]} == {"EMPIRICAL","SYNTHETIC_VERIFICATION"}
    assert any(r.get("relative_median") is not None and r.get("relative_p95") is not None and r.get("relative_max") is not None for r in report["rows"])
    failed = [r for r in report["rows"] if r.get("cohort") == "M7_DROPPED_30_PERCENT" and r["model"] == "lbfgs" and not r["usable"]]
    assert len(failed) == 1 and failed[0]["predicted"] == 0 and failed[0]["expected"] == 25
    assert "cohort_rank" not in failed[0]
    assert len(report["scoped_rankings"]) == 44  # 11 complete cohorts, four models
    assert all(row.get("coverage") == 1 for row in report["rows"] if row.get("cohort_rank"))
    dense = [r for r in report["rows"] if r.get("cohort") == "M7_DENSE_ARBITRAGE_SUPPLEMENT" and r["seed"] == 44]
    assert len(dense) == 1 and dense[0]["calendar"] == 448 and dense[0]["negative_forward"] == 272
    assert any(r.get("butterfly_exam") == 0 and r.get("diagnostics_available") for r in report["rows"] if r["model"] == "scipy")
    with pytest.raises(FileExistsError):
        restore(DEFAULT,output)


def test_schema_and_provenance_quarantine_generic_other_question(tmp_path):
    with pytest.raises(ValueError):
        compare.validate_metrics({"params":{}})
    valid = {"params":{"dataset":"q2_demo","mode":"empirical","model":"MC"},"metrics":{"price":1.2},"fit_seconds":.1,"history":{}}
    directory = tmp_path/"approach1/q2_demo"; directory.mkdir(parents=True)
    (directory/"metrics.json").write_text(json.dumps(valid))
    report = compare.load_comparison(tmp_path)
    assert not report["issues"] and report["rows"][0]["question"] == "q2"
    assert report["rows"][0]["raw_metrics"] == {"price":1.2}
    valid["params"]["data_origin"] = "SYNTHETIC_VERIFICATION"
    (directory/"metrics.json").write_text(json.dumps(valid))
    report = compare.load_comparison(tmp_path)
    assert report["issues"] and not report["rows"]


def test_dense_calendar_violations_are_genuine_by_checkpoint_and_finite_difference(replay):
    result = dense_check(replay[0]/"results/verification/approach2/q1_synthetic_missing_quotes")
    assert result["genuine_model_behavior_not_a_numerical_sign_error"]
    assert result["counts_recomputed"] == {"negative_w_T":402,"calendar_pairs":448,"negative_forward":272}
    assert result["independent_pair"]["later_minus_earlier_w"] < 0
    assert all(row["absolute_difference_from_autograd"] < 1e-4 for row in result["finite_differences"])


def test_markdown_export_roundtrip_preserves_every_message_and_tool_state():
    payload = {"info":{"id":"ses_test"},"messages":[
        {"id":"m1","type":"user","text":"Prompt with ``` and multiline\ncontent"},
        {"id":"m2","type":"assistant","content":[{"type":"text","text":"Answer"},
            {"type":"tool","name":"shell","state":{"input":{"command":"test"},"content":[{"type":"text","text":"actual output"}]}}]}]}
    text = session_export.render(payload)
    assert session_export.recover(text) == payload["messages"]


def test_comparison_html_is_separated_and_self_contained(replay,tmp_path):
    from html.parser import HTMLParser
    output = tmp_path/"comparison.html"
    compare.write_comparison(replay[2],output)
    text = output.read_text(encoding="utf-8")
    class Scripts(HTMLParser):
        def __init__(self):
            super().__init__(); self.external = []
        def handle_starttag(self,tag,attrs):
            if tag == "script":
                self.external.extend(value for key,value in attrs if key == "src")
    parser = Scripts(); parser.feed(text)
    assert not parser.external and "plotly.js" in text
    assert "SYNTHETIC_VERIFICATION" in text and "EMPIRICAL" in text and "448" in text
    with pytest.raises(FileExistsError):
        compare.write_comparison(replay[2],output)
