"""Offline M1 integrity and evidence-boundary tests. Live requests are forbidden."""

import hashlib
import json
from pathlib import Path
import runpy
import socket
import subprocess

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "data/clean/q1/m1_evidence"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("M1 offline verification must not open a network connection")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def test_all_snapshots_have_matching_sha256_and_are_small():
    manifest = read_json(EVIDENCE / "snapshot_manifest.json")
    total = 0
    for entry in manifest["files"]:
        path = (ROOT / entry["path"]).resolve()
        assert path.is_relative_to(ROOT / "data/clean/q1")
        payload = path.read_bytes()
        assert len(payload) == entry["bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert len(payload) < 50_000_000
        total += len(payload)
    assert total == manifest["total_bytes"] < 50_000_000


def test_git_does_not_change_snapshot_line_endings():
    path = "data/clean/q1/m1_audit/snapshots/sse_current_contracts.csv"
    result = subprocess.check_output(["git", "check-attr", "text", "--", path], cwd=ROOT, text=True)
    assert result.strip().endswith(": text: unset")


def test_required_api_signatures_and_actual_calls_are_recorded():
    summary = read_json(EVIDENCE / "api_audit_summary.json")
    required = {"option_current_day_sse", "option_cffex_hs300_list_sina",
                "option_cffex_hs300_daily_sina", "fund_etf_hist_em",
                "index_zh_a_hist", "macro_china_shibor_all"}
    assert required <= {p["api"] for p in summary["probes"]}
    for record in summary["probes"]:
        assert record["signature"]
        assert record["akshare_version"] == "1.20.27"
        assert record["started_at_utc"]
        if "rows" in record:
            assert record["columns"] is not None
            assert record["dtypes"] is not None


def test_known_failures_are_not_successful_coverage():
    summary = read_json(EVIDENCE / "api_audit_summary.json")
    probes = {p["probe_id"]: p for p in summary["probes"]}
    for probe_id in ["sse_expired_discovery_call", "sse_expired_discovery_put"]:
        assert probes[probe_id]["status"] == "FAIL"
        assert probes[probe_id]["error_type"] == "ValueError"
        assert "Length mismatch" in probes[probe_id]["error"]
    assert summary["facts"]["expired_spot_semantically_usable"] is False
    assert summary["network_capture_correction"]["initial_run_capture_complete"] is False


def test_price_rows_are_traceable_without_fabricated_metadata():
    matrix = pd.read_csv(EVIDENCE / "historical_coverage_matrix.csv", dtype=str)
    assert len(matrix) == 604
    unknown = matrix[matrix["source_probe"] == "sse_expired_daily"]
    assert len(unknown) == 23
    assert unknown[["underlying", "expiry_date", "strike", "option_type", "contract_unit"]].isna().all().all()
    assert (matrix["model_ready"] == "False").all()
    for probe_id, group in matrix.groupby("source_probe"):
        source = pd.read_csv(ROOT / group.iloc[0]["source_snapshot"], dtype=str)
        close_column = source.columns[4]
        for _, record in group.iterrows():
            assert record["price"] == source.iloc[int(record["source_row_zero_based"])][close_column]


def test_etf_and_index_rows_remain_separate():
    matrix = pd.read_csv(EVIDENCE / "historical_coverage_matrix.csv", dtype=str)
    index = matrix[matrix["instrument_family"] == "CFFEX_INDEX_OPTION"]
    assert len(index) == 94
    assert (index["underlying"] == "000300").all()
    assert index["contract_unit"].isna().all()
    assert index["expiry_date"].isna().all()
    assert (matrix["underlying"] == "510050").sum() == 487


def test_four_expiries_do_not_become_a_full_month_claim():
    summary = read_json(EVIDENCE / "api_audit_summary.json")
    requirements = {item["requirement"]: item for item in summary["requirement_evidence"]}
    assert requirements["One month of daily complete ETF option chains"]["status"] == "FAIL"
    assert requirements["At least five complete sample chains spanning two years"]["status"] == "FAIL"
    month = pd.read_csv(EVIDENCE / "october_2026_coverage.csv")
    assert len(month) == 17
    assert (month["observed_expiries_with_current_metadata"] == 4).sum() == 2
    assert not month["complete_chain_verified"].any()


def test_observation_dates_and_same_day_alignment():
    targets = pd.read_csv(EVIDENCE / "target_date_coverage.csv")
    assert len(targets) == 5
    assert (pd.to_datetime(targets.iloc[-1]["target_observation_date"]) -
            pd.to_datetime(targets.iloc[0]["target_observation_date"])).days >= 730
    assert not targets["target_is_an_observation"].any()
    assert targets["spot_in_snapshot"].all()
    assert targets["shibor_in_snapshot"].all()
    aligned = pd.read_csv(EVIDENCE / "historical_alignment.csv", dtype={"underlying": str})
    etf = aligned[aligned["underlying"] == "510050"]
    assert len(etf) == 487
    assert etf["spot_unadjusted_positive"].all()
    assert etf["shibor_same_date_all_eight_tenors"].all()


def test_current_adjusted_units_and_dividend_gap_are_retained():
    summary = read_json(EVIDENCE / "api_audit_summary.json")
    assert summary["facts"]["current_sse_contract_units_observed"] == ["10000", "10180"]
    prices = summary["facts"]["etf_old_date_adjustments"]
    assert len({prices["unadjusted"], prices["qfq"], prices["hfq"]}) == 3
    dividend = next(r for r in summary["requirement_evidence"] if r["requirement"].startswith("Dividend/"))
    assert dividend["status"] == "UNVERIFIED"


def test_corrected_retry_path_is_instrumented_and_restored(tmp_path, monkeypatch):
    settings = runpy.run_path(str(ROOT / "code/q1/audit_interfaces.py"))
    audit = settings["Audit"](tmp_path / "instrumentation")
    original = settings["http"].request_with_retry
    calls = []

    def probe(*args, **kwargs):
        assert settings["http"].request_with_retry.__self__ is audit
        calls.append(args[0])
    monkeypatch.setattr(audit, "probe", probe)
    audit.run(supplemental=True)
    assert len(calls) == 3
    assert settings["http"].request_with_retry is original


def test_rebuild_tables_offline_without_overwriting_evidence(tmp_path):
    settings = runpy.run_path(str(ROOT / "code/q1/build_feasibility.py"))
    destination = tmp_path / "rebuilt"
    settings["build"](destination)
    for entry in read_json(EVIDENCE / "api_audit_summary.json")["artifacts"]:
        filename = Path(entry["path"]).name
        assert (destination / filename).read_bytes() == (EVIDENCE / filename).read_bytes()
