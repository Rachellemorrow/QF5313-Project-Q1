"""Offline pipeline verification: labelled synthetic stand-ins and real archive smoke."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import socket

import numpy as np
import pandas as pd
import pytest

from q1 import market_data as md
from q1.provenance import ROOT, local_path, sha256, verify_manifest
from q1.rates import RatePolicy, TENORS


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("M2 processing/verification must not acquire or rediscover sources")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.fixture
def schema_case():
    path = Path(__file__).parent / "fixtures/synthetic/m2_pipeline_cases.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    assert fixture["provenance"].startswith("SYNTHETIC_")
    provenance = {"source_snapshot": path.relative_to(ROOT).as_posix(), "source_sha256": sha256(path.read_bytes()),
                  "provenance_verified": True, "source_probe": "SYNTHETIC_SCHEMA_TEST_ONLY"}
    quote = fixture["quote"] | provenance
    metadata = [fixture["metadata"] | provenance]
    spot = pd.DataFrame([{**provenance, "observation_date": quote["observation_date"], "underlying": quote["underlying"],
                          "instrument_family": "ETF", "spot": fixture["spot"], "adjustment": "unadjusted"}])
    rates = pd.DataFrame([{**provenance, "observation_date": quote["observation_date"],
                           **dict(zip(TENORS, fixture["shibor_percent_levels"]))}])
    policy = md.FilterPolicy(target_underlying="SYNTHETIC_ETF",
                             rates=RatePolicy("zero_continuous", 0.01, "SYNTHETIC_TEST_APPROVAL_ONLY", True))
    nodes = {quote["observation_date"]: fixture["synthetic_tenor_days_not_an_exchange_calendar"]}
    return {"quote": quote, "metadata": metadata, "spots": spot, "rates": rates, "policy": policy, "nodes": nodes}


def prepare(case, quotes=None, *, simulate_trusted_real_loader=False, **changes):
    # This explicit stand-in switch exercises the trusted-loader contract in unit
    # tests only. The fixture file remains labelled synthetic and is never emitted
    # to a real-data pipeline output or counted as recovered market observations.
    quotes = deepcopy(quotes if quotes is not None else [case["quote"]])
    if simulate_trusted_real_loader:
        for quote in quotes:
            quote["data_origin"] = "REAL_ARCHIVED_OBSERVATION"
    arguments = {"metadata": case["metadata"], "spots": case["spots"], "rates": case["rates"],
                 "policy": case["policy"], "verified_tenor_days": case["nodes"]} | changes
    return md.normalize_quotes(quotes, **arguments)


def reasons(frame, row=0):
    return set(json.loads(frame.iloc[row]["rejection_reasons"]))


def test_canonical_schema_and_trusted_loader_state_contract(schema_case):
    frame = prepare(schema_case, simulate_trusted_real_loader=True)
    assert md.validate_canonical_schema(frame)
    assert frame.iloc[0]["row_readiness"] == md.VERIFIED_EMPIRICAL
    assert frame.iloc[0]["eligible_for_empirical"]
    assert frame.iloc[0]["strike"] == 3.0 and frame.iloc[0]["contract_unit"] == 10000
    assert frame.iloc[0]["forward"] == pytest.approx(3*math_exp(0.01*91/365))
    malformed = frame.drop(columns=["contract_id"])
    with pytest.raises(ValueError):
        md.validate_canonical_schema(malformed)


def math_exp(value):
    return float(np.exp(value))


def test_synthetic_origin_is_not_empirical_or_reported_as_real(schema_case):
    frame = prepare(schema_case)
    assert frame.iloc[0]["row_readiness"] == md.PARTIAL_HISTORICAL
    assert not frame.iloc[0]["eligible_for_empirical"]
    assert "NOT_VERIFIED_REAL_OBSERVATION" in reasons(frame)
    assert not md.coverage_report(frame, policy=schema_case["policy"])["input_rows_are_real_not_synthetic"]


@pytest.mark.parametrize("field,value,expected", [
    ("price", -1, "INVALID_OR_NONPOSITIVE_PRICE"), ("price", "bad", "INVALID_NUMERIC_PRICE"),
    ("price", np.inf, "INVALID_OR_NONPOSITIVE_PRICE"), ("volume", -1, "NEGATIVE_VOLUME"),
    ("volume", 0, "LOW_VOLUME"), ("volume", None, "MISSING_OR_INVALID_VOLUME"),
    ("bid", 0.06, "INVALID_OR_CROSSED_BID_ASK"), ("bid", "bad", "INVALID_NUMERIC_BID"),
    ("open_interest", -1, "NEGATIVE_OPEN_INTEREST"), ("high", 0.01, "INCONSISTENT_OHLC"),
    ("observation_date", "2000-02-30", "INVALID_OBSERVATION_DATE"),
    ("quote_timestamp", "2000-01-05T15:00:00+08:00", "QUOTE_TIMESTAMP_DATE_CONFLICT"),
    ("quote_timestamp", "bad", "INVALID_QUOTE_TIMESTAMP"),
])
def test_invalid_quotes_are_retained_with_reasons(schema_case, field, value, expected):
    raw = [schema_case["quote"] | {field: value}]
    frame = prepare(schema_case, raw)
    assert len(frame) == 1 and expected in reasons(frame)
    assert not frame.iloc[0]["eligible_for_empirical"]


def test_duplicate_cleaning_is_deterministic_and_never_chooses_conflicting_price(schema_case):
    raw = [schema_case["quote"], schema_case["quote"] | {"source_row": "1"}]
    first = prepare(schema_case, raw)
    second = prepare(schema_case, list(reversed(raw)))
    pd.testing.assert_frame_equal(first, second)
    assert sum("DUPLICATE_EXACT" in reasons(first, i) for i in range(2)) == 1
    assert first["quote_id"].is_unique
    conflict = prepare(schema_case, [raw[0], raw[1] | {"price": 0.051}])
    assert all("DUPLICATE_CONFLICT" in reasons(conflict, i) for i in range(2))
    identical_source = prepare(schema_case, [raw[0], raw[0]])
    assert identical_source["quote_id"].is_unique


def test_future_and_overlapping_metadata_do_not_leak(schema_case):
    future = deepcopy(schema_case["metadata"])
    future[0]["valid_from"] = "2000-02-01"
    frame = prepare(schema_case, metadata=future)
    assert "NO_METADATA_VERSION_COVERS_DATE" in reasons(frame)
    assert pd.isna(frame.iloc[0]["strike"])
    overlap = prepare(schema_case, metadata=schema_case["metadata"]*2)
    assert "AMBIGUOUS_EFFECTIVE_METADATA" in reasons(overlap)
    assert not overlap.iloc[0]["metadata_effective_valid"]


def test_dated_metadata_wins_over_a_later_current_snapshot(schema_case):
    current = schema_case["metadata"][0] | {"evidence_level": "CURRENT_SNAPSHOT_UNDATED", "valid_from": None,
                                           "valid_to": None, "strike": 9, "contract_unit": 777}
    frame = prepare(schema_case, metadata=[current, *schema_case["metadata"]])
    assert frame.iloc[0]["strike"] == 3 and frame.iloc[0]["contract_unit"] == 10000
    undated = prepare(schema_case, metadata=[current])
    assert "UNDATED_CURRENT_METADATA" in reasons(undated)
    assert not undated.iloc[0]["metadata_effective_valid"]


@pytest.mark.parametrize("changes,reason", [
    ({"expiry_date": None}, "INCOMPLETE_CONTRACT_METADATA"),
    ({"option_type": "X"}, "INCOMPLETE_CONTRACT_METADATA"),
    ({"strike": 0}, "INCOMPLETE_CONTRACT_METADATA"),
    ({"contract_unit": None}, "INCOMPLETE_CONTRACT_METADATA"),
    ({"listing_date": "2000-02-01"}, "QUOTE_BEFORE_LISTING"),
    ({"expiry_date": "2000-01-03"}, "QUOTE_AFTER_EXPIRY"),
    ({"source_sha256": "bad"}, "METADATA_EVIDENCE_UNVERIFIED"),
])
def test_metadata_validation_has_explicit_missing_or_conflicting_reasons(schema_case, changes, reason):
    metadata = [schema_case["metadata"][0] | changes]
    frame = prepare(schema_case, metadata=metadata)
    assert reason in reasons(frame) and not frame.iloc[0]["eligible_for_empirical"]


def test_source_strike_and_underlying_conflicts_are_not_overwritten(schema_case):
    frame = prepare(schema_case, [schema_case["quote"] | {"quoted_strike": 3.2, "underlying": "OTHER_ETF"}])
    assert {"METADATA_STRIKE_CONFLICT", "METADATA_UNDERLYING_CONFLICT"} <= reasons(frame)
    assert not frame.iloc[0]["metadata_effective_valid"]
    assert frame.iloc[0]["quoted_strike"] == 3.2


def test_no_nearest_day_or_adjusted_underlying_join(schema_case):
    spots = schema_case["spots"].copy()
    spots.loc[0, "observation_date"] = "2000-01-05"
    assert "MISSING_SPOT_DATE" in reasons(prepare(schema_case, spots=spots))
    spots = schema_case["spots"].copy()
    spots.loc[0, "adjustment"] = "qfq"
    assert "INVALID_OR_ADJUSTED_SPOT" in reasons(prepare(schema_case, spots=spots))
    duplicate = pd.concat([schema_case["spots"], schema_case["spots"]], ignore_index=True)
    assert "DUPLICATE_SPOT_DATE" in reasons(prepare(schema_case, spots=duplicate))


def test_liquidity_missing_fields_and_optional_moneyness_rules(schema_case):
    raw = [schema_case["quote"] | {"bid": None, "ask": None, "open_interest": None}]
    frame = prepare(schema_case, raw)
    assert {"BID_ASK_MISSING", "OPEN_INTEREST_MISSING"} <= set(json.loads(frame.iloc[0]["warnings"]))
    assert pd.isna(frame.iloc[0]["bid"]) and pd.isna(frame.iloc[0]["open_interest"])
    policy = md.FilterPolicy(target_underlying="SYNTHETIC_ETF", require_bid_ask=True, min_open_interest=1,
                             max_abs_log_moneyness=0.0001, rates=schema_case["policy"].rates)
    strict = prepare(schema_case, raw, policy=policy)
    assert {"BID_ASK_MISSING", "OPEN_INTEREST_MISSING", "OUTSIDE_MONEYNESS_BAND"} <= reasons(strict)


def test_no_implicit_dividend_rate_or_verified_unit_assumptions(schema_case):
    policy = md.FilterPolicy(target_underlying="SYNTHETIC_ETF")
    frame = prepare(schema_case, [schema_case["quote"] | {"price_unit_verified": False}], policy=policy)
    assert {"PRICE_UNIT_UNVERIFIED", "DIVIDEND_OR_RATE_CONVENTIONS_UNAPPROVED"} <= reasons(frame)
    assert frame[["zero_rate", "discount_factor", "forward", "dividend_yield"]].isna().all().all()


def test_manifest_corruption_and_path_escape_fail_before_reading_data(tmp_path):
    artifact = tmp_path/"source.csv"
    artifact.write_bytes(b"SYNTHETIC_TEST_ONLY\n")
    manifest = {"files": [{"path": "source.csv", "bytes": artifact.stat().st_size, "sha256": sha256(artifact.read_bytes())}]}
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    checksum = sha256(path.read_bytes())
    verify_manifest(tmp_path, "manifest.json", checksum)
    artifact.write_bytes(b"corrupted synthetic copy")
    with pytest.raises(ValueError, match="checksum"):
        verify_manifest(tmp_path, "manifest.json", checksum)
    with pytest.raises(ValueError, match="SHA-256"):
        verify_manifest(tmp_path, "manifest.json", "0"*64)
    with pytest.raises(ValueError):
        local_path(tmp_path, "../outside")


@pytest.fixture(scope="module")
def real_pipeline_output(tmp_path_factory):
    output = tmp_path_factory.mktemp("q1_m2_real")/"dataset"
    result = md.build_pipeline(output)
    return output, result


def test_available_real_archive_smoke_and_partial_coverage(real_pipeline_output):
    output, result = real_pipeline_output
    loaded = md.load_offline_pipeline(output, result["manifest_sha256"])
    frame, coverage = loaded["quotes"], loaded["coverage"]
    assert len(frame) == 606
    assert coverage["source_daily_close_rows"] == 604 and coverage["source_timestamped_current_quote_rows"] == 2
    assert coverage["target_etf_observations_not_automatically_eligible"] == {
        "quotes": 489, "dates": 130, "expiries": 4, "contracts": 10, "strikes": 3,
        "calls": 261, "puts": 228, "date_min": "2026-04-23", "date_max": "2026-11-04"}
    assert coverage["strictly_usable_for_empirical_analysis"]["quotes"] == 0
    assert coverage["dataset_readiness"] == md.PARTIAL_HISTORICAL
    assert not coverage["examination_historical_requirement_satisfied"]
    assert not frame["metadata_effective_valid"].any()
    assert frame[["discount_factor", "forward", "zero_rate"]].isna().all().all()
    daily = frame[frame["price_kind"] == "daily_close"]
    assert daily[["bid", "ask", "open_interest"]].isna().all().all()
    assert len(frame[frame["instrument_family"] == "INDEX"]) == 94


def test_output_is_byte_reproducible_and_refuses_overwrite(real_pipeline_output, tmp_path):
    output, first = real_pipeline_output
    repeated = md.build_pipeline(output)
    assert repeated["manifest_sha256"] == first["manifest_sha256"]
    rebuilt = md.build_pipeline(tmp_path/"rebuild")
    assert rebuilt["manifest_sha256"] == first["manifest_sha256"]
    copied = tmp_path/"tampered_output"
    shutil.copytree(output, copied)
    with (copied/"quotes.csv").open("ab") as handle:
        handle.write(b"tampered synthetic test copy")
    with pytest.raises(ValueError, match="checksum"):
        md.load_offline_pipeline(copied, first["manifest_sha256"])
    with pytest.raises(FileExistsError):
        md.build_pipeline(copied)


def test_numeric_offline_reload_is_exact(real_pipeline_output):
    output, result = real_pipeline_output
    bundle = md.load_source_records()
    expected = md.normalize_quotes(bundle["raw_quotes"], bundle["metadata"], bundle["spots"], bundle["rates"])
    actual = md.load_offline_pipeline(output, result["manifest_sha256"])["quotes"]
    pd.testing.assert_frame_equal(actual, expected)


def test_registered_q1_dataset_reloads_without_network():
    result = md.load_registered_q1()
    assert result["coverage"]["dataset_readiness"] == md.PARTIAL_HISTORICAL
    assert not result["quotes"]["eligible_for_empirical"].any()
