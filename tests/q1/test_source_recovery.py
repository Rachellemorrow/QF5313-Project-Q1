"""Offline M1.5 evidence/permission-boundary checks; never query vendor services."""

import csv
import hashlib
import json
from pathlib import Path
import runpy
import socket

import pytest


ROOT = Path(__file__).resolve().parents[2]
RECOVERY = ROOT / "data/clean/q1/m15_recovery"


def load(name):
    return json.loads((RECOVERY / name).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def forbid_live_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Source-recovery regression tests must stay offline")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def test_recovery_audit_record_hashes():
    manifest = load("audit_manifest.json")
    assert "NOT a market-data" in manifest["kind"]
    for entry in manifest["files"]:
        path = (ROOT / entry["path"]).resolve()
        assert path.is_relative_to(RECOVERY)
        payload = path.read_bytes()
        assert len(payload) == entry["bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]


def test_catalog_is_not_promoted_to_recovered_history():
    evidence = load("source_evidence.json")
    assert evidence["overall_verdict"] == "BLOCKED"
    assert not evidence["original_q1_requirements_satisfied"]
    assert evidence["new_qualifying_etf_option_rows"] == 0
    assert evidence["new_price_metadata_joins_verified"] == 0
    assert not evidence["new_market_data_snapshot_created"]
    assert not evidence["m2_started"]
    for provider in evidence["sources"]:
        assert provider["documentary_evidence"] == "VERIFIED"
        assert provider["data_evidence"] == "UNVERIFIED"
        assert provider["access_verdict"].startswith("STOPPED_")
        assert not provider["data_api_attempted"]
        assert not provider["login_attempted"]
        assert provider["verified_market_data_date_range"] is None
        assert provider["provider_total_usable_dates"] is None
        assert provider["verified_expiry_count"] is None
        assert provider["repository_or_grader_redistribution_permission"] == "UNVERIFIED"


def test_empty_new_coverage_does_not_invent_dates_or_expiries():
    with (RECOVERY / "new_verified_coverage.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert {"observation_date", "expiry_date", "contract_id", "multiplier"} <= set(reader.fieldnames)
        assert list(reader) == []
    with (RECOVERY / "source_comparison.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(row["verified_usable_dates_in_new_sample"] == "0" for row in rows)
    assert all(row["provider_total_usable_dates"] == "UNVERIFIED" for row in rows)


def test_bigquant_gaps_and_effective_join_are_explicit():
    bq = load("source_evidence.json")["sources"][0]
    assert bq["provider"] == "BigQuant"
    assert bq["documented_price_metadata_join_key"] == ["date", "instrument"]
    assert ["2024-02-08", "2024-09-24"] in bq["documented_static_missing_intervals"]
    assert "2024-12-26" in bq["documented_missing_bar_dates_for_sse"]
    assert not bq["actual_adjustment_records_verified"]


def test_pdf_retrieval_is_not_a_market_data_sample():
    records = load("public_reference_checks.json")
    pdfs = [record for record in records if record.get("content_type") == "application/pdf"]
    assert len(pdfs) == 3
    assert all(record["http_status"] == 200 for record in pdfs)
    assert all(len(record["sha256"]) == 64 for record in pdfs)
    assert all(record["local_path"].startswith(".tmp/") for record in pdfs)
    assert any(record.get("error_type") == "ConnectionError" and "resolve" in record["error"] for record in records)


class FakeResponse:
    def __init__(self, status, payload=b"%PDF-public-reference"):
        self.status_code = status
        self.is_redirect = 300 <= status < 400
        self.headers = {"Content-Type": "application/pdf"}
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        assert self.status_code == 200

    def iter_content(self, size):
        yield self.payload


def install_fake_session(settings, monkeypatch, status, payload=b"%PDF-public-reference"):
    calls = []

    class Session:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, **kwargs):
            assert not self.trust_env
            assert kwargs["allow_redirects"] is False
            calls.append(url)
            return FakeResponse(status, payload)
    monkeypatch.setattr(settings["requests"], "Session", Session)
    return calls


@pytest.mark.parametrize("status", [302, 401, 402, 403, 429])
def test_access_gates_stop_without_redirect_or_retry(tmp_path, monkeypatch, status):
    settings = runpy.run_path(str(ROOT / "code/q1/inspect_public_archives.py"))
    calls = install_fake_session(settings, monkeypatch, status)
    destination = tmp_path / "never_downloaded.pdf"
    record, payload = settings["download"](settings["PUBLIC_REFERENCES"]["sse_history_product.pdf"], destination)
    assert record["verdict"] == "STOPPED_REDIRECT_OR_ACCESS_GATE"
    assert payload is None and not destination.exists()
    assert len(calls) == 1


def test_public_reference_has_checksum_and_is_bounded(tmp_path, monkeypatch):
    settings = runpy.run_path(str(ROOT / "code/q1/inspect_public_archives.py"))
    install_fake_session(settings, monkeypatch, 200)
    url = settings["PUBLIC_REFERENCES"]["sse_history_product.pdf"]
    record, payload = settings["download"](url, tmp_path / "reference.pdf")
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    record, payload = settings["download"](url, tmp_path / "too_big.pdf", max_bytes=2)
    assert record["verdict"] == "FAILED" and payload is None
    assert not (tmp_path / "too_big.pdf").exists()


def test_reader_cannot_query_gated_vendors_or_write_outside_repo(tmp_path):
    settings = runpy.run_path(str(ROOT / "code/q1/inspect_public_archives.py"))
    with pytest.raises(ValueError, match="allowlist"):
        settings["download"]("https://bigquant.com/data/datasources/cn_option_bar1d", tmp_path / "forbidden")
    with pytest.raises(ValueError, match="repository-local"):
        settings["download"](settings["PUBLIC_REFERENCES"]["sse_history_product.pdf"], ROOT.parent / "forbidden.pdf")


def test_reader_preserves_existing_file(tmp_path, monkeypatch):
    settings = runpy.run_path(str(ROOT / "code/q1/inspect_public_archives.py"))
    install_fake_session(settings, monkeypatch, 200)
    destination = tmp_path / "existing.pdf"
    destination.write_bytes(b"existing work")
    record, payload = settings["download"](settings["PUBLIC_REFERENCES"]["sse_history_product.pdf"], destination)
    assert record["error_type"] == "FileExistsError"
    assert destination.read_bytes() == b"existing work"
