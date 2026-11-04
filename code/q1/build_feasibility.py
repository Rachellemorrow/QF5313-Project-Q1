"""Build M1 evidence tables offline from immutable, checksum-verified API snapshots."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd
from rich.console import Console


ROOT = Path(__file__).resolve().parents[2]
RUNS = ["data/clean/q1/m1_audit", "data/clean/q1/m1_alignment", "data/clean/q1/m1_pairs"]
TENORS = ["O/N", "1W", "2W", "1M", "3M", "6M", "9M", "1Y"]
TARGETS = ["2024-10-08", "2024-10-31", "2025-10-09", "2026-10-08", "2026-10-30"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite_positive(value):
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def iso_date(value):
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    text = str(value)
    return pd.to_datetime(text, format="%Y%m%d" if re.fullmatch(r"\d{8}", text) else None).date().isoformat()


def load_evidence():
    summaries = [read_json(ROOT / run / "api_audit.json") for run in RUNS]
    for summary in summaries:
        for artifact in summary["artifacts"]:
            path = (ROOT / artifact["path"]).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError("External artifact path")
            payload = path.read_bytes()
            if len(payload) != artifact["bytes"] or hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
                raise ValueError(f"Evidence checksum mismatch: {artifact['path']}")
    probes = {p["probe_id"]: p for summary in summaries for p in summary["probes"]}
    return summaries, probes


def table(probes, probe_id, strings=False):
    record = probes[probe_id]
    if record["status"] != "PASS":
        raise ValueError(f"Cannot derive a table from failed probe {probe_id}")
    return pd.read_csv(ROOT / record["snapshot"]["path"], dtype=str if strings else None)


def build(output):
    output = output.resolve()
    summaries, probes = load_evidence()
    if not output.resolve().is_relative_to(ROOT):
        raise ValueError("Output must be repository-local")
    output.mkdir(parents=True, exist_ok=False)
    meta = table(probes, "sse_current_contracts", strings=True)
    meta_map = meta.set_index("合约编码").to_dict(orient="index")
    cffex_chain = table(probes, "cffex_current_spot", strings=True)
    cffex_by_identifier = {}
    for _, contract in cffex_chain.iterrows():
        for label, option_type in [("看涨", "C"), ("看跌", "P")]:
            cffex_by_identifier[contract[f"{label}合约-标识"]] = {
                "strike": contract["行权价"], "option_type": option_type,
            }
    rows = []
    history_probes = [name for name, probe in probes.items()
                      if probe["api"] in {"option_sse_daily_sina", "option_cffex_hs300_daily_sina"}]
    for probe_id in history_probes:
        probe = probes[probe_id]
        if probe["status"] != "PASS":
            continue
        identifier = probe["parameters"]["symbol"]
        sse = probe["api"] == "option_sse_daily_sina"
        contract = meta_map.get(identifier, {}) if sse else cffex_by_identifier.get(identifier, {})
        for source_row, observation in table(probes, probe_id, strings=True).iterrows():
            close = observation["收盘" if sse else "close"]
            row = {
                "observation_date": iso_date(observation["日期" if sse else "date"]),
                "instrument_family": "SSE_ETF_OPTION_ENDPOINT" if sse else "CFFEX_INDEX_OPTION",
                "underlying": None, "contract_id": identifier,
                "expiry_date": None, "strike": None, "option_type": None, "contract_unit": None,
                "price": close, "price_kind": "daily_close", "positive_finite_market_price": finite_positive(close),
                "volume": observation["成交量" if sse else "volume"],
                "bid": None, "ask": None, "open_interest": None,
                "metadata_source": None, "historical_metadata_validity": "UNVERIFIED",
                "model_ready": False, "source_probe": probe_id, "source_row_zero_based": source_row,
                "current_metadata_listing_date": None, "on_or_after_current_listing_date": None,
                "source_snapshot": probe["snapshot"]["path"],
                "underlying_inferred": None, "expiry_month_inferred": None,
                "strike_inferred": None, "option_type_inferred": None,
            }
            if sse and contract:
                match = re.search(r"\((\d{6})\)", contract["标的券名称及代码"])
                row.update({
                    "underlying": match.group(1) if match else None,
                    "expiry_date": iso_date(contract["到期日"]), "strike": contract["行权价"],
                    "option_type": {"认购": "C", "认沽": "P"}.get(contract["类型"]),
                    "contract_unit": contract["合约单位"], "metadata_source": "sse_current_contracts",
                    "current_metadata_listing_date": iso_date(contract["开始日期"]),
                    "on_or_after_current_listing_date": row["observation_date"] >= iso_date(contract["开始日期"]),
                })
            elif not sse:
                row["underlying"] = "000300"  # Named underlying of this specific API, not an ETF.
                if contract:
                    row.update(contract)
                    row["metadata_source"] = "cffex_current_spot"
                match = re.fullmatch(r"io(\d{4})([CP])(\d+)", identifier, re.IGNORECASE)
                if match:
                    row["expiry_month_inferred"] = "20" + match.group(1)
                    row["option_type_inferred"] = match.group(2).upper()
                    row["strike_inferred"] = match.group(3)
            rows.append(row)
    matrix = pd.DataFrame(rows)

    # Quote timestamp is a source field, not inferred from retrieval time.
    inventory = meta.copy()
    inventory.insert(0, "observation_date", None)
    inventory["price_probe_status"] = "NOT_REQUESTED"
    inventory["positive_finite_market_price"] = pd.Series([None] * len(inventory), dtype=object)
    inventory["snapshot_retrieved_at_utc"] = probes["sse_current_contracts"]["started_at_utc"]
    for probe_id in ["sse_current_spot_0", "sse_current_spot_1"]:
        quote = table(probes, probe_id, strings=True).set_index("字段")["值"].to_dict()
        identifier = probes[probe_id]["parameters"]["symbol"]
        selected = inventory["合约编码"] == identifier
        inventory.loc[selected, "observation_date"] = iso_date(quote.get("行情时间"))
        inventory.loc[selected, "price_probe_status"] = "OBSERVED_CURRENT_QUOTE"
        inventory.loc[selected, "positive_finite_market_price"] = finite_positive(quote.get("最新价"))

    spot = table(probes, "etf_unadjusted_history", strings=True).set_index("日期")
    rates = table(probes, "shibor", strings=True).set_index("日期")
    index_spot = table(probes, "index_spot", strings=True).set_index("日期")
    alignment = []
    for row in rows:
        date = row["observation_date"]
        spot_table = spot if row["underlying"] == "510050" else index_spot if row["underlying"] == "000300" else None
        spot_value = spot_table.loc[date, "收盘"] if spot_table is not None and date in spot_table.index else None
        rate_values = {tenor: rates.loc[date, tenor + "-定价"] if date in rates.index else None for tenor in TENORS}
        alignment.append({
            "observation_date": date, "contract_id": row["contract_id"],
            "instrument_family": row["instrument_family"], "underlying": row["underlying"],
            "spot_close": spot_value, "spot_unadjusted_positive": finite_positive(spot_value),
            "shibor_same_date_all_eight_tenors": all(finite_positive(value) for value in rate_values.values()),
            "rates_quote_unit": "percent per annum (API docstring; not converted to zero rates)",
            **{tenor + "_percent": value for tenor, value in rate_values.items()},
        })

    target_rows = []
    for date in TARGETS:
        observed = matrix[(matrix["observation_date"] == date) & (matrix["underlying"] == "510050")]
        target_rows.append({
            "target_observation_date": date, "underlying": "510050", "instrument_family": "ETF",
            "target_is_an_observation": False, "observed_price_rows": len(observed),
            "observed_expiries_with_current_metadata": observed["expiry_date"].nunique(),
            "historically_verified_complete_chain": False,
            "spot_in_snapshot": date in spot.index, "shibor_in_snapshot": date in rates.index,
            "status": "PARTIAL" if len(observed) else "UNVERIFIED",
        })

    month_spot_dates = sorted(date for date in spot.index if date.startswith("2026-10"))
    monthly = []
    for date in month_spot_dates:
        daily = matrix[(matrix["observation_date"] == date) & (matrix["underlying"] == "510050")]
        monthly.append({"observation_date": date, "underlying": "510050",
                        "calendar_basis": "dates actually returned by unadjusted ETF history; NOT an official exchange calendar",
                        "observed_contracts": daily["contract_id"].nunique(),
                        "observed_expiries_with_current_metadata": daily["expiry_date"].nunique(),
                        "observed_calls": int((daily["option_type"] == "C").sum()),
                        "observed_puts": int((daily["option_type"] == "P").sum()),
                        "complete_chain_verified": False})

    artifacts = []
    for name, frame in [
        ("historical_coverage_matrix.csv", matrix),
        ("current_contract_inventory.csv", inventory),
        ("historical_alignment.csv", pd.DataFrame(alignment)),
        ("target_date_coverage.csv", pd.DataFrame(target_rows)),
        ("october_2026_coverage.csv", pd.DataFrame(monthly)),
    ]:
        payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        path = output / name
        with path.open("xb") as handle:
            handle.write(payload)
        artifacts.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(payload).hexdigest(),
                          "bytes": len(payload), "rows": len(frame)})

    old_unadjusted = spot.loc["2024-10-08", "收盘"]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "akshare_version": summaries[0]["akshare_version"],
        "source_audits": [run + "/api_audit.json" for run in RUNS],
        "probes": [probe for summary in summaries for probe in summary["probes"]],
        "artifacts": artifacts,
        "network_capture_correction": {
            "initial_run_capture_complete": False,
            "captured_initial_http_requests": summaries[0]["observed"]["requests"],
            "omission": "index_zh_a_hist's index_code_id_map_em uses fetch_paginated_data -> request_with_retry(fresh_session=True), bypassing the initial http.request hook",
            "additional_page_calls_inferred_from_progress_and_source": 18,
            "actual_total_requests_and_bytes_initial_run": "UNVERIFIED: omitted helper responses were not captured; do not interpret the original limits/observed counts as whole-process totals",
            "correction": "Runner now instruments request_with_retry without retries; original acquisition evidence preserved unchanged; index map NOT downloaded again",
            "supplemental_captured_requests": summaries[1]["observed"]["requests"],
            "four_expiry_pairs_captured_requests": summaries[2]["observed"]["requests"],
        },
        "facts": {
            "historical_price_rows": len(matrix),
            "known_etf_price_rows": int((matrix["underlying"] == "510050").sum()),
            "unknown_metadata_sse_price_rows": int(((matrix["instrument_family"] == "SSE_ETF_OPTION_ENDPOINT") & matrix["underlying"].isna()).sum()),
            "index_price_rows": int((matrix["instrument_family"] == "CFFEX_INDEX_OPTION").sum()),
            "current_sse_metadata_rows": len(meta),
            "current_sse_underlyings": meta["标的券名称及代码"].value_counts().to_dict(),
            "current_sse_contract_units_observed": sorted(meta["合约单位"].unique()),
            "current_510050_expiries": sorted(meta.loc[meta["标的券名称及代码"] == "50ETF(510050)", "到期日"].unique()),
            "etf_old_date_adjustments": {
                "date": "2024-10-08", "unadjusted": old_unadjusted,
                "qfq": table(probes, "etf_qfq_old_date", strings=True).iloc[0]["收盘"],
                "hfq": table(probes, "etf_hfq_old_date", strings=True).iloc[0]["收盘"],
            },
            "shibor_tenors": TENORS,
            "shibor_observed_date_range": probes["shibor"]["date_columns"],
            "shibor_docstring_start_disagrees_with_observed": "Installed docstring says 20170317; response begins 2015-05-08",
            "october_spot_dates": len(month_spot_dates),
            "etf_same_day_spot_and_all_rates_rows": sum(row["underlying"] == "510050" and row["spot_unadjusted_positive"] and row["shibor_same_date_all_eight_tenors"] for row in alignment),
            "expired_spot_semantically_usable": False,
            "current_metadata_listing_date_mismatch_rows": int((matrix["on_or_after_current_listing_date"] == False).sum()),
        },
        "limitations": [
            "Observed positive close is only a candidate market price, not a verified liquid/executable quote",
            "Current metadata joined retrospectively is NOT proof of historical contract-unit/strike validity",
            "Never combine index option rows with ETF rows to claim ETF coverage",
            "Unknown expiry dates, units, types or underlying remain null; code-inferred values are separate columns",
            "All model_ready flags are false: no financial validation or modeling is implemented in M1",
            "Daily history interfaces omit bid/ask and OI; current quote fields cannot backfill historical liquidity",
        ],
        "requirement_evidence": [
            {"requirement": "Installed signatures and all six named interfaces exercised", "status": "PASS", "evidence": "m1_audit/api_audit.json: six named APIs returned actual data"},
            {"requirement": "One month of daily complete ETF option chains", "status": "FAIL", "evidence": "october_2026_coverage.csv: sparse selected contracts only, not complete daily chains"},
            {"requirement": "At least four priced expiries", "status": "PARTIAL", "evidence": "Four current expiries have matched sampled calls/puts, but only one paired strike and five common recent dates"},
            {"requirement": "At least five complete sample chains spanning two years", "status": "FAIL", "evidence": "target_date_coverage.csv: no verified 2024/2025 ETF option chains"},
            {"requirement": "Expired contract discovery", "status": "FAIL", "evidence": "Both 202410 expiry-month discovery responses are empty; installed parser raises ValueError"},
            {"requirement": "Known expired contract prices", "status": "PARTIAL", "evidence": "23 SSE and 54 CFFEX history rows; historical metadata not independently verified"},
            {"requirement": "Historical price and contract metadata pairing", "status": "PARTIAL", "evidence": "487 price rows join current SSE metadata; historical adjustment validity is UNVERIFIED"},
            {"requirement": "Historical unadjusted ETF and index spot availability", "status": "PASS", "evidence": "505 ETF rows in 2024-2026 and four index rows in 2024; not all option dates covered for the index example"},
            {"requirement": "Adjustment modes distinguished", "status": "PASS", "evidence": "2024-10-08 ETF closes: unadjusted 2.978, qfq 2.843, hfq 3.64"},
            {"requirement": "SHIBOR eight-tenor historical availability", "status": "PASS", "evidence": "2398 source dates, 2015-05-08 to 2026-11-04; no missing level quotes"},
            {"requirement": "Same-date option, underlying and rate alignment", "status": "PARTIAL", "evidence": "487 selected 50ETF rows have spot and all eight tenor levels on the same date; older chains remain unverified"},
            {"requirement": "Rate units and discount-curve conventions", "status": "PARTIAL", "evidence": "Installed API labels levels as percent; official day-count, compounding and curve construction require verification/approval"},
            {"requirement": "Corporate actions and historical contract-unit adjustments", "status": "PARTIAL", "evidence": "Current 500ETF A-series units 10180 observed; dated adjustment notices/history absent"},
            {"requirement": "Historical bid/ask and open interest", "status": "FAIL", "evidence": "Both daily history APIs omit these fields; two current quote snapshots cannot backfill history"},
            {"requirement": "Volume field and units", "status": "PARTIAL", "evidence": "Volume returned by daily APIs; unit definition and unusually large expired-example values not independently corroborated"},
            {"requirement": "Dividend/forward assumptions approved by student", "status": "UNVERIFIED", "evidence": "No dividend yield, pricing convention or curve method chosen in M1"},
            {"requirement": "Small reproducible offline evidence snapshots", "status": "PASS", "evidence": "SHA-256 source and derived manifests; Q1-only byte-preserving attributes; total archive below 50 MB"},
            {"requirement": "Complete initial HTTP-request capture", "status": "PARTIAL", "evidence": "Initial index-code pagination bypass disclosed; later runs instrumented correctly; no repeated map download"},
        ],
    }
    with (output / "api_audit_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    paths = sorted({path for folder in [*(ROOT / run for run in RUNS), output]
                    for path in folder.rglob("*") if path.is_file()})
    manifest = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    with (output / "snapshot_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump({"files": manifest, "total_bytes": sum(item["bytes"] for item in manifest)}, handle, indent=2)
    Console().print_json(data=summary["facts"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/clean/q1/m1_evidence"))
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
