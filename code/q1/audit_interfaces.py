"""M1 bounded live API inspection; never prices a derivative or invents a quote.

Use the repository launcher. A live run creates a new immutable output directory;
network failures are evidence, not successful coverage. No requests run on import.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import time
import traceback

import akshare as ak
from akshare.utils import http
from akshare.utils.context import config
import pandas as pd
from rich.console import Console
from rich.table import Table


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = [
    "option_current_day_sse", "option_cffex_hs300_list_sina",
    "option_cffex_hs300_daily_sina", "fund_etf_hist_em",
    "index_zh_a_hist", "macro_china_shibor_all",
]
EXTRA = [
    "option_sse_list_sina", "option_sse_codes_sina", "option_sse_daily_sina",
    "option_daily_stats_sse", "option_sse_spot_price_sina",
    "option_sse_expire_day_sina", "option_cffex_hs300_spot_sina",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


class AuditLimitError(RuntimeError):
    pass


class Audit:
    def __init__(self, output, max_requests=32, max_bytes=8_000_000):
        self.output = output.resolve()
        if not self.output.is_relative_to(ROOT):
            raise ValueError("Audit output must be inside the repository")
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "snapshots").mkdir()
        (self.output / "http").mkdir()
        self.max_requests = max_requests
        self.max_bytes = max_bytes
        self.total_bytes = 0
        self.network = []
        self.probes = []
        self.artifacts = []
        self.frames = {}
        self.active_probe = None
        self.blocked_hosts = set()
        self.original_request = http.request
        self.original_retry = http.request_with_retry

    def artifact(self, name, payload, kind):
        path = self.output / name
        with path.open("xb") as handle:
            handle.write(payload)
        entry = {"path": path.relative_to(ROOT).as_posix(), "bytes": len(payload),
                 "sha256": digest(payload), "kind": kind}
        self.artifacts.append(entry)
        return entry

    def request(self, method, url, **kwargs):
        from urllib.parse import urlsplit

        host = urlsplit(url).hostname
        if host in self.blocked_hosts:
            raise AuditLimitError(f"No repeat request sent to rate-limited host {host}")
        if len(self.network) >= self.max_requests:
            raise AuditLimitError("Audit HTTP request budget exhausted; no request sent")
        if sum(item["probe_id"] == self.active_probe for item in self.network) >= 6:
            raise AuditLimitError("Six-request per-API probe budget exhausted; no further request sent")
        entry = {"probe_id": self.active_probe, "method": method, "url": url,
                 "params": kwargs.get("params"), "retrieved_at_utc": utc_now()}
        self.network.append(entry)
        # Stream/decompress with an explicit size cap, preserving the API's parser.
        kwargs["stream"] = True
        kwargs["timeout"] = (5, 12)
        started = time.perf_counter()
        response = None
        try:
            response = self.original_request(method, url, **kwargs)
            entry.update({"response_url": response.url,
                          "http_status": response.status_code,
                          "content_type": response.headers.get("Content-Type")})
            if response.status_code == 429:
                self.blocked_hosts.add(host)
            chunks = []
            size = 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > 2_000_000 or self.total_bytes + size > self.max_bytes:
                    raise AuditLimitError("Audit response/body byte budget exceeded")
                chunks.append(chunk)
            payload = b"".join(chunks)
            self.total_bytes += len(payload)
            response._content = payload
            response._content_consumed = True
            entry["body"] = self.artifact(
                f"http/{len(self.network):02d}.body", payload,
                "HTTP response body after transfer decompression (not a packet capture)",
            )
            return response
        except Exception as error:
            if type(error).__name__ == "Push2RateLimitError":
                self.blocked_hosts.add(host)
            entry["error_type"] = type(error).__name__
            entry["error"] = str(error)
            raise
        finally:
            if response is not None:
                response.close()
            entry["seconds"] = time.perf_counter() - started

    def probe(self, probe_id, api_name, parameters=None, role="", context=None):
        parameters = parameters or {}
        function = getattr(ak, api_name, None)
        record = {"probe_id": probe_id, "api": api_name, "parameters": parameters,
                  "role": role, "context": context or {}, "started_at_utc": utc_now(),
                  "akshare_version": ak.__version__, "status": "UNVERIFIED"}
        self.probes.append(record)
        self.active_probe = probe_id
        if function is None:
            record.update({"status": "FAIL", "error_type": "AttributeError",
                           "error": f"Installed akshare has no attribute {api_name}"})
            return None
        record["signature"] = str(inspect.signature(function))
        # Validate real parameters before any request; retain exact installed source.
        inspect.signature(function).bind(**parameters)
        source = inspect.getsource(function)
        record["source_sha256"] = digest(source.encode())
        started = time.perf_counter()
        try:
            value = function(**parameters)
            if isinstance(value, pd.DataFrame):
                self.frames[probe_id] = value
                record["rows"] = len(value)
                record["columns"] = [str(column) for column in value.columns]
                record["dtypes"] = {str(c): str(value[c].dtype) for c in value.columns}
                record["null_counts"] = {str(c): int(value[c].isna().sum()) for c in value.columns}
                record["date_columns"] = {}
                for column in value.columns:
                    if str(column) in {"日期", "date", "交易日", "到期日", "期权行权日", "开始日期", "行权交收日"}:
                        dates = pd.to_datetime(value[column], errors="coerce").dropna()
                        record["date_columns"][str(column)] = {
                            "min": dates.min().date().isoformat() if len(dates) else None,
                            "max": dates.max().date().isoformat() if len(dates) else None,
                            "distinct": int(dates.nunique()),
                        }
                payload = value.to_csv(index=False, lineterminator="\n").encode("utf-8")
                record["snapshot"] = self.artifact(f"snapshots/{probe_id}.csv", payload,
                                                   "Unfiltered AKShare-returned table serialized to UTF-8 CSV")
                record["status"] = "PASS" if len(value) else "PARTIAL"
            else:
                record["value"] = value
                payload = json.dumps(value, ensure_ascii=False, indent=2, default=str).encode()
                record["snapshot"] = self.artifact(f"snapshots/{probe_id}.json", payload,
                                                   "AKShare-returned non-tabular result")
                record["status"] = "PASS" if value else "PARTIAL"
            return value
        except Exception as error:
            record.update({"status": "FAIL", "error_type": type(error).__name__,
                           "error": str(error), "traceback": traceback.format_exc(),
                           "columns": None, "dtypes": None, "date_columns": None})
            return None
        finally:
            record["seconds"] = time.perf_counter() - started
            self.active_probe = None
            # Per-probe records survive later errors without overwriting earlier evidence.
            self.artifact(f"{probe_id}.json", json.dumps(record, ensure_ascii=False,
                          indent=2, default=str).encode(), "Individual API audit record")

    def request_without_retry(self, method, url, *, max_retries=1, base_delay=1,
                              fresh_session=False, **kwargs):
        # Some installed helpers use fresh-session retries, bypassing http.request.
        # requests.request already creates a fresh Session. Deliberately no retry.
        return self.request(method, url, **kwargs)

    def run(self, supplemental=False, four_expiry_pairs=False):
        old_timeout = config.timeout
        config.timeout = (5, 12)
        http.request = self.request
        http.request_with_retry = self.request_without_retry
        try:
            if four_expiry_pairs:
                summary = json.loads((ROOT / "data/clean/q1/m1_audit/api_audit.json").read_text(encoding="utf-8"))
                original = next(item for item in summary["probes"] if item["probe_id"] == "sse_current_contracts")
                payload = (ROOT / original["snapshot"]["path"]).read_bytes()
                if digest(payload) != original["snapshot"]["sha256"]:
                    raise ValueError("Metadata snapshot checksum failed")
                metadata = pd.read_csv(ROOT / original["snapshot"]["path"], dtype=str)
                subset = metadata[(metadata["标的券名称及代码"] == "50ETF(510050)") &
                                  (metadata["合约单位"] == "10000")]
                expiries = sorted(subset["到期日"].unique())
                if len(expiries) != 4:
                    raise ValueError("Expected four actually observed current 50ETF expiries")
                common = None
                for expiry in expiries:
                    for kind in ["认购", "认沽"]:
                        strikes = set(subset.loc[(subset["到期日"] == expiry) & (subset["类型"] == kind), "行权价"])
                        common = strikes if common is None else common & strikes
                if not common:
                    raise ValueError("No actually observed common call/put strike across four expiries")
                strikes = sorted(common, key=float)
                strike = strikes[len(strikes)//2]
                for number, expiry in enumerate(expiries):
                    for kind, label in [("认购", "call"), ("认沽", "put")]:
                        contract = subset[(subset["到期日"] == expiry) & (subset["类型"] == kind) &
                                          (subset["行权价"] == strike)].iloc[0]
                        self.probe(f"sse_pair_{number}_{label}", "option_sse_daily_sina",
                                   {"symbol": contract["合约编码"]},
                                   role="One observed standard-unit call/put pair per CURRENT expiry; not a full chain",
                                   context={"metadata_probe": "sse_current_contracts", "expiry_from_current_metadata": expiry,
                                            "strike_from_current_metadata": strike, "type_from_current_metadata": kind})
                return
            if supplemental:
                for adjust, label, end in [("", "unadjusted_history", "20261104"),
                                            ("qfq", "qfq_old_date", "20241008"),
                                            ("hfq", "hfq_old_date", "20241008")]:
                    self.probe(f"etf_{label}", "fund_etf_hist_em", {
                        "symbol": "510050", "period": "daily", "start_date": "20241008",
                        "end_date": end, "adjust": adjust,
                    }, role="Historical alignment and adjustment probe; no inference of option metadata")
                return
            metadata = self.probe("sse_current_contracts", "option_current_day_sse",
                                  role="Current SSE contract metadata, NOT historical chain prices")
            months = self.probe("cffex_current_months", "option_cffex_hs300_list_sina",
                                role="Current CSI300 INDEX option month discovery; not ETF options")
            self.probe("cffex_expired_daily", "option_cffex_hs300_daily_sina",
                       {"symbol": "io2202P4350"}, role="Expired INDEX contract history probe",
                       context={"identifier_provenance": "Installed function's documented default; not independently verified metadata"})
            for adjust, label in [("", "unadjusted"), ("qfq", "qfq"), ("hfq", "hfq")]:
                self.probe(f"etf_{label}", "fund_etf_hist_em", {
                    "symbol": "510050", "period": "daily", "start_date": "20261026",
                    "end_date": "20261030", "adjust": adjust,
                }, role="50ETF spot adjustment comparison, bounded five-calendar-day window")
            self.probe("index_spot", "index_zh_a_hist", {
                "symbol": "000300", "period": "daily", "start_date": "20241008", "end_date": "20241011",
            }, role="CSI300 INDEX spot; function internally fetches broad data before slicing")
            self.probe("shibor", "macro_china_shibor_all",
                       role="Full fixed-size SHIBOR response: API exposes no date bounds")
            sse_months = self.probe("sse_current_months", "option_sse_list_sina", {"symbol": "50ETF"},
                                    role="Current SSE expiry-month discovery, NOT archived identifiers")
            for option_type, label in [("看涨期权", "call"), ("看跌期权", "put")]:
                self.probe(f"sse_expired_discovery_{label}", "option_sse_codes_sina", {
                    "symbol": option_type, "trade_date": "202410", "underlying": "510050",
                }, role="Expired-month identifier discovery; trade_date means EXPIRY MONTH, not quote date")
            self.probe("sse_expired_daily", "option_sse_daily_sina", {"symbol": "10003889"},
                       role="Expired SSE contract history probe",
                       context={"identifier_provenance": "Installed function's default example, not independently matched historical metadata"})
            self.probe("sse_expired_spot", "option_sse_spot_price_sina", {"symbol": "10003889"},
                       role="Check whether an expired example retains metadata/quotes; a nonempty table alone is not proof")
            self.probe("sse_historical_statistics", "option_daily_stats_sse", {"date": "20241008"},
                       role="Historical underlying-level aggregate statistics, NOT per-contract prices")
            # Probe at most two current SSE contracts selected from actual metadata.
            if isinstance(metadata, pd.DataFrame) and "合约编码" in metadata:
                subset = metadata[metadata.astype(str).apply(lambda c: c.str.contains("510050", regex=False)).any(axis=1)]
                selected = []
                for _, row in subset.iterrows():
                    identifier = str(row["合约编码"])
                    if identifier not in selected:
                        selected.append(identifier)
                    if len(selected) == 2:
                        break
                for number, identifier in enumerate(selected):
                    self.probe(f"sse_current_daily_{number}", "option_sse_daily_sina", {"symbol": identifier},
                               role="Daily price history for identifier observed in current SSE metadata",
                               context={"metadata_probe": "sse_current_contracts"})
                    self.probe(f"sse_current_spot_{number}", "option_sse_spot_price_sina", {"symbol": identifier},
                               role="Current bid/ask, OI, volume and quote timestamp for actual SSE identifier")
            if isinstance(sse_months, list) and sse_months:
                self.probe("sse_current_expiry", "option_sse_expire_day_sina",
                           {"trade_date": str(sse_months[0]), "symbol": "50ETF"},
                           role="Exact expiry day for a currently observed expiry month")
            if isinstance(months, dict) and months:
                group = next(iter(months.values()))
                if isinstance(group, list) and group:
                    spot = self.probe("cffex_current_spot", "option_cffex_hs300_spot_sina",
                                      {"symbol": str(group[0])}, role="INDEX chain metadata and live quote fields")
                    if isinstance(spot, pd.DataFrame) and len(spot) and "看涨合约-标识" in spot:
                        self.probe("cffex_current_daily", "option_cffex_hs300_daily_sina",
                                   {"symbol": str(spot.iloc[len(spot)//2]["看涨合约-标识"])},
                                   role="INDEX history for identifier discovered in actual current chain")
        finally:
            http.request = self.original_request
            http.request_with_retry = self.original_retry
            config.timeout = old_timeout

    def finish(self):
        sources = {}
        for name in REQUIRED + EXTRA:
            function = getattr(ak, name, None)
            sources[name] = None if function is None else {
                "signature": str(inspect.signature(function)),
                "docstring": inspect.getdoc(function),
                "source": inspect.getsource(function),
            }
        self.artifact("installed_api_sources.json", json.dumps(sources, ensure_ascii=False,
                      indent=2).encode(), "Installed API signatures, docstrings and exact source")
        summary = {
            "schema_version": 1, "akshare_version": ak.__version__,
            "http_capture_version": 2,
            "completed_at_utc": utc_now(), "probes": self.probes,
            "network_requests": self.network, "artifacts": self.artifacts,
            "limits": {"max_requests": self.max_requests, "max_total_response_bytes": self.max_bytes,
                       "max_response_bytes": 2_000_000, "timeout_connect_read_seconds": [5, 12]},
            "observed": {"requests": len(self.network), "response_bytes": self.total_bytes},
            "status_meaning": "Probe PASS means nonempty API return ONLY; it is not a requirements/coverage PASS",
        }
        with (self.output / "api_audit.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
        table = Table("Probe", "API result", "Rows/value", "Error")
        for item in self.probes:
            table.add_row(item["probe_id"], item["status"], str(item.get("rows", item.get("value", ""))),
                          item.get("error_type", ""))
        Console().print(table)
        Console().print(f"Audit: {len(self.network)} HTTP attempts; {self.total_bytes} response bytes; {self.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly authorize the bounded network probes")
    parser.add_argument("--output", type=Path, required=True, help="New repository-local directory; must not already exist")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--supplemental", action="store_true", help="Only three small ETF alignment/adjustment requests")
    modes.add_argument("--four-expiry-pairs", action="store_true", help="Eight histories selected from the verified current metadata snapshot")
    args = parser.parse_args()
    if not args.live:
        parser.error("Pass --live for acquisition; offline verification is provided separately")
    audit = Audit(args.output)
    try:
        audit.run(supplemental=args.supplemental, four_expiry_pairs=args.four_expiry_pairs)
    finally:
        audit.finish()


if __name__ == "__main__":
    main()
