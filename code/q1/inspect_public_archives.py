"""M1.5 bounded public-reference inspection only; no account or vendor data API.

Downloads go to a NEW repository-local ignored directory. BigQuant, Tushare and
RiceQuant market-data endpoints are deliberately absent: their access is gated.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import requests
from rich.console import Console


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_REFERENCES = {
    "sse_history_product.pdf": "https://www.sseinfo.com/services/assortment/market/hqywwd/wdcpsms/c/10782125/files/f2ba70dea74a4323bf13b76fffce0e40.pdf",
    "sse_history_interface.pdf": "https://www.sseinfo.com/services/assortment/market/hqywwd/wdzsjk/c/10824356/files/4ddcd5a592f9429f9288b0d2cb0c9549.pdf",
    "sse_options_product.pdf": "https://www.sseinfo.com/services/assortment/market/hqywwd/wdcpsms/c/10782127/files/557ef506aa2142ad92752dc3c9e81215.pdf",
    "zenodo_record.json": "https://zenodo.org/api/records/4288593",
}


def download(url, output, max_bytes=3_000_000):
    """One anonymous GET; no retries, credentials, cookies, or login redirects."""
    output = output.resolve()
    if not output.is_relative_to(ROOT / ".tmp"):
        raise ValueError("Reference destination must be repository-local .tmp")
    allowed = {"www.sseinfo.com", "zenodo.org"}
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in allowed or parsed.username:
        raise ValueError("URL is outside the approved public-reference allowlist")
    record = {"url": url, "requested_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        # Do not follow redirects into login, off-site storage, or another service.
        with requests.Session() as session:
            # Never pick up user netrc credentials, cookies or provider sessions.
            # This affects this anonymous reference reader only, not system config.
            session.trust_env = False
            with session.get(url, timeout=(5, 20), stream=True, allow_redirects=False) as response:
                record["http_status"] = response.status_code
                record["content_type"] = response.headers.get("Content-Type")
                if response.is_redirect or response.status_code in {401, 402, 403, 429}:
                    record["verdict"] = "STOPPED_REDIRECT_OR_ACCESS_GATE"
                    return record, None
                response.raise_for_status()
                data = bytearray()
                for chunk in response.iter_content(65536):
                    data.extend(chunk)
                    if len(data) > max_bytes:
                        raise ValueError("Public reference exceeded the bounded download size")
        payload = bytes(data)
        with output.open("xb") as handle:
            handle.write(payload)
        record.update({"local_path": output.relative_to(ROOT).as_posix(), "bytes": len(payload),
                       "sha256": hashlib.sha256(payload).hexdigest(), "verdict": "PUBLIC_REFERENCE_RETRIEVED"})
        return record, payload
    except Exception as error:
        record.update({"verdict": "FAILED", "error_type": type(error).__name__, "error": str(error)})
        return record, None


def inspect(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT / ".tmp"):
        raise ValueError("Reference downloads must stay in the repository's .tmp directory")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for filename, url in PUBLIC_REFERENCES.items():
        record, payload = download(url, output / filename)
        records.append(record)
        if filename != "zenodo_record.json" or payload is None:
            continue
        metadata = json.loads(payload)
        license_id = metadata.get("metadata", {}).get("license", {}).get("id")
        access = metadata.get("metadata", {}).get("access_right")
        # This is an explicitly published small research dataset, not a vendor
        # entitlement bypass. Only this exact public CC-BY record is considered.
        if license_id != "cc-by-4.0" or access != "open":
            record["sample_verdict"] = "STOPPED_NO_CONFIRMED_OPEN_LICENSE"
            continue
        entry = next((item for item in metadata.get("files", [])
                      if item["key"] == "data0322.dta" and item["size"] < 100_000), None)
        if entry is None:
            record["sample_verdict"] = "NO_SMALL_DOCUMENTED_SAMPLE"
            continue
        sample_record, sample = download(entry["links"]["self"], output / "zenodo_data0322.dta", 100_000)
        sample_record["license_asserted_by_publisher"] = license_id
        sample_record["upstream_rights_independently_verified"] = False
        records.append(sample_record)
        if sample is None:
            continue
        expected = entry.get("checksum", "")
        actual_md5 = "md5:" + hashlib.md5(sample).hexdigest()
        sample_record["upstream_checksum"] = expected
        sample_record["upstream_checksum_matches"] = expected == actual_md5
        if expected != actual_md5:
            sample_record["sample_verdict"] = "REJECTED_CHECKSUM_MISMATCH"
            continue
        try:
            frame = pd.read_stata(io.BytesIO(sample), convert_categoricals=False)
            sample_record["rows"] = len(frame)
            sample_record["columns"] = list(frame.columns)
            sample_record["dtypes"] = {name: str(dtype) for name, dtype in frame.dtypes.items()}
            sample_record["sample_values_for_schema_review"] = frame.head(3).to_dict(orient="records")
            sample_record["offline_reload_rows"] = len(pd.read_stata(output / "zenodo_data0322.dta", convert_categoricals=False))
            sample_record["sample_verdict"] = "SCHEMA_INSPECTED_NOT_YET_QUALIFIED_AS_ETF_OPTION_CHAINS"
        except Exception as error:
            sample_record.update({"sample_verdict": "PARSER_FAILED", "error_type": type(error).__name__, "error": str(error)})
    with (output / "reference_checks.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2, default=str)
    Console().print_json(data=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.output)
