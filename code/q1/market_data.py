"""Offline Q1 option normalization, provenance, temporal validation and filtering.

No pricing/calibration imports, network calls, inferred strikes, silent filling,
or promotion of current metadata to historical effective-dated evidence.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .provenance import ROOT, OfflineArchive, local_path, sha256
from .rates import TENORS, RatePolicy, align_rate, date_value


VERIFIED_EMPIRICAL = "VERIFIED_EMPIRICAL"
PARTIAL_HISTORICAL = "PARTIAL_HISTORICAL"
SCHEMA_VERSION = 1
FLOAT_COLUMNS = (
    "price", "open", "high", "low", "volume", "bid", "ask", "open_interest",
    "strike", "contract_unit", "quoted_strike", "spot", "time_to_expiry",
    "shibor_quote_decimal", "zero_rate", "discount_factor", "dividend_yield", "forward", "log_moneyness",
)
BOOL_COLUMNS = ("provenance_verified", "price_unit_verified", "metadata_effective_valid", "metadata_complete",
                "conventions_approved", "eligible_for_empirical", "partial_preview_eligible")
STRING_COLUMNS = (
    "quote_id", "data_origin", "observation_date", "quote_timestamp", "exchange", "instrument_family", "underlying",
    "contract_id", "option_type", "expiry_date", "listing_date", "price_kind", "price_unit", "raw_price",
    "metadata_basis", "metadata_valid_from", "metadata_valid_to", "metadata_source_snapshot", "metadata_source_sha256",
    "source_probe", "source_snapshot", "source_sha256", "source_row", "source_retrieved_at_utc",
    "spot_date", "spot_adjustment", "spot_source_snapshot", "spot_source_sha256", "rate_date", "rate_status",
    "rate_source_snapshot", "rate_source_sha256", "rate_approval_reference", "row_readiness",
    "rejection_reasons", "warnings",
)
CANONICAL_COLUMNS = (*STRING_COLUMNS, *FLOAT_COLUMNS, *BOOL_COLUMNS)


@dataclass(frozen=True)
class FilterPolicy:
    target_family: str = "ETF"
    target_underlying: str = "510050"
    min_volume: float = 1.0  # In source-reported units; not a claim those are verified contracts.
    require_bid_ask: bool = False
    max_relative_spread: float = 0.25
    min_open_interest: float | None = None
    max_abs_log_moneyness: float | None = None
    rates: RatePolicy = field(default_factory=RatePolicy)

    def __post_init__(self):
        values = [self.min_volume, self.max_relative_spread]
        values += [v for v in [self.min_open_interest, self.max_abs_log_moneyness] if v is not None]
        if any(not math.isfinite(float(v)) or v < 0 for v in values):
            raise ValueError("Filter thresholds must be finite and nonnegative")


def _text(value):
    return None if value is None or pd.isna(value) or not str(value).strip() else str(value).strip()


def _number(value):
    text = _text(value)
    if text is None:
        return None
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", text):
        text = text.replace(",", "")
    try:
        number = float(text)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _json_list(values):
    return json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))


def _underlying(value):
    text = _text(value)
    if text is None:
        return None
    return re.sub(r"\.(SH|SZ|XSHG|XSHE)$", "", text.upper())


def normalize_metadata(metadata):
    result = []
    for original in metadata:
        row = dict(original)
        for name in ("contract_id", "exchange", "instrument_family", "source_snapshot", "source_sha256", "evidence_level"):
            row[name] = _text(row.get(name))
        for name in ("exchange", "instrument_family"):
            if row[name]:
                row[name] = row[name].upper()
        row["underlying"] = _underlying(row.get("underlying"))
        kind = _text(row.get("option_type"))
        row["option_type"] = {"认购": "C", "认沽": "P"}.get(kind, kind.upper() if kind else None)
        for name in ("valid_from", "valid_to", "listing_date", "expiry_date"):
            row[name] = date_value(row.get(name))
        for name in ("strike", "contract_unit"):
            row[name] = _number(row.get(name))
        result.append(row)
    return result


def _metadata_match(quote, metadata):
    """Select only a unique covered version, otherwise retain a labelled candidate."""
    candidates = [m for m in metadata if m.get("contract_id") == quote.get("contract_id")
                  and m.get("exchange") == quote.get("exchange")]
    date = quote.get("observation_date")
    covered = [m for m in candidates if date and date_value(m.get("valid_from"))
               and date_value(m.get("valid_to"))
               and date_value(m["valid_from"]) <= date <= date_value(m["valid_to"])]
    if len(covered) > 1:
        return {}, False, ["AMBIGUOUS_EFFECTIVE_METADATA"]
    if covered:
        selected = covered[0]
        valid = (selected.get("evidence_level") == "VERIFIED_DATE_EFFECTIVE"
                 and selected.get("provenance_verified") is True and bool(selected.get("source_snapshot"))
                 and bool(re.fullmatch(r"[0-9a-f]{64}", selected.get("source_sha256") or "")))
        return selected, valid, [] if valid else ["METADATA_EVIDENCE_UNVERIFIED"]
    undated = [m for m in candidates if m.get("evidence_level") == "CURRENT_SNAPSHOT_UNDATED"]
    if len(undated) == 1:
        return undated[0], False, ["UNDATED_CURRENT_METADATA"]
    return {}, False, ["MISSING_EFFECTIVE_METADATA" if not candidates else "NO_METADATA_VERSION_COVERS_DATE"]


def _duplicate_reasons(rows):
    groups = {}
    for i, row in enumerate(rows):
        key = tuple(row.get(name) for name in ("exchange", "contract_id", "observation_date", "price_kind", "quote_timestamp"))
        groups.setdefault(key, []).append(i)
    output = {}
    attributes = ("price", "open", "high", "low", "volume", "bid", "ask", "open_interest", "quoted_strike",
                  "price_unit", "price_unit_verified")
    for members in groups.values():
        if len(members) < 2:
            continue
        signatures = {tuple(rows[i].get(name) for name in attributes) for i in members}
        if len(signatures) == 1:
            for i in members[1:]:
                output[i] = "DUPLICATE_EXACT"
        else:
            for i in members:
                output[i] = "DUPLICATE_CONFLICT"
    return output


def normalize_quotes(raw_quotes, metadata, spots, rates, *, policy=FilterPolicy(), verified_tenor_days=None):
    """Deterministic row-preserving cleaning; all rejections are retained.

    Metadata verification tags/provenance are supplied by a verified loader, not
    inferred from an identifier or from listing/expiry dates. Unit tests use
    separately labelled stand-ins to exercise that contract, not empirical data.
    """
    required_spot = {"observation_date", "underlying", "instrument_family", "spot", "adjustment", "source_snapshot", "source_sha256"}
    required_rates = {"observation_date", *TENORS}
    if not required_spot <= set(spots.columns) or not required_rates <= set(rates.columns):
        raise ValueError("Underlying/rate table schema is incomplete")
    metadata = normalize_metadata(metadata)
    spots, rates = spots.copy(), rates.copy()
    spots["observation_date"] = spots["observation_date"].map(date_value)
    spots["underlying"] = spots["underlying"].map(_underlying)
    spots["instrument_family"] = spots["instrument_family"].map(lambda v: (_text(v) or "").upper())
    rates["observation_date"] = rates["observation_date"].map(date_value)
    normalized = []
    for raw in raw_quotes:
        row = {name: None for name in CANONICAL_COLUMNS}
        for name in STRING_COLUMNS:
            if name in raw:
                row[name] = _text(raw[name])
        row["raw_price"] = _text(raw.get("price"))
        row["observation_date"] = date_value(raw.get("observation_date"))
        row["exchange"] = row["exchange"].upper() if row["exchange"] else None
        row["instrument_family"] = row["instrument_family"].upper() if row["instrument_family"] else None
        for name in FLOAT_COLUMNS:
            row[name] = _number(raw.get(name))
        row["provenance_verified"] = raw.get("provenance_verified") is True
        row["price_unit_verified"] = raw.get("price_unit_verified") is True
        row["_raw"] = raw
        normalized.append(row)
    normalized.sort(key=lambda r: (*tuple(r.get(k) or "" for k in
                    ("observation_date", "exchange", "contract_id", "price_kind", "quote_timestamp", "source_snapshot", "source_row")),
                                  json.dumps(r["_raw"], sort_keys=True, default=str)))
    duplicates = _duplicate_reasons(normalized)
    result = []
    occurrences = Counter()
    for index, row in enumerate(normalized):
        raw = row.pop("_raw")
        reasons, warnings = [], []
        if index in duplicates:
            reasons.append(duplicates[index])
        if row["observation_date"] is None:
            reasons.append("INVALID_OBSERVATION_DATE")
        if row["quote_timestamp"]:
            try:
                stamp = pd.Timestamp(row["quote_timestamp"])
                if stamp.tz is None:
                    warnings.append("TIMESTAMP_ASSUMED_ASIA_SHANGHAI")
                    stamp = stamp.tz_localize("Asia/Shanghai")
                stamp = stamp.tz_convert("Asia/Shanghai")
                if stamp.date().isoformat() != row["observation_date"]:
                    reasons.append("QUOTE_TIMESTAMP_DATE_CONFLICT")
            except (ValueError, TypeError):
                reasons.append("INVALID_QUOTE_TIMESTAMP")
        if not row["contract_id"] or not row["exchange"]:
            reasons.append("MISSING_CONTRACT_IDENTITY")
        if row["price"] is None or row["price"] <= 0:
            reasons.append("INVALID_OR_NONPOSITIVE_PRICE")
        for name in FLOAT_COLUMNS:
            if name in raw and _text(raw[name]) is not None and _number(raw[name]) is None:
                reasons.append("INVALID_NUMERIC_" + name.upper())
        if not row["provenance_verified"]:
            reasons.append("UNVERIFIED_PROVENANCE")
        if row["data_origin"] != "REAL_ARCHIVED_OBSERVATION":
            reasons.append("NOT_VERIFIED_REAL_OBSERVATION")
        if not row["price_unit_verified"]:
            reasons.append("PRICE_UNIT_UNVERIFIED")
        for name in ("open", "high", "low"):
            if row[name] is not None and row[name] < 0:
                reasons.append("INVALID_OHLC")
        if row["low"] is not None and row["high"] is not None:
            if row["low"] > row["high"] or (row["price"] is not None and not row["low"] <= row["price"] <= row["high"]):
                reasons.append("INCONSISTENT_OHLC")
        match, effective, metadata_errors = _metadata_match(row, metadata)
        reasons.extend(metadata_errors)
        row.update({
            "metadata_basis": match.get("evidence_level", "MISSING"),
            "metadata_valid_from": date_value(match.get("valid_from")),
            "metadata_valid_to": date_value(match.get("valid_to")),
            "metadata_source_snapshot": match.get("source_snapshot"),
            "metadata_source_sha256": match.get("source_sha256"),
            "metadata_effective_valid": effective,
            "listing_date": date_value(match.get("listing_date")),
            "expiry_date": date_value(match.get("expiry_date")),
            "strike": _number(match.get("strike")), "contract_unit": _number(match.get("contract_unit")),
            "option_type": match.get("option_type"),
        })
        observed_underlying = _underlying(raw.get("underlying"))
        if match.get("instrument_family") and match["instrument_family"] != row["instrument_family"]:
            reasons.append("METADATA_INSTRUMENT_FAMILY_CONFLICT")
            effective = False
        row["underlying"] = match.get("underlying", observed_underlying)
        if observed_underlying and match.get("underlying") and observed_underlying != match["underlying"]:
            reasons.append("METADATA_UNDERLYING_CONFLICT")
            effective = False
        if row["quoted_strike"] is not None and row["strike"] is not None and not math.isclose(row["quoted_strike"], row["strike"], abs_tol=1e-10, rel_tol=0):
            reasons.append("METADATA_STRIKE_CONFLICT")
            effective = False
        complete = (bool(row["underlying"]) and row["option_type"] in {"C", "P"} and row["strike"] is not None
                    and row["strike"] > 0 and row["contract_unit"] is not None and row["contract_unit"] > 0
                    and row["expiry_date"] is not None and row["listing_date"] is not None)
        row["metadata_complete"] = bool(complete)
        if not complete:
            reasons.append("INCOMPLETE_CONTRACT_METADATA")
        if row["observation_date"]:
            if row["listing_date"] and row["observation_date"] < row["listing_date"]:
                reasons.append("QUOTE_BEFORE_LISTING")
                effective = False
            if row["expiry_date"] and row["observation_date"] > row["expiry_date"]:
                reasons.append("QUOTE_AFTER_EXPIRY")
                effective = False
        if row["listing_date"] and row["expiry_date"] and row["listing_date"] > row["expiry_date"]:
            reasons.append("INVALID_CONTRACT_LIFECYCLE")
            effective = False
        row["metadata_effective_valid"] = bool(effective and complete)
        empirical_structure = bool(effective and complete and row["provenance_verified"] and row["price_unit_verified"]
                                   and row["price"] is not None and row["price"] > 0 and row["observation_date"]
                                   and row["data_origin"] == "REAL_ARCHIVED_OBSERVATION")
        if row["instrument_family"] != policy.target_family:
            reasons.append("NON_TARGET_INSTRUMENT_FAMILY")
        if row["underlying"] != policy.target_underlying:
            reasons.append("MISSING_OR_NON_TARGET_UNDERLYING")
        if row["volume"] is None:
            reasons.append("MISSING_OR_INVALID_VOLUME")
        elif row["volume"] < 0:
            reasons.append("NEGATIVE_VOLUME")
        elif row["volume"] < policy.min_volume:
            reasons.append("LOW_VOLUME")
        warnings.append("VOLUME_UNIT_REQUIRES_SOURCE_VALIDATION")
        bid, ask = row["bid"], row["ask"]
        if bid is None or ask is None:
            (reasons if policy.require_bid_ask else warnings).append("BID_ASK_MISSING")
        elif bid < 0 or ask <= 0 or bid > ask:
            reasons.append("INVALID_OR_CROSSED_BID_ASK")
        elif (ask-bid)/((ask+bid)/2) > policy.max_relative_spread:
            reasons.append("WIDE_RELATIVE_SPREAD")
        if row["open_interest"] is None:
            (reasons if policy.min_open_interest is not None else warnings).append("OPEN_INTEREST_MISSING")
        elif row["open_interest"] < 0:
            reasons.append("NEGATIVE_OPEN_INTEREST")
        elif policy.min_open_interest is not None and row["open_interest"] < policy.min_open_interest:
            reasons.append("LOW_OPEN_INTEREST")
        matched_spot = spots[(spots["observation_date"] == row["observation_date"])
                             & (spots["underlying"] == row["underlying"])
                             & (spots["instrument_family"] == row["instrument_family"])]
        if len(matched_spot) == 1:
            spot = matched_spot.iloc[0]
            row.update(spot=_number(spot["spot"]), spot_date=spot["observation_date"], spot_adjustment=spot["adjustment"],
                       spot_source_snapshot=spot["source_snapshot"], spot_source_sha256=spot["source_sha256"])
            if row["spot"] is None or row["spot"] <= 0 or spot["adjustment"] != "unadjusted":
                reasons.append("INVALID_OR_ADJUSTED_SPOT")
        else:
            reasons.append("MISSING_SPOT_DATE" if not len(matched_spot) else "DUPLICATE_SPOT_DATE")
        date_nodes = None if verified_tenor_days is None else verified_tenor_days.get(row["observation_date"])
        aligned = align_rate(row["observation_date"], row["expiry_date"], rates, policy=policy.rates, node_days=date_nodes)
        row.update(aligned)
        rate_matches = rates[rates["observation_date"] == row["rate_date"]]
        if len(rate_matches) == 1:
            row["rate_source_snapshot"] = rate_matches.iloc[0].get("source_snapshot")
            row["rate_source_sha256"] = rate_matches.iloc[0].get("source_sha256")
        if row["rate_status"] != "APPROVED":
            reasons.append(row["rate_status"])
        if policy.rates.dividend_yield is None or not row["conventions_approved"]:
            reasons.append("DIVIDEND_OR_RATE_CONVENTIONS_UNAPPROVED")
        if row["conventions_approved"] and row["spot"] is not None and row["spot"] > 0:
            try:
                row["forward"] = row["spot"] * math.exp(-float(policy.rates.dividend_yield)*row["time_to_expiry"]) / row["discount_factor"]
                if not math.isfinite(row["forward"]) or row["forward"] <= 0:
                    raise ValueError("Invalid forward")
                if row["strike"] is not None and row["strike"] > 0:
                    row["log_moneyness"] = math.log(row["strike"])-math.log(row["forward"])
            except (ValueError, OverflowError, ZeroDivisionError):
                reasons.append("INVALID_DERIVED_FORWARD")
                row["forward"] = row["log_moneyness"] = None
        if policy.max_abs_log_moneyness is not None:
            if row["log_moneyness"] is None:
                reasons.append("MONEYNESS_UNAVAILABLE")
            elif abs(row["log_moneyness"]) > policy.max_abs_log_moneyness:
                reasons.append("OUTSIDE_MONEYNESS_BAND")
        unresolved = {"UNDATED_CURRENT_METADATA", "METADATA_EVIDENCE_UNVERIFIED", "PRICE_UNIT_UNVERIFIED",
                      "INDICATIVE_UNAPPROVED_CONVENTIONS", "DIVIDEND_OR_RATE_CONVENTIONS_UNAPPROVED"}
        row["partial_preview_eligible"] = bool(complete and not (set(reasons)-unresolved))
        unreliable_observation = any(reason.startswith("INVALID_NUMERIC_") or reason in {
            "INVALID_OBSERVATION_DATE", "INVALID_QUOTE_TIMESTAMP", "QUOTE_TIMESTAMP_DATE_CONFLICT",
            "DUPLICATE_CONFLICT", "INVALID_OHLC", "INCONSISTENT_OHLC"} for reason in reasons)
        row["row_readiness"] = VERIFIED_EMPIRICAL if empirical_structure and not unreliable_observation else PARTIAL_HISTORICAL
        row["eligible_for_empirical"] = not reasons and row["row_readiness"] == VERIFIED_EMPIRICAL
        identity = {name: row.get(name) for name in ["source_snapshot", "source_row", "price_kind", "contract_id", "quote_timestamp"]}
        token = json.dumps(identity, sort_keys=True)
        occurrences[token] += 1
        row["quote_id"] = sha256((token + "|" + str(occurrences[token])).encode())[:24]
        row["rejection_reasons"], row["warnings"] = _json_list(reasons), _json_list(warnings)
        result.append(row)
    frame = pd.DataFrame(result, columns=CANONICAL_COLUMNS)
    for name in STRING_COLUMNS:
        frame[name] = frame[name].astype("string")
    for name in FLOAT_COLUMNS:
        frame[name] = frame[name].astype("Float64")
    for name in BOOL_COLUMNS:
        frame[name] = frame[name].astype("boolean")
    validate_canonical_schema(frame)
    return frame


def validate_canonical_schema(frame):
    if list(frame.columns) != list(CANONICAL_COLUMNS):
        raise ValueError("Canonical quote columns/order do not match the schema")
    for columns, dtype in [(STRING_COLUMNS, "string"), (FLOAT_COLUMNS, "Float64"), (BOOL_COLUMNS, "boolean")]:
        if any(str(frame[name].dtype) != dtype for name in columns):
            raise ValueError("Canonical quote dtypes do not match the schema")
    if frame[list(BOOL_COLUMNS)].isna().any().any() or not frame["quote_id"].is_unique:
        raise ValueError("Canonical flags must be explicit and source-occurrence IDs unique")
    if not frame["row_readiness"].isin([VERIFIED_EMPIRICAL, PARTIAL_HISTORICAL]).all():
        raise ValueError("Unknown readiness state")
    for name in FLOAT_COLUMNS:
        if not np.isfinite(frame[name].dropna().to_numpy(dtype=float)).all():
            raise ValueError("Canonical numbers must be finite or explicitly missing")
    for text in frame["rejection_reasons"]:
        reasons = json.loads(text)
        if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
            raise ValueError("Rejection reasons must be a JSON string array")
    eligible = frame[frame["eligible_for_empirical"]]
    if len(eligible) and (not (eligible["row_readiness"] == VERIFIED_EMPIRICAL).all()
                          or not eligible[["metadata_effective_valid", "price_unit_verified", "conventions_approved"]].all().all()
                          or not (eligible["rejection_reasons"] == "[]").all()):
        raise ValueError("Empirical eligibility contradicts evidence/approval/rejection flags")
    return True


def load_source_records(root=ROOT):
    """Use only actual committed tables. Do not invoke acquisition or audit scripts."""
    archive = OfflineArchive(root)
    probes = {}
    for directory in ["m1_audit", "m1_alignment", "m1_pairs"]:
        summary = archive.json(f"data/clean/q1/{directory}/api_audit.json")
        probes.update({p["probe_id"]: p for p in summary["probes"]})

    def source(probe_id):
        p = probes[probe_id]
        path = p["snapshot"]["path"]
        return archive.csv(path), {"source_probe": probe_id, "source_snapshot": path,
                                  "source_sha256": archive.files[path]["sha256"],
                                  "source_retrieved_at_utc": p["started_at_utc"], "provenance_verified": True,
                                  "data_origin": "REAL_ARCHIVED_OBSERVATION"}

    metadata = []
    master, provenance = source("sse_current_contracts")
    for _, item in master.iterrows():
        underlying = re.search(r"\((\d{6})\)", item["标的券名称及代码"])
        metadata.append({**provenance, "contract_id": item["合约编码"], "exchange": "SSE", "instrument_family": "ETF",
                         "underlying": underlying.group(1) if underlying else None,
                         "option_type": {"认购": "C", "认沽": "P"}.get(item["类型"]),
                         "strike": item["行权价"], "contract_unit": item["合约单位"],
                         "listing_date": date_value(item["开始日期"]), "expiry_date": date_value(item["到期日"]),
                         "valid_from": None, "valid_to": None, "evidence_level": "CURRENT_SNAPSHOT_UNDATED"})
    chain, provenance = source("cffex_current_spot")
    for _, item in chain.iterrows():
        for label, kind in [("看涨", "C"), ("看跌", "P")]:
            metadata.append({**provenance, "contract_id": item[f"{label}合约-标识"], "exchange": "CFFEX", "instrument_family": "INDEX",
                             "underlying": "000300", "option_type": kind, "strike": item["行权价"],
                             "contract_unit": None, "listing_date": None, "expiry_date": None,
                             "valid_from": None, "valid_to": None, "evidence_level": "CURRENT_SNAPSHOT_UNDATED"})
    raw = []
    for probe_id, p in probes.items():
        if p["api"] not in {"option_sse_daily_sina", "option_cffex_hs300_daily_sina"} or p["status"] != "PASS":
            continue
        table, provenance = source(probe_id)
        sse = p["api"] == "option_sse_daily_sina"
        for number, item in table.iterrows():
            raw.append({**provenance, "source_row": str(number), "observation_date": item["日期" if sse else "date"],
                        "contract_id": p["parameters"]["symbol"], "exchange": "SSE" if sse else "CFFEX",
                        "instrument_family": "ETF" if sse else "INDEX", "underlying": None if sse else "000300",
                        "price_kind": "daily_close", "price": item["收盘" if sse else "close"],
                        "open": item["开盘" if sse else "open"], "high": item["最高" if sse else "high"],
                        "low": item["最低" if sse else "low"], "volume": item["成交量" if sse else "volume"],
                        "bid": None, "ask": None, "open_interest": None,
                        "price_unit": "UNVERIFIED_SOURCE_PREMIUM_UNIT", "price_unit_verified": False})
    for probe_id in ["sse_current_spot_0", "sse_current_spot_1"]:
        table, provenance = source(probe_id)
        fields = table.set_index("字段")["值"].to_dict()
        timestamp = pd.Timestamp(fields["行情时间"]).tz_localize("Asia/Shanghai")
        raw.append({**provenance, "source_row": str(table.index[table["字段"] == "最新价"][0]),
                    "observation_date": timestamp.date().isoformat(), "quote_timestamp": timestamp.isoformat(),
                    "contract_id": probes[probe_id]["parameters"]["symbol"], "exchange": "SSE", "instrument_family": "ETF",
                    "underlying": fields["标的股票"], "price_kind": "latest_quote_snapshot", "price": fields["最新价"],
                    "quoted_strike": fields["行权价"], "open": fields["开盘价"], "high": fields["最高价"], "low": fields["最低价"],
                    "volume": fields["成交量"], "bid": fields["买价"], "ask": fields["卖价"], "open_interest": fields["持仓量"],
                    "price_unit": "UNVERIFIED_SOURCE_PREMIUM_UNIT", "price_unit_verified": False})
    spots = []
    for probe_id, underlying, family in [("etf_unadjusted_history", "510050", "ETF"), ("index_spot", "000300", "INDEX")]:
        table, provenance = source(probe_id)
        for _, item in table.iterrows():
            spots.append({**provenance, "observation_date": date_value(item["日期"]), "underlying": underlying,
                          "instrument_family": family, "spot": _number(item["收盘"]), "adjustment": "unadjusted"})
    table, provenance = source("shibor")
    rates = pd.DataFrame([{"observation_date": date_value(item["日期"]), **provenance,
                           **{tenor: _number(item[tenor+"-定价"]) for tenor in TENORS}} for _, item in table.iterrows()])
    return {"archive": archive, "probes": probes, "raw_quotes": raw, "metadata": metadata,
            "spots": pd.DataFrame(spots), "rates": rates}


def _counts(frame):
    dates = frame["observation_date"].dropna()
    return {"quotes": len(frame), "dates": int(dates.nunique()), "expiries": int(frame["expiry_date"].nunique()),
            "contracts": int(frame["contract_id"].nunique()), "strikes": int(frame["strike"].nunique()),
            "calls": int((frame["option_type"] == "C").sum()), "puts": int((frame["option_type"] == "P").sum()),
            "date_min": str(dates.min()) if len(dates) else None,
            "date_max": str(dates.max()) if len(dates) else None}


def coverage_report(quotes, *, policy=FilterPolicy()):
    target = quotes[(quotes["instrument_family"] == policy.target_family)
                    & (quotes["underlying"] == policy.target_underlying)]
    verified = target[target["row_readiness"] == VERIFIED_EMPIRICAL]
    eligible = target[target["eligible_for_empirical"]]
    preview = target[target["partial_preview_eligible"]]
    return {
        "dataset_readiness": PARTIAL_HISTORICAL,
        "examination_historical_requirement_satisfied": False,
        "readiness_reason": "No complete dated chain roster and official calendar over the required historical scope have been supplied",
        "all_real_observations_including_separate_index_and_unknown_metadata_rows": _counts(quotes),
        "target_etf_observations_not_automatically_eligible": _counts(target),
        "verified_empirical_rows_complete_metadata_only": _counts(verified),
        "strictly_usable_for_empirical_analysis": _counts(eligible),
        "partial_diagnostic_preview_NOT_empirical": _counts(preview),
        "source_daily_close_rows": int((quotes["price_kind"] == "daily_close").sum()),
        "source_timestamped_current_quote_rows": int((quotes["price_kind"] == "latest_quote_snapshot").sum()),
        "missing_bid_or_ask_rows": int((quotes["bid"].isna() | quotes["ask"].isna()).sum()),
        "missing_open_interest_rows": int(quotes["open_interest"].isna().sum()),
        "date_effective_metadata_verified_rows": int(quotes["metadata_effective_valid"].sum()),
        "incomplete_contract_attribute_rows": int((~quotes["metadata_complete"]).sum()),
        "input_rows_are_real_not_synthetic": bool(len(quotes) and (quotes["data_origin"] == "REAL_ARCHIVED_OBSERVATION").all()),
        "complete_month_verified": False,
        "five_verified_sample_chains_spanning_two_years": False,
        "complete_chain_expiry_counts_verified": False,
        "index_options_are_not_an_etf_substitute": True,
    }


def _json_payload(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _csv_payload(frame):
    return frame.to_csv(index=False, lineterminator="\n", na_rep="", float_format="%.17g").encode("utf-8")


def build_pipeline(output, *, root=ROOT, policy=FilterPolicy()):
    """Rebuild from pinned local inputs; identical reruns verify, never overwrite.

    The official-coverage acceptance gate remains closed without independently
    supplied expected chain rosters/calendar. A handful of valid rows alone cannot
    certify the examination's entire historical sampling requirement.
    """
    output, root = Path(output).resolve(), Path(root).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Pipeline output must stay inside the repository")
    bundle = load_source_records(root)
    frame = normalize_quotes(bundle["raw_quotes"], bundle["metadata"], bundle["spots"], bundle["rates"], policy=policy)
    coverage = coverage_report(frame, policy=policy)
    rejections, warnings = [], []
    for _, row in frame.iterrows():
        for reason in json.loads(row["rejection_reasons"]):
            rejections.append({"quote_id": row["quote_id"], "reason": reason})
        for reason in json.loads(row["warnings"]):
            warnings.append({"quote_id": row["quote_id"], "warning": reason})
    rejection_counts = pd.DataFrame(sorted(Counter(item["reason"] for item in rejections).items()), columns=["reason", "quote_count"])
    warning_counts = pd.DataFrame(sorted(Counter(item["warning"] for item in warnings).items()), columns=["warning", "quote_count"])
    grouped = []
    keys = ["observation_date", "instrument_family", "underlying", "expiry_date"]
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        grouped.append({**{name: (None if pd.isna(item) else item) for name, item in zip(keys, key)},
                        **_counts(group), "verified_metadata_quotes": int(group["metadata_effective_valid"].sum()),
                        "strictly_usable_quotes": int(group["eligible_for_empirical"].sum()),
                        "partial_preview_quotes": int(group["partial_preview_eligible"].sum()),
                        "complete_chain_verified": False})
    schema = {
        "schema_version": SCHEMA_VERSION,
        "columns": {name: {"dtype": str(frame[name].dtype), "nullable": name not in BOOL_COLUMNS} for name in CANONICAL_COLUMNS},
        "date_format": "ISO YYYY-MM-DD; timestamps retain explicit Asia/Shanghai offset",
        "price_unit_policy": "Raw source premium unchanged; source quote units unverified until independently validated",
        "metadata_policy": "Current master values are candidate attributes only; version bounds are never inferred from listing/expiry",
        "readiness_states": [VERIFIED_EMPIRICAL, PARTIAL_HISTORICAL],
        "verified_row_not_full_dataset": "A verified row is not proof of the required multi-year/monthly coverage",
        "rejection_encoding": "JSON array of all applicable reasons; reason counts overlap",
        "partial_preview_warning": "Diagnostic cleaning preview only, not empirical-ready or examination-compliant",
    }
    code_paths = ["data/fetch.py", "code/q1/market_data.py", "code/q1/rates.py", "code/q1/provenance.py"]
    provenance = {
        "pipeline_schema_version": SCHEMA_VERSION,
        "question": "q1", "mode": "OFFLINE_ONLY_NO_SOURCE_DISCOVERY",
        "policy": asdict(policy),
        "rate_conventions": {"time_to_expiry": "ACT/365F calendar-date difference, indicative until approved",
                             "tenor_dates_default": "Unadjusted calendar offsets; no business-day/spot-lag assumption asserted as verified",
                             "shibor_input_unit": "percent per annum as labelled by installed source API",
                             "default_zero_curve_interpretation": "UNAPPROVED; only raw quote interpolation is emitted",
                             "dividend_default": "UNAPPROVED; no q=0 assumption",
                             "no_default_discount_factor_or_forward": True},
        "source_manifests": [{"path": item["path"], "sha256": item["sha256"]} for item in bundle["archive"].manifests],
        "source_files_consumed": [bundle["archive"].files[name] for name in sorted(bundle["archive"].consumed)],
        "source_api_records": [{key: probe.get(key) for key in ["probe_id", "api", "parameters", "started_at_utc", "columns", "dtypes", "date_columns", "snapshot"]}
                               for probe in bundle["probes"].values() if probe.get("snapshot", {}).get("path") in bundle["archive"].consumed],
        "code_lf_sha256": {name: sha256((ROOT/name).read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")) for name in code_paths},
        "build_timestamp": None,
        "timestamp_policy": "No invented retrieval/build timestamp; source retrieval times retained, reproducible outputs contain no wall-clock run time",
        "prior_initial_http_capture_limitation": "M1 index-map helper pagination was incompletely recorded; preserved audit disclosure remains applicable",
        "license_status": "UNVERIFIED for external redistribution; processes already committed course audit evidence only",
        "excluded_source_kinds": ["SSE aggregate statistics are not per-contract quotes", "Untimestamped CFFEX live-chain prices are not assigned an observation date", "Catalog documentation examples are not market observations"],
    }
    payloads = {
        "quotes.csv": _csv_payload(frame),
        "accepted_quotes.csv": _csv_payload(frame[frame["eligible_for_empirical"]]),
        "partial_diagnostic_quotes.csv": _csv_payload(frame[frame["partial_preview_eligible"]]),
        "contract_metadata.csv": _csv_payload(pd.DataFrame(bundle["metadata"])),
        "underlying_spots.csv": _csv_payload(bundle["spots"]),
        "shibor_quotes.csv": _csv_payload(bundle["rates"]),
        "rejections.csv": _csv_payload(pd.DataFrame(rejections, columns=["quote_id", "reason"])),
        "rejection_counts.csv": _csv_payload(rejection_counts),
        "warning_counts.csv": _csv_payload(warning_counts),
        "coverage_by_date_expiry.csv": _csv_payload(pd.DataFrame(grouped)),
        "coverage_summary.json": _json_payload(coverage),
        "schema.json": _json_payload(schema), "provenance.json": _json_payload(provenance),
    }
    manifest = {"question": "q1", "schema_version": SCHEMA_VERSION, "dataset_readiness": PARTIAL_HISTORICAL,
                "files": [{"name": name, "bytes": len(content), "sha256": sha256(content)} for name, content in sorted(payloads.items())]}
    payloads["manifest.json"] = _json_payload(manifest)
    if output.exists():
        actual = {path.name for path in output.iterdir() if path.is_file()}
        if actual != set(payloads) or any((output/name).read_bytes() != content for name, content in payloads.items()):
            raise FileExistsError("Existing pipeline output differs; inspect it and use a new --output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name, content in payloads.items():
            with (output/name).open("xb") as handle:
                handle.write(content)
    return {"coverage": coverage, "manifest_sha256": sha256(payloads["manifest.json"]), "output": output}


def load_offline_pipeline(output, expected_manifest_sha256):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Pipeline output must stay repository-local")
    payload = (output/"manifest.json").read_bytes()
    if sha256(payload) != expected_manifest_sha256:
        raise ValueError("Pipeline manifest SHA-256 mismatch")
    manifest = json.loads(payload)
    if manifest["question"] != "q1" or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported pipeline schema/question")
    names = set()
    for item in manifest["files"]:
        if Path(item["name"]).name != item["name"] or item["name"] in names:
            raise ValueError("Invalid or duplicate output artifact name")
        names.add(item["name"])
        content = local_path(output, item["name"]).read_bytes()
        if len(content) != item["bytes"] or sha256(content) != item["sha256"]:
            raise ValueError(f"Output artifact checksum mismatch: {item['name']}")
    if not {"quotes.csv", "schema.json", "coverage_summary.json", "provenance.json"} <= names:
        raise ValueError("Incomplete output manifest")
    quotes = pd.read_csv(output/"quotes.csv", dtype=str, keep_default_na=False)
    if list(quotes.columns) != list(CANONICAL_COLUMNS):
        raise ValueError("Canonical quote schema mismatch")
    for name in STRING_COLUMNS:
        quotes[name] = quotes[name].replace("", pd.NA).astype("string")
    for name in FLOAT_COLUMNS:
        parsed = []
        for text in quotes[name]:
            number = None if text == "" else float(text)
            if number is not None and not math.isfinite(number):
                raise ValueError("Nonfinite numeric token in canonical CSV")
            parsed.append(number)
        quotes[name] = pd.array(parsed, dtype="Float64")
    for name in BOOL_COLUMNS:
        if not quotes[name].isin(["True", "False"]).all():
            raise ValueError("Invalid boolean in canonical quotes")
        quotes[name] = quotes[name].map({"True": True, "False": False}).astype("boolean")
    validate_canonical_schema(quotes)
    return {"quotes": quotes, "manifest": manifest,
            "coverage": json.loads((output/"coverage_summary.json").read_bytes())}


def load_registered_q1(root=ROOT):
    root = Path(root).resolve()
    registry = json.loads((root/"data/metadata.json").read_text(encoding="utf-8"))["questions"]["q1"]
    archive = OfflineArchive(root)
    if registry["source_manifests"] != [{"path": item["path"], "sha256": item["sha256"]} for item in archive.manifests]:
        raise ValueError("Registered source manifests do not match the pinned Q1 archive")
    return load_offline_pipeline(local_path(root, registry["output_directory"]), registry["output_manifest_sha256"])
