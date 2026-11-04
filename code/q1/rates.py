"""Q1 rate/date conventions and bounded interpolation, independent of pricing.

SHIBOR fixings are not silently called zero rates. The default produces only an
indicative interpolated deposit quote. Discount factors/forwards require an
explicit approved rate interpretation and dividend convention from the caller.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


TENORS = ("O/N", "1W", "2W", "1M", "3M", "6M", "9M", "1Y")


def date_value(value):
    """Strict date-only ISO or YYYYMMDD; never invent a date from retrieval time."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        stamp = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if len(text) == 8 and text.isdigit():
            stamp = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        elif len(text) == 10 and text[4] == "-" and text[7] == "-":
            stamp = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
        else:
            return None
    return None if pd.isna(stamp) else stamp.date().isoformat()


def tenor_calendar_days(observation_date):
    """Indicative unadjusted calendar offsets; NOT a verified SHIBOR calendar.

    Real spot-lag/business-day/holiday conventions need an approved calendar.
    Callers can instead supply independently verified node day counts explicitly.
    """
    date = date_value(observation_date)
    if date is None:
        raise ValueError("Invalid observation date")
    start = pd.Timestamp(date)
    ends = [start + pd.Timedelta(days=1), start + pd.Timedelta(days=7),
            start + pd.Timedelta(days=14), *[start + pd.DateOffset(months=n) for n in (1, 3, 6, 9, 12)]]
    return np.array([(end-start).days for end in ends], dtype=float)


def interpolate_shibor(node_days, quoted_percent, target_days, *, interpretation="unapproved"):
    """Linear interpolation at actual elapsed calendar days, without extrapolation.

    Returns quoted-rate decimals and (only for explicit interpretations) continuous
    ACT/365F zero rates. `zero_continuous` interprets the input as continuous zeros;
    `simple_act360` converts each deposit quote using D=1/(1+L*days/360) before
    interpolating continuous ACT/365F zeros. Neither interpretation is asserted to
    be an approved empirical SHIBOR curve construction merely by calling this API.
    """
    if any(np.iscomplexobj(item) for item in (node_days, quoted_percent, target_days)):
        raise ValueError("Rate inputs must be real, not complex")
    days = np.asarray(node_days, dtype=float)
    levels = np.asarray(quoted_percent, dtype=float)
    target = float(target_days)
    if days.ndim != 1 or levels.shape != days.shape or len(days) < 2:
        raise ValueError("Curve nodes/levels must be matching one-dimensional arrays")
    if not np.isfinite(days).all() or not np.isfinite(levels).all() or not math.isfinite(target):
        raise ValueError("Curve nodes/levels/target must be finite; missing tenors are not dropped")
    if np.any(days <= 0) or np.any(np.diff(days) <= 0) or target <= 0:
        raise ValueError("Require positive ordered node days and positive target days")
    if target < days[0] or target > days[-1]:
        raise ValueError("TENOR_OUT_OF_RANGE: curve extrapolation is disabled")
    if interpretation not in {"unapproved", "zero_continuous", "simple_act360"}:
        raise ValueError("Unknown interest-rate interpretation")
    decimals = levels / 100.0
    quoted = float(np.interp(target, days, decimals))
    zero = None
    if interpretation == "zero_continuous":
        zero = quoted
    elif interpretation == "simple_act360":
        accumulation = 1 + decimals * days / 360.0
        if np.any(accumulation <= 0) or not np.isfinite(accumulation).all():
            raise ValueError("Invalid simple-interest accumulation factor")
        zeros = np.log1p(decimals * days / 360.0) / (days / 365.0)
        zero = float(np.interp(target, days, zeros))
    return {"shibor_quote_decimal": quoted, "zero_rate": zero,
            "target_days": target, "time_to_expiry": target/365.0,
            "interpretation": interpretation, "quote_unit": "percent_input_decimal_output"}


@dataclass(frozen=True)
class RatePolicy:
    interpretation: str = "unapproved"
    dividend_yield: float | None = None
    approval_reference: str | None = None
    tenor_calendar_approved: bool = False

    def __post_init__(self):
        if self.interpretation not in {"unapproved", "zero_continuous", "simple_act360"}:
            raise ValueError("Unknown rate interpretation")
        if self.dividend_yield is not None and not math.isfinite(float(self.dividend_yield)):
            raise ValueError("Dividend yield must be a finite continuous annual decimal")
        if not isinstance(self.tenor_calendar_approved, bool):
            raise ValueError("tenor_calendar_approved must be an explicit boolean")
        if self.approval_reference is not None and not isinstance(self.approval_reference, str):
            raise ValueError("approval_reference must be a string or None")

    @property
    def approved(self):
        return (self.interpretation != "unapproved" and self.dividend_yield is not None
                and bool(self.approval_reference) and self.tenor_calendar_approved)


def align_rate(observation_date, expiry_date, rates, *, policy=RatePolicy(), node_days=None):
    """Match the SAME date only. No future fixings, carry-forward or invented nodes."""
    date, expiry = date_value(observation_date), date_value(expiry_date)
    result = {"rate_date": None, "rate_status": "MISSING_RATE_DATE", "shibor_quote_decimal": None,
              "zero_rate": None, "time_to_expiry": None, "discount_factor": None,
              "dividend_yield": policy.dividend_yield, "conventions_approved": False,
              "rate_approval_reference": policy.approval_reference}
    if not {"observation_date", *TENORS} <= set(rates.columns):
        result["rate_status"] = "INVALID_RATE_TABLE_SCHEMA"
        return result
    if date is None or expiry is None:
        result["rate_status"] = "MISSING_DATE_OR_EXPIRY"
        return result
    days = (pd.Timestamp(expiry)-pd.Timestamp(date)).days
    result["time_to_expiry"] = days / 365.0
    if days <= 0:
        result["rate_status"] = "NONPOSITIVE_TIME_TO_EXPIRY"
        return result
    matches = rates[rates["observation_date"] == date]
    if len(matches) != 1:
        result["rate_status"] = "MISSING_RATE_DATE" if not len(matches) else "DUPLICATE_RATE_DATE"
        return result
    result["rate_date"] = date
    try:
        numeric = pd.to_numeric(matches.iloc[0][list(TENORS)], errors="coerce")
        if np.iscomplexobj(numeric.to_numpy()):
            raise ValueError("Rate quotes must be real")
        levels = numeric.to_numpy(dtype=float)
        if not np.isfinite(levels).all():
            result["rate_status"] = "MISSING_TENOR_QUOTES"
            return result
        nodes = tenor_calendar_days(date) if node_days is None else np.asarray(node_days, dtype=float)
        if len(nodes) != len(TENORS):
            raise ValueError("Require all eight SHIBOR tenor nodes")
        calculated = interpolate_shibor(nodes, levels, days, interpretation=policy.interpretation)
        result.update({name: calculated[name] for name in ["shibor_quote_decimal", "time_to_expiry"]})
        # Even explicitly requested scenario rates are not an approved discount
        # curve without approval and externally supplied verified tenor dates.
        approved = policy.approved and node_days is not None
        result["conventions_approved"] = approved
        if approved:
            zero = calculated["zero_rate"]
            discount = math.exp(-zero * days / 365.0)
            if discount <= 0 or not math.isfinite(discount):
                raise ValueError("Discount factor is not representable")
            result.update(zero_rate=zero, discount_factor=discount, rate_status="APPROVED")
        else:
            result["rate_status"] = "INDICATIVE_UNAPPROVED_CONVENTIONS"
    except (ValueError, OverflowError) as error:
        result.update(conventions_approved=False, zero_rate=None, discount_factor=None)
        result["rate_status"] = "TENOR_OUT_OF_RANGE" if "TENOR_OUT_OF_RANGE" in str(error) else "INVALID_RATE_CURVE"
    return result
