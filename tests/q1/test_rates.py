"""Synthetic rate/convention verification. No market acquisition or curve approval."""

import math

import numpy as np
import pandas as pd
import pytest

from q1.rates import RatePolicy, TENORS, align_rate, date_value, interpolate_shibor, tenor_calendar_days


def rates_table(date="2000-01-04", level=2):
    return pd.DataFrame([{"observation_date": date, **{tenor: level for tenor in TENORS}}])


def test_strict_date_normalization_and_missing_dates():
    assert date_value("20261030") == "2026-10-30"
    assert date_value("2026-10-30") == "2026-10-30"
    assert date_value("2026-02-30") is None
    assert date_value("01/02/2026") is None
    assert date_value(None) is None


def test_calendar_offsets_are_labelled_indicative_not_business_day_dates():
    actual = tenor_calendar_days("2000-01-04")
    np.testing.assert_array_equal(actual[:3], [1, 7, 14])
    assert actual[3] == 31 and actual[-1] == 366


def test_percent_conversion_and_linear_quote_interpolation():
    result = interpolate_shibor([10, 30, 90], [1, 2, 3], 60)
    assert result["shibor_quote_decimal"] == pytest.approx(0.025)
    assert result["zero_rate"] is None
    explicit = interpolate_shibor([10, 30, 90], [1, 2, 3], 60, interpretation="zero_continuous")
    assert explicit["zero_rate"] == pytest.approx(0.025)


def test_simple_act360_to_continuous_act365_conversion():
    result = interpolate_shibor([30, 365], [2, 3], 365, interpretation="simple_act360")
    assert result["zero_rate"] == pytest.approx(math.log1p(0.03*365/360), abs=1e-15)
    assert math.exp(-result["zero_rate"]) == pytest.approx(1/(1+0.03*365/360), abs=1e-15)


@pytest.mark.parametrize("days,levels,target", [
    ([30, 90], [1, np.nan], 60), ([30, 30], [1, 2], 30), ([90, 30], [1, 2], 60),
    ([30, 90], [1, 2], 10), ([30, 90], [1, 2], 100), ([30, 90], [1], 60),
    ([30, 90], [1, 2], 0), ([30, 90], [1, 2], np.inf),
    ([30, 90], [1+1j, 2], 60),
])
def test_invalid_curve_inputs_and_no_extrapolation(days, levels, target):
    with pytest.raises(ValueError):
        interpolate_shibor(days, levels, target)


def test_negative_rates_are_supported_not_clipped():
    result = interpolate_shibor([30, 365], [-1, -1], 365, interpretation="simple_act360")
    assert result["zero_rate"] < 0
    assert math.exp(-result["zero_rate"]) > 1
    with pytest.raises(ValueError):
        interpolate_shibor([30, 365], [-2000, -2000], 365, interpretation="simple_act360")


def test_no_default_curve_or_dividend_approval():
    result = align_rate("2000-01-04", "2000-04-04", rates_table())
    assert result["rate_status"] == "INDICATIVE_UNAPPROVED_CONVENTIONS"
    assert result["shibor_quote_decimal"] == pytest.approx(0.02)
    assert result["discount_factor"] is None and result["zero_rate"] is None
    assert result["dividend_yield"] is None and not result["conventions_approved"]


def test_explicit_approved_convention_requires_supplied_verified_nodes():
    policy = RatePolicy("zero_continuous", 0.01, "SYNTHETIC_TEST_APPROVAL_ONLY", True)
    unverified = align_rate("2000-01-04", "2000-04-04", rates_table(), policy=policy)
    assert not unverified["conventions_approved"]
    nodes = [1, 7, 14, 31, 91, 182, 273, 365]
    result = align_rate("2000-01-04", "2000-04-04", rates_table(), policy=policy, node_days=nodes)
    assert result["rate_status"] == "APPROVED"
    assert result["discount_factor"] == pytest.approx(math.exp(-0.02*91/365))


def test_no_future_previous_or_duplicate_rate_join():
    assert align_rate("2000-01-05", "2000-04-04", rates_table())["rate_status"] == "MISSING_RATE_DATE"
    duplicate = pd.concat([rates_table(), rates_table()], ignore_index=True)
    assert align_rate("2000-01-04", "2000-04-04", duplicate)["rate_status"] == "DUPLICATE_RATE_DATE"
    missing = rates_table()
    missing.loc[0, "3M"] = np.nan
    assert align_rate("2000-01-04", "2000-04-04", missing)["rate_status"] == "MISSING_TENOR_QUOTES"


def test_bad_expiry_and_outside_curve_are_reported():
    assert align_rate("2000-01-04", None, rates_table())["rate_status"] == "MISSING_DATE_OR_EXPIRY"
    assert align_rate("2000-01-04", "2000-01-04", rates_table())["rate_status"] == "NONPOSITIVE_TIME_TO_EXPIRY"
    assert align_rate("2000-01-04", "2002-01-04", rates_table())["rate_status"] == "TENOR_OUT_OF_RANGE"


def test_invalid_rate_schema_or_approval_never_leaves_approved_flags():
    assert align_rate("2000-01-04", "2000-04-04", rates_table().drop(columns=["3M"]))["rate_status"] == "INVALID_RATE_TABLE_SCHEMA"
    with pytest.raises(ValueError):
        RatePolicy(tenor_calendar_approved="False")
    policy = RatePolicy("zero_continuous", 0.0, "SYNTHETIC_TEST_APPROVAL_ONLY", True)
    failed = align_rate("2000-01-04", "2000-04-04", rates_table(level=-1e10), policy=policy,
                        node_days=[1, 7, 14, 31, 91, 182, 273, 365])
    assert failed["rate_status"] == "INVALID_RATE_CURVE"
    assert not failed["conventions_approved"] and failed["discount_factor"] is None
