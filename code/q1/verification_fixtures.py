"""Explicit synthetic M4 development inputs, never an empirical data source."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from . import pricing, svi


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/q1/fixtures/synthetic/m4_svi_verification.json"
SYNTHETIC_ORIGIN = "SYNTHETIC_M4_SOFTWARE_VERIFICATION_ONLY"


def development_quotes():
    payload = FIXTURE.read_bytes()
    fixture = json.loads(payload)
    if fixture["provenance"] != SYNTHETIC_ORIGIN or fixture["historical_market_data"] is not False:
        raise ValueError("Development fixture is not explicitly labelled synthetic")
    grid = fixture["training_log_moneyness"]
    k = svi.fixed_log_moneyness_grid(grid["lower"], grid["upper"], grid["points"])
    base = fixture["base_svi"]
    rows, truth = [], {}
    for days, scale in zip(fixture["expiry_day_offsets"], fixture["variance_scales"]):
        t = days/365.0
        expiry = (pd.Timestamp(fixture["observation_date"])+pd.Timedelta(days=days)).date().isoformat()
        p = svi.SVIParameters(base["a"]*scale, base["b"]*scale, base["rho"], base["m"], base["eta"])
        truth[expiry] = asdict(p)
        forward = pricing.forward_from_spot(fixture["spot"], fixture["continuous_rate"], fixture["continuous_dividend_yield"], t)
        discount = pricing.discount_factor(fixture["continuous_rate"], t)
        for index, location in enumerate(k):
            strike = forward*math.exp(location)
            sigma = math.sqrt(svi.svi_total_variance(location, p)/t)
            for kind in ["C", "P"]:
                price = pricing.black_scholes(forward, strike, t, sigma, discount, kind)
                half = fixture["synthetic_relative_bid_ask_half_width"]
                rows.append({"quote_id": f"SYNTHETIC_{days}_{index}_{kind}", "contract_id": f"SYNTHETIC_{days}_{index}_{kind}",
                    "data_origin": SYNTHETIC_ORIGIN, "eligible_for_empirical": False,
                    "observation_date": fixture["observation_date"], "expiry_date": expiry,
                    "option_type": kind, "strike": strike, "contract_unit": 10000.0,
                    "price": price, "spot": fixture["spot"], "forward": forward, "discount_factor": discount,
                    "time_to_expiry": t, "zero_rate": fixture["continuous_rate"],
                    "dividend_yield": fixture["continuous_dividend_yield"], "volume": fixture["synthetic_volume"],
                    "bid": price*(1-half), "ask": price*(1+half), "price_kind": "synthetic_verification_price"})
    return pd.DataFrame(rows), {"fixture_path": FIXTURE.relative_to(ROOT).as_posix(),
        "fixture_sha256": hashlib.sha256(payload).hexdigest(), "specification": fixture, "truth_by_expiry": truth}
