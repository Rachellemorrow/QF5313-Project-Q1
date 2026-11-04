"""Q1 classical SVI CLI: strict empirical mode or separately labelled verification."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from rich.console import Console
from rich.table import Table

from q1.approach_a import calibrate_quotes, empirical_gate, make_metrics, plain
from q1.approach_a_figures import write_figures
from q1.classical_svi import CalibrationConfig
from q1.market_data import load_registered_q1
from q1.verification_fixtures import development_quotes


ROOT = Path(__file__).resolve().parents[1]


def save_run(directory, metrics, result, *, mode):
    directory = directory.resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError("Output must remain inside the repository")
    directory.mkdir(parents=True, exist_ok=False)
    if result is not None:
        metrics["history"]["figures"] = write_figures(directory, result, mode=mode)
        with (directory/"diagnostics.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(plain(result), handle, ensure_ascii=False, indent=2, allow_nan=False)
    elif metrics["history"]["eligibility"] is not None:
        with (directory/"eligibility_report.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(metrics["history"]["eligibility"], handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (directory/"metrics.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)
    files = [{"name": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(directory.iterdir()) if path.is_file()]
    with (directory/"artifact_manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump({"mode": mode, "dataset": metrics["params"]["dataset"], "files": files}, handle, indent=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="q1_510050")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    parser.add_argument("--mode", choices=["development", "empirical"], default="empirical")
    parser.add_argument("--max-nfev", type=int, default=2000)
    args = parser.parse_args(argv)
    expected_dataset = "q1_synthetic_svi_verification" if args.mode == "development" else "q1_510050"
    if args.dataset != expected_dataset:
        parser.error(f"For {args.mode} mode use --dataset {expected_dataset}; origins cannot be mixed")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must lie in [0, 2**32)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    config = CalibrationConfig(max_nfev=args.max_nfev)
    base = args.output_dir.resolve()
    if not base.is_relative_to(ROOT):
        parser.error("--output-dir must stay inside the repository")
    directory = base/"verification"/"approach1"/args.dataset if args.mode == "development" else base/"approach1"/args.dataset
    if directory.exists():
        parser.error("Output run already exists; inspect it and use a new --output-dir (no overwrite)")
    fixture = coverage = eligibility = None
    result = None
    if args.mode == "development":
        frame, fixture = development_quotes()
        result = calibrate_quotes(frame, mode=args.mode, config=config)
    else:
        loaded = load_registered_q1()
        coverage = loaded["coverage"]
        frame, eligibility = empirical_gate(loaded["quotes"])
        if len(frame):
            result = calibrate_quotes(frame, mode=args.mode, config=config)
    metrics = make_metrics(result or {}, mode=args.mode, dataset=args.dataset, seed=args.seed,
                           config=config, input_rows=len(frame) if eligibility is None else eligibility["input_rows"],
                           eligibility=eligibility, fixture=fixture, coverage=coverage)
    blocked = args.mode == "empirical" and eligibility["eligible_rows"] == 0
    if args.mode == "development":
        specification = fixture["specification"]
        passed = (metrics["metrics"]["successful_slices"] == len(specification["expiry_day_offsets"])
                  and metrics["metrics"]["quote_preparation_rejections"] == 0
                  and metrics["metrics"]["max_known_parameter_absolute_error"] <= specification["recovery_parameter_absolute_tolerance"]
                  and metrics["metrics"]["max_known_variance_absolute_error"] <= specification["recovery_variance_absolute_tolerance"])
        metrics["history"]["SOFTWARE_VERIFICATION"] = "PASS" if passed else "PARTIAL"
        metrics["history"]["EMPIRICAL_REQUIREMENTS"] = "BLOCKED"
        code = 0 if passed else 1
    else:
        metrics["history"]["SOFTWARE_VERIFICATION"] = "PARTIAL" if blocked or metrics["metrics"]["failed_slices"] else "PASS"
        metrics["history"]["EMPIRICAL_REQUIREMENTS"] = "BLOCKED" if blocked else "PASS" if coverage["examination_historical_requirement_satisfied"] else "PARTIAL"
        code = 2 if blocked else 1 if metrics["metrics"]["failed_slices"] else 0
    save_run(directory, metrics, result, mode=args.mode)
    console = Console()
    console.print("[bold red]"+("SYNTHETIC SOFTWARE VERIFICATION — NOT EMPIRICAL MARKET RESULTS" if args.mode == "development" else "STRICT EMPIRICAL PIPELINE GATE")+"[/bold red]")
    table = Table("Attempted slices", "Successful", "Failed", "Empirical requirements")
    table.add_row(str(metrics["metrics"]["attempted_slices"]), str(metrics["metrics"]["successful_slices"]),
                  str(metrics["metrics"]["failed_slices"]), metrics["history"]["EMPIRICAL_REQUIREMENTS"])
    console.print(table)
    console.print(str(directory))
    if blocked:
        console.print("BLOCKED: zero strictly eligible empirical records; no calibration or empirical figures generated.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
