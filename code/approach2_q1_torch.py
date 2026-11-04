"""Approach B Part 1: CPU float64 differentiable SVI; no neural surface in M5."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from q1.approach_a import empirical_gate, plain
from q1.classical_svi import CalibrationConfig
from q1.market_data import load_registered_q1
from q1.torch_comparison import compare_quotes, comparison_metrics
from q1.torch_comparison_figures import comparison_figures
from q1.torch_svi import TorchOptimizerConfig
from q1.verification_fixtures import development_quotes


ROOT = Path(__file__).resolve().parents[1]


def save_comparison(output, metrics, *, mode):
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must remain repository-local")
    output.mkdir(parents=True, exist_ok=False)
    comparison = metrics["history"]["comparison"]
    if comparison is not None:
        metrics["history"]["figures"] = comparison_figures(output, comparison, mode=mode)
        rows = comparison["comparison_rows"]
        with (output/"optimizer_comparison.csv").open("x", encoding="utf-8", newline="") as handle:
            fields = ["algorithm", "slice_id", "converged", "failure_reason", "objective_value", "objective_difference_from_scipy",
                      "native_iterations", "native_scipy_nfev", "objective_evaluations", "total_scipy_residual_evaluations", "gradient_evaluations",
                      "analytic_residual_jacobian_evaluations", "closure_calls", "fit_seconds", "optimizer_setup_seconds", "optimization_seconds"]
            writer = csv.DictWriter(handle, fieldnames=["origin", *fields], lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({"origin": "SYNTHETIC_VERIFICATION" if mode == "development" else "EMPIRICAL_SAMPLE", **{name: row[name] for name in fields}})
        with (output/"parameter_comparison.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump({"mode": mode, "rows": rows}, handle, indent=2, allow_nan=False)
    elif metrics["history"]["eligibility"] is not None:
        with (output/"eligibility_report.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(metrics["history"]["eligibility"], handle, indent=2, allow_nan=False)
    with (output/"metrics.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(plain(metrics), handle, ensure_ascii=False, indent=2, allow_nan=False)
    artifacts = [{"name": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                 for path in sorted(output.iterdir()) if path.is_file()]
    with (output/"artifact_manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump({"mode": mode, "dataset": metrics["params"]["dataset"], "files": artifacts}, handle, indent=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="q1_510050")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    parser.add_argument("--mode", choices=["development", "empirical"], default="empirical")
    parser.add_argument("--component", choices=["svi"], default="svi", help="M5 implements differentiable SVI only; neural integration is a later milestone")
    parser.add_argument("--adam-lr", type=float, default=0.03)
    parser.add_argument("--adam-max-steps", type=int, default=10000)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-max-iter", type=int, default=1000)
    args = parser.parse_args(argv)
    dataset = "q1_synthetic_svi_optimizer_comparison" if args.mode == "development" else "q1_510050"
    if args.dataset != dataset:
        parser.error("Mode/dataset origin mismatch; use "+dataset)
    if not 0 <= args.seed < 2**32:
        parser.error("seed must lie in [0,2**32)")
    config = CalibrationConfig()
    try:
        optimizer_config = TorchOptimizerConfig(adam_lr=args.adam_lr, adam_max_steps=args.adam_max_steps,
                                                lbfgs_lr=args.lbfgs_lr, lbfgs_max_iter=args.lbfgs_max_iter)
    except ValueError as error:
        parser.error(str(error))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    base = args.output_dir.resolve()
    if not base.is_relative_to(ROOT):
        parser.error("Output must remain inside the repository")
    output = base/"verification"/"approach2"/dataset if args.mode == "development" else base/"approach2"/dataset
    if output.exists():
        parser.error("Run already exists; use a new output directory without overwriting prior evidence")
    fixture = eligibility = coverage = result = None
    if args.mode == "development":
        frame, fixture = development_quotes()
        result = compare_quotes(frame, mode=args.mode, config=config, optimizer_config=optimizer_config, fixture=fixture)
    else:
        loaded = load_registered_q1()
        coverage = loaded["coverage"]
        frame, eligibility = empirical_gate(loaded["quotes"])
        if len(frame):
            result = compare_quotes(frame, mode=args.mode, config=config, optimizer_config=optimizer_config)
    metrics = comparison_metrics(result, mode=args.mode, dataset=dataset, seed=args.seed, config=config,
                                  optimizer_config=optimizer_config, eligibility=eligibility, fixture=fixture, coverage=coverage)
    if result is None:
        software, empirical, code = "PARTIAL", "BLOCKED", 2
    elif args.mode == "development":
        # Surface accuracy and convergence are separate from raw parameter error;
        # all actual differences remain in the machine-readable comparison table.
        passed = all(info["successful_slices"] == len(fixture["truth_by_expiry"])
                     and info["maximum_known_variance_grid_error"] is not None
                     and info["maximum_known_variance_grid_error"] <= 1e-6
                     and info["maximum_known_parameter_error"] <= 1e-4 for info in result["summary"].values())
        software, empirical, code = ("PASS" if passed else "PARTIAL"), "BLOCKED", (0 if passed else 1)
    else:
        passed = all(not info["failed_slice_ids"] for info in result["summary"].values())
        software = "PASS" if passed else "PARTIAL"
        empirical = "PASS" if coverage["examination_historical_requirement_satisfied"] else "PARTIAL"
        code = 0 if passed else 1
    metrics["history"].update(SOFTWARE_VERIFICATION=software, EMPIRICAL_REQUIREMENTS=empirical,
                              surface_verification_tolerance=1e-6, parameter_verification_tolerance=1e-4,
                              neural_surface_implemented=False)
    save_comparison(output, metrics, mode=args.mode)
    console = Console()
    console.print("[bold red]"+("SYNTHETIC CPU OPTIMIZER VERIFICATION — NOT EMPIRICAL FINDINGS" if args.mode == "development" else "STRICT EMPIRICAL ELIGIBILITY GATE")+"[/bold red]")
    report = Table("Algorithm", "Successful/attempted", "Max RSS", "Fit seconds")
    for name, info in (result["summary"].items() if result is not None else []):
        report.add_row(name, f"{info['successful_slices']}/{info['attempted_slices']}", str(info["maximum_final_objective"]), f"{info['fit_seconds']:.6f}")
    console.print(report)
    console.print(f"SOFTWARE_VERIFICATION: {software}; EMPIRICAL_REQUIREMENTS: {empirical}")
    console.print(str(output))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
