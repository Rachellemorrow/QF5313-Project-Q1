"""Matched-cohort CPU SVI comparison, with unchanged M4 SciPy algorithms."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import platform
from time import perf_counter
from unittest.mock import patch

import numpy as np
import scipy
import torch

from . import approach_a, classical_svi, svi
from .approach_a import calibrate_quotes, plain
from .classical_svi import CalibrationConfig, reprice_quote, repricing_summary, verify_surface
from .torch_svi import TorchOptimizerConfig, fit_torch_slice


def input_fingerprint(k, target, weights):
    digest = hashlib.sha256()
    for item in (k, target, weights):
        array = np.asarray(item, dtype="<f8")
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@contextmanager
def scipy_timing_trace():
    """Serial-only transparent residual timing; objective/Jacobian/solver unchanged.

    Records the actual extra post-fit residual call separately from native nfev.
    This is instrumentation overhead, not an uninstrumented benchmark claim.
    """
    original = classical_svi.weighted_residual
    original_fit = approach_a.fit_slice
    active = {"entries": None}

    def traced(theta, k, target, weights):
        result = original(theta, k, target, weights)
        if active["entries"] is not None:
            active["entries"].append({"absolute_perf_time": perf_counter(), "weighted_rss": float(result@result)})
        return result

    def traced_fit(*args, **kwargs):
        entries = []
        active["entries"] = entries
        try:
            fit = original_fit(*args, **kwargs)
        finally:
            active["entries"] = None
        n = fit["objective_evaluations"]
        fit["comparison_history"] = [{"evaluation": i+1, "iteration": None, "weighted_rss": row["weighted_rss"],
                                      "elapsed_since_first_evaluation_seconds": row["absolute_perf_time"]-entries[0]["absolute_perf_time"]}
                                     for i, row in enumerate(entries[:n])]
        fit["native_iterations"] = None
        fit["iteration_note"] = "M4 least_squares does not expose accepted optimizer iterations; nfev is not relabelled as iterations"
        fit["total_residual_evaluations_including_postcheck"] = len(entries)
        fit["post_fit_residual_evaluations"] = max(0, len(entries)-n)
        return fit

    with patch.object(classical_svi, "weighted_residual", traced), patch.object(approach_a, "fit_slice", traced_fit):
        yield


def compare_quotes(frame, *, mode, config=CalibrationConfig(), optimizer_config=TorchOptimizerConfig(), fixture=None):
    """One shared M4 preparation/baseline; Torch gets its exact targets/start/weights."""
    with scipy_timing_trace():
        baseline = calibrate_quotes(frame, mode=mode, config=config)
    algorithms = {"scipy": baseline}
    matched = []
    grouped = {}
    for record in baseline["quote_ledger"]:
        if record["eligible_for_fit"]:
            key = (record["input"]["observation_date"], record["input"]["expiry_date"])
            grouped.setdefault(key, []).append(record)
    torch_slices = {"adam": [], "lbfgs": []}
    for source in baseline["slices"]:
        rows = grouped.get((source["observation_date"], source["expiry_date"]), [])
        k = np.array([row["k"] for row in rows], dtype=np.float64)
        target = np.array([row["total_variance"] for row in rows], dtype=np.float64)
        if source["fit"]["weights"] is not None:
            weights = np.array(source["fit"]["weights"], dtype=np.float64)
        else:
            counts = Counter(row["input"]["strike"] for row in rows)
            weights = np.array([1/counts[row["input"]["strike"]] for row in rows], dtype=np.float64)
        key = input_fingerprint(k, target, weights)
        fit = source["fit"]
        fit["comparison_input_sha256"] = key
        initial = svi.SVIParameters(**fit["initial_parameters"]) if fit["initial_parameters"] is not None else None
        matched.append({"slice_id": source["slice_id"], "input_sha256": key, "quote_count": len(k),
                        "weights": weights.tolist(), "raw_initial_parameters": asdict(initial) if initial else None})
        for name in ["adam", "lbfgs"]:
            fitted = fit_torch_slice(k, target, weights=weights, initial=initial, optimizer=name,
                                     config=config, optimizer_config=optimizer_config)
            fitted["comparison_input_sha256"] = key
            item = {name: value for name, value in source.items() if name not in {"fit", "repricing"}}
            item.update(fit=fitted, repricing=[])
            if fitted["converged"]:
                p = svi.SVIParameters(**fitted["fitted_parameters"])
                item["repricing"] = [reprice_quote(row["input"], p, price_floor=config.near_zero_price_floor) for row in rows]
            torch_slices[name].append(item)
    for name, slices in torch_slices.items():
        algorithms[name] = {"slices": slices, "arbitrage": verify_surface(slices, config),
                            "repricing_summary": repricing_summary(slices)}
    summary, comparison = {}, []
    for name, result in algorithms.items():
        valid = [item for item in result["slices"] if item["fit"]["converged"]]
        failures = [item["slice_id"] for item in result["slices"] if not item["fit"]["converged"]]
        objectives = [item["fit"]["objective_value"] for item in result["slices"] if item["fit"]["objective_value"] is not None]
        successful_objectives = [item["fit"]["objective_value"] for item in valid]
        grid_errors, parameter_errors = [], []
        for item in result["slices"]:
            fit = item["fit"]
            grid_error = parameter_error = None
            if fixture and fit["fitted_parameters"] is not None:
                truth = fixture["truth_by_expiry"][item["expiry_date"]]
                p, expected = svi.SVIParameters(**fit["fitted_parameters"]), svi.SVIParameters(**truth)
                grid_error = float(np.max(np.abs(svi.svi_total_variance(config.grid(), p, require_positive=False)-svi.svi_total_variance(config.grid(), expected))))
                parameter_error = max(abs(fit["fitted_parameters"][field]-truth[field]) for field in truth)
                grid_errors.append(grid_error)
                parameter_errors.append(parameter_error)
            scipy_fit = next(row["fit"] for row in baseline["slices"] if row["slice_id"] == item["slice_id"])
            to_scipy = None
            surface_to_scipy = None
            if fit["fitted_parameters"] is not None and scipy_fit["fitted_parameters"] is not None:
                to_scipy = {field: fit["fitted_parameters"][field]-scipy_fit["fitted_parameters"][field] for field in fit["fitted_parameters"]}
                p = svi.SVIParameters(**fit["fitted_parameters"])
                baseline_p = svi.SVIParameters(**scipy_fit["fitted_parameters"])
                surface_to_scipy = float(np.max(np.abs(svi.svi_total_variance(config.grid(), p, require_positive=False)-svi.svi_total_variance(config.grid(), baseline_p, require_positive=False))))
            comparison.append({"algorithm": name, "slice_id": item["slice_id"], "converged": fit["converged"],
                "failure_reason": fit["failure_reason"], "fitted_parameters": fit["fitted_parameters"],
                "initial_parameters": fit["initial_parameters"], "objective_value": fit["objective_value"],
                "objective_difference_from_scipy": fit["objective_value"]-scipy_fit["objective_value"] if fit["objective_value"] is not None and scipy_fit["objective_value"] is not None else None,
                "parameter_difference_from_scipy": to_scipy,
                "variance_grid_max_abs_difference_from_scipy": surface_to_scipy,
                "native_iterations": fit.get("iterations"), "native_scipy_nfev": fit.get("nfev"),
                "objective_evaluations": fit["objective_evaluations"],
                "total_scipy_residual_evaluations": fit.get("total_residual_evaluations_including_postcheck"),
                "gradient_evaluations": fit.get("gradient_evaluations"), "analytic_residual_jacobian_evaluations": fit.get("njev"),
                "closure_calls": fit.get("lbfgs_closure_calls"), "fit_seconds": fit["fit_seconds"],
                "optimizer_setup_seconds": fit.get("optimizer_setup_seconds"), "optimization_seconds": fit.get("optimization_seconds"),
                "known_variance_grid_max_abs_error": grid_error, "known_parameter_max_abs_error": parameter_error})
        summary[name] = {"attempted_slices": len(result["slices"]), "successful_slices": len(valid), "failed_slice_ids": failures,
            "convergence_success_rate": len(valid)/len(result["slices"]) if result["slices"] else None,
            "maximum_final_objective": max(objectives) if objectives else None,
            "maximum_successful_final_objective": max(successful_objectives) if successful_objectives else None,
            "maximum_candidate_errors_include_nonconverged_fits": True,
            "maximum_known_variance_grid_error": max(grid_errors) if grid_errors else None,
            "maximum_known_parameter_error": max(parameter_errors) if parameter_errors else None,
            "fit_seconds": sum(item["fit"]["fit_seconds"] for item in result["slices"]),
            "repricing": result["repricing_summary"],
            "arbitrage": {"examination_butterfly_violations": sum(item["examination"]["count"] for item in result["arbitrage"]["slices"] if item["examination"]),
                          "complete_butterfly_violations": sum(item["complete"]["count"] for item in result["arbitrage"]["slices"] if item["complete"]),
                          "calendar_violations": sum(item["count"] for item in result["arbitrage"]["calendar"]),
                          "negative_forward_variance_locations": len(result["arbitrage"]["negative_forward_variance"]),
                          "failed_slices_without_diagnostics": result["arbitrage"]["failed_slice_ids"]}}
    return {"algorithms": algorithms, "matched_inputs": matched, "comparison_rows": comparison,
            "summary": summary, "quote_ledger": baseline["quote_ledger"], "parity": baseline["parity"],
            "precision": "float64", "device": "cpu", "torch_num_threads": torch.get_num_threads(),
            "comparison_caveats": ["Smooth feasible domain excludes SciPy boundary faces; not perfectly equivalent",
                "Stopping criteria and coordinate scalings differ; nominal tolerances are not interchangeable",
                "One fixed-order run of four synthetic slices is not a robust performance benchmark",
                "Timing includes logging overhead; first-use Torch optimizer setup is reported and not hidden",
                "SciPy wall-time trace excludes pre-first-residual setup; objective plots distinguish evaluations from iterations"]}


def comparison_metrics(result, *, mode, dataset, seed, config, optimizer_config, eligibility=None, fixture=None, coverage=None):
    summary = result["summary"] if result is not None else {}
    return plain({"params": {"dataset": dataset, "mode": mode, "seed": seed, "device": "cpu", "dtype": "float64",
                             "data_origin": "SYNTHETIC_M5_OPTIMIZER_VERIFICATION_USING_M4_FIXTURE" if mode == "development" else "REAL_ARCHIVED_OBSERVATION",
                             "calibration_config": asdict(config), "torch_optimizer_config": asdict(optimizer_config),
                             "fixture": fixture, "pipeline_coverage": coverage,
                             "empirical_comparison_eligible": mode == "empirical" and any(item["successful_slices"] for item in summary.values()),
                             "split_protocol": "Same M4 dated-expiry slices; matched in-sample quotes; no random holdout",
                             "perfectly_controlled_optimization_comparison": False},
                  "metrics": summary, "fit_seconds": sum(item["fit_seconds"] for item in summary.values()),
                  "history": {"status": "BLOCKED" if result is None else "COMPLETED_WITH_FAILURES" if any(item["failed_slice_ids"] for item in summary.values()) else "COMPLETED",
                              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                                          "torch": torch.__version__, "platform": platform.platform(), "machine": platform.machine(),
                                          "processor": platform.processor(), "torch_num_threads": torch.get_num_threads()},
                              "determinism": "No stochastic initialization or mini-batching; seeded CPU float64. Wall times vary; bitwise portability across library/CPU/thread configurations is not guaranteed.",
                              "eligibility": eligibility, "comparison": result, "figures": [],
                              "historical_requirements_satisfied": bool(mode == "empirical" and coverage and coverage.get("examination_historical_requirement_satisfied"))}})
