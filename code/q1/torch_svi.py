"""Differentiable float64 SVI and honestly counted CPU optimizer runs.

The smooth transformation covers the INTERIOR of M4's feasible domain, not its
closed boundaries. Initial boundary values are rejected, never silently nudged.
No neural surface, market-data loading, arbitrage repair or pricing duplication.
"""

from dataclasses import asdict, dataclass
import math
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from . import svi
from .classical_svi import CalibrationConfig, encode_parameters


@dataclass(frozen=True)
class TorchOptimizerConfig:
    adam_lr: float = 0.03
    adam_max_steps: int = 10000
    adam_decay_interval: int = 2000
    adam_decay_factor: float = 0.5
    adam_eps: float = 1e-12
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 1000
    lbfgs_max_eval: int = 2000
    lbfgs_history_size: int = 50
    lbfgs_tolerance_grad: float = 1e-12
    lbfgs_tolerance_change: float = 1e-18
    objective_tolerance: float = 1e-14
    physical_gradient_tolerance: float = 1e-10

    def __post_init__(self):
        for name in ["adam_max_steps", "adam_decay_interval", "lbfgs_max_iter", "lbfgs_max_eval", "lbfgs_history_size"]:
            item = getattr(self, name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ["adam_lr", "adam_eps", "lbfgs_lr", "lbfgs_tolerance_grad", "lbfgs_tolerance_change", "objective_tolerance", "physical_gradient_tolerance"]:
            item = getattr(self, name)
            if not math.isfinite(item) or item <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.adam_decay_factor <= 1:
            raise ValueError("Adam decay factor must be in (0,1]")


def physical_from_latent(latent, config=CalibrationConfig()):
    """Return differentiable (v_min,b,rho,m,eta), matching M4 coordinate meanings."""
    v_min = config.positive_floor + functional.softplus(latent[0])
    b = functional.softplus(latent[1])
    rho = (1-config.rho_margin)*torch.tanh(latent[2])
    eta = config.eta_floor + functional.softplus(latent[4])
    return torch.stack((v_min, b, rho, latent[3], eta))


def raw_from_physical(physical):
    v_min, b, rho, m, eta = physical.unbind()
    a = v_min-b*eta*torch.sqrt((1-rho)*(1+rho))
    return torch.stack((a, b, rho, m, eta))


def raw_from_latent(latent, config=CalibrationConfig()):
    return raw_from_physical(physical_from_latent(latent, config))


def latent_from_raw(parameters, config=CalibrationConfig()):
    physical = encode_parameters(parameters)
    v_min, b, rho, m, eta = physical
    positive = [v_min-config.positive_floor, b, eta-config.eta_floor]
    ratio = rho/(1-config.rho_margin)
    if any(not math.isfinite(float(x)) or x <= 0 for x in positive) or abs(ratio) >= 1:
        raise ValueError("INITIAL_ON_TRANSFORM_BOUNDARY: smooth inverse requires strict interior parameters")
    inverse = lambda x: x+math.log(-math.expm1(-x))
    values = [inverse(positive[0]), inverse(b), math.atanh(ratio), m, inverse(positive[2])]
    return torch.tensor(values, dtype=torch.float64, device="cpu")


def total_variance_torch(k, parameters):
    """Raw SVI with pure torch operations; gradients flow in k and all parameters."""
    a, b, rho, m, eta = parameters.unbind()
    x = k-m
    return a+b*(rho*x+torch.hypot(x, eta))


def objective_torch(latent, k, target, weights, config=CalibrationConfig()):
    w = total_variance_torch(k, raw_from_latent(latent, config))
    return torch.sum(weights*(w-target).square())


class DifferentiableSVI(nn.Module):
    def __init__(self, initial, config=CalibrationConfig()):
        super().__init__()
        self.config = config
        self.latent = nn.Parameter(latent_from_raw(initial, config))

    def forward(self, k):
        return total_variance_torch(k, raw_from_latent(self.latent, self.config))

    def parameters_raw(self):
        values = raw_from_latent(self.latent, self.config).detach().cpu().tolist()
        return svi.SVIParameters(*values)


def fit_torch_slice(k, observed_w, *, weights, initial, optimizer="adam",
                    config=CalibrationConfig(), optimizer_config=TorchOptimizerConfig()):
    """Run one CPU solver, counting initial, closure and final checks explicitly.

    Adam history records objective/gradient evaluations at actual update counts.
    L-BFGS history includes line-search trial closure evaluations labelled with its
    observed internal n_iter counter; these are NOT invented accepted iterates.
    """
    start = perf_counter()
    record = {"optimizer": optimizer, "device": "cpu", "dtype": "float64", "converged": False,
        "failure_reason": None, "message": None, "initial_parameters": asdict(initial) if initial is not None else None,
        "initial_reconstruction_max_abs_error": None, "fitted_parameters": None, "objective_value": None,
        "objective": "sum(weights*(SVI_total_variance-observed_total_variance)**2)",
        "weights": None, "iterations": 0, "outer_step_calls": 0, "objective_evaluations": 0,
        "gradient_evaluations": 0, "lbfgs_closure_calls": 0, "native_func_evals": None,
        "evaluation_history": [], "final_physical_half_rss_gradient_inf": None,
        "final_latent_rss_gradient_inf": None, "global_minimum_variance": None,
        "optimizer_config": asdict(optimizer_config), "fit_seconds": 0., "optimizer_setup_seconds": 0.,
        "optimization_seconds": 0., "quote_count": 0, "distinct_k_count": 0,
        "native_stop_reason": "NOT_EXPOSED_BY_PYTORCH",
        "feasible_domain_equivalent_to_scipy": False,
        "domain_caveat": "Smooth real-number image excludes b=0 and v_min/eta/rho safety-boundary endpoints allowed by SciPy; finite-precision saturation may reach numerical margins"}
    model = None
    opt = None
    optimizer_start = None
    try:
        if optimizer not in {"adam", "lbfgs"}:
            raise ValueError("optimizer must be adam or lbfgs")
        if any(np.iscomplexobj(item) for item in (k, observed_w, weights)):
            raise ValueError("Inputs must be real")
        k, target, weight = [np.asarray(item, dtype=np.float64) for item in (k, observed_w, weights)]
        if k.ndim != 1 or target.shape != k.shape or weight.shape != k.shape:
            raise ValueError("Require matching one-dimensional observations/weights")
        record.update(quote_count=len(k), distinct_k_count=len(np.unique(k)))
        if not all(np.isfinite(item).all() for item in (k, target, weight)) or np.any(target <= 0) or np.any(weight <= 0):
            raise ValueError("Require positive finite targets/weights and finite k")
        if len(np.unique(k)) < config.min_distinct_strikes:
            record.update(failure_reason="INSUFFICIENT_DISTINCT_STRIKES", message="Same minimum distinct-strike requirement as M4")
            return record
        if initial is None:
            raise ValueError("Matched raw SciPy initialization is required")
        record["weights"] = weight.tolist()
        model = DifferentiableSVI(initial, config)
        reconstructed = model.parameters_raw()
        record["initial_reconstruction_max_abs_error"] = max(abs(asdict(reconstructed)[name]-number) for name, number in asdict(initial).items())
        record["initial_latent_parameters"] = model.latent.detach().tolist()
        tk, tw, target = [torch.as_tensor(item, dtype=torch.float64, device="cpu") for item in (k, weight, target)]
        if optimizer == "adam":
            opt = torch.optim.Adam([model.latent], lr=optimizer_config.adam_lr, eps=optimizer_config.adam_eps)
        else:
            opt = torch.optim.LBFGS([model.latent], lr=optimizer_config.lbfgs_lr,
                max_iter=optimizer_config.lbfgs_max_iter, max_eval=optimizer_config.lbfgs_max_eval,
                tolerance_grad=optimizer_config.lbfgs_tolerance_grad,
                tolerance_change=optimizer_config.lbfgs_tolerance_change,
                history_size=optimizer_config.lbfgs_history_size, line_search_fn="strong_wolfe")
        record["optimizer_setup_seconds"] = perf_counter()-start
        optimizer_start = perf_counter()

        def evaluate(iteration, *, role):
            opt.zero_grad(set_to_none=True)
            physical = physical_from_latent(model.latent, config)
            physical.retain_grad()
            prediction = total_variance_torch(tk, raw_from_physical(physical))
            if not torch.isfinite(prediction).all() or torch.any(prediction <= 0):
                raise ValueError("Nonfinite/nonpositive variance during optimization; no repair applied")
            loss = torch.sum(tw*(prediction-target).square())
            record["objective_evaluations"] += 1
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite objective")
            loss.backward()
            record["gradient_evaluations"] += 1
            if not torch.isfinite(model.latent.grad).all() or not torch.isfinite(physical.grad).all():
                raise ValueError("Nonfinite gradient")
            history = {"evaluation": record["objective_evaluations"], "iteration": int(iteration), "role": role,
                "weighted_rss": float(loss.detach()), "latent_rss_gradient_inf": float(model.latent.grad.detach().abs().max()),
                "physical_half_rss_gradient_inf": float(physical.grad.detach().abs().max()/2),
                "optimization_elapsed_seconds": perf_counter()-optimizer_start,
                "learning_rate": float(opt.param_groups[0]["lr"])}
            record["evaluation_history"].append(history)
            return loss, history

        def reached(item):
            if item["weighted_rss"] <= optimizer_config.objective_tolerance:
                return "OBJECTIVE_TOLERANCE"
            if item["physical_half_rss_gradient_inf"] <= optimizer_config.physical_gradient_tolerance:
                return "PHYSICAL_GRADIENT_TOLERANCE"
            return None

        if optimizer == "adam":
            for iteration in range(optimizer_config.adam_max_steps+1):
                loss, last = evaluate(iteration, role="adam_update_boundary")
                stop = reached(last)
                if stop:
                    record.update(converged=True, message=stop)
                    break
                if iteration == optimizer_config.adam_max_steps:
                    break
                record["outer_step_calls"] += 1
                opt.step()
                record["iterations"] += 1
                if record["iterations"] % optimizer_config.adam_decay_interval == 0:
                    opt.param_groups[0]["lr"] *= optimizer_config.adam_decay_factor
        else:
            def closure():
                record["lbfgs_closure_calls"] += 1
                iteration = opt.state.get(model.latent, {}).get("n_iter", 0)
                return evaluate(iteration, role="lbfgs_closure_including_line_search_trials")[0]
            record["outer_step_calls"] = 1
            opt.step(closure)
            state = opt.state.get(model.latent, {})
            record["iterations"] = int(state["n_iter"]) if "n_iter" in state else None
            record["native_func_evals"] = int(state["func_evals"]) if "func_evals" in state else None
            _, last = evaluate(record["iterations"] or 0, role="post_optimizer_objective_and_gradient_check")
            stop = reached(last)
            record.update(converged=bool(stop), message=stop or "Native L-BFGS returned without meeting the declared post-check tolerance")
        record["optimization_seconds"] = perf_counter()-optimizer_start
        p = model.parameters_raw()
        record.update(fitted_parameters=asdict(p), objective_value=last["weighted_rss"],
            final_physical_half_rss_gradient_inf=last["physical_half_rss_gradient_inf"],
            final_latent_rss_gradient_inf=last["latent_rss_gradient_inf"],
            global_minimum_variance=float(svi.svi_global_minimum(p)))
        positive = svi.variance_positivity(np.r_[config.grid(), k], p)
        if not positive["positive_on_grid"] or not positive["positive_globally"]:
            record.update(converged=False, failure_reason="TOTAL_VARIANCE_POSITIVITY_FAILURE")
        elif not record["converged"]:
            record["failure_reason"] = "ITERATION_LIMIT_OR_NATIVE_STOP_WITHOUT_POSTCHECK_CONVERGENCE"
    except (ValueError, TypeError, RuntimeError, OverflowError, FloatingPointError) as error:
        record.update(converged=False, failure_reason="INVALID_INPUT_OR_NUMERICAL_FAILURE", message=str(error))
    finally:
        if optimizer == "lbfgs" and opt is not None and model is not None:
            state = opt.state.get(model.latent, {})
            record["iterations"] = int(state["n_iter"]) if "n_iter" in state else None
            record["native_func_evals"] = int(state["func_evals"]) if "func_evals" in state else None
        record["fit_seconds"] = perf_counter()-start
        if optimizer_start is not None and record["optimization_seconds"] == 0:
            record["optimization_seconds"] = perf_counter()-optimizer_start
    return record
