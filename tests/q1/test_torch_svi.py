"""Float64 model/objective/gradient verification; fixtures are not empirical data."""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from q1 import svi
from q1.classical_svi import CalibrationConfig, encode_parameters, residual_jacobian, weighted_residual
from q1.torch_svi import (DifferentiableSVI, TorchOptimizerConfig, fit_torch_slice, latent_from_raw,
                           objective_torch, physical_from_latent, raw_from_latent, raw_from_physical, total_variance_torch)


@pytest.fixture
def tensor_case(synthetic_cases):
    p = svi.SVIParameters(**synthetic_cases["svi_smooth"])
    k = np.linspace(-.4, .4, 41)
    target = svi.svi_total_variance(k, p)
    weights = np.linspace(.5, 2., len(k))
    return p, k, target, weights


def test_torch_numpy_value_and_first_second_k_derivatives(tensor_case):
    p, k, _, _ = tensor_case
    tk = torch.tensor(k, dtype=torch.float64, requires_grad=True)
    parameters = torch.tensor(list(asdict(p).values()), dtype=torch.float64, requires_grad=True)
    w = total_variance_torch(tk, parameters)
    first = torch.autograd.grad(w.sum(), tk, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), tk)[0]
    np.testing.assert_allclose(w.detach(), svi.svi_total_variance(k, p), atol=1e-15, rtol=0)
    np.testing.assert_allclose(first.detach(), svi.svi_first_derivative(k, p), atol=1e-14, rtol=0)
    np.testing.assert_allclose(second.detach(), svi.svi_second_derivative(k, p), atol=1e-13, rtol=0)


def test_transform_initialization_and_constraints(tensor_case):
    p, k, _, _ = tensor_case
    latent = latent_from_raw(p)
    actual = raw_from_latent(latent)
    np.testing.assert_allclose(actual, list(asdict(p).values()), atol=1e-15, rtol=0)
    model = DifferentiableSVI(p)
    assert model.latent.dtype == torch.float64 and model.latent.device.type == "cpu"
    assert sum(parameter.numel() for parameter in model.parameters()) == 5
    for values in [torch.zeros(5, dtype=torch.float64), torch.linspace(-30, 30, 5, dtype=torch.float64)]:
        a, b, rho, m, eta = raw_from_latent(values).tolist()
        assert b >= 0 and abs(rho) < 1 and eta > 0
        assert svi.svi_global_minimum(svi.SVIParameters(a,b,rho,m,eta)) > 0
    # Fundamental b>=0 admits flat slices, but the smooth inverse cannot encode
    # the exact boundary. It is rejected rather than nudged for a nicer comparison.
    boundary = svi.SVIParameters(.04, 0, 0, 0, .2)
    with pytest.raises(ValueError, match="BOUNDARY"):
        latent_from_raw(boundary)


def test_objective_equality_and_physical_gradient_matches_analytic_jacobian(tensor_case):
    p, k, target, weights = tensor_case
    initial = svi.SVIParameters(p.a, p.b*1.1, 0, 0, .2)
    latent = latent_from_raw(initial).requires_grad_()
    tensors = [torch.tensor(value, dtype=torch.float64) for value in (k,target,weights)]
    loss = objective_torch(latent, *tensors)
    residual = weighted_residual(encode_parameters(initial), k, target, weights)
    assert float(loss.detach()) == pytest.approx(float(residual@residual), abs=1e-16)
    physical = physical_from_latent(latent).detach().requires_grad_()
    torch.sum(tensors[2]*(total_variance_torch(tensors[0],raw_from_physical(physical))-tensors[1]).square()).backward()
    expected = residual_jacobian(encode_parameters(initial), k, target, weights).T@residual
    np.testing.assert_allclose(physical.grad.detach()/2, expected, atol=1e-13, rtol=0)


def test_latent_autograd_matches_finite_differences(tensor_case):
    p, k, target, weights = tensor_case
    initial = svi.SVIParameters(.03, .1, -.1, 0, .25)
    latent = latent_from_raw(initial).requires_grad_()
    tensors = [torch.tensor(value,dtype=torch.float64) for value in (k,target,weights)]
    objective_torch(latent,*tensors).backward()
    numerical = []
    for i in range(5):
        step = torch.zeros_like(latent)
        step[i] = 1e-6
        numerical.append(float((objective_torch(latent.detach()+step,*tensors)-objective_torch(latent.detach()-step,*tensors))/(2e-6)))
    np.testing.assert_allclose(latent.grad, numerical, atol=1e-9, rtol=0)


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_small_budgets_report_actual_counts_and_nonconvergence(tensor_case, optimizer):
    _, k, target, weights = tensor_case
    initial = svi.SVIParameters(.03,.1,0,0,.2)
    cfg = TorchOptimizerConfig(adam_max_steps=1, lbfgs_max_iter=1, lbfgs_max_eval=5,
                               objective_tolerance=1e-25, physical_gradient_tolerance=1e-20)
    result = fit_torch_slice(k,target,weights=weights,initial=initial,optimizer=optimizer,optimizer_config=cfg)
    assert not result["converged"] and result["failure_reason"]
    assert result["iterations"] == 1
    assert result["objective_evaluations"] == result["gradient_evaluations"] == len(result["evaluation_history"])
    if optimizer == "adam":
        assert result["objective_evaluations"] == 2 and result["lbfgs_closure_calls"] == 0
    else:
        assert result["objective_evaluations"] == result["lbfgs_closure_calls"]+1
        assert result["native_func_evals"] == result["lbfgs_closure_calls"]
    assert result["fitted_parameters"] is not None


@pytest.mark.parametrize("case", ["nan", "negative_weight", "insufficient", "boundary", "missing_initial"])
def test_invalid_inputs_and_initial_domain_are_not_successes(tensor_case, case):
    _, k, target, weights = tensor_case
    initial = svi.SVIParameters(.03,.1,0,0,.2)
    if case == "nan":
        target = target.copy(); target[0] = np.nan
    elif case == "negative_weight":
        weights = weights.copy(); weights[0] = -1
    elif case == "insufficient":
        k,target,weights = k[:3],target[:3],weights[:3]
    elif case == "boundary":
        initial = svi.SVIParameters(.04,0,0,0,.2)
    else:
        initial = None
    result = fit_torch_slice(k,target,weights=weights,initial=initial)
    assert not result["converged"] and result["failure_reason"]
    assert result["objective_evaluations"] == result["gradient_evaluations"] == 0


def test_deterministic_initialization_and_short_adam_run(tensor_case):
    _, k, target, weights = tensor_case
    initial = svi.SVIParameters(.03,.1,0,0,.2)
    cfg = TorchOptimizerConfig(adam_max_steps=2)
    first = fit_torch_slice(k,target,weights=weights,initial=initial,optimizer_config=cfg)
    second = fit_torch_slice(k,target,weights=weights,initial=initial,optimizer_config=cfg)
    for name in ["initial_latent_parameters", "fitted_parameters", "objective_value", "iterations", "objective_evaluations"]:
        assert first[name] == second[name]
    with pytest.raises(ValueError):
        TorchOptimizerConfig(adam_lr=0)
