"""Small float64 mathematical tests only; no full neural training experiments."""

import numpy as np
import pytest
import torch

from q1 import pricing, svi
from q1.neural_surface import (NeuralSurface, SurfaceConfig, arbitrage_penalties,
                                black_scholes_torch, surface_derivatives)
from q1.torch_svi import total_variance_torch


def inputs():
    return np.array([[-.4,.1], [.1,.3], [.4,1.]])


def test_positive_output_layer_count_seed_and_float64():
    a, b = NeuralSurface(inputs()), NeuralSurface(inputs())
    assert a.parameter_count == 337
    assert sum(isinstance(layer, torch.nn.Linear) for layer in a.network) == 3
    x = torch.tensor(inputs(), dtype=torch.float64)
    assert torch.all(a(x[:,0],x[:,1]) > 0)
    assert all(p.dtype == torch.float64 and p.device.type == "cpu" for p in a.parameters())
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    for config in [SurfaceConfig(hidden=(4,), activation="softplus"), SurfaceConfig(hidden=(4,4,4))]:
        assert NeuralSurface(inputs(),config=config).parameter_count > 0
    with pytest.raises(ValueError):
        SurfaceConfig(hidden=(4,4,4,4))
    with pytest.raises(ValueError):
        a(torch.tensor(0.),torch.tensor(0.))


@pytest.mark.parametrize("kind", ["C", "P"])
def test_numpy_torch_pricing_agree_including_tails_and_limits(kind):
    f = np.array([100.,100.,100.,100.,100.,100.])
    strike = np.array([100.,50.,180.,100.,150.,100.])
    t = np.array([1.,.01,.1,0.,2.,1e-8])
    sigma = np.array([.3,.2,.2,.2,0.,1e-6])
    d = np.array([.98,1.01,.99,1.,.95,1.])
    actual = black_scholes_torch(f,strike,t,torch.tensor(sigma),d,kind)
    expected = pricing.black_scholes(f,strike,t,sigma,d,kind)
    np.testing.assert_allclose(actual.detach(),expected,atol=2e-12,rtol=1e-10)


def test_mixed_pricing_loss_has_finite_nonzero_parameter_gradients():
    model = NeuralSurface(inputs())
    x = torch.tensor(inputs(),dtype=torch.float64)
    price = black_scholes_torch(100.,100.*torch.exp(x[:,0]),x[:,1],model(x[:,0],x[:,1]),.98,torch.tensor([True,False,True]))
    price.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.parameters()) > 0


def test_svi_autograd_diagnostics_equal_reused_analytic_formulas():
    p = svi.SVIParameters(.02,.15,-.3,.04,.2)
    raw = torch.tensor(list(vars(p).values()),dtype=torch.float64)
    model = lambda k,t: torch.sqrt(total_variance_torch(k,raw)/t)
    k = torch.linspace(-.4,.4,21,dtype=torch.float64)
    t = torch.full_like(k,.3)
    report = surface_derivatives(model,k,t)
    for key, expected in [("w_k",svi.svi_first_derivative(k.numpy(),p)),
                          ("w_kk",svi.svi_second_derivative(k.numpy(),p)),
                          ("examination",svi.examination_butterfly(k.numpy(),p)),
                          ("complete",svi.complete_butterfly(k.numpy(),p))]:
        np.testing.assert_allclose(report[key].detach(),expected,atol=2e-14,rtol=0)
    assert report["w_T"].detach().abs().max() < 1e-15


def test_analytical_temporal_derivative_and_nonzero_penalties():
    # Deliberately violates calendar monotonicity, independently of any training.
    model = lambda k,t: torch.exp(3*k-4*t)
    k = torch.linspace(-.4,.4,11,dtype=torch.float64)
    t = torch.full_like(k,.4)
    report = surface_derivatives(model,k,t)
    w = report["total_variance"].detach()
    torch.testing.assert_close(report["w_k"].detach(),6*w,atol=1e-14,rtol=0)
    torch.testing.assert_close(report["w_kk"].detach(),36*w,atol=1e-13,rtol=0)
    torch.testing.assert_close(report["w_T"].detach(),w*(1/t-8),atol=1e-14,rtol=0)
    penalty = arbitrage_penalties(model,k,t)
    assert penalty["butterfly"] > 0 and penalty["calendar"] > 0
    torch.testing.assert_close(penalty["calendar"],torch.relu(-report["w_T"]).square().mean())


def test_normalized_network_physical_derivatives_match_finite_differences():
    model = NeuralSurface(inputs())
    k = torch.tensor([-.11,.17],dtype=torch.float64)
    t = torch.tensor([.2,.7],dtype=torch.float64)
    report = surface_derivatives(model,k,t)
    w = lambda k,t: model(k,t).square()*t
    h = 1e-4
    fd_k = (w(k+h,t)-w(k-h,t))/(2*h)
    fd_t = (w(k,t+h)-w(k,t-h))/(2*h)
    fd_kk = (w(k+h,t)-2*w(k,t)+w(k-h,t))/h**2
    torch.testing.assert_close(report["w_k"],fd_k,atol=1e-8,rtol=0)
    torch.testing.assert_close(report["w_T"],fd_t,atol=1e-8,rtol=0)
    torch.testing.assert_close(report["w_kk"],fd_kk,atol=1e-8,rtol=0)


def test_active_penalties_backpropagate_through_input_derivatives():
    alpha = torch.nn.Parameter(torch.tensor(3.,dtype=torch.float64))
    beta = torch.nn.Parameter(torch.tensor(-4.,dtype=torch.float64))
    model = lambda k,t: torch.exp(alpha*k+beta*t)
    k = torch.linspace(-.4,.4,11,dtype=torch.float64)
    t = torch.full_like(k,.4)
    terms = arbitrage_penalties(model,k,t)
    (terms["butterfly"]+terms["calendar"]).backward()
    assert torch.isfinite(alpha.grad) and torch.isfinite(beta.grad)
    assert alpha.grad.abs() > 0 and beta.grad.abs() > 0


def test_independent_diagnostics_retain_intentional_violations():
    from q1.neural_evaluation import evaluate_grid
    from q1.verification_fixtures import development_quotes
    frame,_ = development_quotes()
    diagnostics = evaluate_grid(lambda k,t:torch.exp(3*k-4*t),frame.to_dict(orient="records"))
    assert diagnostics["violations"]["examination"]["count"] > 0
    assert diagnostics["violations"]["w_T"]["count"] > 0
    assert diagnostics["calendar_violation_count"] > 0 and diagnostics["negative_forward_variance_count"] > 0
    assert all(row["magnitude"] > 0 for row in diagnostics["negative_forward_variance"])
