"""CPU float64 neural volatility and differentiable forward Black--Scholes.

No market conventions are inferred. Layer count means affine/Linear layers,
including the output; the maximum is four (at most three hidden layers).
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class SurfaceConfig:
    hidden: tuple[int, ...] = (16, 16)
    activation: str = "tanh"
    volatility_floor: float = 1e-6

    def __post_init__(self):
        if not 1 <= len(self.hidden) <= 3 or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in self.hidden):
            raise ValueError("Require 1--3 positive integer hidden widths; <=4 affine layers including output")
        if self.activation not in {"tanh", "softplus"}:
            raise ValueError("Use smooth tanh or softplus activation")
        if not math.isfinite(self.volatility_floor) or self.volatility_floor <= 0:
            raise ValueError("Volatility floor must be finite and positive")


class NeuralSurface(nn.Module):
    def __init__(self, training_inputs, *, config=SurfaceConfig(), seed=42):
        super().__init__()
        inputs = torch.as_tensor(training_inputs, dtype=torch.float64, device="cpu")
        if inputs.ndim != 2 or inputs.shape[1] != 2 or not len(inputs) or not torch.isfinite(inputs).all() or torch.any(inputs[:, 1] <= 0):
            raise ValueError("Training normalization requires finite (k,T) rows with T>0")
        self.config = config
        self.register_buffer("input_mean", inputs.mean(0))
        scale = inputs.std(0, correction=0)
        self.register_buffer("input_scale", torch.where(scale > 0, scale, torch.ones_like(scale)))
        layers = []
        # Isolate seeded initialization from caller RNG state. No dropout/batchnorm.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            widths = (2, *config.hidden, 1)
            for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
                layers.append(nn.Linear(left, right, dtype=torch.float64, device="cpu"))
                if index < len(widths)-2:
                    layers.append(nn.Tanh() if config.activation == "tanh" else nn.Softplus())
            # Fixed, truth-independent initial output bias; weights stay seeded.
            layers[-1].bias.data.fill_(math.log(math.expm1(.3)))
        self.network = nn.Sequential(*layers)

    def forward(self, k, maturity):
        k, maturity = torch.broadcast_tensors(k, maturity)
        if not torch.isfinite(k).all() or not torch.isfinite(maturity).all() or torch.any(maturity <= 0):
            raise ValueError("Neural surface requires finite k and strictly positive T")
        physical = torch.stack((k, maturity), dim=-1)
        normalized = (physical-self.input_mean)/self.input_scale
        return self.config.volatility_floor+functional.softplus(self.network(normalized).squeeze(-1))

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def black_scholes_torch(forward, strike, maturity, volatility, discount, option_type):
    """Torch counterpart of q1.pricing.black_scholes, including log-CDF OTM side.

    Accepts 'C'/'P' or a boolean tensor (True=call) for mixed batches. Prices are
    premiums per underlying unit. T=0/sigma=0 uses the discounted intrinsic limit.
    """
    sigma = torch.as_tensor(volatility)
    if not sigma.is_floating_point():
        sigma = sigma.to(torch.float64)
    f, strike, t, sigma, discount = torch.broadcast_tensors(*[
        torch.as_tensor(item, dtype=sigma.dtype, device=sigma.device)
        for item in (forward, strike, maturity, sigma, discount)])
    if not all(torch.isfinite(item).all() for item in (f, strike, t, sigma, discount)):
        raise ValueError("Black--Scholes inputs must be finite")
    if torch.any((f <= 0) | (strike <= 0) | (t < 0) | (sigma < 0) | (discount <= 0)) or torch.any((t == 0) & (discount != 1)):
        raise ValueError("Invalid forward/strike/T/sigma/discount")
    if isinstance(option_type, str):
        if option_type not in {"C", "P"}:
            raise ValueError("option_type must be C/P or boolean call mask")
        call_mask = option_type == "C"
    else:
        call_mask = torch.as_tensor(option_type, device=sigma.device)
        if call_mask.dtype != torch.bool:
            raise ValueError("Mixed option types require a boolean call mask")
    std = sigma*torch.sqrt(t)
    if not torch.isfinite(std).all() or torch.any((sigma > 0) & (t > 0) & (std == 0)):
        raise ValueError("Volatility-time product overflow/underflow")
    active = std > 0
    safe_std = torch.where(active, std, torch.ones_like(std))
    log_fk = torch.log(f)-torch.log(strike)
    d1, d2 = log_fk/safe_std+.5*safe_std, log_fk/safe_std-.5*safe_std
    otm_call = f <= strike
    log_a = torch.where(otm_call, torch.log(f)+torch.special.log_ndtr(d1), torch.log(strike)+torch.special.log_ndtr(-d2))
    log_b = torch.where(otm_call, torch.log(strike)+torch.special.log_ndtr(d2), torch.log(f)+torch.special.log_ndtr(-d1))
    saturated = torch.isneginf(log_a)
    difference = torch.where(saturated,torch.zeros_like(log_b),log_b)-torch.where(saturated,torch.zeros_like(log_a),log_a)
    small = torch.exp(log_a)*(-torch.expm1(torch.clamp_max(difference, 0.)))
    small = torch.where(f == strike, f*torch.erf(std/(2*math.sqrt(2))), small)
    small = torch.where(active, small, torch.zeros_like(small))
    small = torch.minimum(torch.clamp_min(small, 0.), torch.minimum(f, strike))
    call, put = small+torch.clamp_min(f-strike, 0.), small+torch.clamp_min(strike-f, 0.)
    return discount*(torch.where(call_mask, call, put) if not isinstance(call_mask, bool) else call if call_mask else put)


def _derivative(value, coordinate, *, create_graph=True):
    if not value.requires_grad:
        return coordinate*0
    result = torch.autograd.grad(value.sum(), coordinate, create_graph=create_graph, retain_graph=True, allow_unused=True)[0]
    return coordinate*0 if result is None else result


def surface_derivatives(model, k, maturity, *, second=True, create_graph=True):
    """Differentiate PHYSICAL inputs; normalization stays inside the graph.

    Each output depends only on its corresponding row (no cross-sample layers).
    Therefore the gradient of the sum gives pointwise derivatives. T derivative
    holds k fixed, not K or normalized coordinates fixed.
    """
    k, t = torch.broadcast_tensors(k, maturity)
    k, t = k.detach().clone().requires_grad_(), t.detach().clone().requires_grad_()
    sigma = model(k, t)
    w = sigma.square()*t
    first = _derivative(w, k)
    temporal = _derivative(w, t, create_graph=create_graph)
    curvature = _derivative(first, k, create_graph=create_graph) if second else None
    exam = (1-k*first/(2*w)).square()-first.square()/4*(1/w+.25)
    return {"k": k, "maturity": t, "volatility": sigma, "total_variance": w,
            "w_k": first, "w_T": temporal, "w_kk": curvature,
            "examination": exam, "complete": exam+curvature/2 if second else None}


def arbitrage_penalties(model, k, maturity):
    report = surface_derivatives(model, k, maturity, second=False)
    return {"butterfly": torch.relu(-report["examination"]).square().mean(),
            "calendar": torch.relu(-report["w_T"]).square().mean()}
