"""Deterministic paired-quote splits, train-only preprocessing and Adam fitting."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import pricing
from .neural_surface import NeuralSurface, SurfaceConfig, arbitrage_penalties, black_scholes_torch


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 600
    batch_size: int = 256
    learning_rate: float = .01
    patience: int = 100
    min_delta: float = 1e-8
    validation_fraction: float = .2
    test_fraction: float = .2
    collocation_k_points: int = 17
    collocation_t_points: int = 7

    def __post_init__(self):
        for key in ["epochs", "batch_size", "patience", "collocation_k_points", "collocation_t_points"]:
            item = getattr(self, key)
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0 or not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("Require finite positive learning rate and nonnegative min_delta")
        if not (0 < self.validation_fraction < 1 and 0 < self.test_fraction < 1 and self.validation_fraction+self.test_fraction < 1):
            raise ValueError("Train, validation and held-out test fractions must all be positive")


def paired_split(rows, *, seed=42, config=TrainingConfig()):
    """Stratify by expiry, keeping the call/put pair at each strike in one fold.

    This is a within-date interpolation split, not temporal validation or the M7
    30%-missing experiment. Held-out test quotes are never used for selection.
    """
    if len({row["observation_date"] for row in rows}) != 1:
        raise ValueError("Fit one dated neural surface at a time")
    if len({row["quote_id"] for row in rows}) != len(rows):
        raise ValueError("Quote IDs must be unique")
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["expiry_date"], {}).setdefault((float(row["strike"]), float(row["contract_unit"])), []).append(index)
    rng = np.random.default_rng(seed)
    folds = {"train": [], "validation": [], "test": []}
    for expiry, strikes in sorted(groups.items()):
        keys = sorted(strikes)
        if len(keys) < 5:
            raise ValueError("Need >=5 paired strikes per expiry for a three-way split")
        nv, nt = max(1, int(len(keys)*config.validation_fraction)), max(1, int(len(keys)*config.test_fraction))
        if nv+nt >= len(keys):
            raise ValueError("Split leaves no training strikes")
        order = rng.permutation(len(keys))
        for position, which in enumerate(order):
            indices = strikes[keys[which]]
            if sorted(rows[i]["option_type"] for i in indices) != ["C", "P"]:
                raise ValueError("Each split group must have exactly one call and one put")
            name = "validation" if position < nv else "test" if position < nv+nt else "train"
            folds[name].extend(indices)
    return {name: sorted(indices) for name, indices in folds.items()}


def validate_split(rows, split):
    if set(split) != {"train", "validation", "test"} or any(not split[name] for name in split):
        raise ValueError("Require three nonempty folds")
    combined = [index for indices in split.values() for index in indices]
    if sorted(combined) != list(range(len(rows))):
        raise ValueError("Folds must partition all rows without overlaps or omissions")
    owners = {}
    for name, indices in split.items():
        for index in indices:
            row = rows[index]
            key = (row["observation_date"], row["expiry_date"], row["strike"], row["contract_unit"])
            if key in owners and owners[key] != name:
                raise ValueError("Call/put pair leaked across folds")
            owners[key] = name


def quote_tensors(rows):
    fields = {key: torch.tensor([row[key] for row in rows], dtype=torch.float64) for key in
              ["forward", "strike", "time_to_expiry", "discount_factor", "price"]}
    fields["k"] = torch.tensor(pricing.log_moneyness(fields["strike"].numpy(), fields["forward"].numpy()), dtype=torch.float64)
    fields["call"] = torch.tensor([row["option_type"] == "C" for row in rows], dtype=torch.bool)
    return fields


def predicted_prices(model, data):
    sigma = model(data["k"], data["time_to_expiry"])
    return black_scholes_torch(data["forward"], data["strike"], data["time_to_expiry"], sigma, data["discount_factor"], data["call"])


def penalty_grid(training_data, config):
    """Mid-cell collocation on TRAIN support only, no validation/test bounds."""
    k, t = training_data["k"], training_data["time_to_expiry"]
    def centers(values, count):
        lo, hi = float(values.min()), float(values.max())
        return torch.full((count,),lo,dtype=torch.float64) if lo == hi else lo+(torch.arange(count,dtype=torch.float64)+.5)*(hi-lo)/count
    kk, tt = torch.meshgrid(centers(k,config.collocation_k_points), centers(t,config.collocation_t_points), indexing="xy")
    return kk.flatten(), tt.flatten()


def state_fingerprint(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode()); digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def train_surface(rows, split, *, lambda_butterfly=0., lambda_calendar=0., seed=42,
                  surface_config=SurfaceConfig(), training_config=TrainingConfig()):
    """Price MSE + the exact required soft penalties; no IV regression/pretraining.

    Return the best VALIDATION-price checkpoint. MAX_EPOCHS is a finite completed
    budget, not claimed mathematical convergence. No test quote is read in fitting.
    """
    start = perf_counter()
    record = {"success": False, "status": "FAILED", "failure_reason": None, "history": [],
              "epochs_completed": 0, "optimizer_updates": 0, "backward_calls": 0,
              "training_objective_evaluations": 0, "validation_price_evaluations": 0,
              "best_epoch": None, "best_validation_price_mse": None, "initial_validation_price_mse": None,
              "fit_seconds": 0., "lambda_butterfly": lambda_butterfly, "lambda_calendar": lambda_calendar,
              "training_penalty_history_status": "COMPUTED" if lambda_butterfly or lambda_calendar else "NOT_COMPUTED_ZERO_COEFFICIENTS",
              "surface_config": asdict(surface_config), "training_config": asdict(training_config), "seed": seed}
    model = None
    try:
        if any(not math.isfinite(number) or number < 0 for number in [lambda_butterfly,lambda_calendar]):
            raise ValueError("Penalty coefficients must be finite and nonnegative")
        validate_split(rows,split)
        training = quote_tensors([rows[i] for i in split["train"]])
        validation = quote_tensors([rows[i] for i in split["validation"]])
        x = torch.stack((training["k"],training["time_to_expiry"]),dim=1)
        model = NeuralSurface(x,config=surface_config,seed=seed)
        record.update(parameter_count=model.parameter_count, initial_state_sha256=state_fingerprint(model),
                      input_mean=model.input_mean.tolist(), input_scale=model.input_scale.tolist())
        kk, tt = penalty_grid(training,training_config)
        record["penalty_grid"] = {"k":kk.tolist(),"maturity":tt.tolist(),"origin":"TRAINING_SUPPORT_MID_CELL_COLLOCATION"}
        optimizer = torch.optim.Adam(model.parameters(),lr=training_config.learning_rate)
        rng = np.random.default_rng(seed)

        def objective(data):
            record["training_objective_evaluations"] += 1
            price_loss = (predicted_prices(model,data)-data["price"]).square().mean()
            # Zero-coefficient ablations do not need a higher-order training graph.
            penalties = arbitrage_penalties(model,kk,tt) if lambda_butterfly or lambda_calendar else {"butterfly":price_loss*0,"calendar":price_loss*0}
            total = price_loss+lambda_butterfly*penalties["butterfly"]+lambda_calendar*penalties["calendar"]
            if not all(torch.isfinite(value) for value in [price_loss,total,*penalties.values()]):
                raise ValueError("NONFINITE_TRAINING_LOSS")
            return total, price_loss, penalties

        def validation_loss():
            record["validation_price_evaluations"] += 1
            with torch.no_grad():
                value = float((predicted_prices(model,validation)-validation["price"]).square().mean())
            if not math.isfinite(value):
                raise ValueError("NONFINITE_VALIDATION_LOSS")
            return value

        total, initial_price, penalties = objective(training)
        initial_val = validation_loss()
        record["initial_validation_price_mse"] = initial_val
        record["history"].append({"epoch":0,"training_total":float(total.detach()),"training_price_mse":float(initial_price.detach()),
                                  "training_butterfly_penalty":float(penalties["butterfly"].detach()) if lambda_butterfly or lambda_calendar else None,
                                  "training_calendar_penalty":float(penalties["calendar"].detach()) if lambda_butterfly or lambda_calendar else None,
                                  "validation_price_mse":initial_val,"gradient_l2":None})
        del total, initial_price, penalties
        best, best_epoch, state = initial_val, 0, deepcopy(model.state_dict())
        patience_reference, stale = best, 0
        stop_reason = "MAX_EPOCHS"
        for epoch in range(1,training_config.epochs+1):
            order = rng.permutation(len(x))
            sums = np.zeros(5)
            for offset in range(0,len(x),training_config.batch_size):
                indices = order[offset:offset+training_config.batch_size]
                batch = {name:value[indices] for name,value in training.items()}
                optimizer.zero_grad(set_to_none=True)
                total, price_loss, penalties = objective(batch)
                record["backward_calls"] += 1
                total.backward()
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    raise ValueError("NONFINITE_OR_MISSING_GRADIENT")
                gradient = math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters()))
                sums += len(indices)*np.array([float(total.detach()),float(price_loss.detach()),float(penalties["butterfly"].detach()),float(penalties["calendar"].detach()),gradient])
                optimizer.step()
                record["optimizer_updates"] += 1
                if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise ValueError("NONFINITE_PARAMETER_UPDATE")
            value = validation_loss()
            sums /= len(x)
            record["epochs_completed"] = epoch
            history = dict(zip(["training_total","training_price_mse","training_butterfly_penalty","training_calendar_penalty","gradient_l2"],sums.tolist()),epoch=epoch,validation_price_mse=value)
            if not (lambda_butterfly or lambda_calendar):
                history.update(training_butterfly_penalty=None,training_calendar_penalty=None)
            record["history"].append(history)
            if value < best:
                best, best_epoch, state = value, epoch, deepcopy(model.state_dict())
            if value < patience_reference-training_config.min_delta:
                patience_reference, stale = value, 0
            else:
                stale += 1
            if stale >= training_config.patience:
                stop_reason = "EARLY_STOPPING_VALIDATION_PRICE_PATIENCE"
                break
        model.load_state_dict(state)
        final, price_loss, penalties = objective(training)
        restored_val = validation_loss()
        record.update(success=True,status="COMPLETED",stop_reason=stop_reason,
                      mathematical_convergence_claimed=False,best_epoch=best_epoch,best_validation_price_mse=best,
                      restored_validation_price_mse=restored_val,restored_training_price_mse=float(price_loss.detach()),
                      restored_training_total=float(final.detach()),final_state_sha256=state_fingerprint(model))
    except (ValueError,RuntimeError,TypeError,OverflowError,FloatingPointError) as error:
        record.update(success=False,status="FAILED",failure_reason=str(error))
    finally:
        record["fit_seconds"] = perf_counter()-start
    return model, record


def save_checkpoint(path, model, record):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or path.exists():
        raise ValueError("Checkpoint must be a new repository-local file")
    if not record["success"]:
        raise ValueError("Do not save failed training as a usable checkpoint")
    payload = {"format":"Q1_M6_NEURAL_SURFACE_V1","surface_config":asdict(model.config),
               "state_dict":model.state_dict(),"metadata":{
                   key:record[key] for key in ["seed","best_epoch","best_validation_price_mse","final_state_sha256","input_mean","input_scale",
                                             "training_config","lambda_butterfly","lambda_calendar"]}}
    with path.open("xb") as handle:
        torch.save(payload,handle)
    return payload["metadata"]


def load_checkpoint(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Checkpoint must be repository-local")
    payload = torch.load(path,map_location="cpu",weights_only=True)
    if payload["format"] != "Q1_M6_NEURAL_SURFACE_V1":
        raise ValueError("Unknown checkpoint format")
    config = SurfaceConfig(**{**payload["surface_config"],"hidden":tuple(payload["surface_config"]["hidden"])})
    model = NeuralSurface([[0.,1.]],config=config,seed=payload["metadata"]["seed"])
    model.load_state_dict(payload["state_dict"],strict=True)
    if state_fingerprint(model) != payload["metadata"]["final_state_sha256"]:
        raise ValueError("Checkpoint tensor fingerprint mismatch")
    model.eval()
    return model, payload["metadata"]
