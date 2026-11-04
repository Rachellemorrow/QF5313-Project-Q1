"""Independent neural grid diagnostics and quote repricing using Q1 primitives."""

import numpy as np
import torch

from . import pricing, svi
from .classical_svi import CalibrationConfig, repricing_summary
from .neural_surface import surface_derivatives
from .neural_training import quote_tensors


def violations(values, k, times):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite diagnostic values; do not claim zero violations")
    indices = np.argwhere(values < 0)
    return {"count":len(indices),"locations":[{"maturity":float(times[i]),"k":float(k[j]),
            "value":float(values[i,j]),"magnitude":float(-values[i,j])} for i,j in indices]}


def evaluate_grid(model, rows, *, config=CalibrationConfig()):
    k = config.grid()
    observed = sorted({float(row["time_to_expiry"]) for row in rows})
    times = np.unique(np.r_[observed,np.linspace(observed[0],observed[-1],13)])
    kk, tt = np.meshgrid(k,times)
    report = surface_derivatives(model,torch.tensor(kk.flatten(),dtype=torch.float64),
                                 torch.tensor(tt.flatten(),dtype=torch.float64),create_graph=False)
    arrays = {name:report[name].detach().numpy().reshape(len(times),len(k)) for name in
              ["volatility","total_variance","examination","complete","w_T"]}
    if any(not np.isfinite(value).all() for value in arrays.values()) or np.any(arrays["volatility"] <= 0) or np.any(arrays["total_variance"] <= 0):
        raise ValueError("Nonfinite/nonpositive independent surface evaluation")
    checks = {name:violations(arrays[name],k,times) for name in ["examination","complete","w_T"]}
    slices = []
    for row in sorted({(r["expiry_date"],r["time_to_expiry"]) for r in rows}):
        expiry, t = row
        index = int(np.searchsorted(times,t))
        item = {"expiry_date":expiry,"maturity":t,"positive_total_variance":bool(np.all(arrays["total_variance"][index] > 0))}
        for label in ["examination","complete"]:
            values = arrays[label][index]
            diagnostic = svi.GridDiagnostics(k,values,values < 0,0.)
            item[label] = {"count":diagnostic.count,"locations":diagnostic.locations.tolist(),"magnitudes":diagnostic.magnitudes.tolist()}
        slices.append(item)
    calendar = []
    w = arrays["total_variance"]
    # All maturity pairs at the SAME k, including independent intermediate nodes.
    for i in range(len(times)):
        for j in range(i+1,len(times)):
            diagnostic = svi.calendar_diagnostics(k,w[i],w[j],times[i],times[j])
            calendar.append({"earlier_maturity":float(times[i]),"later_maturity":float(times[j]),"count":diagnostic.count,
                             "locations":diagnostic.locations.tolist(),"magnitudes":diagnostic.magnitudes.tolist()})
    forward = svi.forward_variance_diagnostics(times,w)
    negative_forward = [{"earlier_maturity":float(times[i]),"later_maturity":float(times[i+1]),"k":float(k[j]),
                         "magnitude":float(-forward.forward_variance[i,j])} for i,j in forward.indices]
    midpoints = (times[:-1]+times[1:])/2
    interpolated = svi.interpolate_total_variance(times,w,midpoints)
    return {"grid_origin":"INDEPENDENT_EVALUATION_NOT_TRAINING_COLLOCATION","k":k.tolist(),"maturities":times.tolist(),
            **{name:value.tolist() for name,value in arrays.items()},"violations":checks,"per_expiry":slices,
            "calendar":calendar,"calendar_violation_count":sum(row["count"] for row in calendar),
            "negative_forward_variance":negative_forward,"negative_forward_variance_count":len(negative_forward),
            "linear_total_variance_diagnostic_interpolation":{"query_maturities":midpoints.tolist(),"total_variance":np.asarray(interpolated).tolist(),
                "note":"Shared linear-w diagnostic, not a replacement for the neural model at intermediate T"}}


def evaluate_quotes(model, rows, split, *, price_floor=1e-8):
    data = quote_tensors(rows)
    with torch.no_grad():
        sigma = model(data["k"],data["time_to_expiry"]).numpy()
    folds = {index:name for name,indices in split.items() for index in indices}
    training = [rows[i] for i in split["train"]]
    support = {}
    for row in training:
        support.setdefault(row["expiry_date"],[]).append(pricing.log_moneyness(row["strike"],row["forward"]))
    predictions = []
    for index, (row,vol) in enumerate(zip(rows,sigma)):
        out = {"quote_id":row["quote_id"],"observation_date":row["observation_date"],"expiry_date":row["expiry_date"],
               "option_type":row["option_type"],"fold":folds[index],"k":float(data["k"][index]),
               "maturity":row["time_to_expiry"],"status":"FAILED","failure_reason":None,
               "market_price":row["price"],"model_price":None,"fitted_iv":float(vol),"absolute_error":None,
               "relative_error":None,"relative_error_status":None}
        out["strike_support"] = "INTERPOLATION" if min(support[row["expiry_date"]]) <= out["k"] <= max(support[row["expiry_date"]]) else "EXTRAPOLATION"
        try:
            model_price = pricing.black_scholes(row["forward"],row["strike"],row["time_to_expiry"],vol,row["discount_factor"],row["option_type"])
            absolute = abs(model_price-row["price"])
            out.update(status="REPRICED",model_price=model_price,absolute_error=absolute,
                       relative_error=absolute/row["price"] if row["price"] >= price_floor else None,
                       relative_error_status="DEFINED" if row["price"] >= price_floor else "NEAR_ZERO_DENOMINATOR_EXCLUDED")
        except (ValueError,TypeError,OverflowError) as error:
            out["failure_reason"] = str(error)
        predictions.append(out)
    summary = {}
    for fold in split:
        selected = [row for row in predictions if row["fold"] == fold]
        slices = []
        for expiry in sorted({row["expiry_date"] for row in rows}):
            records = [row for row in selected if row["expiry_date"] == expiry]
            # Legacy repricing_summary readiness flag only: this is NOT a claim
            # of optimizer convergence, which is separately never asserted.
            slices.append({"slice_id":rows[0]["observation_date"]+"|"+expiry,"observation_date":rows[0]["observation_date"],
                           "expiry_date":expiry,"fit":{"converged":True},"repricing":records})
        errors = [row["absolute_error"]**2 for row in selected if row["status"] == "REPRICED"]
        summary[fold] = {"quote_count":len(selected),"failed_repricing":sum(row["status"] != "REPRICED" for row in selected),
                         "price_mse":float(np.mean(errors)) if len(errors) == len(selected) else None,
                         "relative_error":repricing_summary(slices)}
    return {"predictions":predictions,"fold_metrics":summary}
