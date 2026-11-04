"""Shared missing-quote runner: reuse M4/M5/M6, fit on retained information only."""

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import pricing, svi
from .approach_a import plain, prepare_quotes
from .classical_svi import CalibrationConfig, fit_slice, verify_surface
from .neural_evaluation import evaluate_grid
from .neural_surface import SurfaceConfig
from .neural_training import TrainingConfig, load_checkpoint, save_checkpoint, train_surface
from .robustness_protocol import METHODS, digest, quote_drop_manifest, retained_neural_split, slice_key, support_class
from .torch_svi import TorchOptimizerConfig, fit_torch_slice


ROOT = Path(__file__).resolve().parents[2]


def prepare_universe(frame, *, mode, config=CalibrationConfig()):
    """Freeze CLEANED universe before deletion, not after breaking C/P pairs.

    Individual IV inversion uses the exogenous verified F/D, never parity-fitted
    inputs. Full-data parity is descriptive ONLY and cannot feed any refit.
    """
    ledger, parity = prepare_quotes(frame,mode=mode,config=config)
    rows = [{**record["input"],"_iv":record["iv"]["volatility"],"_k":record["k"],"_w":record["total_variance"]}
            for record in ledger if record["eligible_for_fit"]]
    return sorted(rows,key=lambda row:row["quote_id"]), ledger, parity


class FitCache:
    """Immutable per-fit records. Resume reuses completed jobs, including failures.

    No result is accepted under different retained inputs/configuration. Failed
    optimizer convergence is a completed experimental outcome, not auto-retried.
    An interrupted unsaved job alone is recomputed. Never imports an M6 model.
    """
    def __init__(self, directory=None):
        self.directory = Path(directory).resolve() if directory is not None else None
        if self.directory is not None:
            if not self.directory.is_relative_to(ROOT):
                raise ValueError("Fit cache must remain repository-local")
            self.directory.mkdir(parents=True,exist_ok=True)
        self.hits, self.new_jobs = 0, 0
        source = {name:hashlib.sha256((ROOT/"code/q1"/name).read_bytes()).hexdigest() for name in
                  ["classical_svi.py","torch_svi.py","neural_surface.py","neural_training.py","pricing.py","svi.py"]}
        self.implementation_sha256 = digest(source)
        self.method_implementation = {
            method:digest({name:source[name] for name in names}) for method,names in {
                "scipy":["classical_svi.py","pricing.py","svi.py"],"adam":["torch_svi.py","classical_svi.py","svi.py"],
                "lbfgs":["torch_svi.py","classical_svi.py","svi.py"],"neural":["neural_surface.py","neural_training.py","pricing.py"]}.items()}

    def get(self, key):
        path = self.directory/(key+".json") if self.directory else None
        if path is None or not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["key"] != key or digest(payload["result"]) != payload["result_sha256"]:
            raise ValueError("Fit-cache integrity mismatch")
        self.hits += 1
        return payload["result"]

    def put(self, key, result):
        self.new_jobs += 1
        if self.directory:
            # Atomic rename leaves no partially valid cache record after interruption.
            target = self.directory/(key+".json")
            temporary = self.directory/(key+".pending")
            with temporary.open("w",encoding="utf-8",newline="\n") as handle:
                json.dump({"key":key,"result_sha256":digest(result),"result":plain(result)},handle,indent=2,allow_nan=False)
            temporary.rename(target)

    def svi(self, key, compute):
        record = self.get(key)
        if record is None:
            record = compute()
            self.put(key,record)
        return record

    def neural(self, key, compute):
        result = self.get(key)
        if result is not None:
            model = load_checkpoint(self.directory/result["checkpoint"])[0] if result["training"]["success"] else None
            return model,result["training"]
        model, record = compute()
        checkpoint = None
        if self.directory and record["success"]:
            checkpoint = key+".pt"
            path = self.directory/checkpoint
            # A saved checkpoint without its cache record is not silently replaced.
            if path.exists():
                restored, metadata = load_checkpoint(path)
                if metadata["final_state_sha256"] != record["final_state_sha256"]:
                    raise ValueError("Interrupted checkpoint differs from recomputed model")
                model = restored
            else:
                save_checkpoint(path,model,record)
        self.put(key,{"training":record,"checkpoint":checkpoint})
        return model,record


def retained_arrays(rows):
    counts = Counter(row["strike"] for row in rows)
    return (np.array([row["_k"] for row in rows],dtype=np.float64),
            np.array([row["_w"] for row in rows],dtype=np.float64),
            np.array([1/counts[row["strike"]] for row in rows],dtype=np.float64))


def insufficient_fit(rows):
    return {"converged":False,"failure_reason":"INSUFFICIENT_SUPPORT","fitted_parameters":None,"initial_parameters":None,
            "objective_value":None,"fit_seconds":0.,"quote_count":len(rows),"distinct_k_count":len({row["strike"] for row in rows})}


def full_reference(rows, ledger, *, config, cache):
    key = "reference_"+digest({"rows":rows,"config":asdict(config)})
    cached = cache.get(key)
    if cached is not None:
        return cached
    directory = ROOT/"results/verification/approach1/q1_synthetic_svi_verification"
    if (directory/"artifact_manifest.json").is_file():
        manifest = json.loads((directory/"artifact_manifest.json").read_text())
        entries = {row["name"]:row for row in manifest["files"]}
        names = ["metrics.json","diagnostics.json"]
        if all(name in entries and hashlib.sha256((directory/name).read_bytes()).hexdigest() == entries[name]["sha256"] for name in names):
            previous = json.loads((directory/"diagnostics.json").read_text())
            metrics = json.loads((directory/"metrics.json").read_text())
            if digest(previous["quote_ledger"]) == digest(ledger) and metrics["params"]["config"] == plain(asdict(config)):
                result = {"slices":previous["slices"],"source":"REUSED_HASH_VERIFIED_M4_REFERENCE_FOR_ILLUSTRATION_ONLY",
                          "source_metrics_sha256":entries["metrics.json"]["sha256"],"used_for_refit_or_selection":False}
                cache.put(key,result)
                return result
    grouped = defaultdict(list)
    for row in rows:
        grouped[slice_key(row)].append(row)
    slices = []
    for name,group in sorted(grouped.items()):
        k,target,weights = retained_arrays(group)
        slices.append({"slice_id":name,"observation_date":group[0]["observation_date"],"expiry_date":group[0]["expiry_date"],
                       "time_to_expiry":group[0]["time_to_expiry"],"fit":fit_slice(k,target,weights=weights,config=config)})
    result = {"slices":slices,"source":"SCIPY_REFERENCE_FIT_ONCE_FOR_ILLUSTRATION_ONLY","used_for_refit_or_selection":False}
    cache.put(key,result)
    return result


def common_diagnostics(slices, *, config):
    """Same fixed expiry x k grid and shared M4 diagnostics for all methods."""
    report = verify_surface(slices,config)
    known = sum(row["status"] == "VERIFIED" for row in report["slices"])
    return {"report":report,"diagnosed_expiries":known,
            "missing_expiries":report["failed_slice_ids"],"counts":{
                "examination_butterfly":sum(row["examination"]["count"] for row in report["slices"] if row["examination"]) if known else None,
                "complete_butterfly":sum(row["complete"]["count"] for row in report["slices"] if row["complete"]) if known else None,
                "calendar":sum(row["count"] for row in report["calendar"]) if report["calendar"] else None,
                "negative_forward_variance":len(report["negative_forward_variance"]) if report["calendar"] else None}}


def neural_common_diagnostics(model, retained, templates, usable_expiries, *, config):
    # M6 dense diagnostics remain supplementary. Cross-method comparisons below
    # use ONLY the exact same observed expiries and M4 fixed k nodes.
    dense = evaluate_grid(model,retained,config=config)
    times = dense["maturities"]
    k = np.array(dense["k"])
    valid = [row for row in templates if row["expiry_date"] in usable_expiries]
    valid.sort(key=lambda row:row["time_to_expiry"])
    missing = [row["slice_id"] for row in templates if row["expiry_date"] not in usable_expiries]
    slices,calendar,negative = [],[],[]
    w = []
    for row in valid:
        index = times.index(row["time_to_expiry"])
        item = {"slice_id":row["slice_id"],"status":"VERIFIED","total_variance":dense["total_variance"][index]}
        for label in ["examination","complete"]:
            values = np.array(dense[label][index])
            diagnostic = svi.GridDiagnostics(k,values,values < 0,0.)
            item[label] = {"count":diagnostic.count,"locations":diagnostic.locations.tolist(),"magnitudes":diagnostic.magnitudes.tolist(),"values":values.tolist()}
        slices.append(item)
        w.append(item["total_variance"])
    for i in range(len(valid)):
        for j in range(i+1,len(valid)):
            test = svi.calendar_diagnostics(k,w[i],w[j],valid[i]["time_to_expiry"],valid[j]["time_to_expiry"])
            calendar.append({"earlier_slice":valid[i]["slice_id"],"later_slice":valid[j]["slice_id"],"count":test.count,
                             "locations":test.locations.tolist(),"magnitudes":test.magnitudes.tolist()})
    if len(valid) >= 2:
        fwd = svi.forward_variance_diagnostics([row["time_to_expiry"] for row in valid],w)
        negative = [{"earlier_slice":valid[i]["slice_id"],"later_slice":valid[i+1]["slice_id"],"k":float(k[j]),
                     "magnitude":float(-fwd.forward_variance[i,j])} for i,j in fwd.indices]
    return {"diagnosed_expiries":len(valid),"missing_expiries":missing,"dense_neural_supplement":dense,
            "report":{"grid":k.tolist(),"slices":slices,"calendar":calendar,"negative_forward_variance":negative,"failed_slice_ids":missing},
            "counts":{"examination_butterfly":sum(row["examination"]["count"] for row in slices),
                      "complete_butterfly":sum(row["complete"]["count"] for row in slices),
                      "calendar":sum(row["count"] for row in calendar) if calendar else None,
                      "negative_forward_variance":len(negative) if calendar else None}}


def prediction(row, method, seed, retained, *, parameters=None, model=None, reason=None, actual_training=None, price_floor=1e-8):
    record = {"seed":seed,"method":method,"quote_id":row["quote_id"],"observation_date":row["observation_date"],"expiry_date":row["expiry_date"],
              "option_type":row["option_type"],"k":row["_k"],"maturity":row["time_to_expiry"],"support_class":support_class(row,retained),
              "parameter_training_support_class":support_class(row,actual_training if actual_training is not None else retained),
              "market_iv":row["_iv"],"market_price":row["price"],"predicted_iv":None,"predicted_price":None,
              "iv_error":None,"absolute_iv_error":None,"absolute_price_error":None,
              "relative_price_error":None,"relative_error_status":None,"status":reason or "PREDICTION_FAILED","failure_reason":reason}
    if reason:
        return record
    try:
        if parameters is not None:
            sigma = float(np.sqrt(svi.svi_total_variance(row["_k"],parameters)/row["time_to_expiry"]))
        else:
            with torch.no_grad():
                sigma = float(model(torch.tensor(row["_k"],dtype=torch.float64),torch.tensor(row["time_to_expiry"],dtype=torch.float64)))
        price = pricing.black_scholes(row["forward"],row["strike"],row["time_to_expiry"],sigma,row["discount_factor"],row["option_type"])
        record.update(predicted_iv=sigma,predicted_price=price,status="PREDICTED",iv_error=sigma-row["_iv"],absolute_iv_error=abs(sigma-row["_iv"]),absolute_price_error=abs(price-row["price"]),
                      relative_price_error=abs(price-row["price"])/row["price"] if row["price"] >= price_floor else None,
                      relative_error_status="DEFINED" if row["price"] >= price_floor else "NEAR_ZERO_DENOMINATOR_EXCLUDED")
    except (ValueError,RuntimeError,TypeError,OverflowError) as error:
        record["failure_reason"] = str(error)
    return record


def run_mask(rows, manifest, *, config=CalibrationConfig(), optimizer_config=TorchOptimizerConfig(),
             surface_config=SurfaceConfig(), training_config=TrainingConfig(), cache=None):
    """Four methods receive precisely the same target IDs; held-out rows stay OUT
    of all fit function arguments, cache keys, normalization and early stopping.
    """
    cache = cache or FitCache()
    seed = manifest["seed"]
    retained_ids = {value for item in manifest["slices"] for value in item["retained_ids"]}
    dropped_ids = {value for item in manifest["slices"] for value in item["dropped_ids"]}
    if retained_ids & dropped_ids or retained_ids | dropped_ids != {row["quote_id"] for row in rows}:
        raise ValueError("Mask must partition universe IDs")
    retained = [row for row in rows if row["quote_id"] in retained_ids]
    dropped = [row for row in rows if row["quote_id"] in dropped_ids]
    groups, tests = defaultdict(list),defaultdict(list)
    for row in retained:
        groups[slice_key(row)].append(row)
    for row in dropped:
        tests[slice_key(row)].append(row)
    fits, predictions, slices = [],[],{method:[] for method in METHODS[:3]}
    for item in manifest["slices"]:
        key = item["slice_id"]
        observed, held = groups[key],tests[key]
        template_row = next(row for row in rows if slice_key(row) == key)
        template = {"slice_id":key,"observation_date":template_row["observation_date"],"expiry_date":template_row["expiry_date"],"time_to_expiry":template_row["time_to_expiry"]}
        k,target,weights = retained_arrays(observed)
        base = None
        for method in METHODS[:3]:
            job = digest({"method":method,"seed":seed,"retained":observed,"config":asdict(config),"optimizer":asdict(optimizer_config),"implementation":cache.method_implementation[method]})
            if item["status"] == "INSUFFICIENT_SUPPORT":
                record = insufficient_fit(observed)
            elif method == "scipy":
                record = cache.svi(job,lambda:fit_slice(k,target,weights=weights,config=config))
            else:
                initial = svi.SVIParameters(**base["initial_parameters"]) if base["initial_parameters"] else None
                record = cache.svi(job,lambda:fit_torch_slice(k,target,weights=weights,initial=initial,optimizer=method,config=config,optimizer_config=optimizer_config))
            if method == "scipy":
                base = record
            usable = record["converged"]
            slices[method].append({**template,"fit":record})
            fits.append({"seed":seed,"method":method,**template,"usable":usable,"converged":record["converged"],
                         "failure_reason":record["failure_reason"],"training_count":len(observed),"validation_count":0,"held_out_count":len(held),
                         "training_ids":[row["quote_id"] for row in observed],"validation_ids":[],"held_out_ids":[row["quote_id"] for row in held],
                         "available_strike_bounds":item["retained_strike_bounds"],"fit_seconds_attributed":record["fit_seconds"],
                         "configuration_sha256":job,"fit":record})
            p = svi.SVIParameters(**record["fitted_parameters"]) if usable else None
            predictions.extend(prediction(row,method,seed,observed,parameters=p,reason=None if usable else record["failure_reason"],price_floor=config.near_zero_price_floor) for row in held)
    diagnostics = []
    for method in METHODS[:3]:
        for date in sorted({item["observation_date"] for item in manifest["slices"]}):
            diagnostics.append({"seed":seed,"method":method,"observation_date":date,
                                **common_diagnostics([row for row in slices[method] if row["observation_date"] == date],config=config)})
    neural_training = []
    for date in sorted({item["observation_date"] for item in manifest["slices"]}):
        items = [item for item in manifest["slices"] if item["observation_date"] == date]
        supported = set()
        for item in items:
            n = len({(row["strike"],row["contract_unit"]) for row in retained if slice_key(row) == item["slice_id"]})
            if item["status"] != "INSUFFICIENT_SUPPORT" and n-max(1,int(n*training_config.validation_fraction)) >= 5:
                supported.add(item["slice_id"])
        development = [row for row in retained if slice_key(row) in supported]
        model,record,split = None,None,None
        try:
            split = retained_neural_split(development,seed,fraction=training_config.validation_fraction)
            job = digest({"method":"neural","seed":seed,"retained":development,"split":split,"config":asdict(training_config),"implementation":cache.method_implementation["neural"],
                          "surface":asdict(surface_config),"lambda_butterfly":1.,"lambda_calendar":1.})
            model,record = cache.neural(job,lambda:train_surface(development,split,seed=seed,surface_config=surface_config,
                training_config=training_config,lambda_butterfly=1.,lambda_calendar=1.,retained_only=True))
        except (ValueError,RuntimeError,TypeError) as error:
            record = {"success":False,"failure_reason":str(error),"fit_seconds":0.,"status":"FAILED"}
            job = digest({"method":"neural","seed":seed,"date":date,"retained":development,"failure":str(error)})
        usable = record["success"]
        train_rows = [development[i] for i in split["train"]] if split else []
        validation = [development[i] for i in split["validation"]] if split else []
        neural_training.append({"seed":seed,"observation_date":date,"configuration_sha256":job,"training":record,
                                "training_ids":[row["quote_id"] for row in train_rows],"validation_ids":[row["quote_id"] for row in validation]})
        count = len(items)
        templates = []
        for item in items:
            key = item["slice_id"]
            observed,held = groups[key],tests[key]
            template_row = next(row for row in rows if slice_key(row) == key)
            train = [row for row in train_rows if slice_key(row) == key]
            val = [row for row in validation if slice_key(row) == key]
            ok = usable and key in supported
            reason = None if ok else "INSUFFICIENT_SUPPORT" if key not in supported else record["failure_reason"]
            template = {"slice_id":key,"observation_date":date,"expiry_date":item["expiry_date"],"time_to_expiry":template_row["time_to_expiry"]}
            templates.append(template)
            fits.append({"seed":seed,"method":"neural",**template,"usable":ok,"converged":None,"failure_reason":reason,
                         "training_count":len(train),"validation_count":len(val),"held_out_count":len(held),
                         "training_ids":[row["quote_id"] for row in train],"validation_ids":[row["quote_id"] for row in val],
                         "held_out_ids":[row["quote_id"] for row in held],"available_strike_bounds":item["retained_strike_bounds"],
                         "fit_seconds_attributed":record["fit_seconds"]/count,"configuration_sha256":job,
                         "fit":{"success":ok,"objective_value":record.get("restored_training_total"),"stop_reason":record.get("stop_reason"),
                                "shared_dated_model":True,"mathematical_convergence_claimed":False}})
            predictions.extend(prediction(row,"neural",seed,observed,model=model,reason=reason,actual_training=train,price_floor=config.near_zero_price_floor) for row in held)
        if usable:
            try:
                report = neural_common_diagnostics(model,development,templates,{item["expiry_date"] for item in items if item["slice_id"] in supported},config=config)
            except (ValueError,RuntimeError,TypeError) as error:
                report = {"diagnosed_expiries":0,"missing_expiries":[item["slice_id"] for item in templates],"counts":{key:None for key in
                    ["examination_butterfly","complete_butterfly","calendar","negative_forward_variance"]},"evaluation_failure":str(error)}
        else:
            report = {"diagnosed_expiries":0,"missing_expiries":[item["slice_id"] for item in templates],"counts":{key:None for key in
                    ["examination_butterfly","complete_butterfly","calendar","negative_forward_variance"]}}
        diagnostics.append({"seed":seed,"method":"neural","observation_date":date,**report})
    for record in fits:
        selected = [row for row in predictions if row["method"] == record["method"] and slice_key(row) == record["slice_id"]]
        record["predicted_count"] = sum(row["status"] == "PREDICTED" for row in selected)
        record["prediction_coverage"] = record["predicted_count"]/len(selected) if selected else None
        record["retained_count"] = len(groups[record["slice_id"]])
        record["held_out_ids_sha256"] = digest(sorted(record["held_out_ids"]))
        record["parity_derived_inputs_used"] = False
        actual = [row["strike"] for row in retained if row["quote_id"] in record["training_ids"]]
        record["parameter_training_strike_bounds"] = [min(actual),max(actual)] if actual else None
    return {"fits":fits,"predictions":predictions,"diagnostics":diagnostics,"neural_training":neural_training}
