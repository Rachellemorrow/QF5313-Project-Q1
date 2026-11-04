"""Four prespecified penalty ablations; validation-only selection, no M7 study."""

from dataclasses import asdict
import hashlib
import json
import platform

import numpy as np
import torch

from .approach_a import plain, prepare_quotes
from .neural_evaluation import evaluate_grid, evaluate_quotes
from .neural_surface import SurfaceConfig
from .neural_training import TrainingConfig, paired_split, train_surface


def ablation_configs(nonzero_butterfly=1., nonzero_calendar=1.):
    if any(not np.isfinite(value) or value <= 0 for value in [nonzero_butterfly,nonzero_calendar]):
        raise ValueError("Nonzero ablation coefficients must be finite and positive")
    return {"A":(0.,0.),"B":(nonzero_butterfly,0.),"C":(0.,nonzero_calendar),"D":(nonzero_butterfly,nonzero_calendar)}


def run_ablations(frame, *, mode, seed=42, surface_config=SurfaceConfig(), training_config=TrainingConfig(),
                  nonzero_butterfly=1., nonzero_calendar=1.):
    ledger, parity = prepare_quotes(frame,mode=mode)
    configs = ablation_configs(nonzero_butterfly,nonzero_calendar)
    by_date = {}
    for record in ledger:
        by_date.setdefault(record["input"]["observation_date"],[]).append(record)
    dates, models = [], {}
    for date, records in sorted(by_date.items()):
        rows = [row["input"] for row in records if row["eligible_for_fit"]]
        item = {"observation_date":date,"input_quote_count":len(records),"eligible_quote_count":len(rows),
                "quote_preparation_rejections":len(records)-len(rows),"status":"FAILED","failure_reason":None,
                "ablations":{},"split":None,"selected_ablation":None}
        try:
            split = paired_split(rows,seed=seed,config=training_config)
            item["split"] = {name:[rows[i]["quote_id"] for i in indices] for name,indices in split.items()}
            item["split_sha256"] = hashlib.sha256(json.dumps(item["split"],sort_keys=True).encode()).hexdigest()
            # First run all training/validation. Do not access held-out evaluation
            # prices or diagnostics until coefficient/configuration selection ends.
            fitted = {}
            for name, (lb,lc) in configs.items():
                model, record = train_surface(rows,split,lambda_butterfly=lb,lambda_calendar=lc,seed=seed,
                                              surface_config=surface_config,training_config=training_config)
                item["ablations"][name] = {"training":record,"evaluation":None,"diagnostics":None,"evaluation_failure":None}
                fitted[name] = model
                if record["success"]:
                    models[(date,name)] = model
            successful = [name for name in configs if item["ablations"][name]["training"]["success"]]
            if successful:
                item["selected_ablation"] = min(successful,key=lambda name:(item["ablations"][name]["training"]["best_validation_price_mse"],name))
            for name in successful:
                try:
                    item["ablations"][name]["evaluation"] = evaluate_quotes(fitted[name],rows,split)
                    item["ablations"][name]["diagnostics"] = evaluate_grid(fitted[name],rows)
                except (ValueError,RuntimeError,TypeError,OverflowError) as error:
                    item["ablations"][name]["evaluation_failure"] = str(error)
            item["status"] = "COMPLETED" if len(successful) == 4 and all(item["ablations"][name]["diagnostics"] is not None for name in successful) else "PARTIAL"
        except (ValueError,RuntimeError,TypeError,OverflowError) as error:
            item["failure_reason"] = str(error)
        dates.append(item)
    return {"dates":dates,"quote_ledger":ledger,"parity":parity}, models


def neural_metrics(result, *, mode, dataset, seed, surface_config, training_config, eligibility=None, fixture=None, coverage=None,
                   nonzero_butterfly=1., nonzero_calendar=1.):
    dates = result["dates"] if result is not None else []
    rows = []
    for date in dates:
        for name, run in date["ablations"].items():
            training, evaluation, diagnostics = run["training"],run["evaluation"],run["diagnostics"]
            rows.append({"observation_date":date["observation_date"],"ablation":name,"success":training["success"],
                         "failure_reason":training["failure_reason"] or run["evaluation_failure"],"epochs":training["epochs_completed"],
                         "best_epoch":training["best_epoch"],"stop_reason":training.get("stop_reason"),
                         "fit_seconds":training["fit_seconds"],"best_validation_price_mse":training["best_validation_price_mse"],
                         "test_price_mse":evaluation["fold_metrics"]["test"]["price_mse"] if evaluation else None,
                         "examination_butterfly_violations":diagnostics["violations"]["examination"]["count"] if diagnostics else None,
                         "complete_butterfly_violations":diagnostics["violations"]["complete"]["count"] if diagnostics else None,
                         "calendar_derivative_violations":diagnostics["violations"]["w_T"]["count"] if diagnostics else None,
                         "calendar_pair_violations":diagnostics["calendar_violation_count"] if diagnostics else None,
                         "negative_forward_variance_locations":diagnostics["negative_forward_variance_count"] if diagnostics else None})
    passed = bool(dates) and all(date["status"] == "COMPLETED" and not date["quote_preparation_rejections"]
        and all(run["training"]["best_validation_price_mse"] < run["training"]["initial_validation_price_mse"] for run in date["ablations"].values()) for date in dates)
    software = "PASS" if passed else "PARTIAL"
    empirical = "BLOCKED" if mode == "development" or result is None else "PASS" if coverage and coverage.get("examination_historical_requirement_satisfied") else "PARTIAL"
    return plain({"params":{"dataset":dataset,"mode":mode,"component":"neural","seed":seed,"dtype":"float64","device":"cpu",
                            "surface_config":asdict(surface_config),"training_config":asdict(training_config),
                            "ablation_coefficients":ablation_configs(nonzero_butterfly,nonzero_calendar),"fixture":fixture,"pipeline_coverage":coverage,
                            "empirical_comparison_eligible":mode == "empirical" and bool(dates),
                            "selection_protocol":"Single prespecified positive candidate per coefficient; compare A/B/C/D using best validation price MSE; ties A,B,C,D. No held-out evaluation tuning.",
                            "split_protocol":"Seeded expiry-stratified paired strikes: train/validation/test; fit and normalization use train only. Not M7 missing-quote robustness.",
                            "layer_counting":"Affine layers including output; activations and normalization are not counted",
                            "data_origin":"SYNTHETIC_SOFTWARE_VERIFICATION_ONLY" if mode == "development" else "REAL_ARCHIVED_OBSERVATION"},
                  "metrics":{"ablation_rows":rows,"attempted_dates":len(dates),"completed_dates":sum(date["status"] == "COMPLETED" for date in dates)},
                  "fit_seconds":sum(row["fit_seconds"] for row in rows),
                  "history":{"SOFTWARE_VERIFICATION":software,"EMPIRICAL_REQUIREMENTS":empirical,
                             "status":"BLOCKED" if result is None else "COMPLETED" if passed else "COMPLETED_WITH_FAILURES_OR_NO_VALIDATION_IMPROVEMENT",
                             "software_acceptance":"All four ablations finite, evaluable and validation-improving, with no preparation rejections. Accuracy/arbitrage benchmarks reported separately, not guaranteed.",
                             "eligibility":eligibility,"result":result,"figures":[],"checkpoints":[],
                             "m5_status":"PARTIAL: existing L-BFGS failure and recovery/domain differences unchanged; not re-optimized in M6",
                             "runtime":{"python":platform.python_version(),"numpy":np.__version__,"torch":str(torch.__version__),
                                        "platform":platform.platform(),"processor":platform.processor(),"torch_num_threads":torch.get_num_threads()},
                             "determinism":"Seeded CPU float64, no dropout/batchnorm; wall times vary and bitwise cross-platform reproducibility is not guaranteed."}})
