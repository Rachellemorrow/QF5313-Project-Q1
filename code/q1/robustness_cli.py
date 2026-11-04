"""M7 execution, immutable component cache, split manifests and course metrics."""

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from .approach_a import empirical_gate, plain
from .classical_svi import CalibrationConfig
from .market_data import load_registered_q1
from .neural_surface import SurfaceConfig
from .neural_training import TrainingConfig
from .robustness import FitCache, full_reference, prepare_universe, run_mask
from .robustness_figures import write_robustness_figures
from .robustness_protocol import METHODS, SEEDS, digest, quote_drop_manifest, summarize_predictions, validate_manifest
from .torch_svi import TorchOptimizerConfig
from .verification_fixtures import development_quotes


ROOT = Path(__file__).resolve().parents[2]


def write_json(path, payload):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != plain(payload):
            raise ValueError("Refusing to overwrite different evidence: "+str(path))
        return
    temporary = path.with_suffix(path.suffix+".pending")
    with temporary.open("w",encoding="utf-8",newline="\n") as handle:
        json.dump(plain(payload),handle,indent=2,ensure_ascii=False,allow_nan=False)
    temporary.rename(path)


def validate_run_leakage(rows, manifest, result):
    dropped = {value for item in manifest["slices"] for value in item["dropped_ids"]}
    retained = {value for item in manifest["slices"] for value in item["retained_ids"]}
    for method in METHODS:
        attempts = [row["quote_id"] for row in result["predictions"] if row["method"] == method]
        if set(attempts) != dropped or len(attempts) != len(dropped):
            raise ValueError("Methods must attempt exactly the same held-out quote IDs")
    for fit in result["fits"]:
        development = set(fit["training_ids"]) | set(fit["validation_ids"])
        if development & dropped or not development <= retained or set(fit["training_ids"]) & set(fit["validation_ids"]):
            raise ValueError("Held-out or validation information leaked into fitting")
    if retained | dropped != {row["quote_id"] for row in rows}:
        raise ValueError("Manifest no longer partitions universe")


def execute_study(rows, ledger, parity, *, mode, dataset, seeds, config, optimizer_config, surface_config, training_config,
                  cache, output=None, fixture=None, eligibility=None, coverage=None, smoke=False):
    settings = {"protocol":"Q1_M7_QUOTE_LEVEL_V1","mode":mode,"dataset":dataset,"seeds":list(seeds),"smoke":smoke,
                "requested_drop_fraction":.3,"universe_sha256":digest(rows),"model_implementation_sha256":cache.implementation_sha256,
                "calibration":{**asdict(config),"grid":config.grid().tolist()},"torch_optimizer":asdict(optimizer_config),
                "surface":asdict(surface_config),"neural_training":asdict(training_config),
                "neural_penalty_preset":{"lambda_butterfly":1.,"lambda_calendar":1.,"decision":"USER_APPROVED_FIXED_D_NO_M7_COEFFICIENT_SELECTION"}}
    config_hash = digest(settings)
    settings["configuration_sha256"] = config_hash
    if output is not None:
        write_json(Path(output)/"configuration.json",settings)
        (Path(output)/"split_manifests").mkdir(exist_ok=True)
    manifests, fits, predictions, diagnostics, neural_training = [],[],[],[],[]
    reference = full_reference(rows,ledger,config=config,cache=cache) if rows else None
    for seed in seeds if rows else []:
        manifest = quote_drop_manifest(rows,seed,configuration_sha256=config_hash)
        validate_manifest(rows,manifest,config_hash)
        if output is not None:
            write_json(Path(output)/"split_manifests"/("seed_"+str(seed)+".json"),manifest)
        manifests.append(manifest)
        result = run_mask(rows,manifest,config=config,optimizer_config=optimizer_config,surface_config=surface_config,
                          training_config=training_config,cache=cache)
        validate_run_leakage(rows,manifest,result)
        fits.extend(result["fits"]); predictions.extend(result["predictions"])
        diagnostics.extend(result["diagnostics"]); neural_training.extend(result["neural_training"])
    summary = summarize_predictions(predictions,fits,diagnostics,seeds)
    complete = bool(rows) and all(row["usable"] for row in fits) and all(row["status"] == "PREDICTED" for row in predictions)
    complete = complete and all(not row["missing_expiries"] for row in diagnostics)
    software = "PASS" if complete else "PARTIAL"
    empirical = "BLOCKED" if mode == "development" or not rows else "PASS" if coverage and coverage.get("examination_historical_requirement_satisfied") else "PARTIAL"
    return plain({"params":{**settings,"fixture":fixture,"pipeline_coverage":coverage,
                             "empirical_comparison_eligible":mode == "empirical" and any(row["usable"] for row in fits),
                             "neural_early_stopping":"Retained-only validation, fresh initialization per mask; NEVER reuse M6 weights or full-universe selected C",
                             "fairness_caveat":"SVI fits all retained quotes; neural reserves internal retained validation. Objectives, coordinates, feasible faces and stopping rules differ.",
                             "std_convention":"Across seeds, sample standard deviation ddof=1; unavailable/empty categories null, with seed counts",
                             "relative_error_policy":"Positive market price <1e-8: retain absolute errors, null relative error; denominator never clipped",
                             "reference_warning":"Full-data SVI reference is descriptive only; not a competitive held-out method or initializer"},
                  "metrics":summary,"fit_seconds":sum(row["fit_seconds_attributed"] for row in fits),
                  "history":{"SOFTWARE_VERIFICATION":software,"EMPIRICAL_REQUIREMENTS":empirical,
                             "status":"BLOCKED" if not rows else "COMPLETED" if complete else "COMPLETED_WITH_FIT_FAILURES",
                             "universe":rows,"quote_ledger":ledger,"full_data_parity_illustration_only":parity,
                             "full_reference":reference,"manifests":manifests,"fits":fits,"predictions":predictions,"diagnostics":diagnostics,
                             "neural_training":neural_training,"eligibility":eligibility,"figures":[],
                             "all_held_out_ids_matched":True if rows else None,"cache_hits":cache.hits,"new_cache_jobs":cache.new_jobs,
                             "runtime":{"python":platform.python_version(),"numpy":np.__version__,"torch":str(torch.__version__),
                                        "platform":platform.platform(),"processor":platform.processor(),"torch_num_threads":torch.get_num_threads()},
                             "limitations":["Random quote deletion mostly tests interpolation and may retain opposite-leg near-identical IV information",
                                 "No parity-derived F/D or cross-leg imputation is used; exogenous conventions frozen before deletion",
                                 "No universal/global no-arbitrage or empirical benchmark claim; failed fits retain missing diagnostics",
                                 "Neural finite MAX_EPOCHS termination is not mathematical convergence",
                                 "Single fixed-order logged timings are not isolated repeated performance benchmarks"]}})


def write_outputs(output, metrics):
    output = Path(output)
    if metrics["history"]["universe"]:
        metrics["history"]["figures"] = write_robustness_figures(output,metrics)
    write_json(output/"robustness_summary.json",metrics["metrics"])
    write_json(output/"failure_ledger.json",[row for row in metrics["history"]["fits"] if not row["usable"]])
    write_json(output/"arbitrage_summary.json",[{key:row[key] for key in ["seed","method","observation_date","diagnosed_expiries","missing_expiries","counts"]}
                                               for row in metrics["history"]["diagnostics"]])
    for name, rows in [("held_out_predictions.csv",metrics["history"]["predictions"]),("per_seed_comparison.csv",metrics["metrics"]["per_seed"])]:
        path = output/name
        if rows and not path.exists():
            fields = [key for key,value in rows[0].items() if not isinstance(value,(dict,list))]
            with path.open("x",encoding="utf-8",newline="") as handle:
                writer = csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore",lineterminator="\n")
                writer.writeheader(); writer.writerows(rows)
    write_json(output/"metrics.json",metrics)
    files = [{"name":path.relative_to(output).as_posix(),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_manifest.json" and not path.name.endswith(".pending")]
    write_json(output/"artifact_manifest.json",{"configuration_sha256":metrics["params"]["configuration_sha256"],"files":files})


def verify_artifacts(output):
    output = Path(output)
    manifest = json.loads((output/"artifact_manifest.json").read_text())
    for item in manifest["files"]:
        path = (output/item["name"]).resolve()
        if not path.is_relative_to(output.resolve()) or path.stat().st_size != item["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Artifact integrity failure: "+item["name"])
    metrics = json.loads((output/"metrics.json").read_text(encoding="utf-8"))
    if metrics["params"]["configuration_sha256"] != manifest["configuration_sha256"]:
        raise ValueError("Manifest/metrics configuration mismatch")
    return metrics


def run_robustness(args, parser):
    suffix = "_smoke" if args.smoke else ""
    dataset = "q1_synthetic_missing_quotes"+suffix if args.mode == "development" else "q1_510050_missing_quotes"+suffix
    if args.dataset != dataset:
        parser.error("Robustness mode/dataset mismatch; use "+dataset)
    seeds = tuple(args.seeds)
    if not args.smoke and seeds != SEEDS:
        parser.error("Complete M7 protocol requires seeds 42 43 44 in that order")
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 for seed in seeds):
        parser.error("Require distinct integer seeds in [0,2**32)")
    if args.neural_threads < 1:
        parser.error("neural-threads must be positive")
    try:
        config = CalibrationConfig()
        optimizer = TorchOptimizerConfig(adam_max_steps=2,lbfgs_max_iter=2) if args.smoke else TorchOptimizerConfig()
        training = TrainingConfig(epochs=5,patience=5,collocation_k_points=5,collocation_t_points=3) if args.smoke else TrainingConfig()
        surface = SurfaceConfig()
    except ValueError as error:
        parser.error(str(error))
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("Output must remain repository-local")
    output = output/"verification"/"approach2"/dataset if args.mode == "development" else output/"approach2"/dataset
    if output.exists() and not args.resume:
        parser.error("Run exists; use --resume for this immutable experiment or a new output base")
    if args.resume and (output/"metrics.json").exists() and (output/"artifact_manifest.json").exists():
        metrics = verify_artifacts(output)
        if metrics["params"]["seeds"] != list(seeds) or metrics["params"]["smoke"] != args.smoke:
            parser.error("Resume settings differ from completed experiment")
    else:
        output.mkdir(parents=True,exist_ok=True)
        torch.set_num_threads(args.neural_threads)
        fixture = coverage = eligibility = None
        if args.mode == "development":
            frame,fixture = development_quotes()
        else:
            loaded = load_registered_q1()
            coverage = loaded["coverage"]
            frame,eligibility = empirical_gate(loaded["quotes"])
        if len(frame):
            rows,ledger,parity = prepare_universe(frame,mode=args.mode,config=config)
        else:
            rows,ledger,parity = [],[],[]
        metrics = execute_study(rows,ledger,parity,mode=args.mode,dataset=dataset,seeds=seeds,config=config,optimizer_config=optimizer,
                                surface_config=surface,training_config=training,cache=FitCache(output/"fits"),output=output,
                                fixture=fixture,eligibility=eligibility,coverage=coverage,smoke=args.smoke)
        write_outputs(output,metrics)
        verify_artifacts(output)
    console = Console()
    console.print("[bold red]"+("SYNTHETIC M7 VERIFICATION — NOT EMPIRICAL MARKET RESULTS" if args.mode == "development" else "STRICT EMPIRICAL ROBUSTNESS GATE")+"[/bold red]")
    table = Table("Method","Predictions / targets","Failed expiry fits","Available IV RMSE mean","Available price MAE mean")
    for method,summary in metrics["metrics"]["seed_aggregates"].items():
        pooled = summary["pooled_with_coverage"]
        table.add_row(method,f"{pooled['predicted']}/{pooled['held_out_expected']}",str(summary["failed_expiry_fits"]["mean"]),
                      str(summary["iv_rmse"]["mean"]),str(summary["price_mae"]["mean"]))
    console.print(table)
    console.print(f"SOFTWARE_VERIFICATION: {metrics['history']['SOFTWARE_VERIFICATION']}; EMPIRICAL_REQUIREMENTS: {metrics['history']['EMPIRICAL_REQUIREMENTS']}")
    console.print(str(output))
    return 2 if metrics["history"]["status"] == "BLOCKED" else 0 if metrics["history"]["SOFTWARE_VERIFICATION"] == "PASS" else 1
