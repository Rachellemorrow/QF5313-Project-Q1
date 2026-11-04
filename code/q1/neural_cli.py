"""Neural component of the existing Approach B CLI; preserves M5 output paths."""

import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from .approach_a import empirical_gate, plain
from .market_data import load_registered_q1
from .neural_experiment import ablation_configs, neural_metrics, run_ablations
from .neural_figures import write_neural_figures
from .neural_surface import SurfaceConfig
from .neural_training import TrainingConfig, load_checkpoint, save_checkpoint, state_fingerprint
from .verification_fixtures import development_quotes


ROOT = Path(__file__).resolve().parents[2]


def save_neural_run(output, metrics, models):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must remain repository-local")
    output.mkdir(parents=True,exist_ok=False)
    for (date,case),model in sorted(models.items()):
        training = next(item for item in metrics["history"]["result"]["dates"] if item["observation_date"] == date)["ablations"][case]["training"]
        name = "model_"+str(date)+"_"+case+".pt"
        metadata = save_checkpoint(output/name,model,training)
        restored,_ = load_checkpoint(output/name)
        if state_fingerprint(restored) != state_fingerprint(model):
            raise ValueError("Saved checkpoint restoration failed")
        metrics["history"]["checkpoints"].append({"file":name,"observation_date":date,"ablation":case,"metadata":metadata,"restoration_verified":True})
    if metrics["history"]["result"] is not None:
        if models:
            metrics["history"]["figures"] = write_neural_figures(output,metrics)
        rows = metrics["metrics"]["ablation_rows"]
        if rows:
            with (output/"ablation_comparison.csv").open("x",encoding="utf-8",newline="") as handle:
                writer = csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n")
                writer.writeheader(); writer.writerows(rows)
    with (output/"metrics.json").open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(plain(metrics),handle,indent=2,ensure_ascii=False,allow_nan=False)
    manifest = [{"name":path.name,"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in sorted(output.iterdir()) if path.is_file()]
    with (output/"artifact_manifest.json").open("x",encoding="utf-8",newline="\n") as handle:
        json.dump({"component":"neural","mode":metrics["params"]["mode"],"files":manifest},handle,indent=2)


def run_neural(args, parser):
    dataset = "q1_synthetic_neural_verification" if args.mode == "development" else "q1_510050_neural"
    if args.dataset != dataset:
        parser.error("Neural mode/dataset origin mismatch; use "+dataset)
    if not 0 <= args.seed < 2**32:
        parser.error("seed must lie in [0,2**32)")
    try:
        surface_config = SurfaceConfig(hidden=tuple(args.hidden),activation=args.activation)
        training_config = TrainingConfig(epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.learning_rate,patience=args.patience)
        ablation_configs(args.lambda_butterfly,args.lambda_calendar)
        if args.neural_threads < 1:
            raise ValueError("neural-threads must be positive")
    except ValueError as error:
        parser.error(str(error))
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("Output must remain repository-local")
    output = output/"verification"/"approach2"/dataset if args.mode == "development" else output/"approach2"/dataset
    if output.exists():
        parser.error("Run already exists; choose a new repository-local output base")
    # Process-local CPU control only; no user/environment configuration changes.
    torch.set_num_threads(args.neural_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    fixture = eligibility = coverage = result = None
    models = {}
    if args.mode == "development":
        frame, fixture = development_quotes()
    else:
        loaded = load_registered_q1()
        coverage = loaded["coverage"]
        frame, eligibility = empirical_gate(loaded["quotes"])
    if len(frame):
        result, models = run_ablations(frame,mode=args.mode,seed=args.seed,surface_config=surface_config,training_config=training_config,
                                       nonzero_butterfly=args.lambda_butterfly,nonzero_calendar=args.lambda_calendar)
    metrics = neural_metrics(result,mode=args.mode,dataset=dataset,seed=args.seed,surface_config=surface_config,training_config=training_config,
                             eligibility=eligibility,fixture=fixture,coverage=coverage,
                             nonzero_butterfly=args.lambda_butterfly,nonzero_calendar=args.lambda_calendar)
    save_neural_run(output,metrics,models)
    console = Console()
    console.print("[bold red]"+("SYNTHETIC NEURAL SOFTWARE VERIFICATION — NOT EMPIRICAL RESULTS" if args.mode == "development" else "STRICT EMPIRICAL NEURAL GATE")+"[/bold red]")
    table = Table("Date / ablation","Best validation MSE","Held-out price MSE","Epochs","Failure")
    for row in metrics["metrics"]["ablation_rows"]:
        table.add_row(row["observation_date"]+" / "+row["ablation"],str(row["best_validation_price_mse"]),str(row["test_price_mse"]),str(row["epochs"]),row["failure_reason"] or "")
    console.print(table)
    console.print(f"SOFTWARE_VERIFICATION: {metrics['history']['SOFTWARE_VERIFICATION']}; EMPIRICAL_REQUIREMENTS: {metrics['history']['EMPIRICAL_REQUIREMENTS']}")
    console.print(str(output))
    return 2 if result is None else 0 if metrics["history"]["SOFTWARE_VERIFICATION"] == "PASS" else 1
