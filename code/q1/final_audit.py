"""Targeted checkpoint verification and source/snapshot integrity; no training."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .neural_training import load_checkpoint
from .neural_surface import surface_derivatives
from .neural_evaluation import evaluate_grid
from .provenance import verify_existing_inputs
from .market_data import load_registered_q1


ROOT = Path(__file__).resolve().parents[2]


def dense_check(run):
    run = Path(run).resolve()
    m = json.loads((run/"metrics.json").read_bytes())
    training = next(p for p in m["history"]["neural_training"] if p["seed"] == 44)
    checkpoint = run/"fits"/(training["configuration_sha256"]+".pt")
    model,_ = load_checkpoint(checkpoint)
    saved = next(p["dense_neural_supplement"] for p in m["history"]["diagnostics"] if p["method"] == "neural" and p["seed"] == 44)
    rows = m["history"]["universe"]
    # Evaluation coordinates are frozen from the saved experiment. No labels are
    # used for fitting/selection; inference only on the existing checkpoint.
    measured = evaluate_grid(model,rows)
    keys = {"negative_w_T":("w_T",402),"calendar_pairs":("calendar_violation_count",448),"negative_forward":("negative_forward_variance_count",272)}
    counts = {"negative_w_T":measured["violations"]["w_T"]["count"],"calendar_pairs":measured["calendar_violation_count"],"negative_forward":measured["negative_forward_variance_count"]}
    saved_counts = {"negative_w_T":saved["violations"]["w_T"]["count"],"calendar_pairs":saved["calendar_violation_count"],"negative_forward":saved["negative_forward_variance_count"]}
    point = max(saved["violations"]["w_T"]["locations"],key=lambda p:p["magnitude"])
    k,t = torch.tensor(point["k"],dtype=torch.float64),torch.tensor(point["maturity"],dtype=torch.float64)
    derivative = float(surface_derivatives(model,k,t,second=False)["w_T"].detach())
    steps = []
    with torch.no_grad():
        for h in [1e-3,1e-4,1e-5,1e-6]:
            fd = float((model(k,t+h).square()*(t+h)-model(k,t-h).square()*(t-h))/(2*h))
            steps.append({"step":h,"finite_difference":fd,"absolute_difference_from_autograd":abs(fd-derivative)})
    pair = max((p for p in saved["calendar"] if p["count"]),key=lambda p:max(p["magnitudes"]))
    index = int(np.argmax(pair["magnitudes"]))
    location = pair["locations"][index]
    kk = torch.tensor(location,dtype=torch.float64)
    with torch.no_grad():
        t1,t2 = [torch.tensor(pair[name],dtype=torch.float64) for name in ["earlier_maturity","later_maturity"]]
        difference = float(model(kk,t2).square()*t2-model(kk,t1).square()*t1)
    genuine = counts == saved_counts and derivative < 0 and all(p["finite_difference"] < 0 for p in steps) and difference < 0
    return {"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),"counts_recomputed":counts,"counts_saved":saved_counts,
            "point":point,"autograd_w_T":derivative,"finite_differences":steps,
            "independent_pair":{"k":location,"earlier_T":float(t1),"later_T":float(t2),"later_minus_earlier_w":difference},
            "genuine_model_behavior_not_a_numerical_sign_error":genuine,"training_performed":False}


def data_check():
    manifests = verify_existing_inputs()
    loaded = load_registered_q1()
    return {"source_manifests":len(manifests),"verified_source_files":sum(len(m["files"]) for m in manifests),
            "coverage":loaded["coverage"],"rows":len(loaded["quotes"]),"metadata_verified_rows":loaded["coverage"]["date_effective_metadata_verified_rows"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        parser.error("Choose a new repository-local output file")
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    result = {"data":data_check(),"dense_neural":dense_check(args.run)}
    output.write_text(json.dumps(result,indent=2),encoding="utf-8",newline="\n")
    print(json.dumps({"rows":result["data"]["rows"],"eligible":result["data"]["coverage"]["strictly_usable_for_empirical_analysis"]["quotes"],
                      "genuine_dense_violations":result["dense_neural"]["genuine_model_behavior_not_a_numerical_sign_error"],"output":str(output)}))
