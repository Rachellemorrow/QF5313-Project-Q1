"""Export staged Git source to an empty in-repository checkout and replay Q1.

No original ignored results/model caches are consulted. Shared pre-provisioned
pinned Python dependencies are explicit. No fitting or training is performed.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]


def inside():
    import socket
    socket.socket.connect = lambda *a,**k:(_ for _ in ()).throw(RuntimeError("Network forbidden in isolated Q1 reproduction"))
    sys.path.insert(0,str(ROOT/"code"))
    from q1.validation_bundle import restore
    from q1.market_data import build_pipeline, load_registered_q1
    from q1.final_audit import dense_check
    import importlib.util
    import hashlib
    import torch
    torch.set_num_threads(1)
    start = perf_counter()
    assert not (ROOT/"results").exists(), "Ignored experiment outputs must not preexist"
    evidence = {"source_root":str(ROOT),"python":sys.version,"interpreter":sys.executable,
                "dependencies":"Existing repository-managed pinned environment reused read-only, NOT a fresh offline installation",
                "no_prior_ignored_results":True,"no_model_training":True,"network_connections_blocked":True,"steps":[]}
    t = perf_counter()
    manifest = restore(ROOT/"tests/q1/fixtures/verified_runs",ROOT/"replayed")
    evidence["steps"].append({"action":"restore hash-verified tracked evidence snapshots","seconds":perf_counter()-t,"files":len(manifest["files"])})
    t = perf_counter()
    generated = build_pipeline(ROOT/".tmp/regenerated_m2")
    expected = (ROOT/"data/clean/q1/m2_pipeline/quotes.csv").read_bytes()
    actual = (ROOT/".tmp/regenerated_m2/quotes.csv").read_bytes()
    assert hashlib.sha256(actual).digest() == hashlib.sha256(expected).digest()
    evidence["steps"].append({"action":"rebuild M2 from Git-tracked M1/M1.5 snapshots","seconds":perf_counter()-t,"quote_csv_byte_identical":True,
                              "coverage":generated["coverage"]})
    t = perf_counter()
    spec = importlib.util.spec_from_file_location("comparison",ROOT/"code/compare.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    report = module.load_comparison(ROOT/"replayed/results")
    assert len(report["runs"]) == 8 and not report["issues"]
    module.write_comparison(report,ROOT/"results/comparison.html")
    evidence["steps"].append({"action":"regenerate comparison JSON/CSV/self-contained HTML from replayed numeric evidence","seconds":perf_counter()-t,
                              "runs":len(report["runs"]),"rows":len(report["rows"]),"issues":report["issues"]})
    t = perf_counter()
    check = dense_check(ROOT/"replayed/results/verification/approach2/q1_synthetic_missing_quotes")
    assert check["genuine_model_behavior_not_a_numerical_sign_error"]
    evidence["steps"].append({"action":"checkpoint inference and physical finite-difference validation","seconds":perf_counter()-t,"result":check})
    commands = [["code/approach1_q1_svi.py","--dataset","q1_510050"],
                ["code/approach2_q1_torch.py","--component","svi","--dataset","q1_510050"],
                ["code/approach2_q1_torch.py","--component","neural","--dataset","q1_510050_neural"],
                ["code/approach2_q1_torch.py","--component","robustness","--dataset","q1_510050_missing_quotes"]]
    for args in commands:
        t = perf_counter()
        completed = subprocess.run([sys.executable,*args,"--mode","empirical"],cwd=ROOT,capture_output=True,text=True)
        assert completed.returncode == 2, completed.stderr
        evidence["steps"].append({"action":"regenerate strict BLOCKED gate","command":[sys.executable,*args,"--mode","empirical"],
                                  "seconds":perf_counter()-t,"exit":completed.returncode})
    evidence["limitations"] = ["This is lossless artifact replay/inference and real offline-data rebuild, NOT independent refitting of M4-M7",
        "Original milestone HTML is indexed and preserved in the main checkout; not redundantly regenerated here",
        "A first-time Python/uv/dependency installation still needs network or an independently provisioned wheel/interpreter cache",
        "All snapshots are tracked software evidence/public-source audit records, not a compliant historical ETF panel"]
    evidence["total_seconds"] = perf_counter()-start
    (ROOT/"reproduction_evidence.json").write_text(json.dumps(evidence,indent=2),encoding="utf-8",newline="\n")
    print(json.dumps({"isolated_replay":"PASS","seconds":evidence["total_seconds"],"training":False,"runs":len(report["runs"]),"rows":len(report["rows"])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination",type=Path)
    parser.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inside:
        inside(); return
    if args.destination is None:
        parser.error("Supply a new --destination inside repository")
    destination = args.destination.resolve()
    if not destination.is_relative_to(ROOT) or destination.exists():
        parser.error("Destination must be new and repository-local")
    destination.mkdir(parents=True)
    subprocess.run(["git","checkout-index","--all","--prefix="+destination.as_posix()+"/"],cwd=ROOT,check=True)
    env = os.environ.copy()
    # All generated caches/temporary files stay within the isolated checkout.
    for name,relative in {"PYTHONPYCACHEPREFIX":".cache/pycache","TEMP":".tmp","TMP":".tmp","TMPDIR":".tmp","XDG_CACHE_HOME":".cache"}.items():
        path = destination/relative; path.mkdir(parents=True,exist_ok=True); env[name] = str(path)
    env.pop("PYTHONPATH",None)
    subprocess.run([sys.executable,str(destination/"scripts/q1_isolated_replay.py"),"--inside"],cwd=destination,env=env,check=True)


if __name__ == "__main__":
    main()
