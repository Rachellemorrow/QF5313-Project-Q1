"""Read-only group comparison: strict schema, separate question/dataset/mode/cohort.

Only Q1 has specialized adapters. Other questions retain generic schema-compatible
records and their raw metrics, without invented Q1 interpretation or rankings.
"""

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re

import plotly.graph_objects as go
from rich.console import Console
from rich.table import Table

from q1.comparison import q1_rows


ROOT = Path(__file__).resolve().parents[1]


def validate_metrics(value):
    if not isinstance(value,dict) or not {"params","metrics","history","fit_seconds"} <= value.keys():
        raise ValueError("Required metrics schema: params, metrics, fit_seconds, history")
    if any(not isinstance(value[k],dict) for k in ["params","metrics","history"]):
        raise ValueError("params/metrics/history must be objects")
    if isinstance(value["fit_seconds"],bool) or not isinstance(value["fit_seconds"],(int,float)) or not math.isfinite(value["fit_seconds"]) or value["fit_seconds"] < 0:
        raise ValueError("fit_seconds must be finite and nonnegative")
    return value


def load_comparison(root):
    root = Path(root).resolve()
    records,rows,issues,figures = [],[],[],[]
    for path in sorted(root.rglob("metrics.json")):
        if "fits" in path.relative_to(root).parts:
            continue
        try:
            raw = path.read_bytes()
            data = validate_metrics(json.loads(raw,parse_constant=lambda s:(_ for _ in ()).throw(ValueError("Nonfinite JSON"))))
            params = data["params"]
            dataset = params.get("dataset")
            if not isinstance(dataset,str) or not re.match(r"q[1234]_",dataset):
                raise ValueError("Require explicit question-prefixed dataset identifier")
            question = dataset.split("_",1)[0]
            mode = params.get("mode","UNSPECIFIED")
            origin = str(params.get("data_origin","")).upper()
            synthetic = mode == "development" or "SYNTHETIC" in origin or "synthetic" in dataset.lower()
            if mode == "empirical" and synthetic:
                raise ValueError("Conflicting synthetic/empirical provenance")
            mode = "SYNTHETIC_VERIFICATION" if synthetic else "EMPIRICAL" if mode == "empirical" else "UNSPECIFIED_NOT_EMPIRICAL"
            source = path.relative_to(root).as_posix()
            run_id = hashlib.sha256(raw).hexdigest()[:16]
            base = {"question":question,"dataset":dataset,"mode":mode,"run_id":run_id,"source":source,
                    "software_status":data["history"].get("SOFTWARE_VERIFICATION"),"empirical_status":data["history"].get("EMPIRICAL_REQUIREMENTS"),
                    "empirical_eligible":params.get("empirical_comparison_eligible",False),"run_fit_seconds":data["fit_seconds"]}
            auxiliary = None
            if (path.parent/"diagnostics.json").is_file():
                auxiliary = json.loads((path.parent/"diagnostics.json").read_bytes())
            adapted = q1_rows(data,auxiliary=auxiliary) if question == "q1" else [{"model":params.get("model",params.get("approach","UNSPECIFIED")),
                       "cohort":"OTHER_QUESTION_GENERIC","status":data["history"].get("status"),"raw_metrics":data["metrics"],
                       "other_question_metrics":json.dumps(data["metrics"],ensure_ascii=False),"fit_seconds":data["fit_seconds"]}]
            records.append({**base,"params":params,"metrics":data["metrics"],"status":data["history"].get("status"),"sha256":hashlib.sha256(raw).hexdigest()})
            rows.extend({**base,**row,"errors_conditional_on_available_predictions":True,
                         "complete_prediction_cohort":row.get("expected",0) > 0 and row.get("predicted") == row.get("expected")} for row in adapted)
            for name in data["history"].get("figures",[]):
                figure = (path.parent/name).resolve()
                if not figure.is_relative_to(root):
                    raise ValueError("Figure path escapes input root")
                figures.append({**base,"file":figure.relative_to(root).as_posix(),"exists":figure.is_file(),"caption":name.replace("_"," ").removesuffix(".html")})
        except (ValueError,KeyError,TypeError,OSError) as error:
            issues.append({"source":path.relative_to(root).as_posix(),"error":str(error)})
    groups = {}
    for row in rows:
        if row.get("cohort") == "M7_DROPPED_30_PERCENT":
            key = tuple(row.get(k) for k in ["question","dataset","mode","run_id","seed","observation_date","expiry_date"])
            groups.setdefault(key,[]).append(row)
    rankings = []
    for group in groups.values():
        if {row["model"] for row in group} != {"scipy","adam","lbfgs","neural"}:
            continue
        if not all(row.get("coverage") == 1 and row.get("price_mae") is not None for row in group) or len({row["held_out_ids_sha256"] for row in group}) != 1:
            continue
        for rank,row in enumerate(sorted(group,key=lambda r:(r["price_mae"],r["model"])),start=1):
            row["cohort_rank"] = rank
            row["rank_basis"] = "Price MAE; same run/date/expiry/seed/IDs; ALL four methods have 100% target coverage"
            rankings.append({k:row[k] for k in ["dataset","mode","run_id","seed","observation_date","expiry_date","model","price_mae","cohort_rank"]})
    return {"runs":records,"rows":rows,"issues":issues,"figure_index":figures,"scoped_rankings":rankings,
            "policy":{"rankings":"No global ranking across datasets, modes, cohorts, grid domains or unequal coverage",
                      "duplicates":"Different source runs remain distinct; repeated M5 runs are NOT averaged",
                      "calendar_attribution":"Pair/forward checks attributed to later expiry; no earlier node implies null, not zero"}}


def write_comparison(report, output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Comparison output must remain repository-local")
    if output.exists():
        raise FileExistsError("Preserve existing comparison; choose a new output path")
    output.parent.mkdir(parents=True,exist_ok=True)
    columns = ["question","mode","dataset","run_id","model","seed","observation_date","expiry_date","cohort","observed_quotes","eligible_quotes","training_quotes","validation_quotes",
               "predicted","expected","coverage","relative_median","relative_p95","relative_max","price_mae","iv_mae","iv_rmse",
               "butterfly_exam","butterfly_complete","calendar","calendar_derivative","negative_forward","diagnostic_domain","fit_seconds","converged","failed_fits","status","cohort_rank","rank_basis","other_question_metrics"]
    sections = []
    partitions = sorted({(row["question"],row["mode"]) for row in report["rows"]}) or [("none","NO_RESULTS")]
    for index,(question,mode) in enumerate(partitions):
        selected = [row for row in report["rows"] if row["question"] == question and row["mode"] == mode]
        figure = go.Figure(go.Table(header={"values":columns},cells={"values":[[row.get(key) for row in selected] for key in columns]}))
        figure.update_layout(title=question+" — "+mode,height=max(500,min(2600,50+len(selected)*22)))
        figure.add_annotation(text="Conditional errors require coverage. Null is unavailable, not zero. No cross-cohort ranking. Dense violations are a separate diagnostic domain.",xref="paper",yref="paper",x=0,y=1.08,showarrow=False)
        sections.append(figure.to_html(include_plotlyjs=True if index == 0 else False,full_html=False,div_id="comparison_"+str(index)))
    links = "".join("<li>"+html.escape(item["mode"]+" / "+item["dataset"]+" / "+item["file"])+"</li>" for item in report["figure_index"])
    output.write_text("<!doctype html><html><head><meta charset='utf-8'><title>Separated Q1/group comparison</title></head><body><h1>Comparison: distinct question and provenance partitions</h1>"+
                      "<p>Saved experiments only; no fitting. Repeated runs are not averaged. Inspect run IDs, cohort, grid and prediction coverage before comparing.</p>"+
                      "".join(sections)+"<h2>Original figure index (files preserved)</h2><ul>"+links+"</ul></body></html>",encoding="utf-8",newline="\n")
    output.with_suffix(".json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8",newline="\n")
    with output.with_suffix(".csv").open("x",encoding="utf-8",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=columns,extrasaction="ignore",lineterminator="\n")
        writer.writeheader(); writer.writerows(report["rows"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",type=Path,default=ROOT/"results")
    parser.add_argument("--output",type=Path,default=ROOT/"results/comparison.html")
    args = parser.parse_args(argv)
    if not args.results_dir.resolve().is_relative_to(ROOT):
        parser.error("Input must remain repository-local")
    report = load_comparison(args.results_dir)
    write_comparison(report,args.output)
    Console().print(f"Comparison: {len(report['runs'])} runs, {len(report['rows'])} rows, {len(report['issues'])} schema/provenance issues; no model training. {args.output}")
    table = Table("Scoped complete cohort (first two)","Model","Rank","Price MAE")
    for row in report["scoped_rankings"][:8]:
        table.add_row(str(row["seed"])+" / "+row["expiry_date"],row["model"],str(row["cohort_rank"]),str(row["price_mae"]))
    Console().print(table)
    return 1 if report["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
