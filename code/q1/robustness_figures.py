"""Eight M7 figure groups. Resume keeps completed hash/config-labelled figures."""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import svi
from .robustness_protocol import METHODS


def write_robustness_figures(output, metrics):
    output = Path(output)
    history = metrics["history"]
    rows = history["universe"]
    prefix = "SYNTHETIC M7 SOFTWARE VERIFICATION — NOT EMPIRICAL MARKET RESULTS" if metrics["params"]["mode"] == "development" else "STRICTLY ELIGIBLE REAL ETF QUOTES — COVERAGE LIMITATIONS APPLY"
    full_iv, mask_view, reference, refits, error_k, error_t = [go.Figure() for _ in range(6)]
    neural_reports = [row for row in history["diagnostics"] if row["method"] == "neural"]
    neural = make_subplots(rows=1,cols=max(1,len(neural_reports)),specs=[[{"type":"surface"} for _ in range(max(1,len(neural_reports)))]],
                           subplot_titles=[str(row["seed"])+" "+row["observation_date"] for row in neural_reports])
    grid = np.array(metrics["params"]["calibration"]["grid"])
    groups = sorted({(row["observation_date"],row["expiry_date"]) for row in rows})
    for date,expiry in groups:
        selected = [row for row in rows if row["observation_date"] == date and row["expiry_date"] == expiry]
        label = date+" "+expiry
        full_iv.add_trace(go.Scatter(x=[row["_k"] for row in selected],y=[row["_iv"] for row in selected],mode="markers",name=label,
                                     text=[row["quote_id"] for row in selected]))
    for item in history["full_reference"]["slices"]:
        fit = item["fit"]
        if fit["converged"]:
            w = svi.svi_total_variance(grid,svi.SVIParameters(**fit["fitted_parameters"]))
            reference.add_trace(go.Scatter(x=grid,y=np.sqrt(w/item["time_to_expiry"]),name=item["slice_id"],mode="lines"))
        else:
            reference.add_annotation(text=item["slice_id"]+": FAILED full-data reference",xref="paper",yref="paper",x=0,y=1,showarrow=False)
    for manifest in history["manifests"]:
        for item in manifest["slices"]:
            for name in ["retained","dropped"]:
                ids = set(item[name+"_ids"])
                selected = [row for row in rows if row["quote_id"] in ids]
                mask_view.add_trace(go.Scatter(x=[row["_k"] for row in selected],y=[row["_iv"] for row in selected],mode="markers",
                                               name=f"{manifest['seed']} {item['slice_id']} {name}",text=[row["quote_id"] for row in selected]))
    for fit in history["fits"]:
        if fit["method"] == "neural":
            continue
        label = f"{fit['method']} seed {fit['seed']} {fit['slice_id']}"
        if fit["usable"]:
            w = svi.svi_total_variance(grid,svi.SVIParameters(**fit["fit"]["fitted_parameters"]))
            refits.add_trace(go.Scatter(x=grid,y=np.sqrt(w/fit["time_to_expiry"]),name=label,mode="lines"))
        else:
            refits.add_trace(go.Scatter(x=[],y=[],name=label+" FAILED: "+str(fit["failure_reason"]),mode="lines"))
    for column,report in enumerate(neural_reports,start=1):
        if report.get("dense_neural_supplement"):
            dense = report["dense_neural_supplement"]
            neural.add_trace(go.Surface(x=dense["k"],y=dense["maturities"],z=dense["volatility"],showscale=False,
                                        name=f"seed {report['seed']} {report['observation_date']}"),row=1,col=column)
            neural.update_scenes(xaxis_title="k",yaxis_title="T (years)",zaxis_title="Implied volatility",row=1,col=column)
        else:
            neural.add_annotation(text=f"Seed {report['seed']}: neural diagnostics unavailable",xref="paper",yref="paper",x=0,y=1,showarrow=False)
    predictions = history["predictions"]
    for method in METHODS:
        for seed in metrics["params"]["seeds"]:
            for category in ["INTERPOLATION","EXTRAPOLATION","INSUFFICIENT_SUPPORT"]:
                selected = [row for row in predictions if row["method"] == method and row["seed"] == seed and row["support_class"] == category]
                valid = [row for row in selected if row["status"] == "PREDICTED"]
                name = f"{method} {seed} {category}: {len(valid)}/{len(selected)} predictions"
                hover = [row["quote_id"]+f"; IV abs error={row['absolute_iv_error']:.6g}; relative price={row['relative_price_error']}" for row in valid]
                for figure,key in [(error_k,"k"),(error_t,"maturity")]:
                    figure.add_trace(go.Scatter(x=[row[key] for row in valid],y=[row["absolute_price_error"] for row in valid],mode="markers",name=name,text=hover))
    comparison = metrics["metrics"]["per_seed"]
    headers = ["Seed","Method","Coverage","Fit failures","IV MAE (available)","IV RMSE (available)","Price MAE (available)","Relative MAE (available)","Fit seconds","Missing diagnostics"]
    table_rows = [[row["seed"],row["method"],f"{row['predicted']}/{row['held_out_expected']}",row["failed_expiry_fits"],
                   *[row["available_prediction_errors"][key] for key in ["iv_mae","iv_rmse","price_mae","relative_price_mae"]],
                   row["fit_seconds"],row["missing_diagnostic_expiries"]] for row in comparison]
    table = go.Figure(go.Table(header={"values":headers},cells={"values":list(map(list,zip(*table_rows))) if table_rows else [[] for _ in headers]}))
    figures = {"full_data_implied_volatility":full_iv,"retained_dropped_quotes":mask_view,"full_data_svi_reference":reference,
               "retained_svi_refits":refits,"neural_predictions":neural,"held_out_errors_moneyness":error_k,
               "held_out_errors_maturity":error_t,"robustness_comparison":table}
    paths = []
    config_hash = metrics["params"]["configuration_sha256"]
    for name,figure in figures.items():
        path = output/(name+".html")
        paths.append(path.name)
        if path.exists():
            html = path.read_text(encoding="utf-8")
            if config_hash not in html or "</html>" not in html:
                raise ValueError("Existing figure is incomplete or belongs to another configuration")
            continue
        figure.update_layout(title=prefix+"<br>"+name.replace("_"," "),template="plotly_white",meta={"configuration_sha256":config_hash})
        if name not in {"robustness_comparison"}:
            figure.update_layout(xaxis_title="T (years)" if name == "held_out_errors_maturity" else "k=ln(K/F)",
                                 yaxis_title="Absolute price error" if name.startswith("held_out_errors") else "Implied volatility")
        note = "Failures remain in metrics; plotted errors are conditional on coverage. Soft/grid checks do not prove global no-arbitrage."
        if name == "full_data_svi_reference":
            note = "FULL-DATA ILLUSTRATION ONLY: sees dropped quotes; NEVER used for refit initialization, selection or held-out scoring."
        figure.add_annotation(text=note,xref="paper",yref="paper",x=0,y=-.2,showarrow=False)
        html = figure.to_html(include_plotlyjs=True,full_html=True,div_id="q1_m7_"+name)
        temporary = output/(name+".html.pending")
        temporary.write_text(html,encoding="utf-8",newline="\n")
        temporary.rename(path)
    return paths
