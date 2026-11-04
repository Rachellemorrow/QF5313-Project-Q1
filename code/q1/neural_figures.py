"""Six self-contained neural views, generated once after ablation evaluation."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def write_neural_figures(output, metrics):
    origin = "SYNTHETIC SOFTWARE VERIFICATION — NOT EMPIRICAL MARKET RESULTS" if metrics["params"]["mode"] == "development" else "STRICTLY ELIGIBLE REAL QUOTES — COVERAGE LIMITATIONS APPLY"
    dates = metrics["history"]["result"]["dates"]
    training, smiles, prices = go.Figure(),go.Figure(),go.Figure()
    selected = [date for date in dates if date["selected_ablation"] is not None and
                date["ablations"][date["selected_ablation"]]["diagnostics"] is not None]
    surface = make_subplots(rows=1,cols=max(1,len(selected)),specs=[[{"type":"surface"} for _ in range(max(1,len(selected)))]])
    runs = [(date,case,run) for date in dates for case,run in date["ablations"].items()]
    heatmap = make_subplots(rows=max(1,len(runs)),cols=3,
        subplot_titles=[f"{date['observation_date']} {case}: {label}" for date,case,_ in runs for label in ["examination g", "complete g", "partial_T w"]],
        vertical_spacing=min(.02,.5/max(1,len(runs))))
    for index,(date,case,run) in enumerate(runs):
        label = date["observation_date"]+" "+case
        history = run["training"]["history"]
        for field in ["training_total","training_price_mse","validation_price_mse"]:
            training.add_trace(go.Scatter(x=[row["epoch"] for row in history],y=[row[field] for row in history],mode="lines",name=label+" "+field))
        if run["evaluation"]:
            for fold in ["train","validation","test"]:
                rows = [row for row in run["evaluation"]["predictions"] if row["fold"] == fold]
                prices.add_trace(go.Scatter(x=[row["market_price"] for row in rows],y=[row["model_price"] for row in rows],mode="markers",name=label+" "+fold))
        diagnostics = run["diagnostics"]
        if diagnostics:
            for column,key in enumerate(["examination","complete","w_T"],start=1):
                heatmap.add_trace(go.Heatmap(x=diagnostics["k"],y=diagnostics["maturities"],z=diagnostics[key],colorscale="RdBu",zmid=0,showscale=False),row=index+1,col=column)
        else:
            heatmap.add_annotation(text=label+": DIAGNOSTICS UNAVAILABLE (failed fit/evaluation)",xref="paper",yref="paper",x=0,y=1-index/max(1,len(runs)),showarrow=False)
    for column,date in enumerate(selected,start=1):
        case = date["selected_ablation"]
        diagnostics = date["ablations"][case]["diagnostics"]
        for expiry in diagnostics["per_expiry"]:
            index = diagnostics["maturities"].index(expiry["maturity"])
            smiles.add_trace(go.Scatter(x=diagnostics["k"],y=diagnostics["volatility"][index],mode="lines",name=date["observation_date"]+" "+case+" "+expiry["expiry_date"]))
        surface.add_trace(go.Surface(x=diagnostics["k"],y=diagnostics["maturities"],z=diagnostics["volatility"],name=date["observation_date"]+" "+case,showscale=False),row=1,col=column)
        surface.update_scenes(xaxis_title="k=ln(K/F)",yaxis_title="T (years)",zaxis_title="implied volatility",row=1,col=column)
    ablation = make_subplots(rows=2,cols=3,subplot_titles=["Price MSE (validation / held-out)","Butterfly violations (dense independent grid)",
        "Negative partial_T(w)","Calendar pair violations", "Negative forward variance", "Fit wall seconds"])
    rows = metrics["metrics"]["ablation_rows"]
    names = [row["observation_date"]+" "+row["ablation"] for row in rows]
    for position, fields in enumerate([["best_validation_price_mse","test_price_mse"],
            ["examination_butterfly_violations","complete_butterfly_violations"],["calendar_derivative_violations"],
            ["calendar_pair_violations"],["negative_forward_variance_locations"],["fit_seconds"]]):
        for field in fields:
            ablation.add_trace(go.Bar(x=names,y=[row[field] for row in rows],name=field),row=position//3+1,col=position%3+1)
    training.update_layout(xaxis_title="Epoch (training values: pre-update minibatch averages)",yaxis_title="Loss; validation is price MSE",yaxis_type="log")
    smiles.update_layout(xaxis_title="k=ln(K/F); outer wings may extrapolate training support",yaxis_title="Positive implied volatility")
    prices.update_layout(xaxis_title="Observed option premium",yaxis_title="Predicted premium (per underlying unit)")
    heatmap.update_xaxes(title_text="k"); heatmap.update_yaxes(title_text="T (years)")
    heatmap.update_layout(height=max(550,220*len(runs)))
    figures = {"training_validation":training,"neural_smiles":smiles,"neural_surface":surface,
               "observed_predicted_prices":prices,"arbitrage_heatmaps":heatmap,"penalty_ablation":ablation}
    for name,figure in figures.items():
        figure.update_layout(title=origin+"<br>"+name.replace("_"," "),template="plotly_white")
        figure.add_annotation(text="Soft penalties and finite-grid checks do not guarantee global no-arbitrage; all failures remain in metrics.",xref="paper",yref="paper",x=0,y=-.16,showarrow=False)
        figure.write_html(output/(name+".html"),include_plotlyjs=True,full_html=True,auto_open=False,div_id="q1_m6_"+name)
    return [name+".html" for name in figures]
