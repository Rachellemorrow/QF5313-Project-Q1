"""Self-contained Approach A figures with unavoidable provenance labels."""

import math

import numpy as np
import plotly.graph_objects as go

from . import svi


FIGURE_NAMES = ("volatility_smile", "total_variance", "butterfly", "calendar",
                "volatility_surface", "market_model_prices", "repricing_error_distribution")


def write_figures(output, result, *, mode):
    label = ("SYNTHETIC SOFTWARE VERIFICATION — NOT EMPIRICAL MARKET DATA" if mode == "development"
             else "VERIFIED REAL QUOTE SAMPLE — SEE HISTORICAL COVERAGE LIMITATIONS")
    figures = {name: go.Figure() for name in FIGURE_NAMES}
    grid = np.asarray(result["arbitrage"]["grid"])
    ledger = result["quote_ledger"]
    quote_map = {row["quote_id"]: row for row in ledger}
    for item in result["slices"]:
        if not item["fit"]["converged"]:
            continue
        key = item["slice_id"]
        p = svi.SVIParameters(**item["fit"]["fitted_parameters"])
        w = np.asarray(svi.svi_total_variance(grid, p))
        records = [row for row in ledger if row["eligible_for_fit"] and
                   row["input"]["observation_date"] == item["observation_date"] and row["input"]["expiry_date"] == item["expiry_date"]]
        locations = [row["k"] for row in records]
        symbols = ["circle" if row["input"]["option_type"] == "C" else "diamond" for row in records]
        figures["volatility_smile"].add_trace(go.Scatter(x=locations, y=[row["iv"]["volatility"] for row in records],
            mode="markers", marker_symbol=symbols, name=key+" quote IV (C circle/P diamond)"))
        figures["volatility_smile"].add_trace(go.Scatter(x=grid, y=np.sqrt(w/item["time_to_expiry"]), name=key+" fitted IV"))
        figures["total_variance"].add_trace(go.Scatter(x=locations, y=[row["total_variance"] for row in records],
            mode="markers", marker_symbol=symbols, name=key+" quote total variance"))
        figures["total_variance"].add_trace(go.Scatter(x=grid, y=w, name=key+" SVI w"))
        for repriced in item["repricing"]:
            if repriced["status"] != "REPRICED":
                continue
            row = quote_map[repriced["quote_id"]]
            # Collect below, one trace per side, rather than hundreds of tiny traces.
            row["_figure_repricing"] = repriced
    for label_name in ["C", "P"]:
        records = [row for row in ledger if "_figure_repricing" in row and row["input"]["option_type"] == label_name]
        figures["market_model_prices"].add_trace(go.Scatter(
            x=[row["_figure_repricing"]["market_price"] for row in records],
            y=[row["_figure_repricing"]["model_price"] for row in records], mode="markers", name=label_name,
            text=[row["quote_id"] for row in records]))
        errors = [row["_figure_repricing"]["relative_error"] for row in records if row["_figure_repricing"]["relative_error"] is not None]
        figures["repricing_error_distribution"].add_trace(go.Histogram(x=errors, name=label_name, opacity=0.65))
    for row in ledger:
        row.pop("_figure_repricing", None)
    prices = [row["market_price"] for item in result["slices"] for row in item["repricing"] if row["status"] == "REPRICED"]
    if prices:
        figures["market_model_prices"].add_trace(go.Scatter(x=[min(prices), max(prices)], y=[min(prices), max(prices)], name="identity", mode="lines"))
    for item in result["arbitrage"]["slices"]:
        if item["status"] != "VERIFIED":
            continue
        for name in ["examination", "complete"]:
            diagnostic = item[name]
            figures["butterfly"].add_trace(go.Scatter(x=grid, y=diagnostic["values"], name=item["slice_id"]+" "+name))
            if diagnostic["count"]:
                figures["butterfly"].add_trace(go.Scatter(x=diagnostic["locations"], y=-np.asarray(diagnostic["magnitudes"]),
                    name=item["slice_id"]+" "+name+" violations", mode="markers", marker_color="red"))
    for item in result["arbitrage"]["calendar"]:
        figures["calendar"].add_trace(go.Scatter(x=grid, y=item["difference"],
            name=item["earlier_slice"]+" -> "+item["later_slice"]))
        if item["count"]:
            figures["calendar"].add_trace(go.Scatter(x=item["locations"], y=-np.asarray(item["magnitudes"]),
                name="calendar violations", mode="markers", marker_color="red"))
    surfaces = [item for item in result["arbitrage"]["interpolation"] if item["status"] == "INTERPOLATED_WITHOUT_EXTRAPOLATION"]
    for index, item in enumerate(surfaces):
        times = np.asarray(item["query_maturities"])
        volatility = np.sqrt(np.asarray(item["total_variance"])/times[:, None])
        figures["volatility_surface"].add_trace(go.Surface(x=grid, y=times, z=volatility,
            name=item["observation_date"], visible=index == 0, colorbar_title="IV"))
    if len(surfaces) > 1:
        figures["volatility_surface"].update_layout(updatemenus=[{"buttons": [
            {"label": item["observation_date"], "method": "update", "args": [{"visible": [i == j for j in range(len(surfaces))]}]}
            for i, item in enumerate(surfaces)]}])
    titles = {"volatility_smile": ("Quote vs fitted volatility", "ln(K/F)", "IV"),
              "total_variance": ("Total variance", "ln(K/F)", "w"),
              "butterfly": ("Separate examination and complete butterfly diagnostics", "ln(K/F)", "g(k)"),
              "calendar": ("Calendar total-variance differences", "ln(K/F)", "w_later - w_earlier"),
              "volatility_surface": ("Volatility surface — total-variance interpolation, no time extrapolation", "ln(K/F)", "T"),
              "market_model_prices": ("Input quote vs model price", "input premium", "model premium"),
              "repricing_error_distribution": ("Relative repricing errors — near-zero denominators excluded and recorded", "absolute relative error", "count")}
    paths = []
    for name, figure in figures.items():
        title, x, y = titles[name]
        figure.update_layout(title=label+"<br>"+title, xaxis_title=x, yaxis_title=y, template="plotly_white")
        figure.add_annotation(text=label, x=0, y=1.12, xref="paper", yref="paper", showarrow=False, font_color="red")
        if name in {"butterfly", "calendar"}:
            figure.add_hline(y=0, line_dash="dash", line_color="black")
        if name == "volatility_surface":
            figure.update_layout(scene={"xaxis_title": "ln(K/F)", "yaxis_title": "T (years)", "zaxis_title": "IV"})
        if not figure.data or not any(getattr(trace, "x", None) is not None and len(trace.x) for trace in figure.data):
            figure.add_annotation(text="NO VALID FIT/DATA FOR THIS VIEW — see failure ledger", x=.5, y=.5,
                                  xref="paper", yref="paper", showarrow=False)
        path = output/(name+".html")
        figure.write_html(path, include_plotlyjs=True, full_html=True, auto_open=False, div_id="q1_approach_a_"+name)
        paths.append(path.name)
    return paths
