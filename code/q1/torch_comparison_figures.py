"""Origin-labelled, inline Plotly comparison views; no empirical/synthetic mixing."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def number(value):
    return "UNAVAILABLE" if value is None else f"{value:.8g}" if isinstance(value, float) else str(value)


def table(headers, rows):
    return go.Figure(go.Table(header={"values": headers}, cells={"values": list(map(list, zip(*rows))) if rows else [[] for _ in headers]}))


def comparison_figures(output, comparison, *, mode):
    prefix = "SYNTHETIC SOFTWARE VERIFICATION — NOT EMPIRICAL MARKET RESULTS" if mode == "development" else "VERIFIED REAL QUOTE SAMPLE — COVERAGE LIMITATIONS APPLY"
    iteration = make_subplots(rows=1, cols=2, subplot_titles=["SciPy: residual evaluations, NOT optimizer iterations", "Torch: actual update/native-iteration labels (line-search trials retained)"])
    wall = go.Figure()
    tiny = np.finfo(float).tiny
    for algorithm, result in comparison["algorithms"].items():
        for item in result["slices"]:
            history = item["fit"].get("comparison_history", []) if algorithm == "scipy" else item["fit"]["evaluation_history"]
            if not history:
                continue
            objective = [max(row["weighted_rss"], tiny) for row in history]
            progress = [row["evaluation"] if algorithm == "scipy" else row["iteration"] for row in history]
            iteration.add_trace(go.Scatter(x=progress, y=objective, name=algorithm+" "+item["slice_id"], mode="lines+markers"), row=1, col=1 if algorithm == "scipy" else 2)
            time_name = "elapsed_since_first_evaluation_seconds" if algorithm == "scipy" else "optimization_elapsed_seconds"
            times = [row[time_name]-history[0][time_name] for row in history]
            wall.add_trace(go.Scatter(x=times, y=objective, name=algorithm+" "+item["slice_id"], mode="lines+markers"))
    iteration.update_yaxes(type="log", title_text="weighted RSS")
    iteration.update_xaxes(title_text="actual residual evaluation", row=1, col=1)
    iteration.update_xaxes(title_text="actual update / reported native iteration", row=1, col=2)
    wall.update_layout(xaxis_title="elapsed since first objective evaluation (seconds); setup excluded here, reported in tables", yaxis_title="weighted RSS", yaxis_type="log")
    rows = comparison["comparison_rows"]
    parameters = table(["Algorithm", "Slice", "a", "b", "rho", "m", "eta", "Max |w - SciPy|"], [
        [row["algorithm"], row["slice_id"], *[number((row["fitted_parameters"] or {}).get(name)) for name in ["a", "b", "rho", "m", "eta"]], number(row["variance_grid_max_abs_difference_from_scipy"])] for row in rows])
    convergence = table(["Algorithm", "Slice", "Converged", "Native iterations", "Native SciPy nfev", "Objective evals", "Grad evals", "Analytic J evals", "Closure calls", "Fit seconds", "Setup seconds"], [
        [row["algorithm"], row["slice_id"], row["converged"], number(row["native_iterations"]), number(row["native_scipy_nfev"]),
         row["objective_evaluations"], number(row["gradient_evaluations"]), number(row["analytic_residual_jacobian_evaluations"]),
         number(row["closure_calls"]), number(row["fit_seconds"]), number(row["optimizer_setup_seconds"])] for row in rows])
    repricing = table(["Algorithm", "Complete date maxima: median", "Complete date maxima: p95", "Successful slices", "Failed slices"], [
        [name, number(info["repricing"]["cross_date_median_of_date_maxima"]), number(info["repricing"]["cross_date_p95_of_date_maxima"]), info["successful_slices"], len(info["failed_slice_ids"])] for name, info in comparison["summary"].items()])
    arbitrage = table(["Algorithm", "Examination butterfly", "Complete butterfly", "Calendar locations", "Negative forward locations", "Failed slices without diagnostics"], [
        [name, info["arbitrage"]["examination_butterfly_violations"], info["arbitrage"]["complete_butterfly_violations"],
         info["arbitrage"]["calendar_violations"], info["arbitrage"]["negative_forward_variance_locations"], len(info["arbitrage"]["failed_slices_without_diagnostics"])] for name, info in comparison["summary"].items()])
    figures = {"objective_vs_iteration": iteration, "objective_vs_wall_time": wall, "parameter_comparison": parameters,
               "optimizer_convergence": convergence, "repricing_comparison": repricing, "arbitrage_summary": arbitrage}
    paths = []
    for name, figure in figures.items():
        figure.update_layout(title=prefix+"<br>"+name.replace("_", " "), template="plotly_white")
        figure.add_annotation(text=prefix, x=0, y=1.12, xref="paper", yref="paper", showarrow=False, font_color="red")
        if name.startswith("objective"):
            figure.add_annotation(text="Not a perfectly controlled benchmark. Zero objectives, if any, drawn at float64 tiny for log display; raw history retained.",
                                  x=0, y=-0.2, xref="paper", yref="paper", showarrow=False)
        figure.write_html(output/(name+".html"), include_plotlyjs=True, full_html=True, auto_open=False, div_id="q1_m5_"+name)
        paths.append(name+".html")
    return paths
