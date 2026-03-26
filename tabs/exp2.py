"""
Exp 2 — Sample Size Scaling

Sweeps n_samples at fixed (family, d, η), records
ID estimate ± 95 % CI and wall-clock runtime per run.
"""

import time
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback

from registry.estimators import build_estimator
from registry.datasets import DEFAULT_SAMPLE_SIZES, ensure_dataset, load_dataset

Z95 = 1.96

# ── Layout ─────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([
        dbc.Card(dbc.CardBody(
            dbc.Row([
                dbc.Col([
                    html.Label("Sample sizes (comma-separated)",
                               className="form-label small text-muted mb-1"),
                    dcc.Input(
                        id="exp2-sample-sizes",
                        type="text",
                        value=", ".join(str(n) for n in DEFAULT_SAMPLE_SIZES),
                        debounce=True,
                        placeholder="e.g. 100, 500, 1000, 5000",
                        className="form-control form-control-sm",
                    ),
                ], width=7),
                dbc.Col([
                    html.Label("n_runs", className="form-label small text-muted mb-1"),
                    dcc.Input(id="exp2-n-runs", type="number",
                              value=5, min=1, max=50, step=1,
                              className="form-control form-control-sm"),
                ], width=2),
                dbc.Col([
                    html.Br(),
                    dbc.Button("Run ▶", id="exp2-run-btn", color="primary",
                               size="sm", className="w-100 mt-1"),
                ], width=3),
            ]),
            style={"padding": "12px 16px"},
        ), className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Loading(dcc.Graph(id="exp2-id-plot",
                                          style={"height": "620px"})), width=12),
        ]),
        html.Div(id="exp2-fits", className="mt-2"),
        html.Div(id="exp2-error", className="text-danger small mt-2"),
    ])


# ── Pure computation (no Dash) ─────────────────────────────────────────────────

def compute(est_store: dict, ds_store: dict,
            sample_sizes: list[int], n_runs: int
            ) -> tuple[dict, dict]:
    """
    Returns
    -------
    id_results  : {n -> [id_run_0, ...]}
    rt_results  : {n -> [time_run_0, ...]}
    """
    est_name = est_store["name"]
    params   = est_store.get("params", {})
    family   = ds_store["family"]
    d        = ds_store["d"]
    D        = ds_store.get("D")
    eta      = ds_store["eta"]

    ensure_dataset(family, d, eta, D=D)
    ds     = load_dataset(family, d, eta=eta, D=D)
    X_full = ds["X"]

    id_res, rt_res = {}, {}
    for n in sorted(sample_sizes):
        n = min(int(n), len(X_full))
        ids_run, rts_run = [], []
        for run in range(n_runs):
            rng = np.random.default_rng(42 + run)
            X   = X_full[rng.choice(len(X_full), n, replace=False)]
            est = build_estimator(est_name, params)
            t0  = time.perf_counter()
            est.fit(X)
            rts_run.append(time.perf_counter() - t0)
            ids_run.append(float(est.dimension_))
        id_res[n] = ids_run
        rt_res[n] = rts_run

    return id_res, rt_res


# ── Figure builders (no Dash) ──────────────────────────────────────────────────

def _ci(vals: list[float]) -> float:
    a = np.array(vals)
    return float(Z95 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def _fit_scaling(ns: list, means: list[float], true_d: int) -> dict | None:
    """Fit log|true_d - d_hat| = a + b*log(n) via OLS."""
    errors = [abs(true_d - m) for m in means]
    valid  = [(n, e) for n, e in zip(ns, errors) if e > 0]
    if len(valid) < 2:
        return None
    ns_v, es_v = zip(*valid)
    b, a = np.polyfit(np.log(ns_v), np.log(es_v), 1)
    return dict(a=float(a), b=float(b))


def build_figures(id_res: dict, rt_res: dict,
                  true_d: int, est_name: str) -> tuple:
    ns    = sorted(id_res)
    means = [float(np.mean(id_res[n])) for n in ns]
    cis   = [_ci(id_res[n]) for n in ns]

    fits   = _fit_scaling(ns, means, true_d)
    log_ns = [float(np.log(n)) for n in ns]

    id_fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            f"{est_name} — estimated ID vs log n",
            "log |d̂ − d| vs log n",
        ),
        row_heights=[0.55, 0.45],
    )

    # ── Row 1: ID vs log n ───────────────────────────────────────────────────
    id_fig.add_trace(go.Scatter(
        x=[log_ns[0], log_ns[-1]], y=[true_d, true_d],
        mode="lines", name=f"true d = {true_d}",
        line=dict(color="grey", width=1.5, dash="dash"),
    ), row=1, col=1)
    id_fig.add_trace(go.Scatter(
        x=log_ns + log_ns[::-1],
        y=[m + c for m, c in zip(means, cis)] +
          [m - c for m, c in zip(reversed(means), reversed(cis))],
        fill="toself", fillcolor="rgba(44,160,44,0.12)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    id_fig.add_trace(go.Scatter(
        x=log_ns, y=means, mode="lines+markers", name=est_name,
        line=dict(color="#2ca02c", width=2), marker=dict(size=7),
        error_y=dict(type="data", array=cis, visible=True, color="#2ca02c"),
    ), row=1, col=1)

    # ── Row 2: log|error| vs log n ───────────────────────────────────────────
    errors    = [abs(true_d - m) for m in means]
    valid_pts = [(lx, e) for lx, e in zip(log_ns, errors) if e > 0]
    if valid_pts:
        log_ns_v, es_v = zip(*valid_pts)
        log_es_v = [float(np.log(e)) for e in es_v]

        id_fig.add_trace(go.Scatter(
            x=log_ns_v, y=log_es_v, mode="markers", name="log|error|",
            marker=dict(color="#2ca02c", size=8),
        ), row=2, col=1)

        if fits:
            x0, x1 = log_ns_v[0], log_ns_v[-1]
            id_fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[fits["a"] + fits["b"] * x0, fits["a"] + fits["b"] * x1],
                mode="lines", name="fit",
                line=dict(color="#d62728", width=1.5, dash="dash"),
            ), row=2, col=1)

    id_fig.update_xaxes(title_text="log n", row=2, col=1)
    id_fig.update_yaxes(title_text="Estimated ID", row=1, col=1)
    id_fig.update_yaxes(title_text="log |d̂ − d|", row=2, col=1)
    id_fig.update_layout(
        template="plotly_white",
        margin=dict(t=60, b=40, l=50, r=20),
        height=620,
    )

    return id_fig, fits


# ── Callback ───────────────────────────────────────────────────────────────────

def _fits_card(f: dict):
    """Render the error scaling fit as a Bootstrap card with MathJax."""
    eq = (
        r"$$\log|\hat{d} - d| = "
        + f"{f['a']:+.3f}"
        + r" + "
        + f"{f['b']:.3f}"
        + r"\,\log n$$"
    )
    return dbc.Card(dbc.CardBody([
        html.P("Error scaling fit", className="text-muted small mb-2"),
        dcc.Markdown(eq, mathjax=True, style={"marginBottom": "0"}),
    ]), style={"display": "inline-block", "padding": "4px 8px"})


@callback(
    Output("exp2-id-plot",    "figure"),
    Output("exp2-fits",       "children"),
    Output("exp2-error",      "children"),
    Input("exp2-run-btn",     "n_clicks"),
    State("est-store",        "data"),
    State("dataset-store",    "data"),
    State("exp2-sample-sizes","value"),
    State("exp2-n-runs",      "value"),
    prevent_initial_call=True,
)
def run_exp2(_, est_store, ds_store, sample_sizes_str, n_runs):
    if not est_store or not ds_store:
        return go.Figure(), "", "Estimator or dataset store is empty."
    try:
        sample_sizes = [int(x.strip()) for x in (sample_sizes_str or "").split(",")
                        if x.strip()]
    except ValueError:
        return go.Figure(), "", "Sample sizes must be comma-separated integers."
    if not sample_sizes:
        return go.Figure(), "", "Enter at least one sample size."
    try:
        id_res, rt_res = compute(est_store, ds_store, sample_sizes, int(n_runs or 5))
        id_fig, fits = build_figures(id_res, rt_res, ds_store["d"], est_store["name"])
        fits_div = _fits_card(fits) if fits else ""
        return id_fig, fits_div, ""
    except Exception as exc:
        return go.Figure(), "", f"Error: {exc}"
