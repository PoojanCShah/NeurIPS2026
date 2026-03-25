"""
Exp 2 — Sample Size Scaling

Sweeps n_samples at fixed (family, d, η), records
ID estimate ± 95 % CI and wall-clock runtime per run.
"""

import time
import numpy as np
import plotly.graph_objects as go
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
                                          style={"height": "420px"})), width=12),
        ]),
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


def build_figures(id_res: dict, rt_res: dict,
                  true_d: int, est_name: str) -> tuple:
    ns       = sorted(id_res)
    means    = [float(np.mean(id_res[n])) for n in ns]
    cis      = [_ci(id_res[n]) for n in ns]
    rt_means = [float(np.mean(rt_res[n])) for n in ns]

    # ── ID vs n ───────────────────────────────────────────────────────────────
    id_fig = go.Figure()
    id_fig.add_hline(y=true_d, line_dash="dash", line_color="grey",
                     annotation_text=f"true d = {true_d}",
                     annotation_position="top right")
    id_fig.add_trace(go.Scatter(
        x=ns + ns[::-1],
        y=[m + c for m, c in zip(means, cis)] +
          [m - c for m, c in zip(reversed(means), reversed(cis))],
        fill="toself", fillcolor="rgba(44,160,44,0.12)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    id_fig.add_trace(go.Scatter(
        x=ns, y=means, mode="lines+markers", name=est_name,
        line=dict(color="#2ca02c", width=2), marker=dict(size=7),
        error_y=dict(type="data", array=cis, visible=True, color="#2ca02c"),
    ))
    id_fig.update_layout(
        title=f"{est_name} — estimated ID vs sample size",
        xaxis_title="n_samples (log scale)", yaxis_title="Estimated ID",
        xaxis_type="log",
        template="plotly_white", margin=dict(t=50, b=40, l=50, r=20),
    )

    return id_fig


# ── Callback ───────────────────────────────────────────────────────────────────

@callback(
    Output("exp2-id-plot",    "figure"),
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
        return go.Figure(), "Estimator or dataset store is empty."
    try:
        sample_sizes = [int(x.strip()) for x in (sample_sizes_str or "").split(",")
                        if x.strip()]
    except ValueError:
        return go.Figure(), "Sample sizes must be comma-separated integers."
    if not sample_sizes:
        return go.Figure(), "Enter at least one sample size."
    try:
        id_res, rt_res = compute(est_store, ds_store, sample_sizes, int(n_runs or 5))
        id_fig = build_figures(id_res, rt_res, ds_store["d"], est_store["name"])
        return id_fig, ""
    except Exception as exc:
        return go.Figure(), f"Error: {exc}"
