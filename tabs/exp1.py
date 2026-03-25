"""
Exp 1 — Noise Sensitivity

Sweeps η values at fixed (family, d, n_samples), records
ID estimate ± 95 % CI and wall-clock runtime per run.
"""

import time
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback

from registry.estimators import build_estimator
from registry.datasets import DEFAULT_ETA_SWEEP, ensure_dataset, load_dataset

Z95 = 1.96

# ── Layout ─────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([
        dbc.Card(dbc.CardBody(
            dbc.Row([
                dbc.Col([
                    html.Label("η values (comma-separated)",
                               className="form-label small text-muted mb-1"),
                    dcc.Input(
                        id="exp1-eta-values",
                        type="text",
                        value=", ".join(str(e) for e in DEFAULT_ETA_SWEEP),
                        debounce=True,
                        placeholder="e.g. 0.0, 0.1, 0.5, 1.0",
                        className="form-control form-control-sm",
                    ),
                ], width=7),
                dbc.Col([
                    html.Label("n_runs", className="form-label small text-muted mb-1"),
                    dcc.Input(id="exp1-n-runs", type="number",
                              value=5, min=1, max=50, step=1,
                              className="form-control form-control-sm"),
                ], width=2),
                dbc.Col([
                    html.Br(),
                    dbc.Button("Run ▶", id="exp1-run-btn", color="primary",
                               size="sm", className="w-100 mt-1"),
                ], width=3),
            ]),
            style={"padding": "12px 16px"},
        ), className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Loading(dcc.Graph(id="exp1-id-plot",
                                          style={"height": "420px"})), width=12),
        ]),
        html.Div(id="exp1-error", className="text-danger small mt-2"),
    ])


# ── Pure computation (no Dash) ─────────────────────────────────────────────────

def compute(est_store: dict, ds_store: dict,
            eta_values: list[float], n_runs: int
            ) -> tuple[dict, dict]:
    """
    Returns
    -------
    id_results  : {eta -> [id_run_0, id_run_1, ...]}
    rt_results  : {eta -> [time_run_0, ...]}
    """
    est_name  = est_store["name"]
    params    = est_store.get("params", {})
    family    = ds_store["family"]
    d         = ds_store["d"]
    D         = ds_store.get("D")
    n_samples = ds_store["n_samples"]

    id_res, rt_res = {}, {}
    for eta in sorted(eta_values):
        ensure_dataset(family, d, eta, D=D)
        ds     = load_dataset(family, d, eta=eta, D=D)
        X_full = ds["X"]
        n      = min(n_samples, len(X_full))

        ids_run, rts_run = [], []
        for run in range(n_runs):
            rng = np.random.default_rng(42 + run)
            X   = X_full[rng.choice(len(X_full), n, replace=False)]
            est = build_estimator(est_name, params)
            t0  = time.perf_counter()
            est.fit(X)
            rts_run.append(time.perf_counter() - t0)
            ids_run.append(float(est.dimension_))
        id_res[eta] = ids_run
        rt_res[eta] = rts_run

    return id_res, rt_res


# ── Figure builders (no Dash) ──────────────────────────────────────────────────

def _ci(vals: list[float]) -> float:
    a = np.array(vals)
    return float(Z95 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def build_figures(id_res: dict, rt_res: dict,
                  true_d: int, est_name: str) -> tuple:
    etas     = sorted(id_res)
    means    = [float(np.mean(id_res[e])) for e in etas]
    cis      = [_ci(id_res[e]) for e in etas]
    rt_means = [float(np.mean(rt_res[e])) for e in etas]

    # ── ID vs η ───────────────────────────────────────────────────────────────
    id_fig = go.Figure()
    id_fig.add_hline(y=true_d, line_dash="dash", line_color="grey",
                     annotation_text=f"true d = {true_d}",
                     annotation_position="top right")
    # CI band
    id_fig.add_trace(go.Scatter(
        x=etas + etas[::-1],
        y=[m + c for m, c in zip(means, cis)] +
          [m - c for m, c in zip(reversed(means), reversed(cis))],
        fill="toself", fillcolor="rgba(31,119,180,0.12)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    id_fig.add_trace(go.Scatter(
        x=etas, y=means, mode="lines+markers", name=est_name,
        line=dict(color="#1f77b4", width=2), marker=dict(size=7),
        error_y=dict(type="data", array=cis, visible=True, color="#1f77b4"),
    ))
    id_fig.update_layout(
        title=f"{est_name} — estimated ID vs noise η",
        xaxis_title="Noise η", yaxis_title="Estimated ID",
        template="plotly_white", margin=dict(t=50, b=40, l=50, r=20),
    )

    return id_fig


# ── Callback ───────────────────────────────────────────────────────────────────

@callback(
    Output("exp1-id-plot", "figure"),
    Output("exp1-error",   "children"),
    Input("exp1-run-btn",  "n_clicks"),
    State("est-store",       "data"),
    State("dataset-store",   "data"),
    State("exp1-eta-values", "value"),
    State("exp1-n-runs",     "value"),
    prevent_initial_call=True,
)
def run_exp1(_, est_store, ds_store, eta_values_str, n_runs):
    if not est_store or not ds_store:
        return go.Figure(), "Estimator or dataset store is empty."
    try:
        eta_values = [float(x.strip()) for x in (eta_values_str or "").split(",")
                      if x.strip()]
    except ValueError:
        return go.Figure(), "η values must be comma-separated numbers."
    if not eta_values:
        return go.Figure(), "Enter at least one η value."
    try:
        id_res, rt_res = compute(est_store, ds_store, eta_values, int(n_runs or 5))
        id_fig = build_figures(id_res, rt_res, ds_store["d"], est_store["name"])
        return id_fig, ""
    except Exception as exc:
        return go.Figure(), f"Error: {exc}"
