"""
Exp 3 — Johnson–Lindenstrauss Projection

Sweeps projection dimension k at fixed (family, d, η, n_samples).
For each k, data is projected via a scaled Gaussian JL map;
ID is estimated on the projected data and compared to the
unprojected baseline.
"""

import time
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback

from registry.estimators import build_estimator
from registry.datasets import DEFAULT_JL_DIMS, AMBIENT_DIM, ensure_dataset, load_dataset

Z95 = 1.96

# ── Layout ─────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([
        dbc.Card(dbc.CardBody(
            dbc.Row([
                dbc.Col([
                    html.Label("Projection dims k (comma-separated)",
                               className="form-label small text-muted mb-1"),
                    dcc.Input(
                        id="exp3-jl-dims",
                        type="text",
                        value=", ".join(str(k) for k in DEFAULT_JL_DIMS),
                        debounce=True,
                        placeholder="e.g. 5, 10, 50, 100, 500",
                        className="form-control form-control-sm",
                    ),
                ], width=7),
                dbc.Col([
                    html.Label("n_runs", className="form-label small text-muted mb-1"),
                    dcc.Input(id="exp3-n-runs", type="number",
                              value=3, min=1, max=20, step=1,
                              className="form-control form-control-sm"),
                ], width=2),
                dbc.Col([
                    html.Br(),
                    dbc.Button("Run ▶", id="exp3-run-btn", color="primary",
                               size="sm", className="w-100 mt-1"),
                ], width=3),
            ]),
            style={"padding": "12px 16px"},
        ), className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Loading(dcc.Graph(id="exp3-id-plot",
                                          style={"height": "420px"})), width=12),
        ]),
        html.Div(id="exp3-error", className="text-danger small mt-2"),
    ])


# ── JL projection ──────────────────────────────────────────────────────────────

def _jl_project(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Project X ∈ R^{n×D} → R^{n×k} via a scaled Gaussian JL map.
    Returns X unchanged when k >= D.
    """
    D = X.shape[1]
    if k >= D:
        return X
    A = rng.standard_normal((k, D)) / np.sqrt(k)
    return X @ A.T


# ── Pure computation (no Dash) ─────────────────────────────────────────────────

def compute(est_store: dict, ds_store: dict,
            jl_dims: list[int], n_runs: int
            ) -> tuple[list, dict, dict]:
    """
    Returns
    -------
    orig_ids   : list of ID estimates on unprojected data (length n_runs)
    jl_id_res  : {k -> [id_run_0, ...]}
    jl_rt_res  : {k -> [time_run_0, ...]}
    """
    est_name  = est_store["name"]
    params    = est_store.get("params", {})
    family    = ds_store["family"]
    d         = ds_store["d"]
    D         = ds_store.get("D")
    eta       = ds_store["eta"]
    n_samples = ds_store["n_samples"]

    ensure_dataset(family, d, eta, D=D)
    ds     = load_dataset(family, d, eta=eta, D=D)
    X_full = ds["X"]
    ambient = X_full.shape[1]

    # Only keep k values that are smaller than ambient dim
    valid_ks = sorted(k for k in jl_dims if k < ambient)

    n = min(n_samples, len(X_full))
    orig_ids              = []
    jl_id_res: dict       = {k: [] for k in valid_ks}
    jl_rt_res: dict       = {k: [] for k in valid_ks}

    for run in range(n_runs):
        sub_rng  = np.random.default_rng(42 + run)
        proj_rng = np.random.default_rng(42 + run + 10_000)

        X = X_full[sub_rng.choice(len(X_full), n, replace=False)]

        # Baseline (unprojected)
        est = build_estimator(est_name, params)
        est.fit(X)
        orig_ids.append(float(est.dimension_))

        # Projected
        for k in valid_ks:
            X_proj = _jl_project(X, k, proj_rng)
            est    = build_estimator(est_name, params)
            t0     = time.perf_counter()
            est.fit(X_proj)
            jl_rt_res[k].append(time.perf_counter() - t0)
            jl_id_res[k].append(float(est.dimension_))

    return orig_ids, jl_id_res, jl_rt_res


# ── Figure builders (no Dash) ──────────────────────────────────────────────────

def _ci(vals: list[float]) -> float:
    a = np.array(vals)
    return float(Z95 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def build_figures(orig_ids: list, jl_id_res: dict, jl_rt_res: dict,
                  true_d: int, est_name: str) -> tuple:
    ks       = sorted(jl_id_res)
    means    = [float(np.mean(jl_id_res[k])) for k in ks]
    cis      = [_ci(jl_id_res[k]) for k in ks]
    rt_means = [float(np.mean(jl_rt_res[k])) for k in ks]

    orig_mean = float(np.mean(orig_ids))
    orig_ci   = _ci(orig_ids)

    # ── ID vs k ───────────────────────────────────────────────────────────────
    id_fig = go.Figure()

    # True d
    id_fig.add_hline(y=true_d, line_dash="dash", line_color="grey",
                     annotation_text=f"true d = {true_d}",
                     annotation_position="top right")

    # Baseline (unprojected) as dotted band
    id_fig.add_hrect(y0=orig_mean - orig_ci, y1=orig_mean + orig_ci,
                     fillcolor="rgba(148,103,189,0.10)", line_width=0)
    id_fig.add_hline(y=orig_mean, line_dash="dot", line_color="#9467bd",
                     line_width=1.8,
                     annotation_text="id_original",
                     annotation_position="bottom right")

    # JL CI band
    id_fig.add_trace(go.Scatter(
        x=ks + ks[::-1],
        y=[m + c for m, c in zip(means, cis)] +
          [m - c for m, c in zip(reversed(means), reversed(cis))],
        fill="toself", fillcolor="rgba(31,119,180,0.12)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    # JL mean line
    id_fig.add_trace(go.Scatter(
        x=ks, y=means, mode="lines+markers",
        name=f"{est_name} (JL)",
        line=dict(color="#1f77b4", width=2), marker=dict(size=7),
        error_y=dict(type="data", array=cis, visible=True, color="#1f77b4"),
    ))

    # Mark k = true_d
    if true_d in ks:
        id_fig.add_vline(x=true_d, line_dash="dashdot", line_color="#333",
                         line_width=0.9, opacity=0.5,
                         annotation_text=f"k = d = {true_d}",
                         annotation_position="top left")

    id_fig.update_layout(
        title=f"{est_name} — ID vs JL projection dim k",
        xaxis_title="Projection dim k (log scale)", yaxis_title="Estimated ID",
        xaxis_type="log",
        template="plotly_white", margin=dict(t=50, b=40, l=50, r=20),
    )

    return id_fig


# ── Callback ───────────────────────────────────────────────────────────────────

@callback(
    Output("exp3-id-plot", "figure"),
    Output("exp3-error",   "children"),
    Input("exp3-run-btn",  "n_clicks"),
    State("est-store",     "data"),
    State("dataset-store", "data"),
    State("exp3-jl-dims",  "value"),
    State("exp3-n-runs",   "value"),
    prevent_initial_call=True,
)
def run_exp3(_, est_store, ds_store, jl_dims_str, n_runs):
    if not est_store or not ds_store:
        return go.Figure(), "Estimator or dataset store is empty."
    try:
        jl_dims = [int(x.strip()) for x in (jl_dims_str or "").split(",")
                   if x.strip()]
    except ValueError:
        return go.Figure(), "Projection dims must be comma-separated integers."
    if not jl_dims:
        return go.Figure(), "Enter at least one projection dimension."
    try:
        orig_ids, jl_id_res, jl_rt_res = compute(
            est_store, ds_store, jl_dims, int(n_runs or 3)
        )
        id_fig = build_figures(
            orig_ids, jl_id_res, jl_rt_res, ds_store["d"], est_store["name"]
        )
        return id_fig, ""
    except Exception as exc:
        return go.Figure(), f"Error: {exc}"
