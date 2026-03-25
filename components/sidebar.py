"""
Sidebar: estimator selector + dynamic hyperparameter panel.

Exports:
  layout()  — the sidebar html.Div (call once when building app layout)

Callbacks registered at import time:
  render_params  — rebuilds controls when estimator changes
  sync_est_store — keeps est-store in sync with current selections
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ALL, callback

from registry.estimators import ESTIMATOR_NAMES, PARAM_DEFS

# ── Styles ─────────────────────────────────────────────────────────────────────

_SECTION_LABEL = {
    "fontSize": "10px",
    "fontWeight": "700",
    "letterSpacing": "0.1em",
    "color": "#adb5bd",
    "marginBottom": "8px",
    "textTransform": "uppercase",
}

_SIDEBAR = {
    "background": "#f8f9fa",
    "borderRight": "1px solid #dee2e6",
    "padding": "20px 16px",
    "height": "100vh",
    "overflowY": "auto",
    "position": "sticky",
    "top": 0,
    "width": "240px",
    "flexShrink": "0",
}

# ── Control factory ────────────────────────────────────────────────────────────

def _param_control(est: str, pdef: dict) -> html.Div:
    """Render one labelled form control for a single hyperparameter."""
    pid = {"type": "est-param", "est": est, "param": pdef["id"]}
    t   = pdef["type"]

    if t == "select":
        ctrl = dcc.Dropdown(
            id=pid, clearable=False,
            options=[{"label": o, "value": o} for o in pdef["options"]],
            value=pdef["default"],
            style={"fontSize": "13px"},
        )
    elif t == "bool":
        ctrl = dcc.Dropdown(
            id=pid, clearable=False,
            options=[{"label": "True", "value": 1}, {"label": "False", "value": 0}],
            value=1 if pdef["default"] else 0,
            style={"fontSize": "13px"},
        )
    else:                           # "int" | "float"
        kw = dict(
            id=pid, type="number",
            value=pdef["default"],
            step=pdef.get("step", 1 if t == "int" else 0.001),
            className="form-control form-control-sm",
        )
        if pdef.get("min") is not None: kw["min"] = pdef["min"]
        if pdef.get("max") is not None: kw["max"] = pdef["max"]
        if pdef["default"] is None:     kw["placeholder"] = "auto"
        ctrl = dcc.Input(**kw)

    return html.Div([
        html.Label(
            pdef["label"],
            style={"fontSize": "11px", "color": "#6c757d",
                   "marginBottom": "2px", "display": "block"},
        ),
        ctrl,
    ], style={"marginBottom": "10px"})


# ── Layout ─────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([
        html.Div([
            html.Div("Estimator", style=_SECTION_LABEL),
            dcc.Dropdown(
                id="est-select",
                options=[{"label": e, "value": e} for e in ESTIMATOR_NAMES],
                value="MLE",
                clearable=False,
                style={"fontSize": "14px"},
            ),
        ], style={"marginBottom": "28px"}),

        html.Div([
            html.Div("Hyperparameters", style=_SECTION_LABEL),
            html.Div(id="param-container"),
        ]),
    ], style=_SIDEBAR)


# ── Callbacks ──────────────────────────────────────────────────────────────────

@callback(
    Output("param-container", "children"),
    Input("est-select", "value"),
)
def render_params(est: str):
    if est not in PARAM_DEFS:
        return html.Div("No parameters.", className="text-muted small")
    return [_param_control(est, p) for p in PARAM_DEFS[est]]


@callback(
    Output("est-store", "data"),
    Input("est-select", "value"),
    Input({"type": "est-param", "est": ALL, "param": ALL}, "value"),
    State({"type": "est-param", "est": ALL, "param": ALL}, "id"),
)
def sync_est_store(est_name, values, ids):
    params = {
        pid["param"]: val
        for pid, val in zip(ids, values)
        if pid["est"] == est_name
    }
    return {"name": est_name, "params": params}
