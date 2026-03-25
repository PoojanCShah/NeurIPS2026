"""
Dataset configuration card → dataset-store.

Exports:
  layout()  — dbc.Card with family selector + dynamic param controls + n_samples

Callbacks registered at import time:
  render_ds_params  — rebuilds dimension controls when family changes
  sync_dataset_store — writes {family, d, D, eta, n_samples} to dataset-store
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ALL, callback

from registry.datasets import FAMILIES, ETA_OPTIONS, FAMILY_PARAMS


def layout() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(
            html.Span("Dataset", className="fw-semibold",
                      style={"fontSize": "13px"})
        ),
        dbc.CardBody(
            dbc.Row([
                # Family selector — always visible
                dbc.Col(_ctrl("Family",
                    dcc.Dropdown(
                        id="ds-family",
                        options=[{"label": f.capitalize(), "value": f}
                                 for f in FAMILIES],
                        value="hypercube",
                        clearable=False,
                        style={"fontSize": "13px"},
                    )
                ), width=3),

                # Dynamic parameter controls (d, and D for linear)
                dbc.Col(
                    html.Div(id="ds-param-container",
                             style={"display": "contents"}),
                    width=6,
                ),

                # η — always visible
                dbc.Col(_ctrl("Base noise η",
                    dcc.Dropdown(
                        id="ds-eta",
                        options=[{"label": str(e), "value": e}
                                 for e in ETA_OPTIONS],
                        value=0.0,
                        clearable=False,
                        style={"fontSize": "13px"},
                    )
                ), width=2),

                # n_samples — always visible
                dbc.Col(_ctrl("n_samples",
                    dcc.Input(
                        id="ds-n-samples",
                        type="number",
                        value=2000,
                        min=50, max=25000, step=50,
                        className="form-control form-control-sm",
                    )
                ), width=1),
            ]),
            style={"padding": "12px 16px"},
        ),
    ], className="mb-3")


def _ctrl(label: str, control) -> html.Div:
    return html.Div([
        html.Label(label, className="form-label small text-muted mb-1"),
        control,
    ])


# ── Callbacks ──────────────────────────────────────────────────────────────────

def _param_control(pdef: dict):
    """Render one form control for a single family parameter."""
    pid = {"type": "ds-param", "param": pdef["id"]}
    t   = pdef["type"]

    if t == "select":
        ctrl = dcc.Dropdown(
            id=pid, clearable=False,
            options=[{"label": str(v), "value": v} for v in pdef["options"]],
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
    else:  # "int" | "float"
        kw = dict(
            id=pid, type="number",
            value=pdef["default"],
            step=pdef.get("step", 1 if t == "int" else 0.01),
            className="form-control form-control-sm",
        )
        if pdef.get("min") is not None: kw["min"] = pdef["min"]
        if pdef.get("max") is not None: kw["max"] = pdef["max"]
        if pdef["default"] is None:     kw["placeholder"] = "auto"
        ctrl = dcc.Input(**kw)

    return ctrl


@callback(
    Output("ds-param-container", "children"),
    Input("ds-family", "value"),
)
def render_ds_params(family: str):
    """Rebuild parameter controls whenever the family changes."""
    if not family or family not in FAMILY_PARAMS:
        return []
    col_width = 12 // len(FAMILY_PARAMS[family])
    return dbc.Row([
        dbc.Col(
            _ctrl(p["label"], _param_control(p)),
            width=col_width,
        )
        for p in FAMILY_PARAMS[family]
    ], className="g-2")


@callback(
    Output("dataset-store", "data"),
    Input("ds-family",    "value"),
    Input({"type": "ds-param", "param": ALL}, "value"),
    State({"type": "ds-param", "param": ALL}, "id"),
    Input("ds-eta",       "value"),
    Input("ds-n-samples", "value"),
)
def sync_dataset_store(family, param_values, param_ids, eta, n_samples):
    params = {pid["param"]: val
              for pid, val in zip(param_ids, param_values)
              if val is not None}
    return {
        "family":    family or "hypercube",
        "d":         int(params.get("d", 2)),
        "D":         int(params["D"]) if "D" in params else None,
        "eta":       float(eta) if eta is not None else 0.0,
        "n_samples": int(n_samples) if n_samples else 2000,
    }
