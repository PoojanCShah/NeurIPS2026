"""
Top-level layout for the NeurIPS 2026 ID estimation dashboard.

Assembles sidebar (estimator + hyperparams) + main area
(dataset panel + experiment tabs).
"""

import dash_bootstrap_components as dbc
from dash import html, dcc

from components.sidebar import layout as sidebar_layout
from components.dataset_panel import layout as dataset_panel_layout
from tabs import exp1, exp2, exp3


def build_layout() -> html.Div:
    return html.Div([
        # ── Shared stores ────────────────────────────────────────────────────
        dcc.Store(id="est-store",     storage_type="memory"),
        dcc.Store(id="dataset-store", storage_type="memory"),

        # ── Page shell ───────────────────────────────────────────────────────
        html.Div([
            # Left: sticky sidebar
            sidebar_layout(),

            # Right: main content
            html.Div([
                # Header
                html.Div([
                    html.H5("Intrinsic Dimension Estimation",
                            className="mb-0 fw-semibold",
                            style={"fontSize": "16px"}),
                    html.Span("NeurIPS 2026",
                              className="text-muted",
                              style={"fontSize": "12px"}),
                ], style={
                    "borderBottom": "1px solid #dee2e6",
                    "padding": "12px 20px",
                    "display": "flex",
                    "alignItems": "baseline",
                    "gap": "12px",
                }),

                # Dataset panel
                html.Div(dataset_panel_layout(),
                         style={"padding": "16px 20px 0"}),

                # Experiment tabs
                html.Div(
                    dbc.Tabs([
                        dbc.Tab(exp1.layout(),
                                label="Exp 1 — Noise",
                                tab_id="tab-exp1"),
                        dbc.Tab(exp2.layout(),
                                label="Exp 2 — Sample Size",
                                tab_id="tab-exp2"),
                        dbc.Tab(exp3.layout(),
                                label="Exp 3 — JL Projection",
                                tab_id="tab-exp3"),
                    ],
                    id="exp-tabs",
                    active_tab="tab-exp1",
                    ),
                    style={"padding": "12px 20px 20px"},
                ),

            ], style={
                "flex": "1",
                "minWidth": 0,
                "overflowY": "auto",
                "height": "100vh",
            }),
        ], style={
            "display": "flex",
            "height": "100vh",
            "overflow": "hidden",
        }),
    ])
