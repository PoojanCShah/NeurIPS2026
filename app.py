"""
Entry point for the NeurIPS 2026 ID estimation dashboard.

Run with:
    python app.py
    python app.py --port 8051 --debug
"""

import os
import argparse
import dash
import dash_bootstrap_components as dbc

# Import all tab/component modules so their @callback decorators register
import components.sidebar        # noqa: F401
import components.dataset_panel  # noqa: F401
import tabs.exp1                 # noqa: F401
import tabs.exp2                 # noqa: F401
import tabs.exp3                 # noqa: F401

from layout import build_layout

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="ID Estimation Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server  # expose Flask server for gunicorn

app.layout = build_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  type=int,  default=int(os.environ.get("PORT", 7860)))
    parser.add_argument("--host",  type=str,  default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    app.run(debug=args.debug, host=args.host, port=args.port)
