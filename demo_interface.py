# Copyright 2024 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file stores the HTML layout for the app."""
from __future__ import annotations
from datetime import date, timedelta

from dash import dcc, html
import plotly.graph_objs as go
import yfinance as yf


from demo_configs import (
    BUDGET,
    DESCRIPTION,
    MAIN_HEADER,
    STOCK_OPTIONS,
    TRANSACTION_COST,
    SOLVER_TIME,
    THEME_COLOR_SECONDARY,
    THUMBNAIL,
)
from src.demo_enums import PeriodType, SolverType

SAMPLER_TYPES = {SolverType.CQM: "Quantum Hybrid (CQM)", SolverType.DQM: "Quantum Hybrid (DQM)"}

stock_options = [
    {"label": f"{yf.Ticker(ticker).info['shortName']} ({ticker})", "value": ticker}
    for ticker in STOCK_OPTIONS["options"]
]

def slider(label: str, id: str, config: dict) -> html.Div:
    """Slider element for value selection.

    Args:
        label: The title that goes above the slider.
        id: A unique selector for this element.
        config: A dictionary of slider configerations, see dcc.Slider Dash docs.
    """
    return html.Div(
        className="slider-wrapper",
        children=[
            html.Label(label),
            dcc.Slider(
                id=id,
                className="slider",
                **config,
                marks={
                    config["min"]: str(config["min"]),
                    config["max"]: str(config["max"]),
                },
                tooltip={
                    "placement": "bottom",
                    "always_visible": True,
                },
            ),
        ],
    )


def dropdown(label: str, id: str, options: list) -> html.Div:
    """Dropdown element for option selection.

    Args:
        label: The title that goes above the dropdown.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
    """
    return html.Div(
        className="dropdown-wrapper",
        children=[
            html.Label(label),
            dcc.Dropdown(
                id=id,
                options=options,
                value=options[0]["value"],
                clearable=False,
                searchable=False,
            ),
        ],
    )


def generate_options(options_list: list) -> list[dict]:
    """Generates options for dropdowns, checklists, radios, etc."""
    return [{"label": label, "value": i} for i, label in enumerate(options_list)]


def generate_settings_form() -> html.Div:
    """This function generates settings for selecting the scenario, model, and solver.

    Returns:
        html.Div: A Div containing the settings for selecting the scenario, model, and solver.
    """
    sampler_options = [
        {"label": label, "value": solver_type.value}
        for solver_type, label in SAMPLER_TYPES.items()
    ]

    return html.Div(
        className="settings",
        children=[
            html.Label("Stocks"),
            dcc.Dropdown(
                stock_options,
                STOCK_OPTIONS["value"],
                id="stocks",
                multi=True,
            ),
            html.Label("Date Range"),
            dcc.DatePickerRange(
                id="date-range",
                max_date_allowed=date.today() - timedelta(days=1), # yesterday
                start_date=date.today() - timedelta(days=730), # 2 years from today
                end_date=date.today() - timedelta(days=1),
                minimum_nights=30,
            ),
            html.Label("Budget (USD)"),
            dcc.Input(
                id="budget",
                type="number",
                **BUDGET,
            ),
            slider(
                "Transaction Cost (%)",
                "transaction-cost",
                TRANSACTION_COST,
            ),
            dropdown(
                "Solver",
                "sampler-type-select",
                sorted(sampler_options, key=lambda op: op["value"]),
            ),
            html.Label("Solver Time Limit (seconds)"),
            dcc.Input(
                id="solver-time-limit",
                type="number",
                **SOLVER_TIME,
            ),
        ],
    )


def generate_run_buttons() -> html.Div:
    """Run and cancel buttons to run the optimization."""
    return html.Div(
        id="button-group",
        children=[
            html.Button(id="run-button", children="Run Optimization", n_clicks=0, disabled=False),
            html.Button(
                id="cancel-button",
                children="Cancel Optimization",
                n_clicks=0,
                className="display-none",
            ),
        ],
    )


def generate_problem_details_table_rows(solver: str, time_limit: int) -> list[html.Tr]:
    """Generates table rows for the problem details table.

    Args:
        solver: The solver used for optimization.
        time_limit: The solver time limit.

    Returns:
        list[html.Tr]: List of rows for the problem details table.
    """

    table_rows = (
        ("Solver:", solver, "Time Limit:", f"{time_limit}s"),
        ### Add more table rows here. Each tuple is a row in the table.
    )

    return [html.Tr([html.Td(cell) for cell in row]) for row in table_rows]


def problem_details(index: int) -> html.Div:
    """Generate the problem details section.

    Args:
        index: Unique element id to differentiate matching elements.
            Must be different from left column collapse button.

    Returns:
        html.Div: Div containing a collapsable table.
    """
    return html.Div(
        id={"type": "to-collapse-class", "index": index},
        className="details-collapse-wrapper collapsed",
        children=[
            # Problem details collapsible button and header
            html.Button(
                id={"type": "collapse-trigger", "index": index},
                className="details-collapse",
                children=[
                    html.H5("Problem Details"),
                    html.Div(className="collapse-arrow"),
                ],
            ),
            html.Div(
                className="details-to-collapse",
                children=[
                    html.Table(
                        className="solution-stats-table",
                        children=[
                            # Problem details table header (optional)
                            html.Thead(
                                [
                                    html.Tr(
                                        [
                                            html.Th(
                                                colSpan=2,
                                                children=["Problem Specifics"],
                                            ),
                                            html.Th(
                                                colSpan=2,
                                                children=["Run Time"],
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            # A Dash callback function will generate content in Tbody
                            html.Tbody(id="problem-details"),
                        ],
                    ),
                ],
            ),
        ],
    )


def create_interface() -> html.Div:
    """Set the application HTML."""
    return html.Div(
        id="app-container",
        children=[
            # Below are any temporary storage items, e.g., for sharing data between callbacks.
            dcc.Store(id="run-in-progress", data=False),  # Indicates whether run is in progress
            dcc.Store(id="iteration", data=0),  # Current iteration of results loop
            dcc.Store(id="max-iterations", data=0),  # Max iterations of result loop
            dcc.Store(id="selected-period", data=0),  # The currently selected period option
            # Header brand banner
            html.Div(
                className="banner",
                children=[
                    html.Img(src=THUMBNAIL),
                    html.Div(
                        [
                            html.Div(
                                html.Button(
                                    period.label,
                                    id={"type": "period-option", "index": index},
                                )
                            )
                            for index, period in enumerate(PeriodType)
                        ]
                    ),
                ]
            ),
            # Settings and results columns
            html.Div(
                className="columns-main",
                children=[
                    # Left column
                    html.Div(
                        id={"type": "to-collapse-class", "index": 0},
                        className="left-column",
                        children=[
                            html.Div(
                                className="left-column-layer-1",  # Fixed width Div to collapse
                                children=[
                                    html.Div(
                                        className="left-column-layer-2",  # Padding and content wrapper
                                        children=[
                                            html.H1(MAIN_HEADER),
                                            html.P(DESCRIPTION),
                                            generate_settings_form(),
                                            generate_run_buttons(),
                                        ],
                                    )
                                ],
                            ),
                            # Left column collapse button
                            html.Div(
                                html.Button(
                                    id={"type": "collapse-trigger", "index": 0},
                                    className="left-column-collapse",
                                    children=[html.Div(className="collapse-arrow")],
                                ),
                            ),
                        ],
                    ),
                    # Right column
                    html.Div(
                        className="right-column",
                        children=[
                            dcc.Tabs(
                                id="tabs",
                                value="input-tab",
                                mobile_breakpoint=0,
                                children=[
                                    dcc.Tab(
                                        label="Input",
                                        id="input-tab",
                                        value="input-tab",  # used for switching tabs programatically
                                        className="tab",
                                        children=[
                                            dcc.Loading(
                                                parent_className="input",
                                                className="input-loading",
                                                type="circle",
                                                color=THEME_COLOR_SECONDARY,
                                                # A Dash callback (in app.py) will generate content in the Div below
                                                children=html.Div(
                                                    [
                                                        dcc.Graph(
                                                            # id={"type": f"graph", "index": 0},
                                                            id="input-graph",
                                                            responsive=True,
                                                            config={"displayModeBar": False},
                                                        )
                                                    ],
                                                ),
                                            ),
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Results",
                                        id="results-tab",
                                        className="tab",
                                        disabled=True,
                                        children=[
                                            html.Div(
                                                className="tab-content-results",
                                                children=[
                                                    dcc.Loading(
                                                        parent_className="results",
                                                        type="circle",
                                                        color=THEME_COLOR_SECONDARY,
                                                        # A Dash callback (in app.py) will generate content in the Div below
                                                        children=html.Div(
                                                            # id="results"
                                                            [
                                                                dcc.Graph(
                                                                    # id={"type": f"graph", "index": 0},
                                                                    id="output-graph",
                                                                    responsive=True,
                                                                    config={"displayModeBar": False},
                                                                )
                                                            ],
                                                        ),
                                                    ),
                                                    # Problem details dropdown
                                                    html.Div([html.Hr(), problem_details(1)]),
                                                ],
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )
