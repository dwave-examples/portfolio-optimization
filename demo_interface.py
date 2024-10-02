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

def slider(
    label: str,
    id: str,
    config: dict,
    wrapper_id: str = "",
    marks: dict={},
    show_tooltip: bool=True,
    dots: bool=False
) -> html.Div:
    """Slider element for value selection.

    Args:
        label: The title that goes above the slider.
        id: A unique selector for this element.
        config: A dictionary of slider configerations, see dcc.Slider Dash docs.
        wrapper_id: A unique selector for the wrapper element.
        marks: Optional marks to show instead of max and min.
        show_tooltip: Whether the tooltip should be visible.
        dots: Whether dots should be shown where the important data points are.
    """
    return html.Div(
        className="slider-wrapper",
        id=wrapper_id,
        children=[
            html.Label(label) if label else (),
            dcc.Slider(
                id=id,
                className="slider",
                **config,
                marks=marks if marks else {
                    config["min"]: str(config["min"]),
                    config["max"]: str(config["max"]),
                },
                dots=dots,
                tooltip={
                    "placement": "bottom",
                    "always_visible": True,
                } if show_tooltip else None,
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
            html.Label(label) if label else (),
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
            dropdown(
                "Solver",
                "sampler-type-select",
                sorted(sampler_options, key=lambda op: op["value"]),
            ),
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
                max_date_allowed=date.today().replace(day=1) - timedelta(days=1), # end of last month
                start_date=date.today() - timedelta(days=730), # 2 years prior
                end_date=date.today().replace(day=1) - timedelta(days=1),
                minimum_nights=120,
                # display_format='MM/Y',
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
                "transaction-cost-wrapper",
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


def generate_table(table_dict: dict, comparison: list=[]) -> html.Table:
    """Generates solution table.

    Args:
        table_dict: Dictionary of table values where each key, value pair make a row of the table.
        comparison: A list of comparisons between tables.

    Returns:
        html.Table: A table containing results.
    """

    return html.Table(
        html.Tbody(
            [
                html.Tr(
                    [
                        html.Td(key),
                        html.Td(
                            [
                                table_dict[key],
                                html.Span(
                                    "↑" if comparison[i] else "↓",
                                    className=f"arrow-{comparison[i]}",
                                    style={"visibility": "hidden"} if comparison[i] is None else {}
                                )
                            ]
                            if i < len(comparison) else table_dict[key]
                        )
                    ]
                ) for i, key in enumerate(table_dict)
            ]
        )
    )


def generate_table_group(
    tables_data: list,
    comparisons_data: list=[],
    title: str="",
) -> html.Div:
    """Generates a grouped collection of tables with optional title and comparison data.

    Args:
        tables_data: A list of dictionaries of tables data.
        comparisons_data: List of lists of comparison data between the passed in tables.
        title: The title to display above the tables.

    Returns:
        html.Div: A div containing a title and grouped tables.
    """
    if comparisons_data:
        tables = [
            generate_table(table, comparison)
            for table, comparison in zip(tables_data, comparisons_data)
        ]
    else:
        tables = [generate_table(table) for table in tables_data]

    return html.Div(
        [html.Div(title) if title else (), html.Div(tables, className="results-tables")],
        className="results-comparison"
    )


def generate_dates_slider(dates: list) -> html.Div:
    """Generates date slider to switch between results tables.

    Args:
        dates: A list of the dates in the slider.

    Returns:
        html.Div: A div containing a dates slider.
    """
    last_date = len(dates)-1

    return slider(
        "",
        "results-date-selector",
        {
            "min": 0,
            "max": last_date,
            "value": last_date,
            "step": 1
        },
        marks={0: dates[0], last_date: dates[-1]},
        dots=True,
        show_tooltip=False,
    )


def create_interface() -> html.Div:
    """Set the application HTML."""
    return html.Div(
        id="app-container",
        children=[
            # Below are any temporary storage items, e.g., for sharing data between callbacks.
            dcc.Store(id="run-in-progress", data=False),  # Indicates whether run is in progress
            dcc.Store(id="max-iterations", data=0),  # Max iterations of result loop
            dcc.Store(id="selected-period", data=0),  # The currently selected period option
            dcc.Store(id="results-date-dict"),  # Dictionary of date periods and their solutions
            dcc.Store(id="portfolio"),
            dcc.Store(id="loop-store"),
            dcc.Store(id="settings-store"),
            dcc.Interval(
                id="loop-interval",
                interval=50,  # Interval in milliseconds
                n_intervals=0,
                disabled=True
            ),
            dcc.Store(id="loop-running", data=False),
            dcc.Store(id="iteration", data=3),
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
                                                children=html.Div(
                                                    [
                                                        dcc.Graph(
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
                                        label="Graph",
                                        id="graph-tab",
                                        className="tab",
                                        style={"display": "none"},
                                        children=html.Div(
                                            [
                                                dcc.Graph(
                                                    id="output-graph",
                                                    responsive=True,
                                                    config={"displayModeBar": False},
                                                )
                                            ],
                                        ),
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
                                                    html.Div(
                                                        [
                                                            html.Div(id="dates-slider"),
                                                            html.Div(id="dynamic-results-table")
                                                        ]
                                                    )
                                                ]
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
