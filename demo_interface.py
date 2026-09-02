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
from enum import EnumMeta

from dash import dcc, html
import dash_mantine_components as dmc

from demo_configs import (
    BUDGET,
    DATES_DEFAULT,
    DEFAULT_STOCKS,
    DESCRIPTION,
    MAIN_HEADER,
    SOLVER_TIME,
    THUMBNAIL,
    TRANSACTION_COST,
)
from src.demo_enums import PeriodType, SolverType


THEME_COLOR = "#2d4376"


def slider(label: str, id: str, config: dict, marks: list | None = None, labelAlwaysOn: bool = True) -> html.Div:
    """Slider element for value selection.

    Args:
        label: The title that goes above the slider.
        id: A unique selector for this element.
        config: A dictionary of slider configurations, see dmc.Slider Dash Mantine docs.
    """
    return html.Div(
        className="slider-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.Slider(
                id=id,
                className="slider",
                **config,
                marks=(
                    marks
                    if marks
                    else [
                        {"value": config["min"], "label": f'{config["min"]}'},
                        {"value": config["max"], "label": f'{config["max"]}'},
                    ]
                ),
                labelAlwaysOn=labelAlwaysOn,
                thumbLabel=f"{label} slider",
                color=THEME_COLOR,
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
            html.Label(label, htmlFor=id),
            dmc.Select(
                id=id,
                data=options,
                value=options[0]["value"],
                allowDeselect=False,
            ),
        ],
    )


def multiselect(label: str, id: str, options: list, values: list) -> html.Div:
    """Multiselect element for option selection.

    Args:
        label: The title that goes above the multiselect.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
        values: A list of values that should be preselected in the multiselect.
    """
    return html.Div(
        className="dropdown-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.MultiSelect(
                id=id,
                data=options,
                value=values,
            ),
        ],
    )


def radio(label: str, id: str, options: list, value: str, inline: bool = True) -> html.Div:
    """Radio element for option selection.

    Args:
        label: The title that goes above the radio.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
        value: The value of the radio that should be preselected.
        inline: Whether the options are displayed beside or below each other.
    """
    return html.Div(
        className="radio-wrapper",
        children=[
            dmc.RadioGroup(
                id=id,
                className=f"radio{' radio--inline' if inline else ''}",
                label=label,
                value=value,
                children=dmc.Group(
                    [
                        dmc.Radio(option["label"], value=option["value"], color=THEME_COLOR)
                        for option in options
                    ]
                ),
            ),
        ],
    )


def input(label: str, id: str, configs: dict, type: str="number") -> html.Div:
    """Input element for either text or number input.

    Args:
        label: The title that goes above the input.
        id: A unique selector for this element.
        configs: A dictionary of configurations for the input element.
        type: The type of input, either "number" or "text".
    """
    return html.Div(
        className="input-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.TextInput(
                id=id,
                **configs,
            ) if type == "text" else dmc.NumberInput(
                id=id,
                **configs,
            ),
        ],
    )


def generate_options(options: list | EnumMeta | dict) -> list[dict]:
    """Format options for dropdowns, checklists, radios, etc.

    Args:
        options: A list, EnumMeta, or dictionary of options to format.

    Returns:
        A list of dictionaries with "label" and "value" keys for each option.
    """
    if isinstance(options, EnumMeta):
        return [{"label": option.label, "value": f"{option.value}"} for option in options]

    if isinstance(options, dict):
        return [{"label": f"{key}", "value": f"{value}"} for key, value in options.items()]

    return [{"label": f"{option}", "value": f"{option}"} for option in options]


def generate_settings_form() -> html.Div:
    """Generate settings for selecting the scenario, model, and solver.

    Returns:
        A Div containing the settings for selecting the scenario, model, and solver.
    """
    solver_options = generate_options(SolverType)
    stock_options = generate_options(DEFAULT_STOCKS)

    return html.Div(
        className="settings",
        children=[
            dropdown(
                "Solver",
                "solver-type-select",
                sorted(solver_options, key=lambda op: op["value"]),
            ),
            input(
                "Solver Time Limit (seconds)",
                "solver-time-limit",
                SOLVER_TIME,
            ),
            multiselect(
                "Stocks",
                "stocks",
                sorted(stock_options, key=lambda op: op["value"]),
                DEFAULT_STOCKS,
            ),
            html.P("Please select at least 2 stocks", id="stocks-error", className="display-none"),
            dmc.DatePickerInput(
                id="date-range",
                label="Date Range",
                description="Date range must be at least four months.",
                minDate=date(2020, 8, 5),
                maxDate=date.today().replace(day=1) - timedelta(days=1),  # prev month end
                type="range",
                value=[DATES_DEFAULT[0], DATES_DEFAULT[1]],
                # maw=125,
            ),
            html.P("Date range must be at least four months.", className="date-range-text"),
            input(
                "Budget (USD)",
                "budget",
                BUDGET,
            ),
            html.Div(
                [
                    slider(
                        "Transaction Cost (%)",
                        "transaction-cost",
                        TRANSACTION_COST,
                    ),
                ],
                id="transaction-cost-wrapper",
            ),
            radio(
                "Period",
                "period-options",
                generate_options(PeriodType),
                value=f"{PeriodType.MULTI.value}",
                inline=False,
            ),
        ],
    )


def generate_run_buttons() -> html.Div:
    """Generate run and cancel buttons to run the optimization."""
    return html.Div(
        id="button-group",
        children=[
            html.Button("Run Optimization", id="run-button", className="button"),
            html.Button(
                "Cancel Optimization",
                id="cancel-button",
                className="button",
                style={"display": "none"},
            ),
        ],
    )


def generate_table(table_dict: dict, comparison: list = []) -> html.Table:
    """Generates solution table.

    Args:
        table_dict: Dictionary of table values where each key, value pair make a row of the table.
        comparison: A list of comparisons between tables.

    Returns:
        A table containing results.
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
                                    style={"visibility": "hidden"} if comparison[i] is None else {},
                                ),
                            ]
                            if i < len(comparison)
                            else table_dict[key]
                        ),
                    ]
                )
                for i, key in enumerate(table_dict)
            ]
        )
    )


def generate_table_group(
    tables_data: list,
    comparisons_data: list = [],
    title: str = "",
) -> html.Div:
    """Generates a grouped collection of tables with optional title and comparison data.

    Args:
        tables_data: A list of dictionaries of tables data.
        comparisons_data: List of lists of comparison data between the passed in tables.
        title: The title to display above the tables.

    Returns:
        A div containing a title and grouped tables.
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
        className="results-comparison",
    )


def generate_dates_slider(dates: list) -> html.Div:
    """Generates date slider to switch between results tables.

    Args:
        dates: A list of the dates in the slider.

    Returns:
        A div containing a dates slider.
    """
    last_date = len(dates) - 1

    return slider(
        "",
        "results-date-selector",
        {"min": 0, "max": last_date, "value": last_date, "step": 1},
        marks=[{"value": 0, "label": dates[0]}, {"value": f"{last_date}", "label": f"{dates[-1]}"}],
        labelAlwaysOn=False, 
    )


def create_interface() -> html.Div:
    """Create the main application interface."""
    return html.Div(
        id="app-container",
        children=[
            html.A(  # Skip link for accessibility
                "Skip to main content",
                href="#main-content",
                id="skip-to-main",
                className="skip-link",
                tabIndex=1,
            ),
            # Below are any temporary storage items, e.g., for sharing data between callbacks.
            dcc.Store(id="run-in-progress", data=False),  # Indicates whether run is in progress
            dcc.Store(id="max-iterations", data=0),  # Max iterations of result loop
            dcc.Store(id="results-date-dict"),  # Dictionary of date periods and their solutions
            dcc.Store(id="portfolio"),
            dcc.Store(id="loop-store"),
            dcc.Store(id="settings-store"),
            dcc.Store(id="all-stocks-store"),
            dcc.Interval(
                id="loop-interval",
                interval=50,  # Interval in milliseconds
                n_intervals=0,
                disabled=True,
            ),
            dcc.Store(id="loop-running", data=False),
            dcc.Store(id="iteration", data=3),
            # Settings and results columns
            html.Main(
                className="columns-main",
                id="main-content",
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
                                            html.Div(
                                                [
                                                    html.H1(MAIN_HEADER),
                                                    html.P(DESCRIPTION),
                                                ],
                                                className="title-section",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        html.Div(
                                                            [
                                                                generate_settings_form(),
                                                                generate_run_buttons(),
                                                            ],
                                                            className="settings-and-buttons",
                                                        ),
                                                        className="settings-and-buttons-wrapper",
                                                    ),
                                                    # Left column collapse button
                                                    html.Div(
                                                        html.Button(
                                                            id={
                                                                "type": "collapse-trigger",
                                                                "index": 0,
                                                            },
                                                            className="left-column-collapse",
                                                            title="Collapse sidebar",
                                                            children=[
                                                                html.Div(className="collapse-arrow")
                                                            ],
                                                            **{"aria-expanded": "true"},
                                                        ),
                                                    ),
                                                ],
                                                className="form-section",
                                            ),
                                        ],
                                    )
                                ],
                            ),
                        ],
                    ),
                    # Right column
                    html.Div(
                        className="right-column",
                        children=[
                            dmc.Tabs(
                                id="tabs",
                                value="input-tab",
                                color="white",
                                children=[
                                    html.Header(
                                        className="banner",
                                        children=[
                                            html.Nav(
                                                [
                                                    dmc.TabsList(
                                                        [
                                                            dmc.TabsTab("Input", value="input-tab"),
                                                            dmc.TabsTab(
                                                                "Graph",
                                                                value="graph-tab",
                                                                id="graph-tab",
                                                                disabled=True,
                                                                style={"display": "none"},
                                                            ),
                                                            dmc.TabsTab(
                                                                "Results",
                                                                value="results-tab",
                                                                id="results-tab",
                                                                disabled=True,
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                            html.Img(src=THUMBNAIL, alt="D-Wave logo"),
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value="input-tab",
                                        tabIndex="12",
                                        children=[
                                            html.Div(
                                                className="tab-content-wrapper",
                                                children=[
                                                    html.H3("Historical Stock Data"),
                                                    dcc.Loading(
                                                        parent_className="input",
                                                        className="input-loading",
                                                        type="circle",
                                                        color=THEME_COLOR,
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
                                                ]
                                            )
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value="graph-tab",
                                        tabIndex="13",
                                        children=[
                                            html.Div(
                                                className="tab-content-wrapper",
                                                children=[
                                                    html.Div(
                                                        [
                                                            html.P(id="graph-update-status"),
                                                            dcc.Graph(
                                                                id="output-graph",
                                                                responsive=True,
                                                                config={"displayModeBar": False},
                                                            ),
                                                        ],
                                                    ),
                                                ]
                                            )
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value="results-tab",
                                        tabIndex="14",
                                        children=[
                                            html.Div(
                                                className="tab-content-wrapper",
                                                children=[
                                                    html.Div(
                                                        className="tab-content-results",
                                                        children=[
                                                            html.Div(
                                                                [
                                                                    html.Div(id="dates-slider"),
                                                                    html.Div(id="dynamic-results-table"),
                                                                ]
                                                            )
                                                        ],
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
