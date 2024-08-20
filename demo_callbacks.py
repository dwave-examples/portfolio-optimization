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

from __future__ import annotations

from datetime import date, datetime
from typing import NamedTuple, Union

import dash
from dash import ALL, MATCH, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objs as go
from src.utils import get_live_data

from demo_interface import generate_problem_details_table_rows, generate_solution_table, generate_table
from src.multi_period import MultiPeriod
from src.single_period import SinglePeriod
from src.demo_enums import PeriodType, SolverType
import numpy as np


@dash.callback(
    Output({"type": "to-collapse-class", "index": MATCH}, "className"),
    inputs=[
        Input({"type": "collapse-trigger", "index": MATCH}, "n_clicks"),
        State({"type": "to-collapse-class", "index": MATCH}, "className"),
    ],
    prevent_initial_call=True,
)
def toggle_left_column(collapse_trigger: int, to_collapse_class: str) -> str:
    """Toggles a 'collapsed' class that hides and shows some aspect of the UI.

    Args:
        collapse_trigger (int): The (total) number of times a collapse button has been clicked.
        to_collapse_class (str): Current class name of the thing to collapse, 'collapsed' if not
            visible, empty string if visible.

    Returns:
        str: The new class name of the thing to collapse.
    """

    classes = to_collapse_class.split(" ") if to_collapse_class else []
    if "collapsed" in classes:
        classes.remove("collapsed")
        return " ".join(classes)
    return to_collapse_class + " collapsed" if to_collapse_class else "collapsed"


def generate_graph(
    df: pd.DataFrame = None
) -> go.Figure:
    """Generates graph given df.

    Args:
        df (pd.DataFrame): A DataFrame containing the data to plot.

    Returns:
        go.Figure: A Plotly figure object.
    """
    fig = go.Figure()

    for col in list(df.columns.values):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))

    fig.update_layout(
        title="Historical Stock Data", xaxis_title="Month", yaxis_title="Price"
    )

    return fig


@dash.callback(
    Output("input-graph", "figure"),
    inputs=[
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("stocks", "value"),
    ],
)
def render_initial_state(
    start_date: str,
    end_date: str,
    stocks: list,
) -> go.Figure:
    """Runs on load and any time the value of the slider is updated.
        Add `prevent_initial_call=True` to skip on load runs.

    Args:
        slider_value: The value of the slider.

    Returns:
        str: The content of the input tab.
    """
    num = 0
    dates = [start_date, end_date] if start_date and end_date else ['2010-01-01', '2012-12-31']
    stocks = stocks if stocks else ['AAPL', 'MSFT', 'AAL', 'WMT']
    baseline = ['^GSPC']
    df, stocks, df_baseline = get_live_data(num, dates, stocks, baseline)

    return generate_graph(df)


@dash.callback(
    Output("output-graph", "figure"),
    Output("iteration", "data"),
    inputs=[
        Input("iteration", "data"),
        State("max-iterations", "data"),
    ],
    prevent_initial_call = True,
)
def update_output(
    iteration: int,
    max_iterations: int
) -> go.Figure:
    """Iteratively updates output graph.

    Args:

    Returns:

    """
    if iteration > max_iterations:
        raise PreventUpdate

    return {}, iteration+1


@dash.callback(
    Output({"type": "period-option", "index": ALL}, "className"),
    Output("selected-period", "data"),
    Output("tabs", "value"),
    Output("results-tab", "disabled"),
    inputs=[
        Input({"type": "period-option", "index": ALL}, "n_clicks"),
        State("selected-period", "data"),
    ],
)
def update_selected_period(
    period_options: list[int],
    selected_period: Union[PeriodType, int],
) -> tuple:
    """Updates the period that is selected (SINGLE or MULTI), hides/shows settings accordingly,
        and updates the navigation options to indicate the currently active period option.

    Args:
        period_options: A list containing the number of times each period option has been clicked.
        selected_period: The currently selected period.

    Returns:
        A tuple containing all outputs to be used when updating the HTML
        template (in ``dash_html.py``). These are:
            period_options_class (list): A list of classes for the period navigation options in the header.
            selected_period (int): Either SINGLE (``0`` or ``PeriodType.SINGLE``) or
                MULTI (``1`` or ``PeriodType.MULTI``).
            selected_tab (str): The tab to select.
            results_tab_disabled (bool): Whether the results tab should be disabled.
    """
    if ctx.triggered_id and selected_period == ctx.triggered_id["index"]:
        raise PreventUpdate

    nav_class_names = [""] * len(period_options)
    is_single = not ctx.triggered_id or ctx.triggered_id["index"] is PeriodType.SINGLE.value

    nav_class_names[PeriodType.SINGLE.value if is_single else PeriodType.MULTI.value] = "active"

    return (
        nav_class_names,
        PeriodType.SINGLE.value if is_single else PeriodType.MULTI.value,
        "input-tab",
        True,
    )


@dash.callback(
    Output("transaction-cost-wrapper", "className"),
    inputs=[
        Input("sampler-type-select", "value"),
    ],
)
def update_settings(
    solver_type: Union[SolverType, int],
) -> tuple:
    """Hides the transaction cost when the DQM is selected and shows otherwise.

    Args:
        solver_type: Either Quantum Hybrid (``0`` or ``SolverType.HYBRID``),
            or Classical (``1`` or ``SolverType.CLASSICAL``).

    Returns:
        transaction-cost-wrapper-classname: The class name to hide or show the
            transaction cost selector.
    """
    return "display-none" if solver_type is SolverType.DQM.value else ""



@dash.callback(
    Output("dynamic-results-table", "children"),
    inputs=[
        Input("results-date-selector", "value"),
        State("results-date-dict", "data"),
    ],
    prevent_initial_call=True,
)
def update_results_date_table(
    date_selected: str, date_dict: dict
) -> tuple:
    """Updates the results date table when the value of the results date selector is changed.

    Args:
        date_selected: The date that was just selected to trigger the callback.
        date_dict: The store of period solution data with the date of the period as the key.

    Returns:
        dynamic-results-table: The new table based on the data from the date that was selected.
    """
    solution = date_dict[date_selected]
    table = {
        "Estimated Returns": f"${solution['return']}",
        "Sales Revenue": solution['sales'],
        "Variance": f"${solution['risk']:.2f}",
    }

    return generate_table(table)


class RunOptimizationReturn(NamedTuple):
    """Return type for the ``run_optimization`` callback function."""

    results: str = dash.no_update
    problem_details_table: list = dash.no_update
    # Add more return variables here. Return values for callback functions
    # with many variables should be returned as a NamedTuple for clarity.


@dash.callback(
    # The Outputs below must align with `RunOptimizationReturn`.
    # Output("results", "children"),
    Output("problem-details", "children"),
    Output("solution-table", "children"),
    Output("max-iterations", "data"),
    Output("results-date-dict", "data"),
    background=True,
    inputs=[
        Input("run-button", "n_clicks"),
        State("sampler-type-select", "value"),
        State("solver-time-limit", "value"),
        State("budget", "value"),
        State("transaction-cost", "value"),
        # State("num-stocks", "value"),
        State("selected-period", "data"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
        State("stocks", "value"),
    ],
    running=[
        # Shows cancel button while running.
        (Output("cancel-button", "className"), "", "display-none"),
        (Output("run-button", "className"), "display-none", ""),  # Hides run button while running.
        (Output("results-tab", "disabled"), True, False),  # Disables results tab while running.
        (Output("results-tab", "label"), "Loading...", "Results"),
        (Output("tabs", "value"), "input-tab", "input-tab"),  # Switch to input tab while running.
        (Output("run-in-progress", "data"), True, False),  # Can block certain callbacks.
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True,
)
def run_optimization(
    # The parameters below must match the `Input` and `State` variables found
    # in the `inputs` list above.
    run_click: int,
    solver_type: Union[SolverType, int],
    time_limit: float,
    budget: int,
    transaction_cost: list,
    # num_stocks: int,
    period_value: Union[PeriodType, int],
    start_date: str,
    end_date: str,
    stocks: list,
) -> RunOptimizationReturn:
    """Runs the optimization and updates UI accordingly.

    This is the main function which is called when the `Run Optimization` button is clicked.
    This function takes in all form values and runs the optimization, updates the run/cancel
    buttons, deactivates (and reactivates) the results tab, and updates all relevant HTML
    components.

    Args:
        run_click: The (total) number of times the run button has been clicked.
        solver_type: Either Quantum Hybrid (``0`` or ``SolverType.HYBRID``),
            or Classical (``1`` or ``SolverType.CLASSICAL``).
        time_limit: The solver time limit.
        slider_value: The value of the slider.
        dropdown_value: The value of the dropdown.
        checklist_value: A list of the values of the checklist.
        radio_value: The value of the radio.

    Returns:
        A NamedTuple (RunOptimizationReturn) containing all outputs to be used when updating the HTML
        template (in ``dash_html.py``). These are:

            results: The results to display in the results tab.
            problem-details: List of the table rows for the problem details table.
    """

    # Only run optimization code if this function was triggered by a click on `run-button`.
    # Setting `Input` as exclusively `run-button` and setting `prevent_initial_call=True`
    # also accomplishes this.
    if run_click == 0 or ctx.triggered_id != "run-button":
        raise PreventUpdate

    solver_type = SolverType(solver_type)
    period_type = PeriodType(period_value)

    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
    num_months = (end_datetime.year - start_datetime.year) * 12 + end_datetime.month - start_datetime.month

    if period_type is PeriodType.SINGLE:
        print(f"\nSingle period portfolio optimization run...")

        my_portfolio = SinglePeriod(
            budget=budget,
            sampler_args="{}",
            model_type=solver_type,
            stocks=stocks,
            dates=[start_date, end_date],
            alpha=[0.005],
            t_cost=transaction_cost,
        )
        solution = my_portfolio.run(min_return=0, max_risk=0, num=0)

        output_tables = [
            generate_solution_table(solution["stocks"]),
            generate_solution_table(
                {
                    "Estimated Returns": f"${solution['return']}",
                    "Sales Revenue": solution['sales'],
                    "Purchase Cost": f"${solution['cost']:.2f}",
                    "Transaction Cost": f"${solution['transaction cost']:.2f}",
                    "Variance": f"${solution['risk']:.2f}",
                }
            )
        ]
    else:
        print(f"\nRebalancing portfolio optimization run...")

        my_portfolio = MultiPeriod(
            budget=budget,
            sampler_args="{}",
            gamma=True,
            model_type=solver_type,
            stocks=stocks,
            dates=[start_date, end_date],
            alpha=[0.005],
            baseline="^GSPC",
            t_cost=transaction_cost,
        )

        all_solutions = my_portfolio.run(min_return=0, max_risk=0, num=0)
        output_tables = []
        is_first = True
        dates = [{"label": date.strftime("%B %Y"), "value": date} for date in all_solutions.keys()]
        for date, solution in all_solutions.items():
            if is_first:
                output_tables.append(
                    generate_solution_table(solution["stocks"]),
                )
                table = {
                    "Estimated Returns": f"${solution['return']}",
                    "Sales Revenue": solution['sales'],
                    "Purchase Cost": f"${solution['cost']:.2f}",
                    "Transaction Cost": f"${solution['transaction cost']:.2f}",
                    "Variance": f"${solution['risk']:.2f}",
                }

                output_tables.append(
                    generate_solution_table(table)
                )
            else:
                table = {
                    "Estimated Returns": f"${solution['return']}",
                    "Sales Revenue": solution['sales'],
                    "Variance": f"${solution['risk']:.2f}",
                }

                output_tables.append(
                    generate_solution_table(table, dates)
                )

            if not is_first: break
            is_first = False

    # Generates a list of table rows for the problem details table.
    problem_details_table = generate_problem_details_table_rows(
        num_solutions=solution['number feasible'],
        energy=solution['best energy'],
        solver="CQM" if solver_type is SolverType.CQM else "DQM",
        time_limit=time_limit,
    )

    return problem_details_table, output_tables, num_months, all_solutions if period_type is PeriodType.MULTI else dash.no_update
