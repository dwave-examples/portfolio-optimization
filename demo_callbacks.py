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

import base64
from datetime import datetime
import dill as pickle
from typing import NamedTuple, Union
import numpy as np

import dash
from dash import ALL, MATCH, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objs as go
from src.utils import get_live_data

from demo_interface import generate_problem_details_table_rows, generate_solution_table, generate_table, update_iteration
from src.multi_period import MultiPeriod
from src.single_period import SinglePeriod
from src.demo_enums import PeriodType, SolverType


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


def generate_input_graph(
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

def initialize_output_graph(
    num_months: int,
    df_baseline: pd.DataFrame,
    budget: int,
) -> go.Figure:
    fig = go.Figure()

    # Create a trace for the "Break-even" line
    break_even_trace = go.Scatter(
        x=list(range(0, num_months)),
        y=[0] * num_months,
        mode='lines',
        line=dict(color='red'),
        name='Break-even'
    )

    # Create the figure and add the trace
    fig = go.Figure(data=[break_even_trace])

    # Update y-axis limits (equivalent to plt.ylim)
    fig.update_yaxes(range=[-1.5 * budget, 1.5 * budget])

    # Update x-axis ticks (equivalent to plt.xticks)
    fig.update_xaxes(
        tickvals=list(range(0, num_months, 2)),  # Positions of the ticks
        ticktext=df_baseline.index.strftime("%b")[::2],  # Labels for the ticks
        tickangle=-90  # Equivalent to 'rotation="vertical"' in matplotlib
    )

    # Optional: Set number of bins (ticks) on the x-axis
    fig.update_xaxes(nticks=int(num_months / 2))

    return fig


def update_output_graph(
    # df: pd.DataFrame,
    fig: go.Figure,
    i: int,
    update_values: list,
    baseline_values: list,
    df_all: pd.DataFrame,
) -> go.Figure:
    """Generates graph given df.

    Args:
        df (pd.DataFrame): A DataFrame containing the data to plot.

    Returns:
        go.Figure: A Plotly figure object.
    """
    update_values = np.array(update_values, dtype=object)
    baseline_values = np.array(baseline_values, dtype=object)

    optimized_trace = go.Scatter(
        x=list(range(3, i + 1)),
        y=update_values,
        mode='lines',
        line=dict(color='blue'),
        name='Optimized portfolio',
        showlegend=i == 4,
    )

    fund_trace = go.Scatter(
        x=list(range(3, i + 1)),
        y=baseline_values,
        mode='lines',
        line=dict(color='grey'),
        name='Fund portfolio',
        showlegend=i == 4,
    )

    fig.add_trace(optimized_trace)
    fig.add_trace(fund_trace)

    if i == 4:
        fig.update_layout(
            title=f"Start: {df_all.first_valid_index().date()}, End: {df_all.last_valid_index().date()}",
            legend=dict(
                x=0,
                y=0,
                xanchor='left',
                yanchor='bottom',
            )
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

    return generate_input_graph(df)

# Serializing the object using pickle
def serialize(obj):
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

# Deserializing the object
def deserialize(serialized_obj):
    return pickle.loads(base64.b64decode(serialized_obj.encode('utf-8')))


@dash.callback(
    # Output('interval-component', 'disabled'),
    Output('loop-running', 'data'),
    Output('start-loop', 'data'),
    inputs=[
        Input('loop-interval', 'n_intervals'),
        State('loop-running', 'data')
    ],
    prevent_initial_call = True,
)
def start_loop(iteration, loop_running):
    if loop_running:
        raise PreventUpdate

    return True, True


class UpdateOutputReturn(NamedTuple):
    """Return type for the ``update_output`` callback function."""

    output_graph: go.Figure = dash.no_update
    iteration: int = dash.no_update
    problem_details_table: list = dash.no_update
    solution_table: list = dash.no_update
    results_date_dict: dict = dash.no_update
    portfolio: MultiPeriod = dash.no_update
    baseline_results: dict = dash.no_update
    months: list = dash.no_update
    initial_budget: int = dash.no_update
    loop_running: bool = dash.no_update
    start_loop: bool = dash.no_update
    interval_disabled: bool = dash.no_update
    iteration: int = dash.no_update


@dash.callback(
    Output("output-graph", "figure"),
    Output("iteration", "data"),
    Output("problem-details", "children"),
    Output("solution-table", "children"),
    Output("results-date-dict", "data"),
    Output("portfolio", "data"),
    Output("baseline-results", "data"),
    Output("months", "data"),
    Output("initial-budget", "data"),
    Output('loop-running', 'data', allow_duplicate=True),
    Output('start-loop', 'data', allow_duplicate=True),
    Output('loop-interval', 'disabled'),
    inputs=[
        Input('start-loop', 'data'),
        State("max-iterations", "data"),
        State('iteration', 'data'),
        State("sampler-type-select", "value"),
        State("solver-time-limit", "value"),
        State("budget", "value"),
        State("transaction-cost", "value"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
        State("stocks", "value"),
        State("results-date-dict", "data"),
        State("portfolio", "data"),
        State("baseline-results", "data"),
        State("months", "data"),
        State("initial-budget", "data"),
        State("output-graph", "figure"),
    ],
    prevent_initial_call = True,
)
def update_output(
    start_loop: bool,
    max_iterations: int,
    iteration: int,
    solver_type: Union[SolverType, int],
    time_limit: float,
    budget: int,
    transaction_cost: list,
    start_date: str,
    end_date: str,
    stocks: list,
    results_date_dict: dict,
    portfolio: MultiPeriod,
    baseline_result: dict,
    months: list,
    initial_budget: int,
    fig: go.Figure,
) -> UpdateOutputReturn:
    """Iteratively updates output graph.

    Args:

    Returns:

    """
    solver_type = SolverType(solver_type)

    if not start_loop or iteration > max_iterations+1:
        raise PreventUpdate

    elif iteration == max_iterations+1:
        output_tables = []
        is_first = True
        dates = [{"label": datetime.strptime(date, '%Y-%m-%d').strftime("%B %Y"), "value": date} for date in results_date_dict.keys()]
        for solution in results_date_dict.values():
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

        return UpdateOutputReturn(
            iteration=iteration+1,
            problem_details_table=problem_details_table,
            solution_table=output_tables,
            results_date_dict=results_date_dict,
            loop_running=False,
            start_loop=False,
            interval_disabled=True,
        )

    elif iteration == 3:
        my_portfolio = MultiPeriod(
            budget=budget,
            sampler_args="{}",
            gamma=True,
            model_type=solver_type,
            stocks=stocks,
            dates=[start_date, end_date] if start_date and end_date else ["2010-01-01", "2012-12-31"],
            alpha=[0.005],
            baseline="^GSPC",
            t_cost=transaction_cost,
        )

        my_portfolio.load_data()

        my_portfolio.baseline_values=[0]
        my_portfolio.update_values=[0]
        my_portfolio.opt_results_df=pd.DataFrame(columns=["Date", "Value"] + stocks + ["Variance", "Returns"])
        my_portfolio.price_df = pd.DataFrame(columns=stocks)

        baseline_result, months, all_solutions, initial_budget, df_baseline, df_all = my_portfolio.run_interface(i=iteration)

        fig = initialize_output_graph(max_iterations, df_baseline, budget)

        return UpdateOutputReturn(
            output_graph=fig,
            iteration=iteration+1,
            results_date_dict=all_solutions,
            portfolio=serialize(my_portfolio),
            baseline_results=baseline_result,
            months=months,
            initial_budget=initial_budget,
            loop_running=False,
            start_loop=False,
        )

    portfolio = deserialize(portfolio)
    baseline_result, months, all_solutions, initial_budget, df_baseline, df_all = portfolio.run_interface(
        i=iteration,
        first_purchase=False,
        baseline_result={key: np.array(value) for key, value in baseline_result.items()},
        months=months,
        initial_budget=initial_budget,
        all_solutions=results_date_dict,
    )

    fig = update_output_graph(go.Figure(fig), iteration, portfolio.update_values, portfolio.baseline_values, df_all)
    return UpdateOutputReturn(
        output_graph=fig,
        iteration=iteration+1,
        results_date_dict=all_solutions,
        portfolio=serialize(portfolio),
        baseline_results=baseline_result,
        months=months,
        loop_running=False,
        start_loop=False,
    )


@dash.callback(
    Output({"type": "period-option", "index": ALL}, "className"),
    Output("selected-period", "data"),
    Output("tabs", "value"),
    Output("results-tab", "disabled"),
    Output("graph-tab", "style"),
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
        {"display": "none"} if is_single else {"display": "block"},
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

    problem_details_table: list = dash.no_update
    solution_table: list = dash.no_update
    max_iterations: int = dash.no_update
    interval_disabled: bool = dash.no_update


@dash.callback(
    Output("problem-details", "children", allow_duplicate=True),
    Output("solution-table", "children", allow_duplicate=True),
    Output("max-iterations", "data", allow_duplicate=True),
    Output('loop-interval', 'disabled', allow_duplicate=True),
    background=True,
    inputs=[
        Input("run-button", "n_clicks"),
        State("sampler-type-select", "value"),
        State("solver-time-limit", "value"),
        State("budget", "value"),
        State("transaction-cost", "value"),
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

        # Generates a list of table rows for the problem details table.
        problem_details_table = generate_problem_details_table_rows(
            num_solutions=solution['number feasible'],
            energy=solution['best energy'],
            solver="CQM" if solver_type is SolverType.CQM else "DQM",
            time_limit=time_limit,
        )

        return RunOptimizationReturn(
            problem_details_table=problem_details_table,
            solution_table=output_tables,
        )

    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
    num_months = (end_datetime.year - start_datetime.year) * 12 + end_datetime.month - start_datetime.month

    print(f"\nRebalancing portfolio optimization run...")

    return RunOptimizationReturn(
        max_iterations=num_months,
        interval_disabled=False,
    )
