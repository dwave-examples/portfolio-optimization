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

from datetime import datetime
from typing import NamedTuple, Union
import numpy as np

import dash
from dash import ALL, MATCH, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objs as go
from src.utils import deserialize, generate_input_graph, get_live_data, format_table_data, initialize_output_graph, serialize, update_output_graph

from demo_interface import generate_problem_details_table_rows, generate_solution_table, generate_table
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


@dash.callback(
    Output({"type": "period-option", "index": ALL}, "className"),
    Output("selected-period", "data"),
    Output("tabs", "value"),
    Output("results-tab", "disabled", allow_duplicate=True),
    Output("graph-tab", "disabled"),
    Output("graph-tab", "style"),
    inputs=[
        Input({"type": "period-option", "index": ALL}, "n_clicks"),
        State("selected-period", "data"),
    ],
    prevent_initial_call='initial_duplicate',
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
        False if is_single else True,
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
        State("sampler-type-select", "value"),
    ],
    prevent_initial_call=True,
)
def update_results_date_table(
    date_selected: str, date_dict: dict, solver_type: Union[SolverType, int],
) -> tuple:
    """Updates the results date table when the value of the results date selector is changed.

    Args:
        date_selected: The date that was just selected to trigger the callback.
        date_dict: The store of period solution data with the date of the period as the key.

    Returns:
        dynamic-results-table: The new table based on the data from the date that was selected.
    """
    solver_type = SolverType(solver_type)

    solution = list(date_dict.values())[date_selected]

    return [generate_table(solution["stocks"]), generate_table(format_table_data(solver_type, solution))]


@dash.callback(
    Output('loop-interval', 'disabled', allow_duplicate=True),
    Output("cancel-button", "className", allow_duplicate=True),
    Output("run-button", "className", allow_duplicate=True),
    Output("iteration", "data", allow_duplicate=True),
    Output("results-tab", "label", allow_duplicate=True),
    Input("cancel-button", "n_clicks"),
    prevent_initial_call = True,
)
def cancel(cancel_button_click: int) -> bool:
    """Resets the UI when the cancel button is clicked.

    Args:
        cancel_button_click: The number of times the cancel button has been clicked.

    Returns:
        loop-interval-disabled: Whether to disable the trigger that starts ``update_output``.
        cancel-button-class: The class for the cancel button.
        run-button-class: The class for the run button.
        iteration: The number to reset the iteration store to.
        results-tab-label: The label of the results tab.
    """
    return True, "display-none", "", 3, "Results"


@dash.callback(
    Output('loop-running', 'data'),
    Output('start-loop', 'data'),
    inputs=[
        Input('loop-interval', 'n_intervals'),
        State('loop-running', 'data')
    ],
    prevent_initial_call = True,
)
def start_loop_iteration(interval_trigger, loop_running):
    """Triggers ``update_output`` when Interval is triggered.

    Args:
        interval_trigger: The Dash Interval that triggers this function.
        loop_running: Whether ``update_output`` is currently running.
    Returns:
        loop-running: Whether the loop is running.
        start-loop: Whether to start the loop.
    """
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
    cancel_button_class: str = dash.no_update
    run_button_class: str = dash.no_update
    results_tab_disabled: bool = dash.no_update
    results_tab_label: str = dash.no_update
    graph_tab_disabled: bool = dash.no_update
    init_holdings: list = dash.no_update


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
    Output("cancel-button", "className", allow_duplicate=True),
    Output("run-button", "className", allow_duplicate=True),
    Output("results-tab", "disabled", allow_duplicate=True),
    Output("results-tab", "label", allow_duplicate=True),
    Output("graph-tab", "disabled", allow_duplicate=True),
    Output("init-holdings", "data"),
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
        State("init-holdings", "data"),
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call = True,
)
def update_output(
    start_loop: bool,
    max_iterations: int,
    iteration: int,
    solver_type: Union[SolverType, int],
    time_limit: float,
    budget: int,
    transaction_cost: float,
    start_date: str,
    end_date: str,
    stocks: list,
    results_date_dict: dict,
    portfolio: MultiPeriod,
    baseline_result: dict,
    months: list,
    initial_budget: int,
    fig: go.Figure,
    init_holdings: list,
) -> UpdateOutputReturn:
    """Iteratively updates output graph.

    Args:

    Returns:

    """
    solver_type = SolverType(solver_type)

    if not start_loop or iteration > max_iterations+1:
        raise PreventUpdate

    elif iteration == max_iterations+1:  # Last iteration
        output_tables = []
        dates = {i: datetime.strptime(date, '%Y-%m-%d').strftime("%b %Y") for i, date in enumerate(list(results_date_dict.keys()))}
        solutions = list(results_date_dict.values())

        output_tables = generate_solution_table(
            [solutions[-1]["stocks"], format_table_data(solver_type, solutions[-1])],
            dates
        )

        # Generates a list of table rows for the problem details table.
        if solver_type is SolverType.CQM:
            problem_details_table = generate_problem_details_table_rows(
                num_solutions=solutions[0]['number feasible'],
                energy=solutions[0]['best energy'],
                solver="CQM",
                time_limit=time_limit,
            )
        else:
            problem_details_table = generate_problem_details_table_rows(
                solver="DQM",
                time_limit=time_limit,
            )

        return UpdateOutputReturn(
            iteration=3,
            problem_details_table=problem_details_table,
            solution_table=output_tables,
            results_date_dict=results_date_dict,
            loop_running=False,
            start_loop=False,
            interval_disabled=True,
            cancel_button_class="display-none",
            run_button_class="",
            results_tab_disabled=False,
            results_tab_label="Results",
        )

    elif iteration == 3:  # First iteration
        my_portfolio = MultiPeriod(
            budget=budget,
            sampler_args="{}",
            gamma=100,
            model_type=solver_type,
            stocks=stocks,
            dates=(start_date, end_date) if start_date and end_date else ("2010-01-01", "2012-12-31"),
            alpha=[0.005],
            baseline="^GSPC",
            t_cost=transaction_cost*0.01,
            verbose=False
        )

        my_portfolio.load_data()

        my_portfolio.baseline_values=[0]
        my_portfolio.update_values=[0]
        my_portfolio.opt_results_df=pd.DataFrame(columns=["Date", "Value"] + stocks + ["Variance", "Returns"])
        my_portfolio.price_df = pd.DataFrame(columns=stocks)
        fig = initialize_output_graph(my_portfolio.df_baseline, budget)

        baseline_result, months, all_solutions, df, init_holdings = my_portfolio.run_interface(
            i=iteration
        )

        fig = update_output_graph(
            fig, iteration, my_portfolio.update_values, my_portfolio.baseline_values, df
        )

        return UpdateOutputReturn(
            output_graph=fig,
            iteration=iteration+1,
            results_date_dict=all_solutions,
            portfolio=serialize(my_portfolio),
            baseline_results=baseline_result,
            months=months,
            initial_budget=budget,
            loop_running=False,
            start_loop=False,
            graph_tab_disabled=False,
            init_holdings=init_holdings,
        )

    portfolio = deserialize(portfolio)
    baseline_result, months, all_solutions, df, init_holdings = portfolio.run_interface(
        i=iteration,
        first_purchase=False,
        baseline_result={key: np.array(value) for key, value in baseline_result.items()},
        months=months,
        initial_budget=initial_budget,
        all_solutions=results_date_dict,
        init_holdings=init_holdings
    )

    fig = update_output_graph(
        go.Figure(fig), iteration, portfolio.update_values, portfolio.baseline_values, df
    )
    return UpdateOutputReturn(
        output_graph=fig,
        iteration=iteration+1,
        results_date_dict=all_solutions,
        portfolio=serialize(portfolio),
        baseline_results=baseline_result,
        months=months,
        loop_running=False,
        start_loop=False,
        init_holdings=init_holdings,
    )


@dash.callback(
    Output("cancel-button", "className"),
    Output("run-button", "className"),
    Output("results-tab", "disabled"),
    Output("results-tab", "label"),
    Output("tabs", "value", allow_duplicate=True),
    Output("max-iterations", "data", allow_duplicate=True),
    Output('loop-interval', 'disabled', allow_duplicate=True),
    Output("graph-tab", "disabled", allow_duplicate=True),
    inputs=[
        Input("run-button", "n_clicks"),
        State("selected-period", "data"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
    ],
    prevent_initial_call=True,
)
def run_optimization(
    run_click: int,
    period: Union[PeriodType, int],
    start_date: str,
    end_date: str,
) -> tuple[int, bool]:
    """Updates UI to prepare for optimization run.

    Args:
        run_click: The (total) number of times the run button has been clicked.
        period: The currently selected PeriodType either single-period or multi-period.
        start_date: The start date to query stock data from Yahoo Finance.
        end_date: The end date to query stock data from Yahoo Finance.

    Returns:
        A tuple containing all outputs to be used when updating the HTML
        template (in ``dash_html.py``). These are:
            cancel-button-class: The class for the cancel button.
            run-button-class: The class for the run button.
            results-tab-label: The label of the results tab.
            tabs-value: Which tab should be selected.
            max-iterations: The number of months between start and end date, which is the number of
                times to run ``update_output`` (minus 3).
            loop-interval-disabled: Whether to disable the trigger that starts ``update_output``.
            graph-tab-disabled: Whether to disable the graph tab.

    """
    return_vals = ("", "display-none", True, "Loading...", "input-tab")

    if period is PeriodType.SINGLE.value:
        return  *return_vals, dash.no_update, dash.no_update, dash.no_update

    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
    num_months = (end_datetime.year - start_datetime.year) * 12 + end_datetime.month - start_datetime.month

    return *return_vals, num_months, False, True


@dash.callback(
    Output("problem-details", "children", allow_duplicate=True),
    Output("solution-table", "children", allow_duplicate=True),
    Output("cancel-button", "className", allow_duplicate=True),
    Output("run-button", "className", allow_duplicate=True),
    Output("results-tab", "disabled", allow_duplicate=True),
    Output("results-tab", "label", allow_duplicate=True),
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
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True,
)
def run_optimization_single(
    run_click: int,
    solver_type: Union[SolverType, int],
    time_limit: float,
    budget: int,
    transaction_cost: float,
    period: Union[PeriodType, int],
    start_date: str,
    end_date: str,
    stocks: list,
) -> tuple[list, list]:
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
        budget: The budget for the problem run.
        transaction_cost: The cost in percentage for each transaction
        period: The currently selected PeriodType either single-period or multi-period.
        start_date: The start date to query stock data from Yahoo Finance.
        end_date: The end date to query stock data from Yahoo Finance.
        stocks: A list of the selected stock symbols.

    Returns:
        A tuple containing all outputs to be used when updating the HTML
        template (in ``dash_html.py``). These are:
            problem-details: TODO
            solution-table: TODO
            cancel-button-class: The class for the cancel button.
            run-button-class: The class for the run button.
            results-tab-disabled: Whether the results tab should be disabled.
            results-tab-label: The label of the results tab.

    """
    if period is PeriodType.MULTI.value:
        raise PreventUpdate

    solver_type = SolverType(solver_type)

    my_portfolio = SinglePeriod(
        budget=budget,
        sampler_args="{}",
        model_type=solver_type,
        stocks=stocks,
        dates=[start_date, end_date],
        alpha=[0.005],
        t_cost=transaction_cost*0.01,
    )
    solution = my_portfolio.run(min_return=0, max_risk=0, num=0)

    table = format_table_data(solver_type, solution)

    output_tables = generate_solution_table([solution["stocks"], table])

    # Generates a list of table rows for the problem details table.
    if solver_type is SolverType.CQM:
        problem_details_table = generate_problem_details_table_rows(
            num_solutions=solution['number feasible'],
            energy=solution['best energy'],
            solver="CQM",
            time_limit=time_limit,
        )
    else:
        problem_details_table = generate_problem_details_table_rows(
            solver="DQM",
            time_limit=time_limit,
        )

    return problem_details_table, output_tables, "display-none", "", False, "Results"
