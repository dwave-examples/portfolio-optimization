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

from demo_interface import generate_dates_slider, generate_problem_details_table_rows, generate_table, generate_table_group
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
    """Takes the selected dates and stocks and updates the stocks graph.

    Args:
        start_date: The selected start date.
        end_date: The selected end date.
        stocks: The selected stocks.

    Returns:
        input-graph: The input stocks graph.
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
) -> tuple[str, int, str, bool, bool, dict]:
    """Updates the period that is selected (SINGLE or MULTI), hides/shows settings accordingly,
        and updates the navigation options to indicate the currently active period option.

    Args:
        period_options: A list containing the number of times each period option has been clicked.
        selected_period: The currently selected period.

    Returns:
        period-options-class (list): A list of classes for the header period navigation options.
        selected-period (int): Either SINGLE (``0`` or ``PeriodType.SINGLE``) or
            MULTI (``1`` or ``PeriodType.MULTI``).
        selected-tab (str): The tab to select.
        results-tab-disabled (bool): Whether the results tab should be disabled.
        graph-tab-disabled (bool): Whether the graph tab should be disabled.
        graph-tab-style (dict): The style settings for the graph tab.
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
        not is_single,
        {"display": "none" if is_single else "block"},
    )


@dash.callback(
    Output("transaction-cost-wrapper", "className"),
    inputs=[
        Input("sampler-type-select", "value"),
    ],
)
def update_settings(
    solver_type: Union[SolverType, int],
) -> str:
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
    date_selected: str, date_results: dict, solver_type: Union[SolverType, int],
) -> list:
    """Updates the results date table when the value of the results date selector is changed.

    Args:
        date_selected: The date that was just selected to trigger the callback.
        date_results: The store of period solution data with the date of the period as the key.
        solver_type: Either Quantum Hybrid (``0`` or ``SolverType.HYBRID``),
            or Classical (``1`` or ``SolverType.CLASSICAL``).

    Returns:
        dynamic-results-table: The new table based on the data from the date that was selected.
    """
    solver_type = SolverType(solver_type)

    date_keys_list = list(date_results.keys())

    date = datetime.strptime(date_keys_list[date_selected], '%Y-%m-%d').strftime("%B %Y")
    date_values_list = list(date_results.values())
    solution = date_values_list[date_selected]

    if date_selected > 0:
        prev_date = datetime.strptime(date_keys_list[date_selected-1], '%Y-%m-%d').strftime("%B %Y")
        prev_solution = date_values_list[date_selected-1]
        compare_stocks = [
            None if prev==curr else prev < curr
            for prev, curr in zip(prev_solution["stocks"].values(), solution["stocks"].values())
        ]

        solution_keys = ['return', 'sales'] if solver_type is SolverType.CQM else ['return']
        compare_solution = [prev_solution[key] < solution[key] for key in solution_keys]
        return [
            generate_table_group(
                tables_data=[prev_solution["stocks"], format_table_data(solver_type, prev_solution)],
                title=prev_date
            ),
            generate_table_group(
                tables_data=[solution["stocks"], format_table_data(solver_type, solution)],
                comparisons_data=[compare_stocks, compare_solution],
                title=date
            )
        ]

    return generate_table_group(
        tables_data=[solution["stocks"], format_table_data(solver_type, solution)], title=date
    )


@dash.callback(
    Output('loop-interval', 'disabled', allow_duplicate=True),
    Output("cancel-button", "className", allow_duplicate=True),
    Output("run-button", "className", allow_duplicate=True),
    Output("iteration", "data", allow_duplicate=True),
    Output("results-tab", "label", allow_duplicate=True),
    Input("cancel-button", "n_clicks"),
    prevent_initial_call = True,
)
def cancel(cancel_button_click: int) -> tuple[bool, str, str, int, str]:
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
    inputs=[
        Input('loop-interval', 'n_intervals'),
        State('loop-running', 'data')
    ],
    prevent_initial_call = True,
)
def start_loop_iteration(interval_trigger, is_loop_running) -> bool:
    """Triggers ``update_output`` when Interval is triggered.

    Args:
        interval_trigger: The Dash Interval that triggers this function.
        is_loop_running: Whether ``update_output`` is currently running.

    Returns:
        loop-running: Whether the loop is running.
    """
    if is_loop_running:
        raise PreventUpdate

    return True


class UpdateOutputReturn(NamedTuple):
    """Return type for the ``update_output`` callback function."""

    output_graph: go.Figure = dash.no_update
    iteration: int = dash.no_update
    problem_details_table: list = dash.no_update
    dates_slider: list = dash.no_update
    solution_tables: list = dash.no_update
    results_date_dict: dict = dash.no_update
    portfolio: MultiPeriod = dash.no_update
    baseline_results: dict = dash.no_update
    months: list = dash.no_update
    initial_budget: int = dash.no_update
    is_loop_running: bool = dash.no_update
    interval_disabled: bool = dash.no_update
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
    Output("dates-slider", "children"),
    Output("dynamic-results-table", "children", allow_duplicate=True),
    Output("results-date-dict", "data"),
    Output("portfolio", "data"),
    Output("baseline-results", "data"),
    Output("months", "data"),
    Output("initial-budget", "data"),
    Output('loop-running', 'data', allow_duplicate=True),
    Output('loop-interval', 'disabled'),
    Output("cancel-button", "className", allow_duplicate=True),
    Output("run-button", "className", allow_duplicate=True),
    Output("results-tab", "disabled", allow_duplicate=True),
    Output("results-tab", "label", allow_duplicate=True),
    Output("graph-tab", "disabled", allow_duplicate=True),
    Output("init-holdings", "data"),
    inputs=[
        Input('loop-running', 'data'),
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
    is_loop_running: bool,
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
        start_loop: Whether the loop should start.
        max_iterations: The maximum times to call run_update().
        iteration: Which iteration of run_update is currently executing.
        solver_type: Which solver was selected.
        time_limit: The time limit for the selected solver.
        budget: The budget for the run.
        transaction_cost: The selected transaction cost.
        start_date: The selected start date.
        end_date: The selected end date.
        stocks: The list of selected stocks.
        results_date_dict: The store of period solution data with the date of the period as the key.
        portfolio: The portfolio object.
        baseline_result: TODO
        months: TODO
        initial_budget: TODO
        fig: The graph to update.
        init_holdings: TODO

    Returns:
        A tuple UpdateOutputReturn that contains the following:
            output_graph: The updated output graph.
            iteration: The next iteration count of the function.
            problem_details_table: A table of details from the last run.
            dates_slider: A slider of dates that updates the visible solution_table.
            solution_tables: The solution tables generated by generate_table_group.
            results_date_dict: A dictionary of date keys and solution values.
            portfolio: The MultiPeriod portfolio object.
            baseline_results: TODO
            months: TODO
            initial_budget: TODO
            is_loop_running: True if the loop is executing, False otherwise.
            interval_disabled: False if interval is running, True otherwise.
            cancel_button_class: The class for the cancel button.
            run_button_class: The class for the run button.
            results_tab_disabled: Whether the results tab should be disabled.
            results_tab_label: The label of the results tab.
            graph_tab_disabled: Whether the graph tab should be disabled.
            init_holdings: TODO
    """
    solver_type = SolverType(solver_type)

    if not is_loop_running or iteration > max_iterations:
        raise PreventUpdate

    if iteration == 3:  # First iteration
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
        my_portfolio.opt_results_df=pd.DataFrame(columns=["Date", "Value"]+stocks+["Variance", "Returns"])
        my_portfolio.price_df = pd.DataFrame(columns=stocks)
        fig = initialize_output_graph(my_portfolio.df_baseline, budget)

        baseline_result, months, all_solutions, init_holdings = my_portfolio.initiate_run_update(
            i=iteration
        )

        fig = update_output_graph(
            fig, iteration, my_portfolio.update_values, my_portfolio.baseline_values, my_portfolio.df_all
        )

        return UpdateOutputReturn(
            output_graph=fig,
            iteration=iteration+1,
            results_date_dict=all_solutions,
            portfolio=serialize(my_portfolio),
            baseline_results=baseline_result,
            months=months,
            initial_budget=budget,
            is_loop_running=False,
            graph_tab_disabled=False,
            init_holdings=init_holdings,
        )

    portfolio = deserialize(portfolio)
    baseline_result, months, all_solutions, init_holdings = portfolio.initiate_run_update(
        i=iteration,
        first_purchase=False,
        baseline_result={key: np.array(value) for key, value in baseline_result.items()},
        months=months,
        initial_budget=initial_budget,
        all_solutions=results_date_dict,
        init_holdings=init_holdings
    )

    fig = update_output_graph(
        go.Figure(fig), iteration, portfolio.update_values, portfolio.baseline_values, portfolio.df_all
    )

    if iteration == max_iterations:  # Last iteration
        dates = [datetime.strptime(date, '%Y-%m-%d').strftime("%b %Y") for date in results_date_dict.keys()]
        solutions = list(results_date_dict.values())

        output_tables = generate_table_group(
            tables_data=[solutions[-1]["stocks"], format_table_data(solver_type, solutions[-1])],
            title=dates[-1]
        )

        dates_slider = generate_dates_slider(dates) if dates and len(dates) > 1 else []

        problem_details_table = generate_problem_details_table_rows(
            solver=solver_type.label, time_limit=time_limit,
        )

        return UpdateOutputReturn(
            output_graph=fig,
            iteration=3,
            problem_details_table=problem_details_table,
            dates_slider=dates_slider,
            solution_tables=output_tables,
            results_date_dict=all_solutions,
            is_loop_running=False,
            interval_disabled=True,
            cancel_button_class="display-none",
            run_button_class="",
            results_tab_disabled=False,
            results_tab_label="Results",
        )

    # Regular iteration
    return UpdateOutputReturn(
        output_graph=fig,
        iteration=iteration+1,
        results_date_dict=all_solutions,
        portfolio=serialize(portfolio),
        baseline_results=baseline_result,
        months=months,
        is_loop_running=False,
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
    Output("dynamic-results-table", "children", allow_duplicate=True),
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

    output_tables = [generate_table(solution["stocks"]), generate_table(table)]

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
