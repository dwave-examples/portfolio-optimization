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

import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import ALL, MATCH, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from demo_configs import BASELINE
from demo_interface import generate_dates_slider, generate_table_group
from src.demo_enums import PeriodType, SolverType
from src.multi_period import MultiPeriod
from src.single_period import SinglePeriod
from src.utils import (
    deserialize,
    format_table_data,
    generate_input_graph,
    get_live_data,
    initialize_output_graph,
    serialize,
    update_output_graph,
)


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
    dates = [start_date, end_date] if start_date and end_date else ["2010-01-01", "2012-12-31"]
    stocks = stocks if stocks else ["AAPL", "MSFT", "AAL", "WMT"]
    df, stocks, df_baseline = get_live_data(dates, stocks, [BASELINE])

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
    prevent_initial_call="initial_duplicate",
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
    new_period = ctx.triggered_id["index"] if ctx.triggered_id else PeriodType.SINGLE.value

    nav_class_names[new_period] = "active"

    return (
        nav_class_names,
        new_period,
        "input-tab",
        True,
        new_period is PeriodType.MULTI.value,
        {"display": "none" if new_period is PeriodType.SINGLE.value else "block"},
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
        State("settings-store", "data"),
    ],
    prevent_initial_call=True,
)
def update_results_date_table(
    date_selected: str,
    date_results: dict,
    settings_store: dict,
) -> list:
    """Updates the results date table when the value of the results date selector is changed.

    Args:
        date_selected: The date that was just selected to trigger the callback.
        date_results: The store of period solution data with the date of the period as the key.
        settings_store: The settings that have been selected for this run.

    Returns:
        dynamic-results-table: The new table based on the data from the date that was selected.
    """
    solver_type = SolverType(settings_store["solver type"])

    date_keys_list = list(date_results.keys())
    date_values_list = list(date_results.values())

    date = datetime.strptime(date_keys_list[date_selected], "%Y-%m-%d").strftime("%B %Y")
    solution = date_values_list[date_selected]

    if date_selected > 0:
        prev_date = datetime.strptime(date_keys_list[date_selected - 1], "%Y-%m-%d").strftime(
            "%B %Y"
        )
        prev_solution = date_values_list[date_selected - 1]
        compare_stocks = [
            None if prev == curr else prev < curr
            for prev, curr in zip(prev_solution["stocks"].values(), solution["stocks"].values())
        ]

        solution_keys = ["return", "sales"] if solver_type is SolverType.CQM else ["return"]
        compare_solution = [prev_solution[key] < solution[key] for key in solution_keys]
        return [
            generate_table_group(
                tables_data=[
                    prev_solution["stocks"],
                    format_table_data(solver_type, prev_solution),
                ],
                title=prev_date,
            ),
            generate_table_group(
                tables_data=[solution["stocks"], format_table_data(solver_type, solution)],
                comparisons_data=[compare_stocks, compare_solution],
                title=date,
            ),
        ]

    return generate_table_group(
        tables_data=[solution["stocks"], format_table_data(solver_type, solution)], title=date
    )


@dash.callback(
    Output("loop-interval", "disabled", allow_duplicate=True),
    Output("cancel-button", "className", allow_duplicate=True),
    Output("run-button", "className", allow_duplicate=True),
    Output("iteration", "data", allow_duplicate=True),
    Output("results-tab", "label", allow_duplicate=True),
    Input("cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def cancel(cancel_button_click: int) -> tuple[bool, str, str, int, str]:
    """Resets the UI when the cancel button is clicked.

    Args:
        cancel_button_click: The number of times the cancel button has been clicked.

    Returns:
        loop-interval-disabled: Whether to disable the trigger that starts ``update_multi_output``.
        cancel-button-class: The class for the cancel button.
        run-button-class: The class for the run button.
        iteration: The number to reset the iteration store to.
        results-tab-label: The label of the results tab.
    """
    return True, "display-none", "", 3, "Results"


@dash.callback(
    Output("loop-running", "data"),
    inputs=[Input("loop-interval", "n_intervals"), State("loop-running", "data")],
    prevent_initial_call=True,
)
def start_loop_iteration(interval_trigger, is_loop_running) -> bool:
    """Triggers ``update_multi_output`` when Interval is triggered.

    Args:
        interval_trigger: The Dash Interval that triggers this function.
        is_loop_running: Whether ``update_multi_output`` is currently running.

    Returns:
        loop-running: Whether the loop is running.
    """
    if is_loop_running:
        raise PreventUpdate

    return True


class UpdateMultiOutputReturn(NamedTuple):
    """Return type for the ``update_multi_output`` callback function."""

    output_graph: go.Figure = dash.no_update
    iteration: int = dash.no_update
    dates_slider: list = dash.no_update
    solution_tables: list = dash.no_update
    results_date_dict: dict = dash.no_update
    portfolio: MultiPeriod = dash.no_update
    loop_store: dict = dash.no_update
    is_loop_running: bool = dash.no_update
    interval_disabled: bool = dash.no_update
    cancel_button_class: str = dash.no_update
    run_button_class: str = dash.no_update
    results_tab_disabled: bool = dash.no_update
    results_tab_label: str = dash.no_update
    graph_tab_disabled: bool = dash.no_update


@dash.callback(
    Output("output-graph", "figure"),
    Output("iteration", "data"),
    Output("dates-slider", "children"),
    Output("dynamic-results-table", "children", allow_duplicate=True),
    Output("results-date-dict", "data"),
    Output("portfolio", "data"),
    Output("loop-store", "data"),
    Output("loop-running", "data", allow_duplicate=True),
    Output("loop-interval", "disabled"),
    Output("cancel-button", "className", allow_duplicate=True),
    Output("run-button", "className", allow_duplicate=True),
    Output("results-tab", "disabled", allow_duplicate=True),
    Output("results-tab", "label", allow_duplicate=True),
    Output("graph-tab", "disabled", allow_duplicate=True),
    inputs=[
        Input("loop-running", "data"),
        State("max-iterations", "data"),
        State("iteration", "data"),
        State("settings-store", "data"),
        State("results-date-dict", "data"),
        State("portfolio", "data"),
        State("loop-store", "data"),
        State("output-graph", "figure"),
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True,
)
def update_multi_output(
    is_loop_running: bool,
    max_iterations: int,
    iteration: int,
    settings_store: dict,
    results_date_dict: dict,
    portfolio: MultiPeriod,
    loop_store: dict,
    fig: go.Figure,
) -> UpdateMultiOutputReturn:
    """Iteratively updates output graph.

    Args:
        is_loop_running: Whether the loop is running.
        max_iterations: The maximum times to call run_update().
        iteration: Which iteration of run_update is currently executing.
        settings_store: The settings that have been selected for this run.
        results_date_dict: The store of period solution data with the date of the period as the key.
        portfolio: The portfolio object.
        loop_store: A dictionary of variables to pass between iterations.
        fig: The graph to update.

    Returns:
        A NamedTuple ``UpdateMultiOutputReturn`` that contains the following:
            output_graph: The updated output graph.
            iteration: The next iteration count of the function.
            dates_slider: A slider of dates that updates the visible solution_table.
            solution_tables: The solution tables generated by generate_table_group.
            results_date_dict: A dictionary of date keys and solution values.
            portfolio: The MultiPeriod portfolio object.
            loop_store: A dictionary of variables to pass between iterations.
            is_loop_running: True if the loop is executing, False otherwise.
            interval_disabled: False if interval is running, True otherwise.
            cancel_button_class: The class for the cancel button.
            run_button_class: The class for the run button.
            results_tab_disabled: Whether the results tab should be disabled.
            results_tab_label: The label of the results tab.
            graph_tab_disabled: Whether the graph tab should be disabled.
    """
    solver_type = SolverType(settings_store["solver type"])
    stocks = settings_store["stocks"]
    budget = settings_store["budget"]

    if not is_loop_running or iteration > max_iterations:
        raise PreventUpdate

    if iteration == 3:  # First iteration
        my_portfolio = MultiPeriod(
            budget=budget,
            sampler_args="{}",
            gamma=100,
            model_type=solver_type,
            stocks=stocks,
            dates=settings_store["dates"],
            alpha=[0.005],
            baseline=BASELINE,
            t_cost=settings_store["transaction cost"] * 0.01,
            verbose=False,
        )

        my_portfolio.load_data()

        my_portfolio.baseline_values = [0]
        my_portfolio.update_values = [0]
        my_portfolio.opt_results_df = pd.DataFrame(
            columns=["Date", "Value"] + stocks + ["Variance", "Returns"]
        )
        my_portfolio.price_df = pd.DataFrame(columns=stocks)

        fig = initialize_output_graph(my_portfolio.df_baseline, budget)

        baseline_result, months, all_solutions, init_holdings = my_portfolio.initiate_run_update(
            i=iteration
        )

        fig = update_output_graph(
            fig,
            iteration,
            my_portfolio.update_values,
            my_portfolio.baseline_values,
            my_portfolio.df_all,
        )

        return UpdateMultiOutputReturn(
            output_graph=fig,
            iteration=iteration + 1,
            results_date_dict=all_solutions,
            portfolio=serialize(my_portfolio),
            loop_store={
                "baseline": baseline_result,
                "months": months,
                "budget": budget,
                "holdings": init_holdings,
            },
            is_loop_running=False,
            graph_tab_disabled=False,
        )

    portfolio = deserialize(portfolio)
    baseline_result, months, all_solutions, init_holdings = portfolio.initiate_run_update(
        i=iteration,
        first_purchase=False,
        baseline_result={key: np.array(value) for key, value in loop_store["baseline"].items()},
        months=loop_store["months"],
        initial_budget=loop_store["budget"],
        all_solutions=results_date_dict,
        init_holdings=loop_store["holdings"],
    )

    fig = update_output_graph(
        go.Figure(fig),
        iteration,
        portfolio.update_values,
        portfolio.baseline_values,
        portfolio.df_all,
    )

    if iteration == max_iterations:  # Last iteration
        dates = [
            datetime.strptime(date, "%Y-%m-%d").strftime("%b %Y")
            for date in results_date_dict.keys()
        ]
        solutions = list(results_date_dict.values())

        output_tables = generate_table_group(
            tables_data=[solutions[-1]["stocks"], format_table_data(solver_type, solutions[-1])],
            title=dates[-1],
        )

        dates_slider = generate_dates_slider(dates) if dates and len(dates) > 1 else []

        return UpdateMultiOutputReturn(
            output_graph=fig,
            iteration=3,
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

    loop_store.update({"baseline": baseline_result, "months": months, "holdings": init_holdings})

    # Regular iteration
    return UpdateMultiOutputReturn(
        output_graph=fig,
        iteration=iteration + 1,
        results_date_dict=all_solutions,
        portfolio=serialize(portfolio),
        loop_store=loop_store,
        is_loop_running=False,
    )


class RunOptimizationReturn(NamedTuple):
    """Return type for the ``run_optimization`` callback function."""

    cancel_button_class: str = ""
    run_button_class: str = "display-none"
    results_tab_disabled: bool = True
    results_tab_label: str = "Loading..."
    tabs_value: str = "input-tab"
    settings_store: dict = dash.no_update
    max_iterations: int = dash.no_update
    loop_interval_disabled: bool = dash.no_update
    graph_tab_disabled: bool = dash.no_update


@dash.callback(
    Output("cancel-button", "className"),
    Output("run-button", "className"),
    Output("results-tab", "disabled"),
    Output("results-tab", "label"),
    Output("tabs", "value", allow_duplicate=True),
    Output("settings-store", "data"),
    Output("max-iterations", "data", allow_duplicate=True),
    Output("loop-interval", "disabled", allow_duplicate=True),
    Output("graph-tab", "disabled", allow_duplicate=True),
    inputs=[
        Input("run-button", "n_clicks"),
        State("selected-period", "data"),
        State("sampler-type-select", "value"),
        State("budget", "value"),
        State("transaction-cost", "value"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
        State("stocks", "value"),
    ],
    prevent_initial_call=True,
)
def run_optimization(
    run_click: int,
    period: Union[PeriodType, int],
    solver_type: Union[SolverType, int],
    budget: int,
    transaction_cost: float,
    start_date: str,
    end_date: str,
    stocks: list,
) -> RunOptimizationReturn:
    """Updates UI and triggers optimization run.

    Args:
        run_click: The (total) number of times the run button has been clicked.
        period: The currently selected PeriodType either single-period or multi-period.
        solver_type: Which solver was selected.
        budget: The budget for the run.
        transaction_cost: The selected transaction cost.
        start_date: The selected start date.
        end_date: The selected end date.
        stocks: The list of selected stocks.

    Returns:
        A NamedTuple ``RunOptimizationReturn`` containing:
            cancel-button-class: The class for the cancel button.
            run-button-class: The class for the run button.
            results-tab-disabled: Whether the results tab should be disabled.
            results-tab-label: The label of the results tab.
            tabs-value: Which tab should be selected.
            settings-store: Storing all the settings for the run.
            max-iterations: The number of months between start and end date, which is the number of
                times to run ``update_multi_output`` (minus 3).
            loop-interval-disabled: Whether to disable the trigger that starts ``update_multi_output``.
            graph-tab-disabled: Whether to disable the graph tab.

    """
    settings_store = {
        "solver type": solver_type,
        "budget": budget,
        "transaction cost": transaction_cost,
        "dates": (
            [start_date, end_date] if start_date and end_date else ["2010-01-01", "2012-12-31"]
        ),
        "stocks": stocks,
    }

    if period is PeriodType.SINGLE.value:
        return RunOptimizationReturn(settings_store=settings_store)

    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    num_months = (
        (end_datetime.year - start_datetime.year) * 12 + end_datetime.month - start_datetime.month
    )

    return RunOptimizationReturn(
        settings_store=settings_store,
        max_iterations=num_months,
        loop_interval_disabled=False,
        graph_tab_disabled=True,
    )


@dash.callback(
    Output("dynamic-results-table", "children", allow_duplicate=True),
    Output("cancel-button", "className", allow_duplicate=True),
    Output("run-button", "className", allow_duplicate=True),
    Output("results-tab", "disabled", allow_duplicate=True),
    Output("results-tab", "label", allow_duplicate=True),
    background=True,
    inputs=[
        Input("settings-store", "data"),
        State("selected-period", "data"),
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True,
)
def run_optimization_single(
    settings_store: dict,
    period: Union[PeriodType, int],
) -> tuple[list, str, str, bool, str]:
    """Runs the single period optimization and updates UI accordingly.

    Args:
        settings_store: The settings that have been selected for this run.
        period: The currently selected PeriodType either single-period or multi-period.

    Returns:
        solution-table: The tables to display the solution.
        cancel-button-class: The class for the cancel button.
        run-button-class: The class for the run button.
        results-tab-disabled: Whether the results tab should be disabled.
        results-tab-label: The label of the results tab.
    """
    if period is PeriodType.MULTI.value:
        raise PreventUpdate

    solver_type = SolverType(settings_store["solver type"])

    my_portfolio = SinglePeriod(
        budget=settings_store["budget"],
        sampler_args="{}",
        model_type=solver_type,
        stocks=settings_store["stocks"],
        dates=settings_store["dates"],
        alpha=[0.005],
        t_cost=settings_store["transaction cost"] * 0.01,
    )
    solution = my_portfolio.run()

    table = format_table_data(solver_type, solution)

    output_tables = generate_table_group(tables_data=[solution["stocks"], table])

    return output_tables, "display-none", "", False, "Results"
