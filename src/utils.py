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

import base64
import math
import random
from typing import Any

import dill as pickle
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf

from src.demo_enums import SolverType


def get_live_data(dates, stocks, baseline, num=0) -> pd.DataFrame:
    """Gets live data from Yahoo Finance and generates a dataframe.

    Args:
        dates: The dates to get the stock data for.
        stocks: The stocks to get data for.
        baseline: The baseline stock for multi-period portfolio optimization run.
        num: The number of random stocks to get data for.

    Returns:
        pd.DataFrame: A dataframe containing the finance data.
    """
    print(f"\nLoading live data from the web from Yahoo Finance from {dates[0]} to {dates[1]}...")

    # Generating random list of stocks
    if num > 0:
        if dates[0] < "2010-01-01":
            raise Exception("Start date must be >= '2010-01-01' when using option 'num'.")

        symbols_df = pd.read_csv("data/stocks_symbols.csv")
        stocks = random.sample(list(symbols_df.loc[:, "Symbol"]), num)

    # Read in daily data; resample to monthly
    panel_data = yf.download(stocks, start=dates[0], end=dates[1])
    panel_data = panel_data.resample("BME").last()
    df_all = pd.DataFrame(index=panel_data.index, columns=stocks)

    if len(stocks) == 1:
        df_all[stocks[0]] = panel_data[["Close"]]
    else:
        for i in stocks:
            df_all[i] = panel_data[[("Close", i)]]

    nan_columns = df_all.columns[df_all.isna().any()].tolist()
    if nan_columns:
        print("The following tickers are dropped due to invalid data: ", nan_columns)
        df_all = df_all.dropna(axis=1)
        if len(df_all.columns) < 2:
            raise Exception("There must be at least 2 valid stock tickers.")
        stocks = list(df_all.columns)

    # Read in baseline data; resample to monthly
    index_df = yf.download(baseline, start=dates[0], end=dates[1])
    index_df = index_df.resample("BME").last()
    df_baseline = pd.DataFrame(index=index_df.index)

    for i in baseline:
        df_baseline[i] = index_df[[("Close")]]

    return df_all, stocks, df_baseline


def serialize(obj: Any) -> str:
    """Serialize the object using pickle"""
    return base64.b64encode(pickle.dumps(obj)).decode("utf-8")


def deserialize(obj: str) -> Any:
    """Deserialize the object"""
    return pickle.loads(base64.b64decode(obj.encode("utf-8")))


def generate_input_graph(df: pd.DataFrame = None) -> go.Figure:
    """Generates graph given df.

    Args:
        df (pd.DataFrame): A DataFrame containing the data to plot.

    Returns:
        go.Figure: A Plotly figure object.
    """
    fig = go.Figure()

    for col in list(df.columns.values):
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], mode="lines", name=col, hovertemplate="$%{y:.2f}")
        )

    fig.update_layout(
        title="Historical Stock Data",
        xaxis_title="Month",
        yaxis_title="Price",
        hovermode="x",
        xaxis_tickformat="%b %Y",
        xaxis_tickvals=df.index[::2],
    )

    return fig


def initialize_output_graph(
    df: pd.DataFrame,
    budget: int,
) -> go.Figure:
    """Creates a go.Figure given a dataframe.

    Args:
        df (pd.DataFrame): A DataFrame containing the data to plot.
        budget (int): The budget setting for the run.

    Returns:
        go.Figure: A Plotly figure object.
    """
    fig = go.Figure(
        go.Scatter(
            x=df.index,
            y=[0] * (df.shape[0]),
            mode="lines",
            line=dict(color="red"),
            name="Break-even",
            hoverinfo="none",
        )
    )

    fig.update_layout(
        title=f"{df.first_valid_index().date().strftime('%B %Y')} - {df.last_valid_index().date().strftime('%B %Y')}<br><sup><i>double click graph to rescale</i></sup>",
        xaxis_tickformat="%b %Y",
        xaxis_tickvals=df.index[::2],
        hovermode="x",
    )

    fig.update_yaxes(range=[-1.5 * budget, 1.5 * budget])

    return fig


def update_output_graph(
    fig: go.Figure,
    i: int,
    update_values: list,
    baseline_values: list,
    df: pd.DataFrame,
) -> go.Figure:
    """Updates a go.Figure with new values.

    Args:
        fig: The go.Figure to update.
        i: The index to add to the go.Figure.
        update_values: The new y values for the optimized portfolio.
        baseline_values: The new y values for the fund portfolio.
        df (pd.DataFrame): A DataFrame containing the data to plot.

    Returns:
        go.Figure: A Plotly figure object.
    """
    if i == 3:
        optimized_trace = go.Scatter(
            x=(df.index[3],),
            y=update_values,
            mode="lines",
            line=dict(color="blue"),
            name="Optimized portfolio",
            hovertemplate="$%{y:.2f}",
        )

        fund_trace = go.Scatter(
            x=(df.index[3],),
            y=baseline_values,
            mode="lines",
            line=dict(color="grey"),
            name="Fund portfolio",
            hovertemplate="$%{y:.2f}",
        )

        fig.add_trace(optimized_trace)
        fig.add_trace(fund_trace)
    else:
        fig.data[1].x += (str(df.index[i]),)
        fig.data[1].y = update_values

        fig.data[2].x += (str(df.index[i]),)
        fig.data[2].y = baseline_values

    return fig


def format_table_data(solver_type: SolverType, solution: dict) -> dict[str, str]:
    """Formats solution data into dict for easy table display.

    Args:
        solver_type: The solver type of the run.
        solution: The solution from the last run.

    Returns:
        table_data: dict of str keys and str values.
    """
    table_data = {"Estimated Returns": f"${solution['return']}"}
    if solver_type is SolverType.CQM:
        table_data.update({"Sales Revenue": f"${solution['sales']:.2f}"})

    table_data.update({"Purchase Cost": f"${solution['cost']:.2f}"})
    if solver_type is SolverType.CQM:
        table_data.update({"Transaction Cost": f"${solution['transaction cost']:.2f}"})
    table_data.update({"Standard Deviation": f"${math.sqrt(solution['risk']):.2f}"})

    return table_data
