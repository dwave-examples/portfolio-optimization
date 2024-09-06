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
import random

from src.demo_enums import SolverType
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import dill as pickle

import yfinance as yf


def get_live_data(num, dates, stocks, baseline) -> pd.DataFrame:
    print(
        f"\nLoading live data from the web from Yahoo Finance",
        f"from {dates[0]} to {dates[1]}...",
    )

    # Generating randomn list of stocks
    if num > 0:
        if dates[0] < "2010-01-01":
            raise Exception(
                f"Start date must be >= '2010-01-01' " f"when using option 'num'."
            )
        symbols_df = pd.read_csv("data/stocks_symbols.csv")
        stocks = random.sample(list(symbols_df.loc[:, "Symbol"]), num)

    # Read in daily data; resample to monthly
    panel_data = yf.download(stocks, start=dates[0], end=dates[1])
    panel_data = panel_data.resample("BM").last()
    df_all = pd.DataFrame(index=panel_data.index, columns=stocks)

    for i in stocks:
        df_all[i] = panel_data[[("Adj Close", i)]]

    nan_columns = df_all.columns[df_all.isna().any()].tolist()
    if nan_columns:
        print("The following tickers are dropped due to invalid data: ", nan_columns)
        df_all = df_all.dropna(axis=1)
        if len(df_all.columns) < 2:
            raise Exception(f"There must be at least 2 valid stock tickers.")
        stocks = list(df_all.columns)

    # Read in baseline data; resample to monthly
    index_df = yf.download(baseline, start=dates[0], end=dates[1])
    index_df = index_df.resample("BM").last()
    df_baseline = pd.DataFrame(index=index_df.index)

    for i in baseline:
        df_baseline[i] = index_df[[("Adj Close")]]

    return df_all, stocks, df_baseline

# Serializing the object using pickle
def serialize(obj):
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

# Deserializing the object
def deserialize(serialized_obj):
    return pickle.loads(base64.b64decode(serialized_obj.encode('utf-8')))

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

    break_even_trace = go.Scatter(
        x=list(range(0, num_months)),
        y=[0] * num_months,
        mode='lines',
        line=dict(color='red'),
        name='Break-even'
    )

    fig = go.Figure(data=[break_even_trace])

    fig.update_yaxes(range=[-1.5 * budget, 1.5 * budget])

    fig.update_xaxes(
        tickvals=list(range(0, num_months, 2)),
        ticktext=df_baseline.index.strftime("%b")[::2],
        tickangle=-90
    )

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

def format_table_data(solver_type: SolverType, solution: dict) -> dict[str, str]:
    table = {"Estimated Returns": f"${solution['return']}"}
    if solver_type is SolverType.CQM:
        table.update({"Sales Revenue": f"${solution['sales']:.2f}"})

    table.update({"Purchase Cost": f"${solution['cost']:.2f}"})
    if solver_type is SolverType.CQM:
        table.update({"Transaction Cost": f"${solution['transaction cost']:.2f}"})
    table.update({"Variance": f"${solution['risk']:.2f}"})

    return table
