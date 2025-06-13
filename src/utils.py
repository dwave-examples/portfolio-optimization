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
import os
import random
from typing import Any

from demo_configs import BASELINE
import dill as pickle
import pandas as pd
import plotly.graph_objs as go

from src.demo_enums import SolverType

PROJECT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean_stock_data(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Modifies stock dataframe created from downloaded data.

    Args:
        df: The dataframe to clean.
        col_name: The name of the stock column.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    # Convert Date column to datetime to get 1 per month
    df['Date'] = pd.to_datetime(df['Date'])
    monthly_stocks = df.groupby(df['Date'].dt.to_period('M'), as_index=False).first()

    # Rename Close column to stock name and remove unnecessary columns
    monthly_stocks.rename({"Close/Last": col_name}, axis=1, inplace=True)
    monthly_stocks.drop(["Open", "High", "Low"], axis=1, inplace=True)

    monthly_stocks = monthly_stocks.set_index("Date")

    if monthly_stocks[col_name].dtype == 'object':
        monthly_stocks[col_name] = monthly_stocks[col_name].str.replace('$', '').astype('float')

    return monthly_stocks


def get_stock_data() -> tuple[pd.DataFrame, list[str]]:
    """Reads stock CSVs and returns stock dataframe.

    Returns:
        pd.DataFrame: A dataframe containing all stock data in historical_data directory.
        list[str]: A list of all the stock names in the dataframe.
    """
    print("\nReading in all stock data.")

    stock_names = []
    df_all_stocks = pd.DataFrame()

    historical_data = os.path.join(PROJECT_DIRECTORY, "data/historical_data")
    directory = os.fsencode(historical_data)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            # Get stock name from csv file name
            stock_name = " ".join([word.capitalize() for word in filename.split(".")[0].split("_")])
            stock_names.append(stock_name)

            # Read csv to a dataframe
            df_stock = pd.read_csv(os.path.join(historical_data, filename))

            monthly_stocks = clean_stock_data(df_stock, stock_name)
            monthly_stocks.drop(["Volume"], axis=1, inplace=True)

            # Add all stocks to one dataframe
            if df_all_stocks.empty:
                df_all_stocks = monthly_stocks
            else:
                df_all_stocks = pd.merge(df_all_stocks, monthly_stocks, left_on='Date', right_on='Date', how='left')

        else:
            raise ValueError("Expected CSV stock data file type.")

    return df_all_stocks, stock_names


def get_baseline_data(dates: list) -> pd.DataFrame:
    """Reads baseline S&P 500 CSV data and returns baseline dataframe.

    Args:
        dates: The dates to get the data for.

    Returns:
        pd.DataFrame: A dataframe containing the baseline S&P 500 data.
    """
    print("\nReading in baseline data.")

    requested_start = pd.to_datetime(dates[0])
    requested_end = pd.to_datetime(dates[1])

    # Read baseline data from file
    baseline_filename = os.path.join(PROJECT_DIRECTORY, "data/baseline.csv")
    df_baseline = pd.read_csv(baseline_filename)
    df_baseline = clean_stock_data(df_baseline, BASELINE)

    df_baseline = df_baseline[(df_baseline.index >= requested_start) & (df_baseline.index <= requested_end)]

    return df_baseline


def get_requested_stocks(df: pd.DataFrame, dates: list, stocks: list=[], num_stocks: int=0) -> pd.DataFrame:
    """Given a dataframe of stocks and stocks/dates to get, returns a dataframe containing
    requested stocks stocks as columns and dates as rows.

    Args:
        df: The dataframe of all stocks.
        dates: The dates to get the stock data for.
        stocks: The stocks to get data for.
        num_stocks: The number of random stocks to get data for.

    Returns:
        pd.DataFrame: A dataframe containing the finance data.
    """
    print(f"\nGetting stock data for {dates[0]} to {dates[1]}...")

    first_date = df.index[0]
    last_date = df.index[-1]
    requested_start = pd.to_datetime(dates[0])
    requested_end = pd.to_datetime(dates[1])

    if requested_start < first_date or requested_end > last_date:
        raise Exception("Data does not exist for requested date.")
    
    df_date_filtered = df[(df.index >= requested_start) & (df.index <= requested_end)]

    if not len(stocks):
        # Generating random list of stocks
        if num_stocks == 0: num_stocks = 3

        all_stocks = list(df.columns)
        df_date_filtered = df_date_filtered[random.sample(all_stocks, num_stocks)]
    else:
        df_date_filtered = df_date_filtered[stocks]

    return df_date_filtered


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
