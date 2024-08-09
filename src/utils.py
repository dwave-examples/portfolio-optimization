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

import random

import pandas as pd

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
    print(stocks)
    print(dates)
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
