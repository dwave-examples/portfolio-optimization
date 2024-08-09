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

"""This file stores input parameters for the app."""

# THEME_COLOR is used for the button, text, and banner and should be dark
# and pass accessibility checks with white: https://webaim.org/resources/contrastchecker/
# THEME_COLOR_SECONDARY can be light or dark and is used for sliders, loading icon, and tabs
THEME_COLOR = "#074C91"  # D-Wave dark blue default #074C91
THEME_COLOR_SECONDARY = "#2A7DE1"  # D-Wave blue default #2A7DE1

THUMBNAIL = "static/dwave_logo.svg"

APP_TITLE = "Portfolio Optimization"
MAIN_HEADER = "Portfolio Optimization"
DESCRIPTION = """\
Optimizing a portfolio of stocks is a challenging problem that looks to identify the optimal number 
of shares of each stock to purchase in order to minimize risk (variance) and maximize returns, 
while staying under some specified spending budget.
"""

#######################################
# Sliders, buttons and option entries #
#######################################

TRANSACTION_COST = {
    "min": 0,
    "max": 5,
    "step": 0.1,
    "value": 2.5,
}

NUM_STOCKS = {
    "min": 1,
    "max": 10,
    "step": 1,
    "value": 5,
}

PERIOD_OPTIONS = ["Single-Period", "Multi-Period Rebalancing"]

STOCK_OPTIONS = {
    "options": [
        {"label": "Amazon (AMZN)", "value": "AMZN"},
        {"label": "American Airlines (AAL)", "value": "AAL"},
        {"label": "Apple (AAPL)", "value": "AAPL"},
        {"label": "Coca-Cola (KO)", "value": "KO"},
        {"label": "Disney (DIS)", "value": "DIS"},
        {"label": "Google (GOOG)", "value": "GOOG"},
        {"label": "Johnson & Johnson (JNJ)", "value": "JNJ"},
        {"label": "Mastercard (MA)", "value": "MA"},
        {"label": "Meta (META)", "value": "META"},
        {"label": "Microsoft (MSFT)", "value": "MSFT"},
        {"label": "Netflix (NFLX)", "value": "NFLX"},
        {"label": "Tesla (TSLA)", "value": "TSLA"},
        {"label": "Visa (V)", "value": "V"},
        {"label": "Walmart (WMT)", "value": "WMT"},
        {"label": "Wells Fargo (WFC)", "value": "WFC"},
    ],
    "value": ['AAPL', 'MSFT', 'AAL', 'WMT'],
}

BUDGET = {
    "min": 1000,
    "max": 100000,
    "step": 100,
    "value": 5000,
}

# solver time limits in seconds (value means default)
SOLVER_TIME = {
    "min": 10,
    "max": 300,
    "step": 5,
    "value": 10,
}
