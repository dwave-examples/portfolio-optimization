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
    "max": 10,
    "step": 0.5,
    "value": 0,
}

# Options:
# ['Social Media', 'Car Company', 'Animation Company', 'Search Engine', 'Online Marketplace',
# 'Another Computer Company', 'Large Retailer', 'Streaming Service', 'Coffee Company',
# 'Computer Company']
DEFAULT_STOCKS = ["Search Engine", "Large Retailer", "Coffee Company"]

BASELINE = "^GSPC"  # The S&P 500 symbol

# The canned data was downloaded from 2020-05-29 to 2025-05-23.
# The dates below should fall between the two dates above.
DATES_DEFAULT = ["2023-01-01", "2024-12-31"]

BUDGET = {
    "min": 5000,
    "max": 100000,
    "step": 100,
    "value": 10000,
}

# solver time limits in seconds (value means default)
SOLVER_TIME = {
    "min": 5,
    "max": 300,
    "step": 5,
    "value": 5,
}
