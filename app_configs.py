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
    "min": 1,
    "max": 100,
    "step": 1,
    "value": 5,
}

NUM_STOCKS = {
    "min": 1,
    "max": 10,
    "step": 1,
    "value": 5,
}

# an example dropdown
DROPDOWN = ["Option 1", "Option 2"]

# an example checklist
CHECKLIST = ["Option 1", "Option 2"]

# an example radio list
RADIO = ["Option 1", "Option 2"]

BUDGET = {
    "min": 100,
    "max": 1000,
    "step": 5,
    "value": 100,
}

# solver time limits in seconds (value means default)
SOLVER_TIME = {
    "min": 10,
    "max": 300,
    "step": 5,
    "value": 10,
}
