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

import argparse

import dash
import diskcache
from dash import DiskcacheManager

from demo_configs import APP_TITLE, THEME_COLOR, THEME_COLOR_SECONDARY
from demo_interface import create_interface

# Essential for initializing callbacks. Do not remove.
import demo_callbacks

# Fix Dash long callbacks crashing on macOS 10.13+ (also potentially not working
# on other POSIX systems), caused by https://bugs.python.org/issue33725
# (aka "beware of multithreaded process forking").
#
# Note: default start method has already been changed to "spawn" on darwin in
# the `multiprocessing` library, but its fork, `multiprocess` still hasn't caught up.
# (see docs: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)
import multiprocess

if multiprocess.get_start_method(allow_none=True) is None:
    multiprocess.set_start_method("spawn")

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    prevent_initial_callbacks="initial_duplicate",
    background_callback_manager=background_callback_manager,
)
app.title = APP_TITLE

app.config.suppress_callback_exceptions = True

# Parse debug argument
parser = argparse.ArgumentParser(description="Dash debug setting.")
parser.add_argument(
    "--debug",
    action="store_true",
    help="Add argument to see Dash debug menu and get live reload updates while developing.",
)

args = parser.parse_args()
DEBUG = args.debug

print(f"\nDebug has been set to: {DEBUG}")
if not DEBUG:
    print(
        "The app will not show live code updates and the Dash debug menu will be hidden.",
        "If editting code while the app is running, run the app with `python app.py --debug`.\n",
        sep="\n",
    )

# Generates css file and variable using THEME_COLOR and THEME_COLOR_SECONDARY settings
css = f"""/* Automatically generated theme settings css file, see app.py */
:root {{
    --theme: {THEME_COLOR};
    --theme-secondary: {THEME_COLOR_SECONDARY};
}}
"""
with open("assets/__generated_theme.css", "w") as f:
    f.write(css)


if __name__ == "__main__":
    # Imports the Dash HTML code and sets it in the app.
    # Creates the visual layout and app (see `demo_interface.py`)
    app.layout = create_interface()

    # Run the server
    app.run_server(debug=DEBUG)
