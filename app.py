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
from typing import NamedTuple, Union

import dash
import diskcache
from dash import MATCH, DiskcacheManager, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app_configs import APP_TITLE, THEME_COLOR, THEME_COLOR_SECONDARY
from dash_html import generate_problem_details_table_rows, set_html
from src.enums import SamplerType

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

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

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    prevent_initial_callbacks="initial_duplicate",
    background_callback_manager=background_callback_manager,
)
app.title = APP_TITLE

server = app.server
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
        sep="\n"
    )

# Generates css file and variable using THEME_COLOR and THEME_COLOR_SECONDARY settings
css = f"""/* Automatically generated theme settings css file, see app.py */
:root {{
    --theme: {THEME_COLOR};
    --theme-secondary: {THEME_COLOR_SECONDARY};
}}
"""
with open("assets/custom_00_theme.css", "w") as f:
    f.write(css)


@app.callback(
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


@app.callback(
    Output("input", "children"),
    inputs=[
        Input("slider", "value"),
    ],
)
def render_initial_state(slider_value: int) -> str:
    """Runs on load and any time the value of the slider is updated.
        Add `prevent_initial_call=True` to skip on load runs.

    Args:
        slider_value: The value of the slider.

    Returns:
        str: The content of the input tab.
    """
    return f"Put demo input here. The current slider value is {slider_value}."


class RunOptimizationReturn(NamedTuple):
    """Return type for the ``run_optimization`` callback function."""

    results: str = dash.no_update
    problem_details_table: list = dash.no_update
    # Add more return variables here. Return values for callback functions
    # with many variables should be returned as a NamedTuple for clarity.


@app.callback(
    # The Outputs below must align with `RunOptimizationReturn`.
    Output("results", "children"),
    Output("problem-details", "children"),
    background=True,
    inputs=[
        Input("run-button", "n_clicks"),
        State("sampler-type-select", "value"),
        State("solver-time-limit", "value"),
        State("slider", "value"),
        State("dropdown", "value"),
        State("checklist", "value"),
        State("radio", "value"),
    ],
    running=[
        # Shows cancel button while running.
        (Output("cancel-button", "className"), "", "display-none"),
        (Output("run-button", "className"), "display-none", ""),  # Hides run button while running.
        (Output("results-tab", "disabled"), True, False),  # Disables results tab while running.
        (Output("results-tab", "label"), "Loading...", "Results"),
        (Output("tabs", "value"), "input-tab", "input-tab"),  # Switch to input tab while running.
        (Output("run-in-progress", "data"), True, False),  # Can block certain callbacks.
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True,
)
def run_optimization(
    # The parameters below must match the `Input` and `State` variables found
    # in the `inputs` list above.
    run_click: int,
    sampler_type: Union[SamplerType, int],
    time_limit: float,
    slider_value: int,
    dropdown_value: int,
    checklist_value: list,
    radio_value: int,
) -> RunOptimizationReturn:
    """Runs the optimization and updates UI accordingly.

    This is the main function which is called when the `Run Optimization` button is clicked.
    This function takes in all form values and runs the optimization, updates the run/cancel
    buttons, deactivates (and reactivates) the results tab, and updates all relevant HTML
    components.

    Args:
        run_click: The (total) number of times the run button has been clicked.
        sampler_type: Either Quantum Hybrid (``0`` or ``SamplerType.HYBRID``),
            or Classical (``1`` or ``SamplerType.CLASSICAL``).
        time_limit: The solver time limit.
        slider_value: The value of the slider.
        dropdown_value: The value of the dropdown.
        checklist_value: A list of the values of the checklist.
        radio_value: The value of the radio.

    Returns:
        A NamedTuple (RunOptimizationReturn) containing all outputs to be used when updating the HTML
        template (in ``dash_html.py``). These are:

            results: The results to display in the results tab.
            problem-details: List of the table rows for the problem details table.
    """

    # Only run optimization code if this function was triggered by a click on `run-button`.
    # Setting `Input` as exclusively `run-button` and setting `prevent_initial_call=True`
    # also accomplishes this.
    if run_click == 0 or ctx.triggered_id != "run-button":
        raise PreventUpdate

    if isinstance(sampler_type, int):
        sampler_type = SamplerType(sampler_type)

    print(
        f"The form has the following values:\n\
        Example Slider: {slider_value}\n\
        Example Dropdown: {dropdown_value}\n\
        Example Checklist: {checklist_value}\n\
        Example Radio: {radio_value}\n\
        Solver: {sampler_type}\n\
        Time Limit: {time_limit}"
    )

    ###########################
    ### YOUR CODE GOES HERE ###
    ###########################

    # Generates a list of table rows for the problem details table.
    problem_details_table = generate_problem_details_table_rows(
        solver="Classical" if sampler_type is SamplerType.CLASSICAL else "Quantum Hybrid",
        time_limit=time_limit,
    )

    return RunOptimizationReturn(
        results="Put demo results here.",
        problem_details_table=problem_details_table,
    )


# Imports the Dash HTML code and sets it in the app.
# Creates the visual layout and app (see `dash_html.py`)
set_html(app)

# Run the server
if __name__ == "__main__":
    app.run_server(debug=DEBUG)
