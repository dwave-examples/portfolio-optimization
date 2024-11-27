# Copyright 2021 D-Wave Systems Inc.
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

import matplotlib
import numpy as np
import pandas as pd

from src.demo_enums import SolverType

matplotlib.use("agg")
import matplotlib.pyplot as plt
from dwave.system import LeapHybridCQMSampler, LeapHybridDQMSampler

from src.single_period import SinglePeriod


class MultiPeriod(SinglePeriod):
    """Solve the multi-period (dynamic) portfolio optimization problem."""

    def __init__(
        self,
        stocks=("AAPL", "MSFT", "AAL", "WMT"),
        budget=1000,
        bin_size=None,
        gamma=None,
        file_path=None,
        dates=None,
        model_type=SolverType.CQM,
        time_limit=5,
        alpha=0.005,
        baseline="^GSPC",
        sampler_args=None,
        t_cost=0.01,
        verbose=True,
    ):
        """Class constructor.

        Args:
            stocks (list of str): List of stocks.
            budget (int): Portfolio budget.
            bin_size (int): Maximum number of intervals for each stock.
            gamma (float or int or list or tuple): Budget constraint penalty coefficient(s).
                If gamma is a tuple/list and model is DQM, grid search will be done;
                otherwise, no grid search.
            file_path (str): Full path of CSV file containing stock data.
            dates (list of str): Pair of strings for start date and end date.
            model_type (str): CQM or DQM.
            time_limit (int): The time limit for the runs.
            alpha (float or int or list or tuple): Risk aversion coefficient.
                If alpha is a tuple/list and model is DQM, grid search will be done;
                otherwise, no grid search.
            baseline (str): Stock baseline for rebalancing model.
            sampler_args (dict): Sampler arguments.
            t_cost (float): transaction cost; percentage of transaction dollar value.
            verbose (bool): Flag to enable additional output.
        """
        super().__init__(
            stocks=stocks,
            budget=budget,
            t_cost=t_cost,
            bin_size=bin_size,
            gamma=gamma,
            file_path=file_path,
            dates=dates,
            model_type=model_type,
            time_limit=time_limit,
            alpha=alpha,
            baseline=baseline,
            sampler_args=sampler_args,
            verbose=verbose,
        )

    def run(self, max_risk=0, min_return=0, num=0, init_holdings=None):
        """Solve the rebalancing portfolio optimization problem.

        Args:
            max_risk (int): Maximum risk for the CQM risk bounding formulation.
            min_return (int): Minimum return for the CQM return bounding formulation.
        """
        if not self.dates:
            self.dates = ["2010-01-01", "2012-12-31"]

        self.load_data()

        num_months = len(self.df_all)
        first_purchase = True
        baseline_result = {}
        self.baseline_values = [0]
        self.update_values = [0]
        months = []
        initial_budget = 0

        # Define dataframe to save output data
        headers = ["Date", "Value"] + self.stocks + ["Variance", "Returns"]
        self.opt_results_df = pd.DataFrame(columns=headers)
        all_solutions = {}

        self.price_df = pd.DataFrame(columns=self.stocks)

        # Initialize the plot
        plt.ylim(ymax=1.5 * self.budget, ymin=-1.5 * self.budget)
        plt.xticks(
            list(range(0, num_months, 2)),
            self.df_baseline.index.strftime("%b")[::2],
            rotation="vertical",
        )
        plt.locator_params(axis="x", nbins=num_months / 2)
        plt.plot(
            list(range(0, num_months)),
            [0] * (num_months),
            color="red",
            label="Break-even",
            linewidth=0.5,
        )

        for i in range(3, num_months):

            baseline_result, months, all_solutions, init_holdings, initial_budget = self.run_update(
                i,
                first_purchase=first_purchase,
                initial_budget=initial_budget,
                baseline_result=baseline_result,
                months=months,
                all_solutions=all_solutions,
                max_risk=max_risk,
                min_return=min_return,
                init_holdings=init_holdings,
                cli_run=True,
            )

            first_purchase = False

        print(self.opt_results_df)

        plt.savefig("portfolio.png")
        plt.show(block=False)

        return all_solutions

    def initiate_run_update(
        self,
        i: int,
        baseline_result: dict,
        months: list,
        all_solutions: dict,
        first_purchase: bool = True,
        initial_budget: float = 0,
        init_holdings: list = None,
    ):
        """Solve the rebalancing portfolio optimization problem.

        Args:
            i: Current loop iteration.
            first_purchase: Whether this is the first loop iteration.
            initial_budget: The budget going into this iteration.
            baseline_result: The baseline stock for multi-period portfolio optimization run.
            months: A list of the months for solutions already found.
            all_solutions: A dict of month, solution key-value pairs.
            init_holdings: The stocks to start the run with.
        """

        self.sampler = {
            "CQM": LeapHybridCQMSampler(**self.sampler_args),
            "DQM": LeapHybridDQMSampler(**self.sampler_args),
        }

        baseline_result, months, all_solutions, init_holdings, initial_budget = self.run_update(
            i,
            first_purchase=first_purchase,
            initial_budget=initial_budget,
            baseline_result=baseline_result,
            months=months,
            all_solutions=all_solutions,
            init_holdings=init_holdings,
        )

        # Removing the following values to be able to serialize
        self.model = {}
        self.sampler = {}
        self.sample_set["CQM"] = {}
        self.sample_set["DQM"] = {}

        return baseline_result, months, all_solutions, init_holdings

    def run_update(
        self,
        i: int,
        baseline_result: dict,
        months: list,
        all_solutions: dict,
        first_purchase: bool = True,
        initial_budget: float = 0,
        max_risk: float = 0,
        min_return: float = 0,
        init_holdings: list = None,
        cli_run: bool = False,
    ):
        """Solve the rebalancing portfolio optimization problem.

        Args:
            i: Current loop iteration.
            first_purchase: Whether this is the first loop iteration.
            initial_budget: The budget going into this iteration.
            baseline_result: The baseline stock for multi-period portfolio optimization run.
            months: A list of the months for solutions already found.
            all_solutions: A dict of month, solution key-value pairs.
            max_risk: Maximum risk for the CQM risk bounding formulation.
            min_return: Minimum return for the CQM return bounding formulation.
            init_holdings: The stocks to start the run with.
            cli_run: Prints to command line interface and outputs portfolio png if True.
        """
        # Look at just the data up to the current month
        df = self.df_all.iloc[0 : i + 1, :].copy()
        baseline_df_current = self.df_baseline.iloc[0 : i + 1, :]
        curr_date = df.last_valid_index()
        months.append(curr_date.date())

        if cli_run:
            print("\nDate:", curr_date)

        if first_purchase:
            budget = self.budget
            initial_budget = self.budget
            baseline_shares = budget / baseline_df_current.iloc[-1]
            baseline_result = {self.baseline[0]: baseline_shares}
        else:
            solution = all_solutions[list(all_solutions)[-1]]

            # Compute profit of current portfolio
            budget = sum([df.iloc[-1][s] * solution["stocks"][s] for s in self.stocks])
            self.update_values.append(budget - initial_budget)

            # Compute profit of fund portfolio
            fund_value = sum(
                [baseline_df_current.iloc[-1][s] * baseline_result[s] for s in self.baseline]
            )

            self.baseline_values.append(*fund_value - initial_budget)

            self.budget = budget

        self.load_data(df=df)

        self.price_df.loc[i - 2] = list(self.price.values)

        if cli_run:
            # Output for user on command-line and plot
            update_values = np.array(self.update_values, dtype=object)
            baseline_values = np.array(self.baseline_values, dtype=object)
            plt.plot(range(3, i + 1), update_values, color="blue", label="Optimized portfolio")
            plt.plot(
                range(3, i + 1),
                baseline_values,
                color="gray",
                label="Fund portfolio",
                linewidth=0.5,
            )

            if first_purchase:
                plt.legend(loc="lower left")
                plt.title(
                    "Start: {start}, End: {end}".format(
                        start=self.df_all.first_valid_index().date(),
                        end=self.df_all.last_valid_index().date(),
                    )
                )

            plt.savefig("portfolio.png")
            plt.pause(0.05)

            print(f"\nMulti-Period {self.model_type.label} Run...")

        # Making solve run
        if self.model_type is SolverType.DQM:
            self.build_dqm()
            solution = self.solve_dqm()
        else:
            # Set budget to 0 to enforce that portfolio is self-financing
            if self.t_cost and not first_purchase:
                self.budget = 0

            solution = self.solve_cqm(
                max_risk=max_risk, min_return=min_return, init_holdings=init_holdings
            )

            init_holdings = solution["stocks"]

        all_solutions[curr_date.date().strftime("%Y-%m-%d")] = solution

        if cli_run:
            self.print_results(solution=solution)

        # Print results to command-line
        value = sum([self.price[s] * solution["stocks"][s] for s in self.stocks])
        returns = solution["return"]
        variance = solution["risk"]

        row = (
            [months[-1].strftime("%Y-%m-%d"), value]
            + [solution["stocks"][s] for s in self.stocks]
            + [variance, returns]
        )
        self.opt_results_df.loc[i - 2] = row

        return baseline_result, months, all_solutions, init_holdings, initial_budget
