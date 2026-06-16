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

import json
import os
from itertools import product

import numpy as np
import pandas as pd
from dimod import Binary, ConstrainedQuadraticModel, DiscreteQuadraticModel, Integer, quicksum
from dwave.system import LeapHybridCQMSampler, LeapHybridDQMSampler, LeapHybridNLSampler
from dwave.optimization import Model
from dwave.optimization.mathematical import add, safe_divide
from dwave.optimization.symbols import IntegerVariable, BinaryVariable

from src.demo_enums import SolverType
from src.utils import get_baseline_data, get_requested_stocks, get_stock_data


class SinglePeriod:
    """Define and solve a  single-period portfolio optimization problem."""

    def __init__(
        self,
        stocks,
        budget=1000,
        bin_size=None,
        gamma=None,
        file_path="data/basic_data.csv",
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
            model_type (str): CQM, Stride or DQM.
            time_limit (int): The time limit for the runs.
            alpha (float or int or list or tuple): Risk aversion coefficient.
                If alpha is a tuple/list and model is DQM, grid search will be done;
                otherwise, no grid search.
            baseline (str): Stock baseline for rebalancing model.
            sampler_args (dict): Sampler arguments.
            t_cost (float): transaction cost; percentage of transaction dollar value.
            verbose (bool): Flag to enable additional output.
        """
        self.stocks = list(stocks)
        self.budget = budget
        self.init_budget = budget
        self.gamma_list = []
        self.file_path = file_path
        self.dates = dates
        self.model_type = model_type
        self.time_limit = time_limit
        self.alpha_list = []
        self.baseline = [baseline]
        self.verbose = verbose
        self.t_cost = t_cost
        self.init_holdings = {s: 0 for s in self.stocks}

        if isinstance(alpha, (list, tuple)):
            self.alpha = alpha[0]
            self.alpha_list = list(alpha)
        elif isinstance(alpha, (int, float)):
            self.alpha = alpha
        else:
            raise TypeError("Use integer or float for alpha, or a list or tuple of either type.")

        if gamma:
            if isinstance(gamma, (list, tuple)):
                self.gamma = gamma[-1]
                self.gamma_list = list(gamma)
            elif isinstance(gamma, (int, float)):
                self.gamma = gamma
            else:
                raise TypeError(
                    "Use integer or float for gamma, or a list or tuple of either type."
                )
        else:
            self.gamma = 100

        if bin_size:
            self.bin_size = bin_size
        else:
            self.bin_size = 10

        self.model = {"CQM": None, "DQM": None, "Stride": None}

        self.sample_set = {}
        if sampler_args:
            self.sampler_args = json.loads(sampler_args)
        else:
            self.sampler_args = {}

        self.sampler = {
            "CQM": LeapHybridCQMSampler(**self.sampler_args),
            "DQM": LeapHybridDQMSampler(**self.sampler_args),
            "Stride": LeapHybridNLSampler(**self.sampler_args),
        }

        self.solution = {}

        self.precision = 2

    def load_data(self, df=None, num=0):
        """Load the relevant stock data from files or dataframe.

        Args:
            df (dataframe): Table of stock prices.
            num (int): Number of stocks to be randomnly generated.
        """
        if df is not None:
            print("\nLoading data from DataFrame...")
            self.df = df
            self.stocks = df.columns.tolist()
        elif self.dates:
            print("\nLoading data from stock files...")
            all_stocks_df, _ = get_stock_data()

            self.df = get_requested_stocks(all_stocks_df, self.dates, self.stocks, num)
            self.df_baseline = get_baseline_data(self.dates)
            self.df_all = self.df

        else:
            print("\nLoading data from provided CSV file...")
            project_dir = os.getcwd()
            data_csv = os.path.join(project_dir, self.file_path)
            self.df = pd.read_csv(data_csv, index_col=0)
            self.stocks = self.df.columns.tolist()

        self.init_holdings = {s: 0 for s in self.stocks}

        self.max_num_shares = (self.budget / self.df.iloc[-1]).astype(int)
        if self.verbose:
            print("\nMax shares we can afford with a budget of", self.budget)
            print(self.max_num_shares.to_string())

        self.shares_intervals = {}
        for stock in self.stocks:
            if self.max_num_shares[stock] + 1 <= self.bin_size:
                self.shares_intervals[stock] = list(range(self.max_num_shares[stock] + 1))
            else:
                span = (self.max_num_shares[stock] + 1) / self.bin_size
                self.shares_intervals[stock] = [int(i * span) for i in range(self.bin_size)]

        self.price = self.df.iloc[-1]
        self.monthly_returns = self.df[list(self.stocks)].pct_change().iloc[1:]
        self.avg_monthly_returns = self.monthly_returns.mean(axis=0)
        self.covariance_matrix = covariance_matrix = self.monthly_returns.cov()

        # convert any NaNs in the covariance matrix to 0s
        covariance_matrix.replace(np.nan, 0)

    def build_cqm(self, max_risk=None, min_return=None, init_holdings=None):
        """Build and store a CQM.
        This method allows the user a choice of 3 problem formulations:
            1) max return - alpha*risk (default formulation)
            2) max return s.t. risk <= max_risk
            3) min risk s.t. return >= min_return

        Args:
            max_risk (int): Maximum risk for the risk bounding formulation.
            min_return (int): Minimum return for the return bounding formulation.
            init_holdings (float): Initial holdings, or initial portfolio state.
        """
        # Instantiating the CQM object
        cqm = ConstrainedQuadraticModel()

        # Defining and adding variables to the CQM model
        x = {
            s: Integer("%s" % s, lower_bound=0, upper_bound=self.max_num_shares[s])
            for s in self.stocks
        }

        # Defining risk expression
        risk = 0
        for s1, s2 in product(self.stocks, self.stocks):
            coeff = self.covariance_matrix[s1][s2] * self.price[s1] * self.price[s2]
            risk = risk + coeff * x[s1] * x[s2]

        # Defining the returns expression
        returns = 0
        for s in self.stocks:
            returns = returns + self.price[s] * self.avg_monthly_returns[s] * x[s]

        # Adding budget and related constraints
        if not init_holdings:
            init_holdings = self.init_holdings
        else:
            self.init_holdings = init_holdings

        if not self.t_cost:
            cqm.add_constraint(
                quicksum([x[s] * self.price[s] for s in self.stocks]) <= self.budget,
                label="upper_budget",
            )
            cqm.add_constraint(
                quicksum([x[s] * self.price[s] for s in self.stocks]) >= 0.997 * self.budget,
                label="lower_budget",
            )
        else:
            # Modeling transaction cost
            x0 = init_holdings

            y = {s: Binary("Y[%s]" % s) for s in self.stocks}

            lhs = 0
            for s in self.stocks:
                lhs = (
                    lhs
                    + 2 * self.t_cost * self.price[s] * x[s] * y[s]
                    + self.price[s] * (1 - self.t_cost) * x[s]
                    - 2 * self.t_cost * self.price[s] * x0[s] * y[s]
                    - self.price[s] * (1 - self.t_cost) * x0[s]
                )

            cqm.add_constraint(lhs <= self.budget, label="upper_budget")
            cqm.add_constraint(lhs >= self.budget - 0.003 * self.init_budget, label="lower_budget")

            # indicator constraints
            for s in self.stocks:
                cqm.add_constraint(x[s] - x0[s] * y[s] >= 0, label=f"indicator_constraint_gte_{s}")
                cqm.add_constraint(
                    x[s] - x[s] * y[s] <= x0[s], label=f"indicator_constraint_lte_{s}"
                )

        if max_risk:
            # Adding maximum risk constraint
            cqm.add_constraint(risk <= max_risk, label="max_risk")

            # Objective: maximize return
            cqm.set_objective(-1 * returns)
        elif min_return:
            # Adding minimum returns constraint
            cqm.add_constraint(returns >= min_return, label="min_return")

            # Objective: minimize risk
            cqm.set_objective(risk)
        else:
            # Objective: minimize mean-variance expression
            cqm.set_objective(self.alpha * risk - returns)

        cqm.substitute_self_loops()

        self.model["CQM"] = cqm

    def solve_cqm(self, max_risk=None, min_return=None, init_holdings=None):
        """Solve CQM.
        This method allows the user to solve one of 3 cqm problem formulations:
            1) max return - alpha*risk (default formulation)
            2) max return s.t. risk <= max_risk
            3) min risk s.t. return >= min_return

        Args:
            max_risk (int): Maximum risk for the risk bounding formulation.
            min_return (int): Minimum return for the return bounding formulation.
            init_holdings (float): Initial holdings, or initial portfolio state.

        Returns:
            solution (dict): This is a dictionary that saves solutions in desired format
                e.g., solution = {'stocks': {'IBM': 3, 'WMT': 12}, 'risk': 10, 'return': 20}
        """
        self.build_cqm(max_risk, min_return, init_holdings)

        self.sample_set["CQM"] = self.sampler["CQM"].sample_cqm(
            self.model["CQM"], label="Example - Portfolio Optimization", time_limit=self.time_limit
        )
        n_samples = len(self.sample_set["CQM"].record)

        feasible_samples = self.sample_set["CQM"].filter(lambda d: d.is_feasible)

        if not feasible_samples:
            raise Exception("No feasible solution could be found for this problem instance.")

        best_feasible = feasible_samples.first

        solution = {}
        solution["stocks"] = {k: int(best_feasible.sample[k]) for k in self.stocks}
        solution["return"], solution["risk"] = self.compute_risk_and_returns(solution["stocks"])

        cost = sum(
            [
                self.price[s] * max(0, solution["stocks"][s] - self.init_holdings[s])
                for s in self.stocks
            ]
        )
        sales = sum(
            [
                self.price[s] * max(0, self.init_holdings[s] - solution["stocks"][s])
                for s in self.stocks
            ]
        )

        transaction = self.t_cost * (cost + sales)

        solution.update(
            {
                "energy": best_feasible.energy,
                "sales": sales,
                "cost": cost,
                "transaction cost": transaction,
                "number feasible": len(feasible_samples),
                "number sampled": n_samples,
                "best energy": self.sample_set["CQM"].first.energy,
            }
        )

        return solution

    def build_dqm(self, alpha=None, gamma=None):
        """Build DQM.

        Args:
            alpha (float): Risk aversion coefficient.
            gamma (int): Penalty coefficient for budgeting constraint.
        """
        if gamma:
            self.gamma = gamma

        if alpha:
            self.alpha = alpha

        # Defining DQM
        dqm = DiscreteQuadraticModel()

        # Build the DQM starting by adding variables
        for s in self.stocks:
            dqm.add_variable(len(self.shares_intervals[s]), label=s)

        # Objective 1: minimize variance
        for s1, s2 in product(self.stocks, self.stocks):
            coeff = self.covariance_matrix[s1][s2] * self.price[s1] * self.price[s2]
            if s1 == s2:
                for k in range(dqm.num_cases(s1)):
                    num_s1 = self.shares_intervals[s1][k]
                    dqm.set_linear_case(
                        s1, k, dqm.get_linear_case(s1, k) + self.alpha * coeff * num_s1 * num_s1
                    )
            else:
                for k in range(dqm.num_cases(s1)):
                    for m in range(dqm.num_cases(s2)):
                        num_s1 = self.shares_intervals[s1][k]
                        num_s2 = self.shares_intervals[s2][m]

                        dqm.set_quadratic_case(
                            s1,
                            k,
                            s2,
                            m,
                            dqm.get_quadratic_case(s1, k, s2, m)
                            + coeff * self.alpha * num_s1 * num_s2,
                        )

        # Objective 2: maximize return
        for s in self.stocks:
            for j in range(dqm.num_cases(s)):
                dqm.set_linear_case(
                    s,
                    j,
                    dqm.get_linear_case(s, j)
                    - self.shares_intervals[s][j] * self.price[s] * self.avg_monthly_returns[s],
                )

        # Scaling factor to guarantee that all coefficients are integral
        # needed in order to use add_linear_inequality_constraint method
        factor = 10**self.precision

        min_budget = round(factor * 0.997 * self.budget)
        budget = int(self.budget)

        terms = [
            (s, j, int(self.shares_intervals[s][j] * factor * self.price[s]))
            for s in self.stocks
            for j in range(dqm.num_cases(s))
        ]

        dqm.add_linear_inequality_constraint(
            terms,
            constant=0,
            lb=min_budget,
            ub=factor * budget,
            lagrange_multiplier=self.gamma,
            label="budget",
        )

        self.model["DQM"] = dqm

    def solve_dqm(self):
        """Solve DQM.

        Returns:
            solution (dict): This is a dictionary that saves solutions in desired format
                e.g., solution = {'stocks': {'IBM': 3, 'WMT': 12}, 'risk': 10, 'return': 20}
        """
        if not self.model["DQM"]:
            self.build_dqm()

        self.sample_set["DQM"] = self.sampler["DQM"].sample_dqm(
            self.model["DQM"], label="Example - Portfolio Optimization", time_limit=self.time_limit
        )

        sample = self.sample_set["DQM"].first.sample

        solution = {}
        solution["stocks"] = {s: self.shares_intervals[s][sample[s]] for s in self.stocks}
        solution["return"], solution["risk"] = self.compute_risk_and_returns(solution["stocks"])

        cost = sum([self.price[s] * solution["stocks"][s] for s in self.stocks])

        solution.update(
            {
                "cost": cost,
                "alpha": self.alpha,
                "gamma": self.gamma,
            }
        )

        return solution

    def dqm_grid_search(self):
        """Execute parameter (alpha, gamma) grid search for DQM."""
        alpha = self.alpha_list
        gamma = self.gamma_list

        data_matrix = np.zeros((len(alpha), len(gamma)))

        if self.verbose:
            print("\nGrid search results:")

        for i in range(len(alpha)):
            for j in range(len(gamma)):

                alpha_i = alpha[i]
                gamma_j = gamma[j]

                self.build_dqm(alpha_i, gamma_j)

                # Solve the problem using the DQM solver
                solution = self.solve_dqm()

                data_matrix[i, j] = solution["return"] / np.sqrt(solution["risk"])

        n_opt = np.argmax(data_matrix)

        self.alpha = alpha[n_opt // len(gamma)]
        self.gamma = gamma[n_opt - (n_opt // len(gamma)) * len(gamma)]

        print(f"DQM Grid Search Completed: alpha={self.alpha}, gamma={self.gamma}.-")

    def compute_risk_and_returns(self, solution):
        """Compute the risk and return values of solution."""
        variance = 0.0
        for s1, s2 in product(solution, solution):
            variance += (
                solution[s1]
                * self.price[s1]
                * solution[s2]
                * self.price[s2]
                * self.covariance_matrix[s1][s2]
            )

        est_return = 0
        for stock in solution:
            est_return += solution[stock] * self.price[stock] * self.avg_monthly_returns[stock]

        return round(est_return, 2), round(variance, 2)

    def print_results(self, solution: dict):
        """Print results to the console given a solution dictionary."""
        is_cqm_run = self.model_type is SolverType.CQM
        is_dqm_run = self.model_type is SolverType.DQM
        is_stride_run = self.model_type is SolverType.Stride

        if self.verbose and is_cqm_run:
            print(
                f"Number of feasible solutions: {solution['number feasible']} out of {solution['number sampled']} sampled.",
                f"\nBest energy: {solution['best energy']:.2f}",
                f"Best energy (feasible): {solution['energy']:.2f}",
                sep="\n",
            )

        if is_dqm_run:
            print(f"\nSolution for alpha = {solution['alpha']} and gamma = {solution['gamma']}:")

        print(
            f"\nBest feasible solution:",
            "\n".join(f"{k}\t{v:>3}" for k, v in solution["stocks"].items()),
            f"\nEstimated Returns: {solution['return']}",
            sep="\n",
        )

        if is_cqm_run or is_stride_run:
            print(f"Sales Revenue: {solution['sales']:.2f}")

        print(f"Purchase Cost: {solution['cost']:.2f}")

        if is_cqm_run or is_stride_run:
            print(f"Transaction Cost: {solution['transaction cost']:.2f}")

        print(f"Variance: {solution['risk']}\n")

    def solve_stride(self, max_risk=None, min_return=None, init_holdings=None):
        """Build and solve a Stride formulation.
        This method allows the user a choice of 3 problem formulations:
            1) max return - alpha*risk (default formulation)
            2) max return s.t. risk <= max_risk
            3) min risk s.t. return >= min_return

        Args:
            max_risk (int): Maximum risk for the risk bounding formulation.
            min_return (int): Minimum return for the return bounding formulation.
            init_holdings (float): Initial holdings, or initial portfolio state.
        """
        # Instantiating the Stride object
        stride_nl = Model()

        # Required constants
        stock_prices = np.array(list(self.price))
        avg_returns = np.array(list(self.avg_monthly_returns))
        StockPrices = stride_nl.constant(stock_prices)
        BudgetUpper = stride_nl.constant(self.budget)
        BudgetLower = stride_nl.constant(0.997*self.budget)
        cov = self.covariance_matrix.loc[self.stocks, self.stocks].to_numpy()
        risk_coeff = cov * np.outer(stock_prices, stock_prices)
        RiskCoeff = stride_nl.constant(risk_coeff)
        AvgReturns = stride_nl.constant(avg_returns*stock_prices)
        NegOne = stride_nl.constant(-1)
        Alpha = stride_nl.constant(self.alpha)
        Zero = stride_nl.constant(0)
        TransCost = stride_nl.constant(self.t_cost)
        One = stride_nl.constant(1)
        Two = stride_nl.constant(2)
        BudgetLowerWithTrans = stride_nl.constant(self.budget - 0.003 * self.init_budget)

        # Defining and adding variables to the Stride model
        x = stride_nl.integer(len(self.stocks), lower_bound=0, upper_bound=list(self.max_num_shares))

        # Defining risk expression
        risk = []
        for (i1, s1), (i2, s2) in product(enumerate(self.stocks), enumerate(self.stocks)):
            risk.append(RiskCoeff[i1, i2] * x[i1] * x[i2])

        # Defining the returns expression
        returns = AvgReturns * x

        # Adding budget and related constraints
        if not init_holdings:
            init_holdings = self.init_holdings
        else:
            self.init_holdings = init_holdings

        if not self.t_cost:
            stock_purchases = x*StockPrices
            stride_nl.add_constraint(stock_purchases.sum() <= BudgetUpper)
            stride_nl.add_constraint(stock_purchases.sum() >= BudgetLower)
        else:
            # Modeling transaction cost
            y = stride_nl.binary(len(self.stocks))  # dummy binary variable
            x0 = stride_nl.constant(np.array(list(init_holdings.values())))
            lhs = []
            for i, s in enumerate(self.stocks):
                curr_return = StockPrices[i] * (One - TransCost) * x[i]
                prev_return = StockPrices[i] * (One - TransCost) * x0[i]
                curr_cost = Two * TransCost * StockPrices[i] * x[i] * y[i]
                prev_cost = Two * TransCost * StockPrices[i] * x0[i] * y[i]
                lhs.append(curr_cost + curr_return - prev_cost - prev_return)
                # Indicator linking constraints
                stride_nl.add_constraint(x[i] - x0[i]*y[i] >= Zero)
                stride_nl.add_constraint(x[i] - x[i]*y[i] <= x0[i])
            stride_nl.add_constraint(add(*lhs) <= BudgetUpper)
            stride_nl.add_constraint(add(*lhs) >= BudgetLowerWithTrans)

        if max_risk:
            print("Maximize returns s.t. an upper bound of risk")
            # Adding maximum risk constraint
            MaxRisk = stride_nl.constant(max_risk)
            stride_nl.add_constraint(add(*risk) <= MaxRisk)
            # Objective: maximize return
            stride_nl.minimize(NegOne*returns.sum())
        elif min_return:
            print("Minimize risk s.t. a lower bound of return")
            # Adding minimum returns constraint
            MinReturn = stride_nl.constant(min_return)
            stride_nl.add_constraint(returns.sum() >= MinReturn)
            # Objective: minimize risk
            stride_nl.minimize(add(*risk))
        else:
            print("Minimize mean-variance expression")
            # Objective: minimize mean-variance expression
            stride_nl.minimize(Alpha*add(*risk) - returns.sum())
            """
            Alternative objective function - minimize the ratio of risk per unit return. This can be also done in Stride.
            obj_expr = safe_divide((Alpha * add(*risk)), returns.sum())
            stride_nl.minimize(obj_expr)
            """

        self.model["Stride"] = stride_nl
        self.sample_set["Stride"] = self.sampler["Stride"].sample(stride_nl, label="Example - Portfolio Optimization", time_limit=self.time_limit)
        stride_nl.lock()
        solution = {}
        if stride_nl.feasible():
            model_des = list(sym for sym in stride_nl.iter_decisions())
            x_var = None
            for i in range(len(model_des)):
                if type(model_des[i]) == IntegerVariable:
                    x_var = model_des[i]
                elif type(model_des[i]) == BinaryVariable:
                    y_var = model_des[i]
            solution["stocks"] = {s: int(x_var.state(0)[i]) for i, s in enumerate(self.stocks)}
            solution["return"], solution["risk"] = self.compute_risk_and_returns(solution["stocks"])
            cost = sum(
                [
                    self.price[s] * max(0, solution["stocks"][s] - self.init_holdings[s])
                    for s in self.stocks
                ]
            )
            sales = sum(
                [
                    self.price[s] * max(0, self.init_holdings[s] - solution["stocks"][s])
                    for s in self.stocks
                ]
            )
            transaction = self.t_cost * (cost + sales)
            solution.update(
                {
                    "sales": sales,
                    "cost": cost,
                    "transaction cost": transaction,
                }
            )
        else:
            raise Exception("No feasible solution could be found for this problem instance.")

        return solution

    def run(
        self, min_return: float = 0, max_risk: float = 0, num: int = 0, init_holdings: float = None
    ):
        """Execute sequence of load_data --> build_model --> solve.

        Args:
            max_risk (int): Maximum risk for the risk bounding formulation.
            min_return (int): Minimum return for the return bounding formulation.
            num (int): Number of stocks to be randomnly generated.
            init_holdings (float): Initial holdings, or initial portfolio state.

        Returns:
            solution (dict): A dictionary containing solution data.
        """
        self.load_data(num=num)
        if self.model_type is SolverType.CQM:
            print(f"\nCQM run...")
            solution = self.solve_cqm(
                min_return=min_return, max_risk=max_risk, init_holdings=init_holdings
            )
        elif self.model_type is SolverType.Stride:
            print(f"\nStride run...")
            solution = self.solve_stride(min_return=min_return, max_risk=max_risk, init_holdings=init_holdings)
        else:
            print(f"\nDQM run...")
            if len(self.alpha_list) > 1 or len(self.gamma_list) > 1:
                print("\nStarting DQM Grid Search...")
                self.dqm_grid_search()

            self.build_dqm()
            solution = self.solve_dqm()

        self.print_results(solution=solution)
        return solution
