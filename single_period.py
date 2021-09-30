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

import itertools
import numpy as np
import pandas as pd
from pandas_datareader import data

from dimod import Integer
from dimod import quicksum 
from dimod import ConstrainedQuadraticModel, DiscreteQuadraticModel

from dwave.system import LeapHybridDQMSampler, LeapHybridCQMSampler 


class SinglePeriod: 
    """Define and solve a  single-period portfolio optimization problem.
    """
    sampler = {'CQM': LeapHybridCQMSampler(profile='cqm_alpha'),
               'DQM': LeapHybridDQMSampler()}

    def __init__(self, stocks=['IBM', 'SEHI', 'WMT'] , budget=1000, 
                 bin_size=10, gamma=[10], file_path='data/basic_data.csv', 
                 dates=[], model_type='CQM', alpha=[0.0005], baseline='^GSPC', 
                 verbose=True):
        self.stocks = stocks 
        self.budget = budget 
        self.bin_size = bin_size
        self.gamma_list = gamma
        self.file_path = file_path
        self.dates = dates 
        self.model_type = model_type
        self.alpha_list = alpha

        self.baseline = [baseline] 

        self.verbose = verbose 

        self.alpha = alpha[-1]
        self.gamma = gamma[-1]
        
        self.model = {'CQM': None, 'DQM': None}
        self.sample_set = {}

        self.precision =  2

        # This is a dictionary that saves solutions in desired format 
        # e.g., solution = {'CQM':{'stocks': {'IBM': 3, 'WMT': 12}, 
        #                          'risk': 10, 
        #                          'return': 20}
        #                   }
        self.solution = {}
        
    def load_data(self, file_path='', dates=[], df=pd.DataFrame()):
        """Load the relevant stock data from file, dataframe, or Yahoo!. 

        Args:
            file_path (string): 
                Full path of csv file containing stock price data for single
                period problem.
            dates (list): 
                [Start_Date, End_Date] to query data from Yahoo!.
            df (dataframe): 
                Table of stock prices.   

        Returns:
            None.
        """
        if not df.empty:
            self.df = df 
        elif dates or self.dates: 
            if dates:
                self.dates = dates 
            else:
                dates = self.dates
            # Read in daily data; resample to monthly
            panel_data = data.DataReader(self.stocks, 'yahoo', dates[0], dates[1])
            panel_data = panel_data.resample('BM').last()
            self.df_all = pd.DataFrame(index=panel_data.index, columns=self.stocks)

            for i in self.stocks:
                self.df_all[i] = panel_data[[('Adj Close',  i)]]

            # Read in baseline data; resample to monthly
            index_df = data.DataReader(self.baseline, 'yahoo', dates[0], dates[1])
            index_df = index_df.resample('BM').last()
            self.df_baseline = pd.DataFrame(index=index_df.index, columns=self.baseline)

            for i in self.baseline:
                self.df_baseline[i] = index_df[[('Adj Close',  i)]]

            self.df = self.df_all 
        else:
            if file_path:
                self.file_path = file_path
            else:
                file_path = self.file_path

            self.df = pd.read_csv(file_path, index_col=0)

        self.max_num_shares = (self.budget/self.df.iloc[-1]).astype(int)
        if self.verbose:
            print("\nMax shares we can afford with a budget of", self.budget)
            print(self.max_num_shares.to_string())

        self.shares_intervals = {}
        for stock in self.stocks:
            if self.max_num_shares[stock]+1 <= self.bin_size:
                self.shares_intervals[stock] = list(range(self.max_num_shares[stock] + 1))
            else:
                span = (self.max_num_shares[stock]+1) / self.bin_size
                self.shares_intervals[stock] = [int(i*span) 
                                        for i in range(self.bin_size)]

        self.price = self.df.iloc[-1]
        self.monthly_returns = self.df[self.stocks].pct_change().iloc[1:]
        self.ave_monthly_returns = self.monthly_returns.mean(axis=0)
        self.covariance_matrix = self.monthly_returns.cov()

    def build_cqm(self, max_risk=None, min_return=None):
        """Build and store a CQM. 
        This method allows the user a choice of 3 problem formulations: 
            1) max return - alpha*risk (default formulation)
            2) max return s.t. risk <= max_risk 
            3) min risk s.t. return >= min_return  

        Args:
            max_risk (int): 
                Maximum risk for the risk bounding problem formulation.
            min_return (int): 
                Minimum return for the return bounding problem formulation.

        Returns:
            None.
        """
        # Instantiating the CQM object 
        cqm = ConstrainedQuadraticModel()

        # Defining and adding variables to the CQM model 
        x = {s: Integer("%s" %s, lower_bound=0, 
                        upper_bound=self.max_num_shares[s]) for s in self.stocks}

        # Adding budget constraint 
        cqm.add_constraint(quicksum([x[s]*self.price[s] for s in self.stocks]) <= self.budget)

        if max_risk: 
            # Adding maximum risk constraint 
            expression = 0
            for s1 in self.stocks:
                for s2 in self.stocks:
                    coeff = (self.covariance_matrix[s1][s2] * self.price[s1] * self.price[s2])
                    expression = expression + coeff*x[s1]*x[s2]
            cqm.add_constraint(expression <= max_risk)

            # Objective: maximize return 
            for s in self.stocks:
                cqm.objective.add_linear(x[s].variables[0], - self.price[s] * self.ave_monthly_returns[s])

            if self.verbose:
                print(f"\nCQM Formulation: maximize return s.t. risk <= max_risk\n")
        elif min_return: 
            # Adding minimum return constraint 
            cqm.add_constraint(quicksum([x[s]*self.price[s]*self.ave_monthly_returns[s] for s in self.stocks]) >= min_return)

            # Objective: minimize risk 
            for s1 in self.stocks:
                for s2 in self.stocks:
                    coeff = (self.covariance_matrix[s1][s2] * self.price[s1] * self.price[s2])
                    cqm.objective.add_quadratic(x[s1].variables[0], x[s2].variables[0], coeff)

            if self.verbose:
                print(f"\nCQM Formulation: mininimize risk s.t. return >= min_return\n")
        else:
            # Objective 1: minimize variance
            for s1 in self.stocks:
                for s2 in self.stocks:
                    coeff = (self.covariance_matrix[s1][s2] * self.price[s1] * self.price[s2])
                    cqm.objective.add_quadratic(x[s1].variables[0], x[s2].variables[0], self.alpha*coeff)
                            
            # Objective 2: maximize return
            for s in self.stocks:
                cqm.objective.add_linear(x[s].variables[0], - self.price[s] * self.ave_monthly_returns[s])

            if self.verbose:
                print(f"\nCQM Formulation: maximize return - alpha*risk\n")

        cqm.substitute_self_loops()

        self.model['CQM'] = cqm 

    def solve_cqm(self, max_risk=None, min_return=None):
        """Solve CQM.  
        This method allows the user a choice of 3 problem formulations: 
            1) max return - alpha*risk (default formulation)
            2) max return s.t. risk <= max_risk 
            3) min risk s.t. return >= min_return  

        Args:
            max_risk (int): 
                Maximum risk for the risk bounding problem formulation.
            min_return (int): 
                Minimum return for the return bounding problem formulation.

        Returns:
            solution (dict):
                This is a dictionary that saves solutions in desired format 
                e.g., solution = {'CQM':{'stocks': {'IBM': 3, 'WMT': 12}, 'risk': 10, 'return': 20}, 
                                  'DQM':{'stocks': {'APL': 5, 'IBM': 24}, 'risk': 15, 'return': 43}}                   
        """
        self.build_cqm(max_risk, min_return)

        self.sample_set['CQM'] = self.sampler['CQM'].sample_cqm(self.model['CQM'])

        best_feasible = next(itertools.filterfalse(lambda d: not getattr(d,'is_feasible'), 
                                                   list(self.sample_set['CQM'].data())))

        feasible_records = [rec for rec in self.sample_set['CQM'].record if rec.is_feasible]
        n_samples = len(self.sample_set['CQM'].record)

        solution = {}

        solution['stocks'] = {k:best_feasible.sample[k] for k in self.stocks}

        solution['return'], solution['risk'] = self.compute_risk_and_returns(solution['stocks'])

        if self.verbose:
            print(f'CQM -- Number of feasible solutions: {len(feasible_records)} out of {n_samples} sampled.')
            print(f'CQM -- Best energy: {self.sample_set["CQM"].first.energy}')
            print(f'CQM -- Best energy (feasible): {min([rec.energy for rec in feasible_records])}')  
            print(f'CQM -- Best feasible solution: {solution}\n')

        return solution 

    def build_dqm(self, alpha=None, gamma=None):
        """Build DQM.  

        Args:
            alpha (float): 
                Risk aversion coefficient.
            gamma (int): 
                Penalty coefficient for budgeting constraint.

        Returns:
            None.     
        """
        if not gamma:
            gamma = self.gamma

        if not alpha:
            alpha = self.alpha

        self.alpha = alpha 
        self.gamma = gamma

         # Defining DQM 
        dqm = DiscreteQuadraticModel() 

        # Build the DQM starting by adding variables
        for s in self.stocks:
            dqm.add_variable(len(self.shares_intervals[s]), label=s)

        # Objective 1: minimize variance
        for s1 in self.stocks:
            for s2 in self.stocks:
                coeff = (self.covariance_matrix[s1][s2]
                        * self.price[s1] * self.price[s2])
                if s1 == s2:
                    for k in range(dqm.num_cases(s1)):
                        num_s1 = self.shares_intervals[s1][k]
                        dqm.set_linear_case(
                                    s1, k, 
                                    dqm.get_linear_case(s1,k) 
                                    + alpha*coeff*num_s1*num_s1)
                else:
                    for k in range(dqm.num_cases(s1)):
                        for m in range(dqm.num_cases(s2)):
                            num_s1 = self.shares_intervals[s1][k]
                            num_s2 = self.shares_intervals[s2][m]

                            dqm.set_quadratic_case(
                                s1, k, s2, m, 
                                dqm.get_quadratic_case(s1,k,s2,m)
                                + coeff*alpha*num_s1*num_s2) 
                        
        # Objective 2: maximize return
        for s in self.stocks:
            for j in range(dqm.num_cases(s)):
                dqm.set_linear_case(
                                s, j, dqm.get_linear_case(s,j)
                                - self.shares_intervals[s][j]*self.price[s]
                                * self.ave_monthly_returns[s])

        # Scaling factor to guarantee that all coefficients are integral
        # needed in order to use add_linear_inequality_constraint method 
        factor = 10**self.precision

        min_budget = round(factor*0.99*self.budget)
        budget = int(self.budget)

        terms = [(s, j, int(self.shares_intervals[s][j]
                *factor*self.price[s])) 
                for s in self.stocks 
                for j in range(dqm.num_cases(s))]

        dqm.add_linear_inequality_constraint(terms, 
                                            constant=0, 
                                            lb=min_budget, 
                                            ub=factor*budget, 
                                            lagrange_multiplier=gamma, 
                                            label="budget")

        self.model['DQM'] = dqm 

    def solve_dqm(self):
        """Solve DQM.

        Returns:
            solution (dict):
                This is a dictionary that saves solutions in desired format 
                e.g., solution = {'CQM':{'stocks': {'IBM': 3, 'WMT': 12}, 'risk': 10, 'return': 20}, 
                                  'DQM':{'stocks': {'APL': 5, 'IBM': 24}, 'risk': 15, 'return': 43}}                    
        """
        if not self.model['DQM']:
            self.build_dqm()

        self.sample_set['DQM'] = self.sampler['DQM'].sample_dqm(self.model['DQM'])

        solution = {}

        solution['stocks'] = {s:self.shares_intervals[s][self.sample_set['DQM'].first.sample[s]] for s in self.stocks}
        
        solution['return'], solution['risk'] = self.compute_risk_and_returns(solution['stocks'])

        print(f'DQM -- solution for alpha == {self.alpha} and gamma == {self.gamma}: {solution}\n')

        return solution 

    def dqm_grid_search(self):
        """Execute parameter (alpha, gamma) grid search for DQM.
        """
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

                data_matrix[i,j] = solution['return'] / np.sqrt(solution['risk'])

        n_opt = np.argmax(data_matrix)

        self.alpha = alpha[n_opt%len(alpha)]
        self.gamma = gamma[n_opt//len(gamma)]

        print(f"DQM Grid Search Completed: alpha={self.alpha}, gamma={self.gamma}.-")

    def compute_risk_and_returns(self, solution):
        """Compute the risk and return values of solution.
        """
        variance = 0.0
        for s1 in solution:
            for s2 in solution:
                variance += (solution[s1] * self.price[s1] 
                            * solution[s2] * self.price[s2]  
                            * self.covariance_matrix[s1][s2])

        est_return = 0
        for stock in solution:
            est_return += solution[stock]*self.price[stock]*self.ave_monthly_returns[stock]

        return round(est_return, 2), round(variance, 2)

    def run(self, min_return=0, max_risk=0): 
        """Execute sequence of load_data --> build_model --> solve.
        """
        self.load_data()
        if self.model_type=='CQM': 
            print(f"\nCQM run...")
            self.solution['CQM'] = self.solve_cqm(min_return=min_return, max_risk=max_risk)
        else:
            print(f"\nDQM run...")
            if len(self.alpha_list) > 1 or len(self.gamma_list) > 1:
                print("\nStarting DQM Grid Search...")
                self.dqm_grid_search()

            self.build_dqm()
            self.solution['DQM'] = self.solve_dqm()
