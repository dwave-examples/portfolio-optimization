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

import pandas as pd 
import numpy as np 
import matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

from single_period import SinglePeriod 

class MultiPeriod(SinglePeriod):
    """Solve the multi-period (dynamic) portfolio optimization problem.
    """

    def run(self, max_risk=0, min_return=0): 
        """Solve the rebalancing portfolio optimization problem.
        """
        if not self.dates:
            self.dates = ['2010-01-01', '2012-12-31']
        self.load_data()

        num_months = len(self.df_all)
        first_purchase = True
        result = {}
        baseline_result = {}
        self.baseline_values = [0]
        self.update_values = [0]
        months = []
        precision = 2 

        # Define dataframe to save output data 
        headers = ['Date', 'Budget', 'Spending'] + self.stocks + ['Variance', 'Returns']
        self.opt_results_df = pd.DataFrame(columns = headers)
        row = []

        self.price_df = pd.DataFrame(columns = self.stocks)

        # Initialize the plot
        plt.ylim(ymax = 1.0*self.budget, ymin = -1.0*self.budget)
        plt.xticks(list(range(0, num_months, 2)), 
                self.df_baseline.index.strftime('%b')[::2], rotation='vertical')
        plt.locator_params(axis='x', nbins=num_months/2)
        plt.plot(list(range(0, num_months)), [0]*(num_months), 
                color='red', label="Break even", linewidth=0.5)

        for i in range(3, num_months):

            # Look at just the data up to the current month
            df = self.df_all.iloc[0:i+1,:].copy()
            baseline_df_current = self.df_baseline.iloc[0:i+1,:]
            print("\nDate:", df.last_valid_index())
            months.append(df.last_valid_index().date()) 

            if first_purchase:
                budget = self.budget
                baseline_shares = (budget / baseline_df_current.iloc[-1])
                baseline_result = {self.baseline[0]: baseline_shares} 
            else:
                # Compute profit of current portfolio
                budget = sum([df.iloc[-1][s]*result['stocks'][s] for s in self.stocks]) 
                self.update_values.append(budget - spending)

                # Compute profit of fund portfolio
                fund_value = sum([baseline_df_current.iloc[-1][s]*baseline_result[s] 
                                 for s in self.baseline]) 
                self.baseline_values.append(fund_value - self.budget)

            self.budget = budget 
            self.load_data(df=df)

            self.price_df.loc[i-2] = list(self.price.values)

            # Output for user on command-line and plot
            update_values = np.array(self.update_values, dtype=object)
            baseline_values = np.array(self.baseline_values, dtype=object)
            plt.plot(range(3, i+1), update_values, 
                    color='blue', label="Updated portfolio")
            plt.plot(range(3, i+1), baseline_values, 
                    color='gray', label="Fund portfolio", linewidth=0.5)
            
            if first_purchase:
                plt.legend(loc="lower left")
                plt.title("Start: {start}, End: {end}".format\
                        (start=self.df_all.first_valid_index().date(), 
                         end=self.df_all.last_valid_index().date()))

            plt.savefig("portfolio.png")
            plt.pause(0.05)
            
            # Making solve run
            if self.model_type == 'DQM':
                print(f"\nMulti-Period DQM Run...")

                self.build_dqm()
                self.solution['DQM'] = self.solve_dqm()
                result = self.solution['DQM']
            else:
                print(f"\nMulti-Period CQM Run...")

                self.solution['CQM'] = self.solve_cqm(max_risk=max_risk, min_return=min_return)
                result = self.solution['CQM']

            # Print results to command-line
            spending = sum([self.price[s]*result['stocks'][s] for s in self.stocks])
            returns = result['return']
            variance = result['risk']     

            row = [months[-1].strftime('%Y-%m-%d'), budget, spending] + \
                [result['stocks'][s] for s in self.stocks] + \
                [variance, returns] 
            self.opt_results_df.loc[i-2] = row 
                
            first_purchase = False

        print(self.opt_results_df)
        print(f'\nRun completed.\n')
        print(f'\nPlease close the plot dialog to return to the terminal.\n')

        plt.savefig("portfolio.png")
        plt.show()
        