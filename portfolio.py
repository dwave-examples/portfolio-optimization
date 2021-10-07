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

import click 

from multi_period import MultiPeriod
from single_period import SinglePeriod

@click.command()
@click.option('-s', '--stocks', default=['IBM', 'SEHI', 'WMT'], multiple=True,
              show_default=True, help='Stock name to be included')
@click.option('-b', '--budget', default=1000, show_default=True,
              help='portfolio budget')
@click.option('-n', '--bin-size', default=10, show_default=True,
              help='Maximum number of intervals for each stock')
@click.option('-g', '--gamma', default=[10], multiple=True, show_default=True,
               help='Penalty coefficient for budget constraint')
@click.option('-a', '--alpha', default=[0.0005], multiple=True, show_default=True,
              help='Risk aversion coefficient')
@click.option('-f', '--file-path', default='data/basic_data.csv', show_default=True,
              help='Full path of csv file containing input stock data')
@click.option('-z', '--baseline', default='^GSPC', show_default=True,
              help='Basiline stock for comparison in multi-period run')
@click.option('-u', '--max-risk', default=0, help='Upper bound on risk/variance')
@click.option('-l', '--min-return', default=0, help='Lower bound on the returns')
@click.option('-d', '--dates', default=(['2010-01-01', '2012-12-31']), 
              nargs=2, type=click.Tuple([str, str]), show_default=True,
              help='Start and end date to query stock data from Yahoo! Finance')
@click.option('-m', '--model-type', default='CQM', multiple=False, 
              type=click.Choice(['CQM', 'DQM'], case_sensitive=False),
              show_default=True, help='Model type, cqm or dqm')
@click.option('-r', '--rebalance', is_flag=True, default=False,
              help='Make a multi-period rebalancing portfolio optimization run; '
                   'otherwise, make a single-period portfolio optimization run')
@click.option('-p', '--params', default="{}", 
              help='Pass sampler arguments such as profile and solver; '
                    'usage: -p \'{"profile": "test"}\'')
@click.option('-v', '--verbose', is_flag=True, 
              help='Enable additional program console output')
def main(stocks, budget, bin_size, gamma, params,  
         file_path, max_risk, min_return,baseline,
         dates, model_type, rebalance, alpha, verbose):
    if max_risk or min_return:
        model_type = 'CQM'

    if rebalance:
        print(f"\nRebalancing portfolio optimization run...")
        if 'SEHI' in stocks:
            stocks = ['AAPL', 'MSFT', 'AAL', 'WMT']
        my_portfolio = MultiPeriod(stocks=stocks, budget=budget, 
                                    sampler_args=params, 
                                    bin_size=bin_size, dates=dates,
                                    file_path=file_path, gamma=gamma,
                                    model_type=model_type, alpha=alpha, 
                                    verbose=verbose, baseline=baseline)
    else:
        print(f"\nSingle period portfolio optimization run...")
        my_portfolio = SinglePeriod(stocks=list(stocks), budget=budget,
                                    bin_size=bin_size, gamma=gamma, 
                                    file_path=file_path, dates=[], 
                                    model_type=model_type, alpha=alpha, 
                                    verbose=verbose, sampler_args=params)
    
    my_portfolio.run(min_return=min_return, max_risk=max_risk)

if __name__ == '__main__':
    main()