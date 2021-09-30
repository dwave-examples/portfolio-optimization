# Portfolio Optimization

Optimizing a portfolio of stocks is a challenging problem that looks to identify the optimal number of shares of each stock to purchase in order to minimize risk (variance) and maximize returns, while staying under some specified spending budget.

## Problem Definition and Mathematical Formulation
Consider a set of $n$ types of stocks to choose from, with an average monthly return per dollar spent of `r_i` for each stock `i`. Furthermore, let `sigma_ij` be the covariance of the returns of stocks $i$ and $j$. For a spending budget of `B` dollars, let `x_i` denote the number of shares of stock `i` purchased at price `p_i` per share. Then, this portfolio optimization problem can be represented as 

![Model Formulation](readme_imgs/equation.png)

Here, `0 < gamma < 1` is a tunable parameter that helps weigh the importance of each of the two objectives (variance and returns). Notice that while we are minimizing the variance, we are also minimizing the negative of the returns (which is equivalent to maximizing the returns). 


## Running the demos

There are two main demos included in this repository. For each demo, two main modeling approaches are showcased for the formulation of the portfolio problem:
- Discrete Quadratic Modeling (DQM)
- Constrained Quadratic Modeling (CQM)

### Single-Period Demo

#### CQM Runs 

The single-period demo determines the optimal number of shares to purchase from 3 stocks based on the historical price data provided. To run the demo, type:

```python portfolio.py```

This runs the single-period portfolio optimization problem, as formulated above, using default data stored in `basic_data.csv`, builds the constrained quadratic model (CQM), and runs the CQM on D-Wave's hybrid solver. The output of the run is printed to the console as follows.

```bash
Single period portfolio optimization run...

Max shares we can afford with a budget of 1000
IBM       8
WMT      18
SEHI    555

CQM run...

CQM Formulation: maximize returns - alpha * risk

CQM -- Number of feasible solutions: 24 out of 48 sampled.
CQM -- Best energy: -96.47695257089694
CQM -- Best energy (feasible): -41.65133986113378
CQM -- Best feasible solution: {'stocks': {'IBM': 2.0, 'SEHI': 427.0, 'WMT': 0.0}, 'return': 62.67, 'risk': 42035.77}
```

##### CQM Bounding Formulations 

The demo allows the user to choose among two additional CQM formulations for the portfolio optimization problem: 
- The risk-bounding formulation solves the problem `maximize returns s.t. risk <= max_risk`
- The return-bounding formulation solves the problem `minimize risk s.t. returns >= min_return`

To run the single-period demo with the CQM risk bounding formulation, type:

`python portfolio.py -m 'CQM' --max-risk 10000`

```bash
Single period portfolio optimization run...

Max shares we can afford with a budget of 1000
IBM       8
WMT      18
SEHI    555

CQM run...

CQM Formulation: maximize return s.t. risk <= max_risk

CQM -- Number of feasible solutions: 27 out of 46 sampled.
CQM -- Best energy: -98.13313059630488
CQM -- Best energy (feasible): -33.29878623848638
CQM -- Best feasible solution: {'stocks': {'IBM': 3.0, 'SEHI': 183.0, 'WMT': 0.0}, 'return': 33.3, 'risk': 9993.17}
```

We can do similarly for the return-bounding formulation: 

`python portfolio.py -m 'CQM' --min-return 20`

```bash
Single period portfolio optimization run...

Max shares we can afford with a budget of 1000
IBM       8
WMT      18
SEHI    555

CQM run...

CQM Formulation: mininimize risk s.t. return >= min_return

CQM -- Number of feasible solutions: 31 out of 50 sampled.
CQM -- Best energy: -1.6764367671839864e-13
CQM -- Best energy (feasible): 3612.171397072142
CQM -- Best feasible solution: {'stocks': {'IBM': 2.0, 'SEHI': 106.0, 'WMT': 0.0}, 'return': 20.08, 'risk': 3612.17}
```

#### DQM Runs 

The user can select to build a disctrete quadratic model (DQM) instead with the following command:

```python portfolio.py -m 'DQM'```

This buils a DQM for the single-period portfolio optimization problem and solves it on the D-Wave's hybrid solver. The DQM uses a binning approach where the range of the shares of each stock is divied into equally-spaced intervals. 

The output of the default DQM run is printed on the console as follows. 

```bash 
Single period portfolio optimization run...

Max shares we can afford with a budget of 1000
IBM       8
WMT      18
SEHI    555

DQM run...
DQM -- solution with alpha == 0.0005 and gamma == 10: {'stocks': {'IBM': 1, 'SEHI': 333, 'WMT': 5}, 'return': 49.42, 'risk': 27437.82}
```

For the DQM single-period problem formulation, this demo gives the user the option to run a grid search on the objective parameters, `alpha` and `gamma`. Note that `alpha` is the risk coefficient introduced in the original formulatio of the problem whereas `gamma` is a penalty coefficient used in DQM to enforced the budget inequality. 

The user can opt to run a grid search with DQM by providing a list of values for `alpha` (here `[0.05, 0.005`),and `gamma` (here `[10, 100]`) as follows: 

``python portfolio.py  -m 'DQM' -a 0.5 -a 0.0005 -g 3 -g 100``

The output is printed to the console at each stage in the grid search, as shown below.
```bash
Single period portfolio optimization run...

Max shares we can afford with a budget of 1000
IBM       8
WMT      18
SEHI    555

DQM run...

Starting DQM Grid Search...

Grid search results:
DQM -- solution for alpha == 0.5 and gamma == 3: {'stocks': {'IBM': 0, 'SEHI': 278, 'WMT': 9}, 'return': 40.9, 'risk': 21204.73}

DQM -- solution for alpha == 0.5 and gamma == 100: {'stocks': {'IBM': 6, 'SEHI': 166, 'WMT': 0}, 'return': 40.06, 'risk': 15641.24}

DQM -- solution for alpha == 0.0005 and gamma == 3: {'stocks': {'IBM': 1, 'SEHI': 333, 'WMT': 5}, 'return': 49.42, 'risk': 27437.82}

DQM -- solution for alpha == 0.0005 and gamma == 100: {'stocks': {'IBM': 6, 'SEHI': 166, 'WMT': 0}, 'return': 40.06, 'risk': 15641.24}

DQM Grid Search Completed: alpha=0.0005, gamma=3.-
DQM -- solution for alpha == 0.0005 and gamma == 3: {'stocks': {'IBM': 2, 'SEHI': 333, 'WMT': 3}, 'return': 51.54, 'risk': 27454.44}
```

### Multiple-Period Demo

The multiple-period demo provides an analysis of portfolio value over time: rebalancing the portfolio at each time period until the target time. To run the demo with default data and settings, type:

`python portfolio.py -r`

The program will pull historical data on four stocks from Yahoo finance over a period of 3 years. For each month in the year, the optimal portfolio is computed and the funds are reinvested. A plot shows the trend over time of the portfolio value, as compared to a fund portfolio (investing all of the budget into a fund, such as S&P500).

Note that this demo will take longer to run and will make many calls to the hybrid solvers, as it is running our DQM each month over a 3-year period, resulting in optimizing over 30 different portfolios.

At the end of the analysis, the plot is saved as a .png file, as shown below.

![Example Plot](readme_imgs/portfolio.png)

## Problem Set Up

### Variance

The variance is the product of amount spent within each of two stocks multiplied by the covariance of the two stocks, summed over all pairs of stocks (with repetition).

`sum(stocks i and j) cov(i,j)*purchase(i)*purchase(j)`

### Returns

To compute the returns for a given purchase profile, we add up the expected returns for each stock: amount spent on the stock times the average monthly return of the stock.

`sum(stock i) return(i)*purchase(i)`

### Budget

We cannot spend more than the budget allocated. This is a simple `<=` constraint (natively enforced in CQM) This requirement can also be modeled as a double inequality with a lower bound on the expenditures (as we do in the case of DQM): 
`minimum expenses <= expenses <= budget`
Here, the quantity `minimum expenses` can be selected to be a relatively close fraction of the budget (e.g., 90% of the budget). This encoding of the budget requirement has practical implications in that it allows the investor to specify a minium amount to spend. 

## Code 

### Options 

The `portfolio.py` program can be called with these additional options:
- -s, --stocks: list of stocks for the problem 
- -b, --budget: problem budget 
- -n, --bin-size: bin size for dqm binning 
- -f, --file-path: full path of file with csv stock data 
- -z, --baseline: baseline stock for dynamic portfolio optimization run 
- -d, --dates: list of [start_date, end_date] for multi period/rebalancing portfolio optimization problem 
- -v, --verbose: to turn on or off additional code output 

### DQM

#### Variables and Cases

A constant is defined called `bin_size`. This constant states how many different purchase quantity options every stock will have. All variables will have a maximum number of cases less than or equal to `bin_size`, and they are evenly spaced out in the interval from 0 to the maximum number of shares possible to purchase given the price and budget.

#### Grid Search

For DQM, the bugeting constraint is added as a component of the objective function expression with an associated penalty coefficient $\gamma$. The code is formatted to allow for a grid search to obtain the optimal values of the penalty coefficients $\gamma$ and the risk-tendency coefficient $\alpha$. 

### CQM 

#### Bounding Formulation 

The risk-bouding formulation fits naturally with CQM since it involves adding a quadratic constraint to the problem. 

## References

https://www2.isye.gatech.edu/~sahmed/isye6669/notes/portfolio.pdf
