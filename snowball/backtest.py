import pandas as pd
from .components import Universe, Rule, Portfolio
from .rules import EqualWeight, RiskParity, ConstantWeight
from .report import log_report, perf_report


def run_backtest(prices, schedule, rule=None, start='1900-01-01', end='2099-12-31', verbose=True):
    '''Run backtest.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily prices/index values of the assets.
    schedule : string('EOM', 'EOQ',...) or list of date
    rule : string('EqualWeight', 'RiskParity') or dictionary or subclass of Rule
    start : string 'YYYY-MM-DD' or datetime
    end : string 'YYYY-MM-DD' or datetime
    verbose : boolean

    Returns
    -------
    portfolio : Portfolio object
        Contains the universe data, backtest policies and the backtest result.
    '''

    universe = Universe('Universe', prices)

    if rule == 'EqualWeight':
        rule = EqualWeight(prices.columns)
    elif rule == 'RiskParity':
        rule = RiskParity(prices.columns)
    elif isinstance(rule, dict):
        rule = ConstantWeight(rule)
    # elif issubclass(rule, Rule):
    #     rule = rule
    # else:
    #     raise ValueError('rule is not valid.')

    portfolio = Portfolio('Portfolio', universe, schedule, rule)
    portfolio.backtest(start=start, end=end, verbose=verbose)
    return portfolio