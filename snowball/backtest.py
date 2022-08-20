import pandas as pd
from .components import Universe, Rule, Portfolio
from .rules import EqualWeight, RiskParity, ConstantWeight


def run_backtest(prices, schedule, rule, cost=0, start='1900-01-01', end='2099-12-31', verbose=True):
    '''
    Run backtest.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price data of the assets.
    schedule : string('EOM', 'EOQ',...) or list
    rule : string('EqualWeight', 'RiskParity') or dictionary or class instance
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
    elif issubclass(type(rule), Rule):
        pass
    else:
        raise ValueError('rule is not valid.')

    portfolio = Portfolio('Portfolio', universe, schedule, rule, cost)
    portfolio.backtest(start=start, end=end, verbose=verbose)
    return portfolio