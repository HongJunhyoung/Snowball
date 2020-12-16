import pandas as pd
from .components import Universe, Scheduler, Constructor, Backtester
from .report import log_report, perf_report

class Portfolio(object):
    def __init__(self, universe, scheduler, constructor):
        self._backtester = Backtester(universe, scheduler, constructor)

    @property
    def log(self):
        return self._backtester.log

    @property
    def returns(self):
        return self._backtester.result.returns

    @property
    def weights(self):
        return self._backtester.result.weights

    @property
    def trades(self):
        return self._backtester.result.trades

    @property
    def stats(self):
        return self._backtester.result.stats

    def report(self, start=None, end=None, benchmark=None, relative=False):
        '''
        Report performance metrics and charts.

        Parameters
        ----------
        start : string 'YYYY-MM-DD' or datetime
        end : string 'YYYY-MM-DD' or datetime
        benchamrk : pd.Series
            daily index value or price, not daily return
        relative : boolean
            If True, excess returns will be analyzed.
        '''
        rtns = self._backtester.result.returns['return']
        wgts = self._backtester.result.weights['weight']
        if benchmark is not None:
            bm = benchmark.pct_change()
            bm.index = pd.to_datetime(bm.index, utc=True)
            bm = bm.reindex(rtns.index).fillna(0)
            t0 = rtns.index[0]
            if rtns.loc[t0] == 0:
                bm.loc[t0] = 0 # To align with portfolio return
            if relative:
                rtns = rtns - bm
        else:
            bm = None
        trds = self._backtester.result.trades['trade']

        perf_report(rtns, trds, wgts, bm)

# Main Function
def backtest(prices, schedule, weight=None, risk_budget=None, start='1900-01-01', end='2099-12-31', verbose=True):
    '''
    Run backtest.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily prices/index values of the assets.
    schedule : string or list
    weight : dictionary
    risk_budget : dictionary
    start : string 'YYYY-MM-DD' or datetime
    end : string 'YYYY-MM-DD' or datetime
    verbose : boolean
    benchamrk : pd.Series
        daily index value or price, not daily return
    verbose : boolean

    Returns
    -------
    portfolio : Portfolio object
        Contains the universe data, backtest policies and the backtest result.
    '''
    # input check
    if (weight is None and risk_budget is None) or (weight is not None and risk_budget is not None):
        raise ValueError('You should decide rebalancing rule. weigt/risk_budget')

    prices.index = pd.to_datetime(prices.index, utc=True)
    pr = prices.stack().rename('price')
    dr = prices.pct_change().stack().rename('return')
    pricing = pd.concat([pr, dr], axis=1)
    universe = Universe()
    universe.add_pricing(pricing)
    scheduler = Scheduler(universe.calendar, schedule)
    if weight is not None:
        constructor = Constructor(method='mix', params=weight)
    else:
        constructor = Constructor(method='risk_budgeting', params=risk_budget)
    portfolio = Portfolio(universe, scheduler, constructor)
    portfolio._backtester.run(start=start, end=end, initial_portfolio=None, verbose=verbose)
    if verbose:
        log_report(portfolio.log)
    return portfolio