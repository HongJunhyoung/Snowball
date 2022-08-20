from abc import ABCMeta, abstractmethod
from sqlite3 import Timestamp
import numpy as np
import pandas as pd
import math
from tqdm.auto import tqdm
from .report import calc_stats, report_log, report_perf

class Universe(object):
    def __init__(self, name, prices=None):
        self.name = name
        if prices is not None:
            _prices = prices.copy() 
            _prices.index = pd.to_datetime(_prices.index)
            pr = _prices.stack().rename('price')
            dr = _prices.pct_change().stack().rename('return')
            pricing = pd.concat([pr, dr], axis=1).sort_index()
            self._pricing = pricing
            self._calendar = self._build_calendar(pricing)
        else:
            self._pricing = None
            self._calendar = None
        self.blind_after = None

    def __repr__(self):
        return f'<UNIVERSE> {self.name}'

    @property
    def pricing(self):
        if self._pricing is None:
            return None
        else:
            return self._pricing.loc[:self.blind_after]

    @property
    def calendar(self):
        if self._calendar is None:
            return None
        else:
            return self._calendar.loc[:self.blind_after]

    def _build_calendar(self, pricing):
        bd = pricing.index.get_level_values(0).unique().sort_values()
        start, end = bd[0], bd[-1]
        bsd = bd.to_frame(name='TD')
        bsd['ND'] = bsd.shift(-1) # next business day
        bsd.loc[end, 'ND'] = bsd.loc[end, 'TD'] + pd.Timedelta(days=1) # fill NaN with next day
        bsd['EOD'] = True
        bsd['EOM'] = bsd.apply(lambda r: True if r['TD'].month != r['ND'].month else False, axis=1)
        bsd['EOQ'] = bsd.apply(lambda r: True if r['TD'].month != r['ND'].month and r['TD'].month in (3, 6, 9, 12) else False, axis=1)
        bsd['EOH'] = bsd.apply(lambda r: True if r['TD'].month != r['ND'].month and r['TD'].month in (6, 12) else False, axis=1)
        bsd['EOY'] = bsd.apply(lambda r: True if r['TD'].month != r['ND'].month and r['TD'].month == 12 else False, axis=1)
        cal = pd.date_range(start, end).to_frame(name='CD') # build calendar 
        cal = pd.concat([cal, bsd], axis=1).fillna(False)   # fill holidays with False
        cal = cal[['EOD', 'EOM', 'EOQ', 'EOH', 'EOY']]
        return cal

    def add_pricing(self, data):
        pricing = self.pricing.append(data).sort_index()
        pr = pricing['price'].unstack().fillna(method='ffill').stack().rename('price')
        dr = pricing['return'].unstack().fillna(0).stack().rename('return')
        self._pricing = pd.concat([pr, dr], axis=1)
        self._calendar = self._build_calendar(self._pricing)

    def set_blind_after(self, date):
        self.blind_after = date


class Scheduler(object):
    def __init__(self, calendar, rule_or_list='EOM'):
        self._business_days = calendar[calendar.EOD == 1]

        if isinstance(rule_or_list, list):
            self._rule = 'LIST'
            self._rebalance_dates = pd.DatetimeIndex(rule_or_list)
        elif isinstance(rule_or_list, pd.DatetimeIndex):
            self._rule = 'LIST'
            self._rebalance_dates = rule_or_list
        elif isinstance(rule_or_list, str):  # EOM, EOQ,..
            self._rule = rule_or_list
            stdday = self._rule[:3]
            offset = 0 if len(self._rule) == 3 else int(self._rule[3:])
            self._rebalance_dates = self._business_days[self._business_days.shift(offset)[stdday] == 1].index
        else:
            raise ValueError('Input must be a keyword or a list of date')

    def __repr__(self):
        return f'<Scheduler>\nRebalance Rule: {self._rule}\nRebalance Dates: {self._rebalance_dates}'

    @property
    def rebalance_dates(self):
        return self._rebalance_dates

    def business_days(self, start, end):
        return self._business_days.loc[start:end].index

    def is_rebalance_date(self, date, *args, **kwargs):
        if self.rebalance_dates is not None:   # periodic rebalancing
            return date in self._rebalance_dates
        else:                                  # threshold rebalancing
            # not implemented
            return False


class Fund(object):
    def __init__(self):
        self.is_initiated = False
        self._nav = 100
        self._weights = pd.Series(None, dtype=float).rename('weight')

    def __repr__(self):
        return f'NAV: {self.nav}\nWeights:\n{self.weights.to_string()}'

    @property
    def nav(self):
        return self._nav

    @property
    def weights(self):
        return self._weights

    def rebalance(self, weights):
        if weights is not None:
            self.is_initiated = True
            new_portfolio = weights.rename('weight')
            old_portfolio = self._weights
            _trades = new_portfolio.subtract(old_portfolio, fill_value=0).rename('trade')
            self._weights = new_portfolio
        else:
            _trades = None
        return _trades

    def update(self, date, pricing, logger):
        _pricing = pricing.xs(date)
        _portfolio_return = np.float64(0)
        _asset_returns = []
        for asset in self._weights.index:
            # Price does not exist in the universe: cash out
            try:
                _asset_return = _pricing.loc[asset, 'return']
            except KeyError:
                logger.write('No price data', date, f'{asset} : cash out')
                self._weights.drop(asset, inplace=True)
                continue
            # To check if there are abnormal data.
            if math.isnan(_asset_return):
                logger.write('Daily return is NaN', date, f'{asset} : changed to zero')
                _asset_return = 0
            if abs(_asset_return) > 0.30:
                logger.write('Large price change', date, f'{asset} : {_asset_return:.2%}')
            _asset_returns.append([asset, _asset_return])
            _portfolio_return += self._weights.loc[asset] * _asset_return
        _asset_returns = pd.DataFrame(_asset_returns, columns=['asset', 'return']).set_index('asset')
        self._nav *= (1 + _portfolio_return)
        self._weights = self._weights.mul((1 + _asset_returns['return']), axis=0) / (1 + _portfolio_return)
        self._weights = self._weights.rename('weight')
        return _portfolio_return


class Rule(metaclass=ABCMeta):
    @abstractmethod
    def calculate(self, date, universe, fund):
        pass


class BacktestLogger(object):
    def __init__(self):
        self._log = []

    def initialize(self):
        self._log = []

    def write(self, event, date, message):
        self._log.append([event, date, message])

    def finalize(self):
        self._log = pd.DataFrame(self._log, columns=['event', 'date', 'message'])


class Portfolio(object):
    def __init__(self, name, universe, schedule, rule, cost=0):
        self.name = name
        self.universe = universe
        self.scheduler = Scheduler(universe._calendar, schedule)
        self.rule = rule
        self.cost = cost

        self.gross_returns = pd.Series(None, dtype=float).rename('return')
        self.returns = None
        self.weights = None
        self.trades = None
        self.stats = None
        self._logger = BacktestLogger()

    def __repr__(self):
        return self.name

    @property
    def log(self):
        return self._logger._log

    def _record(self, date, returns, weights, trades):
        def _to_multiindex_with_date(dt, sr):
            df = sr.to_frame().reset_index()
            df.columns = ['asset'] + [sr.name]
            df['date'] = dt
            df.set_index(['date', 'asset'], inplace=True)
            return df[sr.name]
        self.gross_returns.loc[date] = returns
        self.weights = pd.concat([self.weights, _to_multiindex_with_date(date, weights)])
        if trades is not None:
            self.trades = pd.concat([self.trades, _to_multiindex_with_date(date, trades)])

    def _calc_net_returns(self):
        # compute and subtract transaction cost from the portfolio returns
        if self.trades is None:
            self.returns = self.gross_returns.copy()
        else:
            turnover = self.trades.abs().groupby(self.trades.index.get_level_values(0)).sum().astype(float)
            self.returns = self.gross_returns.sub(turnover * self.cost, fill_value=0)
            self.stats = None # initialize for re-adjustment

    def _evaluate(self):
        self.stats = calc_stats(self.returns, self.trades)
        return self.stats

    def update(self, item, target=None, value=None):
        if item == 'rebalance_date':
            self.scheduler._rebalance_dates = self.scheduler._rebalance_dates.map(
                lambda d: pd.Timestamp(value) if d == pd.Timestamp(target) else d
            )
        else:
            raise ValueError('Not defined')

    def report(self, start='1900-01-01', end='2099-01-01', benchmark=None, relative=False, charts='interactive'):
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
        rtns = self.returns
        g_rtns = self.gross_returns
        wgts = self.weights
        if benchmark is not None:
            bm = benchmark.pct_change()
            bm.index = pd.to_datetime(bm.index)
            bm = bm.reindex(rtns.index).fillna(0)
            t0 = rtns.index[0]
            if rtns.loc[t0] == 0:
                bm.loc[t0] = 0 # To be aligned with portfolio return
            if relative:
                rtns = rtns - bm
        else:
            bm = None
        trds = self.trades

        report_perf(rtns, g_rtns, trds, wgts, bm, charts)

    def backtest(self, start='1900-01-01', end='2099-12-31', initial_weights=None, verbose=True):
        # initialize for re-run
        self.returns = pd.Series(None, dtype=float).rename('return')
        self.weights = None
        self.trades = None
        self.stats = None
        self.universe.set_blind_after(None) 
        self._logger.initialize()

        fund = Fund()
        fund.rebalance(initial_weights)

        bar_format='{percentage:3.0f}% {bar} ({desc}) {n_fmt}/{total_fmt} | \
                    Elapsed {elapsed} | Remaining {remaining} | {rate_inv_fmt}'
        business_days_iterator = tqdm(self.scheduler.business_days(start, end),
                                      bar_format=bar_format,
                                      desc='DATE', disable=(not verbose))
        for td in business_days_iterator:
            business_days_iterator.desc = td.strftime('%Y-%m-%d')

            fund_return = fund.update(td, self.universe._pricing, self._logger)

            # Rebalance
            if self.scheduler.is_rebalance_date(td):
                self.universe.set_blind_after(td) # prevent look-ahead bias 방지
                weights = self.rule.calculate(td, self.universe, fund)
                trades = fund.rebalance(weights)
                if trades is not None:
                    self._logger.write('Rebalancing', td, f'{len(trades)} trades')
            else:
                trades = None

            if fund.is_initiated:
                self._record(td, fund_return, fund.weights, trades)

        business_days_iterator.close()

        self._calc_net_returns()
        self._evaluate()
        self._logger.finalize()

        if verbose:
            report_log(self._logger._log)

