from abc import ABCMeta, abstractmethod
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
        cal = pd.concat([cal, bsd], axis=1)
        cal = cal[['EOD', 'EOM', 'EOQ', 'EOH', 'EOY']].astype('boolean').fillna(False)   # fill holidays with False
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


class DailyReturns(object):
    """Daily asset returns unstacked into a (date x asset) array for fast lookups.

    `has_price` marks whether the asset has a row in the pricing data on that date.
    A missing row means the asset is not tradable (cash out); a row with a NaN return
    is kept and later treated as a zero return.
    """
    def __init__(self, pricing):
        returns = pricing['return'].unstack()
        has_price = pd.Series(True, index=pricing.index).unstack(fill_value=False)
        has_price = has_price.reindex(index=returns.index, columns=returns.columns, fill_value=False)
        self._returns = returns.to_numpy(dtype=float)
        self._has_price = has_price.to_numpy(dtype=bool)
        self._dates = returns.index
        self._col = {a: j for j, a in enumerate(returns.columns)}

    def row(self, date):
        i = self._dates.get_loc(date)
        return self._returns[i], self._has_price[i]

    def col(self, asset):
        return self._col.get(asset)


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
        """Apply the daily asset returns of `date` to the fund.

        pricing : DailyReturns or pricing DataFrame (only the requested date is converted)
        """
        if not isinstance(pricing, DailyReturns):
            pricing = DailyReturns(pricing.loc[[date]])
        row_returns, row_has_price = pricing.row(date)
        assets = self._weights.index
        weights = self._weights.to_numpy(dtype=float)
        kept, kept_returns = [], []
        _portfolio_return = np.float64(0)
        for k, asset in enumerate(assets):
            j = pricing.col(asset)
            # Price does not exist in the universe: cash out
            if j is None or not row_has_price[j]:
                logger.write('No price data', date, f'{asset} : cash out')
                continue
            _asset_return = row_returns[j]
            # To check if there are abnormal data.
            if math.isnan(_asset_return):
                logger.write('Daily return is NaN', date, f'{asset} : changed to zero')
                _asset_return = 0.0
            if abs(_asset_return) > 0.30:
                logger.write('Large price change', date, f'{asset} : {_asset_return:.2%}')
            kept.append(k)
            kept_returns.append(_asset_return)
            _portfolio_return += weights[k] * _asset_return
        self._nav *= (1 + _portfolio_return)
        new_weights = weights[kept] * (1 + np.asarray(kept_returns, dtype=float)) / (1 + _portfolio_return)
        index = assets[kept]
        if index.name != 'asset':
            index = index.rename(None)
        self._weights = pd.Series(new_weights, index=index, name='weight')
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
        self._records = None

    def __repr__(self):
        return self.name

    @property
    def log(self):
        return self._logger._log

    def _record(self, date, returns, weights, trades):
        # Collected in lists and assembled once in _finalize_records(); building pandas objects
        # every day makes the backtest quadratic in the number of days.
        rec = self._records
        rec['dates'].append(date)
        rec['returns'].append(returns)
        rec['weights'].append((date, weights.index, weights.to_numpy(copy=True)))
        if trades is not None:
            rec['trades'].append((date, trades.index, trades.to_numpy(copy=True)))

    @staticmethod
    def _stack_records(chunks, name):
        """[(date, assets, values), ...] -> Series indexed by (date, asset)."""
        if not chunks:
            return None
        lengths = [len(assets) for _, assets, _ in chunks]
        dates = pd.DatetimeIndex([d for d, _, _ in chunks]).repeat(lengths)
        assets = np.concatenate([np.asarray(a, dtype=object) for _, a, _ in chunks])
        values = np.concatenate([v for _, _, v in chunks])
        index = pd.MultiIndex.from_arrays([dates, assets], names=['date', 'asset'])
        return pd.Series(values, index=index, name=name)

    def _finalize_records(self):
        rec = self._records
        new = pd.Series(rec['returns'], index=pd.DatetimeIndex(rec['dates']), dtype=float, name='return')
        old = self.gross_returns
        if len(old) == 0:
            self.gross_returns = new
        else:  # re-run: overwrite overlapping dates, append the rest
            updated = old.copy()
            common = new.index.intersection(old.index)
            updated.loc[common] = new.loc[common]
            self.gross_returns = pd.concat([updated, new[~new.index.isin(old.index)]]).rename('return')
        self.weights = self._stack_records(rec['weights'], 'weight')
        self.trades = self._stack_records(rec['trades'], 'trade')
        self._records = None

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
        self._records = {'dates': [], 'returns': [], 'weights': [], 'trades': []}
        daily_returns = DailyReturns(self.universe._pricing)

        fund = Fund()
        fund.rebalance(initial_weights)

        bar_format='{percentage:3.0f}% {bar} ({desc}) {n_fmt}/{total_fmt} | \
                    Elapsed {elapsed} | Remaining {remaining} | {rate_inv_fmt}'
        business_days_iterator = tqdm(self.scheduler.business_days(start, end),
                                      bar_format=bar_format,
                                      desc='DATE', disable=(not verbose))
        for td in business_days_iterator:
            business_days_iterator.desc = td.strftime('%Y-%m-%d')

            fund_return = fund.update(td, daily_returns, self._logger)

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

        self._finalize_records()
        self._calc_net_returns()
        self._evaluate()
        self._logger.finalize()

        if verbose:
            report_log(self._logger._log)
