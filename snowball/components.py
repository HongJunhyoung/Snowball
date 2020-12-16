import numpy as np
import pandas as pd
import math
from tqdm.auto import tqdm
from .optimizer import risk_budgeting
from .report import calc_stats

class Universe(object):
    def __init__(self, name=None, pricing=None, calendar=None):
        self._name = name
        self._pricing = pricing
        self._calendar = calendar

    def __repr__(self):
        return f'<UNIVERSE> {self._name}'

    @property
    def name(self):
        return self._name

    @property
    def pricing(self):
        if self._pricing is None:
            return pd.DataFrame(None)
        else:
            return self._pricing

    @property
    def calendar(self):
        return self._calendar

    def _build_calendar(self, pricing):
        bd = pricing.index.get_level_values(0).unique().sort_values()
        start, end = bd[0], bd[-1]
        bsd = bd.to_frame(name='TD')
        bsd['ND'] = bsd.shift(-1) # next business day
        bsd.loc[end, 'ND'] = bsd.loc[end, 'TD'] + pd.Timedelta(days=1) # fill NaN with next day
        bsd['BSD'] = True
        bsd['EOM'] = bsd.apply(lambda r: True if r['TD'].month != r['ND'].month else False, axis=1)
        bsd['EOQ'] = bsd.apply(lambda r: True if r['TD'].month != r['ND'].month and r['TD'].month in (3, 6, 9, 12) else False, axis=1)
        bsd['EOH'] = bsd.apply(lambda r: True if r['TD'].month != r['ND'].month and r['TD'].month in (6, 12) else False, axis=1)
        bsd['EOY'] = bsd.apply(lambda r: True if r['TD'].month != r['ND'].month and r['TD'].month == 12 else False, axis=1)
        cal = pd.date_range(start, end).to_frame(name='CD') # build calendar 
        cal = pd.concat([cal, bsd], axis=1).fillna(False)   # fill holidays with False
        cal = cal[['BSD', 'EOM', 'EOQ', 'EOH', 'EOY']]
        return cal

    def add_pricing(self, data):
        pricing = self.pricing.append(data).sort_index()
        pr = pricing['price'].unstack().fillna(method='ffill').stack().rename('price')
        dr = pricing['return'].unstack().fillna(0).stack().rename('return')
        self._pricing = pd.concat([pr, dr], axis=1)
        self._calendar = self._build_calendar(self._pricing)


class Scheduler(object):
    def __init__(self, calendar, rule_or_list='EOM'):
        self._business_days = calendar[calendar.BSD == 1]

        if isinstance(rule_or_list, list):
            self._rule = 'DIRECT'
            self._rebalance_dates = pd.DatetimeIndex(rule_or_list)
        elif isinstance(rule_or_list, pd.DatetimeIndex):
            self._rule = 'DIRECT'
            self._rebalance_dates = rule_or_list
        elif isinstance(rule_or_list, str):  # periodic rebalancing
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
        self._weights = pd.DataFrame(None, columns=['item', 'weight']).set_index('item')

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
            new_portfolio = weights['weight']
            old_portfolio = self._weights['weight']
            _trades = new_portfolio.subtract(old_portfolio, fill_value=0).to_frame(name='trade')
            self._weights = weights
        else:
            _trades = None
        return _trades

    def update(self, date, pricing, logger):
        _pricing = pricing.xs(date)
        _portfolio_return = np.float64(0)
        _item_returns = []
        for item in self._weights.index:
            # Price does not exist in the universe: cash out
            try:
                _item_return = _pricing.loc[item, 'return']
            except KeyError:
                logger.write('No price data', date, f'{item} : cash out')
                self._weights.drop(item, inplace=True)
                continue
            # To check if there are abnormal data.
            if math.isnan(_item_return):
                logger.write('Daily return is NaN', date, f'{item} : changed to zero')
                _item_return = 0
            if abs(_item_return) > 0.30:
                logger.write('Large price change', date, f'{item} : {_item_return:.2%}')
            _item_returns.append([item, _item_return])
            _portfolio_return += self._weights.loc[item, 'weight'] * _item_return
        _item_returns = pd.DataFrame(_item_returns, columns=['item', 'return']).set_index('item')
        self._nav *= (1 + _portfolio_return)
        self._weights = self._weights.mul((1 + _item_returns['return']), axis=0) / (1 + _portfolio_return)
        return _portfolio_return


class Constructor(object):
    def __init__(self, method, params):
        self._method = method
        self._params = params # TODO: params..?

        # Validation
        if self._method == 'mix':
            total_weight = np.sum([val for _, val in self._params.items()])
            if not math.isclose(total_weight, 1):  # to avoid floating point error
                raise ValueError('Sum of the weight must be 1.')
        elif self._method == 'risk_budgeting':
            total_risk = np.sum([val for _, val in self._params.items()])
            if not math.isclose(total_risk, 1):  # to avoid floating point error
                raise ValueError('Sum of the risk budget must be 1.')

    def calculate(self, date, universe, *args, **kwargs):
        if self._method == 'mix':
            weights = pd.DataFrame.from_dict(self._params, orient='index', columns=['weight']).rename_axis(index='item')
        elif self._method == 'risk_budgeting':
            returns_1y = universe.pricing['return'].unstack()[:date]
            returns_1y = returns_1y[-252:]
            returns = []
            budget = []
            for key, val in self._params.items():
                returns.append(returns_1y[key])
                budget.append(val)
            covmat = pd.concat(returns, axis=1).cov()
            weights = risk_budgeting(covmat.values, budget) 
            weights = pd.DataFrame(weights, index=covmat.columns, columns=['weight']).rename_axis(index='item')
        elif self._method == 'func':
            weights = self._params(date, universe, *args, **kwargs)
        else:
            raise ValueError('Not supported method')
        return weights


class BacktestResult(object):
    def __init__(self):
        self.returns = pd.DataFrame(None, columns=['date', 'return']).set_index('date')
        self.weights = None
        self.trades = None
        self.stats = None

    def add(self, date, returns, weights, trades):
        def _to_multiindex_with_date(dt, df):
            ddf = df.copy()
            ddf['date'] = dt
            old_index = df.index.name
            new_index = ['date', old_index]
            ddf.reset_index(inplace=True)
            ddf.set_index(new_index, inplace=True)
            return ddf
        self.returns.loc[date] = returns
        self.weights = pd.concat([self.weights, _to_multiindex_with_date(date, weights)])
        if trades is not None:
            self.trades = pd.concat([self.trades, _to_multiindex_with_date(date, trades)])

    def finalize(self):
        rtns = self.returns['return']
        self.stats = calc_stats(rtns)


class BacktestLogger(object):
    def __init__(self):
        self._log = []

    def write(self, event, date, message):
        self._log.append([event, date, message])

    def finalize(self):
        self._log = pd.DataFrame(self._log, columns=['event', 'date', 'message'])


class Backtester(object):
    def __init__(self, universe, scheduler, constructor):
        self._universe = universe
        self._scheduler = scheduler
        self._constructor = constructor
        self._backtest_result = BacktestResult()
        self._backtest_logger = BacktestLogger()

    def __repr__(self):
        desc = f'Backtester'
        return desc

    @property
    def universe(self):
        return self._universe

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def constructor(self):
        return self._constructor

    @property
    def result(self):
        return self._backtest_result

    @property
    def log(self):
        return self._backtest_logger._log

    @universe.setter
    def universe(self, universe):
        self._universe = universe

    @scheduler.setter
    def scheduler(self, scheduler):
        self._scheduler = scheduler

    @constructor.setter
    def constructor(self, constructor):
        self._constructor = constructor

    def run(self, start='1900-01-01', end='2099-12-31', initial_portfolio=None, verbose=True):
        fund = Fund()
        fund.rebalance(initial_portfolio)

        bar_format='{percentage:3.0f}% {bar} ({desc}) {n_fmt}/{total_fmt} | \
                    Elapsed {elapsed} | Remaining {remaining} | {rate_inv_fmt}'
        business_days_iterator = tqdm(self._scheduler.business_days(start, end),
                                      bar_format=bar_format,
                                      desc='DATE', disable=(not verbose))
        for td in business_days_iterator:
            business_days_iterator.desc = td.strftime('%Y-%m-%d')

            # Calculate portfolio's daily return and adjust weights based on items' daily return.
            fund_return = fund.update(td, self._universe.pricing, self._backtest_logger)

            # Rebalance
            if self._scheduler.is_rebalance_date(td):
                weights = self.constructor.calculate(date=td, universe=self._universe)
                trades = fund.rebalance(weights)
                self._backtest_logger.write('Rebalancing', td, f'{len(trades)} trades')
            else:
                trades = None

            if fund.is_initiated:
                self._backtest_result.add(td, fund_return, fund.weights, trades)
        business_days_iterator.close()

        self._backtest_result.finalize()
        self._backtest_logger.finalize()
