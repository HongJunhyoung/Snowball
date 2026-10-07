import os
import types

import numpy as np
import pandas as pd
import pytest

import snowball as sb
from snowball.components import BacktestLogger, DailyReturns, Fund
from snowball.rules import ledoit_wolf

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(TEST_DIR, 'data', 'etfs_prices.csv')


@pytest.fixture(scope='module')
def sample_prices():
    print('\n--- Fixture: Starting sample_prices load ---')
    prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    prices.index.name = 'Date'
    prices = prices.rename_axis(None, axis=1)
    print('--- Fixture: sample_prices load complete ---')
    return prices


def test_backtest(sample_prices):
    print('--- Test: test_backtest running ---')
    bt = sb.run_backtest(
        prices=sample_prices,
        schedule='EOM',
        rule={'069500': 0.6, '114820': 0.4},
        cost=0.002,
        start='2020-01-01',
        end='2024-12-31',
    )
    actual_value = bt.stats['CAGR']
    expected_value = -0.03308272002801749
    assert actual_value == pytest.approx(expected_value, abs=1e-10), (
        f'Backtest result is not {expected_value}. Actual result: {actual_value}'
    )
    print('--- Test: test_backtest completed ---')


def test_calc_stats(sample_prices):
    print('--- Test: test_calc_stats running ---')
    dummy_returns = sample_prices['069500'].pct_change().iloc[1:]
    stats = sb.calc_stats(dummy_returns)
    actual_value = stats['CAGR']
    expected_value = -0.1231127835849668
    assert actual_value == pytest.approx(expected_value, abs=1e-10), (
        f'Backtest result is not {expected_value}. Actual result: {actual_value}'
    )
    print('--- Test: test_calc_stats completed ---')


@pytest.mark.parametrize('timezone', ['Asia/Seoul', 'America/New_York'])
def test_backtest_preserves_timezone(sample_prices, timezone):
    prices = sample_prices.copy()
    prices.index = prices.index.tz_localize(timezone)
    options = dict(
        schedule='EOM',
        rule={'069500': 0.6, '114820': 0.4},
        cost=0.002,
        start='2020-01-01',
        end='2024-12-31',
        verbose=False,
    )
    expected = sb.run_backtest(prices=sample_prices, **options)
    actual = sb.run_backtest(prices=prices, **options)

    for field in ['gross_returns', 'returns', 'weights', 'trades']:
        result = getattr(actual, field).copy()
        if isinstance(result.index, pd.MultiIndex):
            dates = result.index.get_level_values('date')
            assert str(dates.tz) == timezone
            result.index = pd.MultiIndex.from_arrays(
                [dates.tz_localize(None), result.index.get_level_values('asset')],
                names=result.index.names,
            )
        else:
            assert str(result.index.tz) == timezone
            result.index = result.index.tz_localize(None)
        pd.testing.assert_series_equal(
            result, getattr(expected, field), check_exact=True, check_freq=False
        )


@pytest.mark.parametrize('cached', [False, True])
def test_fund_update_pricing_keyword_and_missing_returns(cached):
    date = pd.Timestamp('2020-01-02')
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp('2020-01-01'), 'A'), (date, 'A'), (date, 'B')], names=['date', 'asset']
    )
    pricing = pd.DataFrame({'return': [0.0, 0.04, float('nan')]}, index=index)
    if cached:
        pricing = DailyReturns(pricing)
    fund = Fund()
    fund.rebalance(pd.Series({'A': 0.4, 'B': 0.4, 'C': 0.2}))
    logger = BacktestLogger()

    daily_return = fund.update(date, pricing=pricing, logger=logger)

    assert daily_return == pytest.approx(0.016)
    assert fund.nav == pytest.approx(101.6)
    expected = pd.Series({'A': 0.4 * 1.04 / 1.016, 'B': 0.4 / 1.016}, name='weight')
    pd.testing.assert_series_equal(fund.weights, expected)
    assert logger._log == [
        ['Daily return is NaN', date, 'B : changed to zero'],
        ['No price data', date, 'C : cash out'],
    ]


@pytest.mark.parametrize('timezone', [None, 'Asia/Seoul'])
@pytest.mark.parametrize('cached', [False, True])
def test_fund_update_accepts_string_date(timezone, cached):
    dates = pd.date_range('2020-01-01', periods=2, tz=timezone)
    index = pd.MultiIndex.from_product([dates, ['A', 'B']], names=['date', 'asset'])
    pricing = pd.DataFrame({'return': [0.9, 0.9, 0.04, 0.015]}, index=index)
    if cached:
        pricing = DailyReturns(pricing)
    fund = Fund()
    fund.rebalance(pd.Series({'A': 0.4, 'B': 0.6}))
    logger = BacktestLogger()

    daily_return = fund.update('2020-01-02', pricing=pricing, logger=logger)

    assert daily_return == pytest.approx(0.025)
    assert fund.nav == pytest.approx(102.5)
    expected = pd.Series({'A': 0.4 * 1.04 / 1.025, 'B': 0.6 * 1.015 / 1.025}, name='weight')
    pd.testing.assert_series_equal(fund.weights, expected)
    assert logger._log == []


def test_portfolio_records_are_snapshots_and_buffer_is_released(sample_prices):
    portfolio = sb.Portfolio('P', sb.Universe('U', sample_prices), 'EOM', None)
    portfolio._records = {'dates': [], 'returns': [], 'weights': [], 'trades': []}
    date = pd.Timestamp('2020-01-02')
    weights = pd.Series({'A': 0.6, 'B': 0.4}, name='weight')
    trades = weights.rename('trade')
    portfolio._record(date, 0.01, weights, trades)
    weights.iloc[:] = 0.0
    trades.iloc[:] = 0.0

    portfolio._finalize_records()

    assert portfolio.weights.loc[date].to_dict() == {'A': 0.6, 'B': 0.4}
    assert portfolio.trades.loc[date].to_dict() == {'A': 0.6, 'B': 0.4}
    assert portfolio.gross_returns.loc[date] == 0.01
    assert portfolio._records is None


def _synthetic_universe():
    rs = np.random.RandomState(0)
    vols = np.array([0.005, 0.01, 0.015, 0.02, 0.008])
    cov = np.outer(vols, vols) * (0.3 + 0.7 * np.eye(5))
    rets = rs.standard_normal((400, 5)) @ np.linalg.cholesky(cov).T
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rets, axis=0),
        columns=list('ABCDE'),
        index=pd.bdate_range('2020-01-01', periods=400),
    )
    pricing = prices.stack().rename('price').to_frame()
    pricing.index.names = ['date', 'asset']
    return types.SimpleNamespace(pricing=pricing)


def test_ledoit_wolf_matches_sklearn_formula():
    X = np.random.RandomState(1).standard_normal((60, 5)) * 0.01
    Xc = X - X.mean(axis=0)
    emp_cov = Xc.T @ Xc / len(X)
    mu = np.trace(emp_cov) / 5
    shrunk = ledoit_wolf(X)
    # Shrunk covariance is a convex combination of the sample covariance and mu * I
    off_diag = ~np.eye(5, dtype=bool)
    shrinkage = 1 - shrunk[off_diag][0] / emp_cov[off_diag][0]
    expected = (1 - shrinkage) * emp_cov + shrinkage * mu * np.eye(5)
    np.testing.assert_allclose(shrunk, expected, rtol=1e-12)
    assert 0 < shrinkage < 1


# Reference weights computed with PyPortfolioOpt 1.5.6
# (CovarianceShrinkage.ledoit_wolf + EfficientFrontier.min_volatility)
@pytest.mark.parametrize(
    'window, expected',
    [
        (60, [0.5503139367848389, 0.163794479235014, 0.0262203183695346, 0.0, 0.2596712656106126]),
        (252, [0.6895650322841379, 0.1240458831317242, 0.0002475503942849, 0.0, 0.186141534189853]),
    ],
)
def test_minimum_variance_matches_pypfopt(window, expected):
    universe = _synthetic_universe()
    date = universe.pricing.index.get_level_values('date').max()
    weights = sb.MinimumVariance(list('ABCDE'), window=window).calculate(date, universe, None)
    assert list(weights.index) == list('ABCDE')
    assert weights.sum() == pytest.approx(1, abs=1e-12)
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.values, expected, atol=1e-6)


@pytest.mark.parametrize(
    'rule',
    [
        sb.MinimumVariance(list('ABCDE'), window=60),
        sb.TopNbyMomentum(list('ABCDE'), top_n=2, period=60),
    ],
)
def test_rules_ignore_prices_after_date(rule):
    universe = _synthetic_universe()
    dates = universe.pricing.index.get_level_values('date').unique()
    date = dates[-50]
    blinded = types.SimpleNamespace(pricing=universe.pricing.loc[:date])

    expected = rule.calculate(date, blinded, None)
    actual = rule.calculate(date, universe, None)

    pd.testing.assert_series_equal(actual, expected)


def test_universe_add_pricing(sample_prices):
    universe = sb.Universe('U', sample_prices.iloc[:100])
    universe.add_pricing(sb.Universe('N', sample_prices.iloc[100:])._pricing)

    full = sb.Universe('F', sample_prices)
    expected = full._pricing.copy()
    expected['return'] = expected['return'].fillna(0)
    expected.loc[sample_prices.index[100], 'return'] = 0.0  # first day of the added data
    pd.testing.assert_frame_equal(universe._pricing, expected)
    pd.testing.assert_frame_equal(universe._calendar, full._calendar)


@pytest.mark.parametrize('charts', [None, 'interactive'])
def test_report(sample_prices, charts):
    bt = sb.run_backtest(
        prices=sample_prices,
        schedule='EOM',
        rule={'069500': 0.6, '114820': 0.4},
        cost=0.002,
        verbose=False,
    )
    bt.report(charts=charts, benchmark=sample_prices['069500'])
    bt.report(charts=charts, benchmark=sample_prices['069500'], relative=True)


def test_backtest_rerun_merges_returns(sample_prices):
    universe = sb.Universe('U', sample_prices)
    assert universe.calendar.index[0] == sample_prices.index[0]
    portfolio = sb.Portfolio('P', universe, 'EOM', sb.EqualWeight(list(sample_prices.columns)))
    portfolio.backtest(start='2024-01-01', end='2024-06-30', verbose=False)
    first = portfolio.gross_returns.copy()
    portfolio.backtest(start='2024-04-01', end='2024-12-31', verbose=False)

    merged = portfolio.gross_returns
    assert merged.index.is_unique
    assert merged.index[0] == first.index[0]
    assert merged.index[-1] == pd.Timestamp('2024-12-30')
    # Dates before the second run keep the first run's returns
    before = first.index < pd.Timestamp('2024-04-01')
    pd.testing.assert_series_equal(merged.loc[first.index[before]], first[before])


def test_schedule_index_and_update(sample_prices):
    dates = pd.DatetimeIndex(['2024-03-29', '2024-06-28', '2024-09-30'])
    portfolio = sb.Portfolio(
        'P', sb.Universe('U', sample_prices), dates, sb.ConstantWeight({'069500': 1.0})
    )
    portfolio.update('rebalance_date', target='2024-06-28', value='2024-07-01')
    assert list(portfolio.scheduler.rebalance_dates) == list(
        pd.DatetimeIndex(['2024-03-29', '2024-07-01', '2024-09-30'])
    )
    portfolio.backtest(verbose=False)
    assert portfolio.trades.index.get_level_values(0).unique().tolist() == list(
        pd.DatetimeIndex(['2024-03-29', '2024-07-01', '2024-09-30'])
    )


def test_fund_logs_large_price_change_and_repr():
    date = pd.Timestamp('2020-01-02')
    index = pd.MultiIndex.from_tuples([(date, 'A')], names=['date', 'asset'])
    pricing = pd.DataFrame({'return': [0.5]}, index=index)
    fund = Fund()
    fund.rebalance(pd.Series({'A': 1.0}))
    logger = BacktestLogger()

    fund.update(date, pricing=pricing, logger=logger)

    assert logger._log == [['Large price change', date, 'A : 50.00%']]
    assert repr(fund).startswith('NAV: 150')


def test_report_aligns_benchmark_without_cost(sample_prices):
    bt = sb.run_backtest(sample_prices, 'EOM', {'069500': 1.0}, verbose=False)
    assert bt.returns.iloc[0] == 0
    bt.report(charts=None, benchmark=sample_prices['114820'], relative=True)
