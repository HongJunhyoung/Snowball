import pandas as pd
import pytest
import os
import snowball as sb
from snowball.components import BacktestLogger, DailyReturns, Fund

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(TEST_DIR, 'data', 'etfs_prices.csv')

@pytest.fixture(scope="module")
def sample_prices():
    print("\n--- Fixture: Starting sample_prices load ---")
    prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    prices.index.name = 'Date'
    prices = prices.rename_axis(None, axis=1)
    print("--- Fixture: sample_prices load complete ---")
    return prices

def test_backtest(sample_prices):
    print("--- Test: test_backtest running ---")
    bt = sb.run_backtest(prices=sample_prices, 
                         schedule='EOM',
                         rule={'069500': 0.6, '114820': 0.4},
                         cost=0.002,
                         start='2020-01-01', end='2024-12-31')
    actual_value = bt.stats['CAGR']
    expected_value = -0.03308272002801749
    assert actual_value == pytest.approx(expected_value, abs=1e-10), \
        f"Backtest result is not {expected_value}. Actual result: {actual_value}"
    print("--- Test: test_backtest completed ---")

def test_calc_stats(sample_prices):
    print("--- Test: test_calc_stats running ---")
    dummy_returns = sample_prices['069500'].pct_change().iloc[1:]
    stats = sb.calc_stats(dummy_returns)
    actual_value = stats['CAGR']
    expected_value = -0.1231127835849668
    assert actual_value == pytest.approx(expected_value, abs=1e-10), \
        f"Backtest result is not {expected_value}. Actual result: {actual_value}"
    print("--- Test: test_calc_stats completed ---")


@pytest.mark.parametrize('timezone', ['Asia/Seoul', 'America/New_York'])
def test_backtest_preserves_timezone(sample_prices, timezone):
    prices = sample_prices.copy()
    prices.index = prices.index.tz_localize(timezone)
    options = dict(schedule='EOM', rule={'069500': 0.6, '114820': 0.4},
                   cost=0.002, start='2020-01-01', end='2024-12-31', verbose=False)
    expected = sb.run_backtest(prices=sample_prices, **options)
    actual = sb.run_backtest(prices=prices, **options)

    for field in ['gross_returns', 'returns', 'weights', 'trades']:
        result = getattr(actual, field).copy()
        if isinstance(result.index, pd.MultiIndex):
            dates = result.index.get_level_values('date')
            assert str(dates.tz) == timezone
            result.index = pd.MultiIndex.from_arrays(
                [dates.tz_localize(None), result.index.get_level_values('asset')],
                names=result.index.names)
        else:
            assert str(result.index.tz) == timezone
            result.index = result.index.tz_localize(None)
        pd.testing.assert_series_equal(result, getattr(expected, field),
                                       check_exact=True, check_freq=False)


@pytest.mark.parametrize('cached', [False, True])
def test_fund_update_pricing_keyword_and_missing_returns(cached):
    date = pd.Timestamp('2020-01-02')
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp('2020-01-01'), 'A'), (date, 'A'), (date, 'B')],
        names=['date', 'asset'])
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
    expected = pd.Series({'A': 0.4 * 1.04 / 1.025, 'B': 0.6 * 1.015 / 1.025},
                         name='weight')
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
