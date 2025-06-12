import pandas as pd
import pytest
import os
import snowball as sb

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
