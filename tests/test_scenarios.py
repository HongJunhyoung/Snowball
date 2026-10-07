"""End-to-end backtests on synthetic prices with listings, gaps and holidays.

Expected values in data/scenarios.json were produced with pandas 2.3 before the
pandas 1-3 compatibility refactoring. Results must not depend on the pandas version.
Regenerate with `python tests/test_scenarios.py` only when a change in results is intended.
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

import snowball as sb

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'scenarios.json')
MOMENTUM_ASSETS = ['A', 'B', 'C', 'E']


def make_prices():
    rs = np.random.RandomState(42)
    dates = pd.bdate_range('2019-01-01', '2021-12-31')
    dates = dates.delete([5, 40, 41, 300])  # holidays
    n, k = len(dates), 5
    vols = np.array([0.006, 0.01, 0.014, 0.02, 0.008])
    cov = np.outer(vols, vols) * (0.3 + 0.7 * np.eye(k))
    rets = rs.standard_normal((n, k)) @ np.linalg.cholesky(cov).T + 0.0002
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates, columns=list('ABCDE'))
    prices.iloc[:250, 3] = np.nan  # D listed later
    prices.iloc[400:403, 4] = np.nan  # E missing prices
    prices.iloc[600, 1] = np.nan  # B missing one day
    return prices


def make_scenarios():
    # name: (schedule, rule, start)
    return {
        'equal_eom': ('EOM', 'EqualWeight', '2019-06-03'),
        'riskparity_eoq': ('EOQ', 'RiskParity', '2020-03-01'),
        'constant_eoy': ('EOY', {'A': 0.5, 'B': 0.3, 'E': 0.2}, '2019-06-03'),
        'constant_eom2': ('EOM-2', {'A': 0.4, 'C': 0.6}, '2019-06-03'),
        'constant_list': (
            ['2019-03-15', '2020-06-30', '2021-02-01'],
            {'B': 0.5, 'C': 0.5},
            '2019-06-03',
        ),
        'momentum_eom': (
            'EOM',
            sb.TopNbyMomentum(MOMENTUM_ASSETS, top_n=2, period=60),
            '2019-06-03',
        ),
        'minvar_eoq': ('EOQ', sb.MinimumVariance(MOMENTUM_ASSETS, window=120), '2019-06-03'),
        'pipeline_eoh': (
            'EOH',
            sb.Pipeline(
                [
                    sb.TopNbyMomentum(MOMENTUM_ASSETS, top_n=3, period=60),
                    sb.RiskParity(window=120),
                ]
            ),
            '2019-06-03',
        ),
    }


def summarize(bt):
    last_date = bt.weights.index.get_level_values(0).max()
    return {
        'stats': {
            k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in bt.stats.items()
        },
        'n_days': int(len(bt.returns)),
        'final_weights': {str(k): float(v) for k, v in bt.weights.loc[last_date].items()},
        'turnover': float(bt.trades.abs().sum()),
        'log': {str(k): int(v) for k, v in bt.log['event'].value_counts().sort_index().items()},
    }


def run_scenario(name):
    schedule, rule, start = make_scenarios()[name]
    bt = sb.run_backtest(make_prices(), schedule, rule, cost=0.001, start=start, verbose=False)
    return summarize(bt)


def assert_close(actual, expected, path=''):
    if isinstance(expected, dict):
        assert sorted(actual) == sorted(expected), path
        for k in expected:
            assert_close(actual[k], expected[k], f'{path}/{k}')
    elif isinstance(expected, float):
        # Optimizer-based rules (RiskParity, MinimumVariance) vary slightly across scipy versions
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6), path
    else:
        assert actual == expected, path


with open(DATA_PATH) as f:
    EXPECTED = json.load(f)


@pytest.mark.parametrize('name', sorted(make_scenarios()))
def test_scenario(name):
    assert_close(run_scenario(name), EXPECTED[name])


if __name__ == '__main__':
    result = {name: run_scenario(name) for name in make_scenarios()}
    with open(DATA_PATH, 'w') as f:
        json.dump(result, f, indent=1, sort_keys=True, default=float)
    sys.stdout.write(f'Wrote {DATA_PATH}\n')
