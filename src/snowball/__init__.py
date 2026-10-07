from importlib.metadata import PackageNotFoundError, version

from .backtest import run_backtest
from .components import (
    Portfolio,
    Rule,
    Universe,
)
from .report import (
    calc_stats,
    report_perf,
)
from .rules import (
    ConstantWeight,
    EqualWeight,
    MinimumVariance,
    Pipeline,
    RiskParity,
    TopNbyMomentum,
)

try:
    __version__ = version('Snowball')
except PackageNotFoundError:
    __version__ = 'unknown'

__all__ = [
    'run_backtest',
    'Portfolio',
    'Universe',
    'Rule',
    'EqualWeight',
    'RiskParity',
    'ConstantWeight',
    'Pipeline',
    'TopNbyMomentum',
    'MinimumVariance',
    'calc_stats',
    'report_perf',
]
