from .backtest import (
    run_backtest
)

from .components import (
    Portfolio,
    Universe,
    Rule,
)

from .rules import (
    Pipeline,
    EqualWeight,
    ConstantWeight,
    RiskParity,
    TopNbyMomentum,
    MinimumVariance,
)

from .report import (
    calc_stats,
    perf_report
)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
