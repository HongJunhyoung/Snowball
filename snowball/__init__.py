from .backtest import (
    run_backtest
)

from .components import (
    Portfolio,
    Universe,
    Rule,
)

from .rules import (
    EqualWeight,
    RiskParity,
    ConstantWeight,
    Pipeline,
    TopNbyMomentum,
    MinimumVariance,
)

from .report import (
    calc_stats,
    report_perf,
)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
