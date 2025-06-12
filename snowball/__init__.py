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

from . import _version
__version__ = _version.get_versions()['version']
