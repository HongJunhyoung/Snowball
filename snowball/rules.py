import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .components import Rule

from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier


def risk_budgeting(covmat, budget):
    def objective(w, cov, rb) :
        var = w.T @ cov @ w
        sgm = var ** 0.5
        mrc = 1 / sgm * (cov @ w)
        rc = w * mrc
        rr = rc / sgm # Relative risk contribution
        return np.sum(np.square(rr - rb))

    covmat[covmat < 0] = 0  # Negative corelation is adjusted to zero.

    cnt = covmat.shape[0]
    w0 = [1 / cnt] * cnt
    constraints = ({'type': 'eq', 'fun': lambda x: x.sum() - 1},
                {'type': 'ineq', 'fun': lambda x: x})
    options = {'ftol': 1e-20, 'maxiter': 10000}
    result = minimize(fun=objective,
                    x0=w0,
                    args=(covmat, budget),
                    method='SLSQP',
                    constraints=constraints,
                    options=options)
    return(result.x)


class Pipeline(Rule):
    def __init__(self, rules):
        self.rules = rules

    def calculate(self, date, universe, fund):
        assets = self.rules[0].assets
        for rule in self.rules:
            rule.set_assets(assets)
            weights = rule.calculate(date, universe, fund)
            assets = weights.keys().tolist()

        return weights


class ConstantWeight(Rule):
    def __init__(self, weights):
        self.weights = pd.Series(weights)

        # assert sum(self.weights) == 1

    def calculate(self, date, universe, fund):
        return self.weights


class EqualWeight(Rule):
    def __init__(self, assets):
        n_assets = len(assets)
        weights = 1 / n_assets
        self.weights = pd.Series([weights] * n_assets, index=assets)

    def calculate(self, date, universe, fund):
        return self.weights

 
class RiskParity(Rule):
    def __init__(self, assets=None, window=252):
        self.assets = assets
        self.window = window
    
    def set_assets(self, assets):
        self.assets = assets

    def calculate(self, date, universe, fund):
        n_assets = len(self.assets)
        returns = universe.pricing['return'].unstack()[self.assets]
        returns = returns.loc[:date].iloc[-self.window:]
        covmat = returns.cov().values
        budget = [1 / n_assets] * n_assets 
        weights = risk_budgeting(covmat, budget)
        weights = pd.Series(weights, index=self.assets)
        return weights


class TopNbyMomentum(Rule):
    def __init__(self, assets=None, top_n=1, period=126):
        self.assets = assets
        self.top_n = top_n
        self.period = period

    def set_assets(self, assets):
        self.assets = assets

    def calculate(self, date, universe, fund):
        prices = universe.pricing['price'].unstack()[self.assets]
        momentums = prices.iloc[-1] / prices.iloc[-self.period] - 1
        selected = momentums.sort_values(ascending=False).iloc[:self.top_n].index
        weight = 1 / len(selected)
        weights = pd.Series([weight] * len(selected), index=selected)
        return weights


class MinimumVariance(Rule):
    def __init__(self, assets=None, window=252):
        self.assets = assets
        self.window = window

    def set_assets(self, assets):
        self.assets = assets

    def calculate(self, date, universe, fund):
        prices = universe.pricing['price'].unstack()[self.assets].iloc[-self.window:]
        mu = mean_historical_return(prices)
        S = CovarianceShrinkage(prices).ledoit_wolf()
        ef = EfficientFrontier(mu, S)
        weights = ef.min_volatility()
        weights = pd.Series(weights, index=self.assets)
        return weights
