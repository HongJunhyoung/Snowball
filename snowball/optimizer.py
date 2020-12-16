import numpy as np
import pandas as pd
from scipy.optimize import minimize

def risk_budgeting(covmat, budget):
    def objective(w, cov, rb) :
        var = w.T @ cov @ w
        sgm = var ** 0.5
        mrc = 1 / sgm * (cov @ w)
        rc = w * mrc  
        rr = rc / sgm # relative risk contribution
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