import numpy as np


from benchopt import BaseObjective


class Objective(BaseObjective):

    name = "Sparse Logistic Regression"

    parameters = {
        'reg': [1, 1e-1, 1e-2, 5*1e-3]
    }

    def __init__(self, reg=.1):
        self.reg = reg

    def set_data(self, X, y):
        if set(y) != set([-1, 1]):
            raise ValueError(
                f"y must contain only -1 or 1 as values. Got {set(y)}"
            )
        self.X, self.y = X, y
        self.lmbd = self.reg * self._get_lambda_max()

    def compute(self, solution):
        beta = solution[:-1]
        intercept = solution[-1]

        beta = beta.flatten().astype(np.float64)
        y_X_beta = self.y * (self.X @ beta + intercept)
        l1 = abs(beta).sum()
        return np.log(1 + np.exp(-y_X_beta)).sum() + self.lmbd * l1

    def _get_lambda_max(self):
        return abs(self.X.T @ self.y).max() / 2

    def to_dict(self):
        return dict(X=self.X, y=self.y, lmbd=self.lmbd)
