import numpy as np
from numpy.linalg import norm
import blitzl1
from benchopt.datasets import make_correlated_data


rho = 1.
n_samples, n_features = 10, 20

X, y, _ = make_correlated_data(n_samples, n_features, random_state=2)
y = np.sign(y)

alpha_max = norm(X.T @ y, ord=np.inf) / 2

blitzl1.set_use_intercept(True)
blitzl1.set_tolerance(1e-9)
problem = blitzl1.LogRegProblem(X, y)

n_iter = 0
alpha = rho * alpha_max
solution = problem.solve(alpha, max_iter=n_iter)

print(solution.x)
print(solution.intercept)
