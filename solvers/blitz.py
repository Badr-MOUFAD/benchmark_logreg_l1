from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    import blitzl1


class Solver(BaseSolver):

    name = 'blitz'

    # 'pip:git+https://github.com/tbjohns/blitzl1.git@master'

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        blitzl1.set_use_intercept(True)
        blitzl1.set_tolerance(1e-9)
        self.problem = blitzl1.LogRegProblem(self.X, self.y)

    def run(self, n_iter):
        solution = self.problem.solve(self.lmbd, max_iter=n_iter)
        self.coef_ = solution.x
        self.intercept = solution.intercept

    def get_result(self):
        return np.append(self.coef_.flatten(), self.intercept)
