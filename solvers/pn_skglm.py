from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from skglm.datafits import Logistic
    from skglm.penalties.separable import L1
    from skglm.solvers.prox_newton import ProxNewton
    from skglm.utils import compiled_clone


class Solver(BaseSolver):

    name = "PN-skglm"

    def __init__(self):
        self.tol = 1e-9  # scale tol

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        self.log_datafit = compiled_clone(Logistic())
        self.l1_penalty = compiled_clone(L1(lmbd / len(y)))
        self.pn_solver = ProxNewton(fit_intercept=True, tol=1e-9)

        # Cache Numba compilation
        self.run(5)

    def run(self, n_iter):
        self.pn_solver.max_iter = n_iter
        self.w = self.pn_solver.solve(self.X, self.y,
                                      self.log_datafit, self.l1_penalty)[0]

    def get_result(self):
        return self.w
