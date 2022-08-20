from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from skglm.datafits import Logistic
    from skglm.penalties.separable import L1
    from skglm.solvers.pn_solver_improved import pn_solver_improved
    from skglm.utils import compiled_clone


class Solver(BaseSolver):

    name = "PN-skglm-improved"

    def __init__(self):
        self.tol = 1e-9  # scale tol

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        self.log_datafit = compiled_clone(Logistic())
        self.l1_penalty = compiled_clone(L1(lmbd / len(y)))

        # Cache Numba compilation
        self.run(5)

    def run(self, n_iter):
        self.coef = pn_solver_improved(self.X, self.y, self.log_datafit,
                                       self.l1_penalty, p0=100,
                                       max_iter=n_iter, tol=self.tol)[0]

    def get_result(self):
        return self.coef
