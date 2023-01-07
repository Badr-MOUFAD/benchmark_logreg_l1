from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from skglm.py_numba_blitz.solver import py_blitz


class Solver(BaseSolver):
    name = 'py-Blitz'
    stopping_strategy = 'iteration'
    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/Badr-MOUFAD/skglm.git@py_blitz'
    ]

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        # cache numba compilation
        self.run(2)

    def run(self, n_iter):
        self.coef_ = py_blitz(self.lmbd, self.X, self.y, max_iter=n_iter)

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def get_result(self):
        return self.coef_.flatten()
