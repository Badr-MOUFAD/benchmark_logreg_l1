from benchopt.benchmark import Benchmark
from benchopt import run_benchmark


bench = Benchmark('./')

run_benchmark(bench, max_runs=25,
              n_jobs=12, n_repetitions=3,
              solver_names=[
                  "Blitz",
                  "py-Blitz",
              ],
              dataset_names=[
                  'simulated',
                  #   'libsvm',
                  #   'leukemia'
              ]
              )
