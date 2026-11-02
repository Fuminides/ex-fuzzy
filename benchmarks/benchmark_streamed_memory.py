"""Peak memory and time of scoring candidates with and without ``batch_size``.

Each setting runs in its own process so the process maximum RSS is its own.
The driver refuses to report anything unless every setting gave the identical
scores.

    python benchmarks/benchmark_streamed_memory.py --samples 2000000
    python benchmarks/benchmark_streamed_memory.py --fuzzy-type t2 --batch-sizes none 100000 20000

``peak_mb`` is the ``tracemalloc`` peak while scoring one candidate, above the
problem's own data (``X``, labels, memberships); ``maxrss_mb`` is the whole
process. The data are synthetic: 10 normal features and three classes.
"""
import argparse
import json
import resource
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run_one(samples: int, batch_size, fuzzy_type: str, candidates: int) -> dict:
    sys.path.insert(0, str(ROOT))
    from ex_fuzzy import evolutionary_fit as evf, fuzzy_sets as fs, utils

    kind = {'t1': fs.FUZZY_SETS.t1, 't2': fs.FUZZY_SETS.t2}[fuzzy_type]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(samples, 10))
    y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(scale=0.5, size=samples) > 0).astype(int) + (X[:, 3] > 1)
    problem = evf.FitRuleBase(X, y, 30, 4, 3, linguistic_variables=utils.construct_partitions(X, kind),
                              fuzzy_type=kind, tolerance=0.01, batch_size=batch_size)
    genes = np.random.default_rng(1).integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                                              size=(candidates, problem.n_var))
    problem._array_score(genes[0].copy())  # warm-up: streamed layouts are built once per problem
    tracemalloc.start()
    scores, peaks, seconds = [], [], []
    for gene in genes:
        tracemalloc.reset_peak()
        start = time.perf_counter()
        scores.append(problem._array_score(gene.copy()))
        seconds.append(time.perf_counter() - start)
        peaks.append(tracemalloc.get_traced_memory()[1])
    return {'samples': samples, 'batch_size': batch_size, 'fuzzy_type': fuzzy_type, 'scores': scores,
            'peak_mb': max(peaks) / 2 ** 20, 'median_seconds': float(np.median(seconds)),
            'maxrss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--samples', type=int, default=2_000_000)
    parser.add_argument('--fuzzy-type', choices=['t1', 't2'], default='t1')
    parser.add_argument('--batch-sizes', nargs='+', default=['none', '100000', '20000'])
    parser.add_argument('--candidates', type=int, default=4)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    sizes = [None if size == 'none' else int(size) for size in args.batch_sizes]
    if args.child:
        print(json.dumps(run_one(args.samples, sizes[0], args.fuzzy_type, args.candidates)))
        return
    results = []
    for size in args.batch_sizes:
        output = subprocess.run(
            [sys.executable, __file__, '--child', '--samples', str(args.samples), '--fuzzy-type',
             args.fuzzy_type, '--batch-sizes', size, '--candidates', str(args.candidates)],
            check=True, capture_output=True, text=True).stdout
        results.append(json.loads(output.strip().splitlines()[-1]))
    if any(result['scores'] != results[0]['scores'] for result in results):
        raise SystemExit('Scores differ between batch sizes; refusing to report timings.')
    print(f"{args.samples} samples, {args.fuzzy_type}, identical scores for all batch sizes")
    for result in results:
        print(f"batch_size={str(result['batch_size']):>8}  peak {result['peak_mb']:8.1f} MB  "
              f"time {result['median_seconds']:6.2f} s  maxrss {result['maxrss_mb']:7.0f} MB")


if __name__ == '__main__':
    main()
