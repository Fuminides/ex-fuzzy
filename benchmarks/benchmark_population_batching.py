"""Compare C01 population batching with the current scalar array evaluator.

Each timed fit runs in a fresh process. The driver refuses to report timings
unless the complete seeded search result is identical, including the final
population and PyMoo's logical evaluation count.

    python benchmarks/benchmark_population_batching.py
    python benchmarks/benchmark_population_batching.py --samples 150 400
"""
import argparse
from contextlib import nullcontext
import json
import platform
from pathlib import Path
import random
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
from sklearn.datasets import make_classification


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ex_fuzzy import evolutionary_fit as evf, fuzzy_sets as fs, utils


def _worker(config):
    X, y = make_classification(
        n_samples=config['samples'], n_features=10, n_informative=9,
        n_redundant=0, n_classes=3, random_state=42)
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    model = evf.BaseFuzzyRulesClassifier(
        nRules=20, nAnts=4, linguistic_variables=partitions, tolerance=0.0)
    context = (nullcontext() if config['mode'] == 'batch' else patch.object(
        evf.FitRuleBase, '_evaluate_elementwise',
        evf.Problem._evaluate_elementwise))
    with context:
        started = time.perf_counter()
        model.fit(
            X, y, n_gen=config['generations'], pop_size=config['population'],
            random_state=7, patience=None)
        seconds = time.perf_counter() - started
    result = model.optimization_result_
    try:
        import resource
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mb = peak_rss / (1024 ** 2 if sys.platform == 'darwin' else 1024)
    except ImportError:
        peak_rss_mb = None
    return {
        'seconds': seconds,
        'peak_rss_mb': peak_rss_mb,
        'performance': float(model.performance),
        'best': np.asarray(result['X']).tolist(),
        'population_x': np.asarray(result['pop'].get('X')).tolist(),
        'population_f': np.asarray(result['pop'].get('F')).tolist(),
        'rule_scores': np.asarray(model.rule_base.get_scores()).tolist(),
        'predictions': np.asarray(model.predict(X)).tolist(),
        'n_eval': int(result['algorithm'].evaluator.n_eval),
    }


def _run_one(config):
    completed = subprocess.run(
        [sys.executable, __file__, '--worker', json.dumps(config)],
        cwd=str(ROOT), capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-4000:])
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _outcome(run):
    return {key: value for key, value in run.items() if key not in ('seconds', 'peak_rss_mb')}


def main(args):
    import pymoo
    print(json.dumps({'environment': {
        'python': platform.python_version(), 'numpy': np.__version__,
        'pymoo': pymoo.__version__, 'machine': platform.machine()},
        'settings': vars(args)}), flush=True)
    rng = random.Random(17)
    for samples in args.samples:
        runs = {'scalar': [], 'batch': []}
        peaks = {'scalar': [], 'batch': []}
        order = ['scalar', 'batch'] * args.repeats
        rng.shuffle(order)
        expected = None
        for mode in order:
            config = {
                'mode': mode, 'samples': samples,
                'generations': args.generations,
                'population': args.population,
            }
            observed = _run_one(config)
            outcome = _outcome(observed)
            if expected is None:
                expected = outcome
            elif outcome != expected:
                raise RuntimeError(
                    'population batching changed the seeded fit at '
                    f'{samples} samples ({mode})')
            runs[mode].append(observed['seconds'])
            if observed['peak_rss_mb'] is not None:
                peaks[mode].append(observed['peak_rss_mb'])
        scalar = float(np.median(runs['scalar']))
        batch = float(np.median(runs['batch']))
        print(json.dumps({
            'samples': samples,
            'scalar_seconds': round(scalar, 6),
            'batch_seconds': round(batch, 6),
            'speedup': round(scalar / batch, 3),
            'scalar_range_seconds': [min(runs['scalar']), max(runs['scalar'])],
            'batch_range_seconds': [min(runs['batch']), max(runs['batch'])],
            'peak_rss_mb': {mode: max(values) if values else None
                            for mode, values in peaks.items()},
            'identical': True,
            'logical_evaluations': expected['n_eval'],
        }), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, nargs='+', default=[150, 400, 1000])
    parser.add_argument('--generations', type=int, default=20)
    parser.add_argument('--population', type=int, default=40)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--worker')
    parsed = parser.parse_args()
    if parsed.worker:
        print(json.dumps(_worker(json.loads(parsed.worker))))
    else:
        main(parsed)
