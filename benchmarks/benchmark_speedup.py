"""Reproduce the README's Ex-Fuzzy 2.0 reference vs Ex-Fuzzy 3.0 CPU charts.

Run serially on an otherwise idle machine. Each fit starts in a fresh process;
imports/data/partition construction are outside the timer, while route probing,
training and finalization are inside it. Both variants include the same small
population-hashing instrumentation cost. This is a preserved-reference evaluator
comparison within this checkout, not a comparison of published release wheels.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess

from benchmark_evaluator_variants import ROOT, _run

LABELS = ('Ex-Fuzzy 2.0', 'Ex-Fuzzy 3.0')


def config(samples, kind, fixed, seed, **overrides):
    result = dict(samples=samples, features=10, rules=20, antecedents=4,
                  population=40, generations=5, fuzzy_type=kind, fixed=fixed,
                  fit_seed=seed, trace=True, disable=[])
    result.update(overrides)
    return result


def compare_outcomes(left, right):
    """Reject any changed objective, search trajectory or fitted phenotype."""
    for key in left.keys() | right.keys():
        if key != 'seconds' and left.get(key) != right.get(key):
            raise AssertionError(f'Exact parity failed: {key}')
    if not left.get('trace'):
        raise AssertionError('Missing evaluated-population trace')


def plot(report, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (kind, fixed) in zip(axes.flat,
                                [(k, f) for k in ('t1', 't2') for f in (False, True)]):
        rows = [r for r in report['results'] if r['fuzzy_type'] == kind and r['fixed'] == fixed]
        x = np.arange(len(rows))
        for index, label in enumerate(LABELS):
            values = [r['median_seconds'][label] for r in rows]
            low = [v - min(r['seconds'][label]) for r, v in zip(rows, values)]
            high = [max(r['seconds'][label]) - v for r, v in zip(rows, values)]
            ax.bar(x + (index - .5) * .36, values, .36, label=label,
                   color=('#536878', '#007d73')[index], yerr=[low, high], capsize=3)
        for i, r in enumerate(rows):
            top = max(max(v) for v in r['seconds'].values())
            ax.text(i, top * 1.06, f"{r['speedup']:.1f}×", ha='center', weight='bold')
        ax.set_xticks(x, [str(r['samples']) for r in rows])
        ax.set_xlabel('Training samples')
        ax.set_ylabel('Complete fit (seconds)')
        ax.set_title(f"{kind.upper()} · {'fixed' if fixed else 'optimized'} partitions")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper left')
    fig.suptitle('Ex-Fuzzy 2.0 reference vs Ex-Fuzzy 3.0\n'
                 'Median complete CPU fit; whiskers show min–max across seeds')
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def run(args):
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  baseline='Preserved full object evaluator in the current checkout; not a release wheel',
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  python=platform.python_version(), platform=platform.platform(),
                  cpu=next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                            if line.startswith('model name')), platform.processor()) if Path('/proc/cpuinfo').exists() else platform.processor(),
                  packages={p: importlib.metadata.version(p) for p in ('numpy', 'pymoo', 'scikit-learn')},
                  threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                  seeds=args.seeds, results=[])
    for kind in ('t1', 't2'):
        for fixed in (False, True):
            for samples in args.samples:
                base = config(samples, kind, fixed, 7)
                schedule = [(seed, label) for seed in args.seeds for label in LABELS]
                random.Random(11).shuffle(schedule)
                outcomes = {}
                for seed, label in schedule:
                    print(f'{kind} fixed={fixed} n={samples} seed={seed} {label}', flush=True)
                    outcomes[seed, label] = _run(dict(base, fit_seed=seed,
                                                     reference=label == LABELS[0]))
                for seed in args.seeds:
                    compare_outcomes(outcomes[seed, LABELS[0]], outcomes[seed, LABELS[1]])
                seconds = {label: [outcomes[seed, label]['seconds'] for seed in args.seeds]
                           for label in LABELS}
                medians = {label: statistics.median(values) for label, values in seconds.items()}
                report['results'].append(dict(samples=samples, fuzzy_type=kind, fixed=fixed,
                    settings=base, seconds=seconds, median_seconds=medians,
                    speedup=medians[LABELS[0]] / medians[LABELS[1]], exact_parity=True,
                    evidence={str(seed): {k: v for k, v in outcomes[seed, LABELS[1]].items()
                                         if k in ('trace', 'n_eval', 'performance')}
                              for seed in args.seeds}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    plot(report, args.output.with_suffix('.svg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, nargs='+', default=[100, 1000, 5000])
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 41])
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/performance/speedup.json')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if min(args.samples) < 10 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use sample counts >= 10 and distinct seeds')
    if args.plot_only:
        plot(json.loads(args.output.read_text()), args.output.with_suffix('.svg'))
    else:
        run(args)
