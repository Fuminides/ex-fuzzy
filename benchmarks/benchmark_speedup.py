"""Reproduce the README's Ex-Fuzzy 2.0 reference vs Ex-Fuzzy 3.0 CPU charts.

Run serially on an otherwise idle machine. Each fit starts in a fresh process;
imports/data/partition construction are outside the timer, while route probing,
training and finalization are inside it. Both variants include the same small
population-hashing instrumentation cost. This is a preserved-reference evaluator
comparison within this checkout, not a comparison of published release wheels.
"""
import argparse
import gzip
import hashlib
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess

from benchmark_evaluator_variants import ROOT, WORKER, _run

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
    groups = sorted({(r['fuzzy_type'], r['settings']['features']) for r in report['results']})
    scaling = len({d for _, d in groups}) > 1
    fig, axes = plt.subplots(len(groups), 2, figsize=(12, 4 * len(groups)),
                             squeeze=False, constrained_layout=True)
    for ax, (kind, features, fixed) in zip(axes.flat,
            [(k, d, f) for k, d in groups for f in (False, True)]):
        rows = sorted([r for r in report['results'] if r['fuzzy_type'] == kind
                       and r['fixed'] == fixed and r['settings']['features'] == features],
                      key=lambda r: r['samples'])
        x = np.array([r['samples'] for r in rows]) if scaling else np.arange(len(rows))
        for index, label in enumerate(LABELS):
            values = [r['median_seconds'][label] for r in rows]
            low = [v - min(r['seconds'][label]) for r, v in zip(rows, values)]
            high = [max(r['seconds'][label]) - v for r, v in zip(rows, values)]
            color = ('#536878', '#007d73')[index]
            if scaling:
                ax.errorbar(x, values, yerr=[low, high], label=label, color=color,
                            marker=('s', 'o')[index], capsize=4, linewidth=2)
            else:
                ax.bar(x + (index - .5) * .36, values, .36, label=label,
                       color=color, yerr=[low, high], capsize=3)
        for i, r in enumerate(rows):
            top = max(max(v) for v in r['seconds'].values())
            ax.text(x[i], top * (1.35 if scaling else 1.06),
                    f"{r['speedup']:.1f}×", ha='center', weight='bold')
        if scaling:
            from matplotlib.ticker import NullLocator
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlim(x.min() / 1.5, x.max() * 1.5)
            ax.grid(axis='y', alpha=.2)
        ax.set_xticks(x, [f"{r['samples']:,}" for r in rows])
        ax.set_xlabel('Training samples')
        ax.set_ylabel('Complete fit (seconds)')
        ax.set_title(f"{kind.upper()} · {features} features · {'fixed' if fixed else 'optimized'} partitions")
        if scaling:
            all_times = [t for r in rows for times in r['seconds'].values() for t in times]
            ax.set_ylim(min(all_times) / 2, max(all_times) * 2.5)
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper left')
    fig.suptitle('Ex-Fuzzy 2.0 reference vs Ex-Fuzzy 3.0\n'
                 + ('Log scales · ' if scaling else '')
                 + 'Median complete CPU fit; whiskers show min–max across seeds')
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def source_digest():
    """Identify the actual evaluator and worker code, including uncommitted edits."""
    digest = hashlib.sha256(WORKER.encode())
    for path in sorted((ROOT / 'ex_fuzzy').rglob('*.py')):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cached_run(cfg, directory, provenance, resume=False):
    """Persist each completed fit atomically; reuse only identical inputs/code/env."""
    identity = dict(config=cfg, provenance=provenance)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = directory / (key + '.json.gz')
    if path.exists() and resume:
        with gzip.open(path, 'rt') as stream:
            saved = json.load(stream)
        if saved['identity'] != identity:
            raise ValueError('Checkpoint identity mismatch')
        print('  reused completed fit', flush=True)
        return saved['outcome']
    outcome = _run(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.gz')
    with gzip.open(temporary, 'wt') as stream:
        json.dump(dict(identity=identity, outcome=outcome), stream)
    temporary.replace(path)
    print(f"  completed in {outcome['seconds']:.3f} fit seconds", flush=True)
    return outcome


def workloads(args):
    """Generate the full Cartesian grid, with smaller data matrices first."""
    return sorted([config(n, kind, fixed, 7, features=d)
                   for kind in args.fuzzy_types for fixed in (False, True)
                   for n in args.samples for d in args.features],
                  key=lambda c: (c['samples'] * c['features'], c['samples'],
                                 c['features'], c['fuzzy_type'], c['fixed']))


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
    report['source_sha256'] = source_digest()
    report['grid'] = dict(samples=args.samples, features=args.features,
                          fuzzy_types=args.fuzzy_types)
    provenance = {k: report[k] for k in ('python', 'platform', 'cpu', 'packages',
                                        'threads', 'source_sha256')}
    profile = os.environ.get('EX_FUZZY_DISPATCH_PROFILE') or str(
        Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) /
        'ex-fuzzy/dispatch_profile.json')
    provenance['dispatch_profile_sha256'] = (hashlib.sha256(Path(profile).read_bytes()).hexdigest()
                                            if Path(profile).is_file() else None)
    report['dispatch_profile_sha256'] = provenance['dispatch_profile_sha256']
    directory = args.output.parent / '.runs' / args.output.stem
    for base in workloads(args):
        schedule = [(seed, label) for seed in args.seeds for label in LABELS]
        random.Random(11).shuffle(schedule)
        outcomes = {}
        for seed, label in schedule:
            print(f"{base['fuzzy_type']} fixed={base['fixed']} n={base['samples']} "
                  f"features={base['features']} seed={seed} {label}", flush=True)
            outcomes[seed, label] = cached_run(
                dict(base, fit_seed=seed, reference=label == LABELS[0]),
                directory, provenance, args.resume)
        for seed in args.seeds:
            compare_outcomes(outcomes[seed, LABELS[0]], outcomes[seed, LABELS[1]])
            observed = outcomes[seed, LABELS[1]]
            if observed['n_eval'] != base['population'] * base['generations']:
                raise AssertionError('Unexpected evaluation count')
            if len(observed['trace']) != base['generations']:
                raise AssertionError('Incomplete generation trace')
        seconds = {label: [outcomes[seed, label]['seconds'] for seed in args.seeds]
                   for label in LABELS}
        medians = {label: statistics.median(values) for label, values in seconds.items()}
        report['results'].append(dict(samples=base['samples'], fuzzy_type=base['fuzzy_type'],
            fixed=base['fixed'], settings=base, seconds=seconds, median_seconds=medians,
            speedup=medians[LABELS[0]] / medians[LABELS[1]], exact_parity=True,
            evidence={str(seed): {k: v for k, v in outcomes[seed, LABELS[1]].items()
                                 if k in ('trace', 'n_eval', 'performance')}
                      for seed in args.seeds}))
        print(f"  exact parity passed; median speedup {report['results'][-1]['speedup']:.2f}x", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    plot(report, args.output.with_suffix('.svg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, nargs='+', default=[100, 1000, 5000])
    parser.add_argument('--features', type=int, nargs='+', default=[10])
    parser.add_argument('--fuzzy-types', choices=['t1', 't2'], nargs='+', default=['t1', 't2'])
    parser.add_argument('--resume', action='store_true', help='Reuse completed fits with identical settings, source and environment.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 41])
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/performance/speedup.json')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if min(args.samples) < 10 or min(args.features) < 4:
        parser.error('Use sample counts >= 10 and features >= 4')
    if any(len(set(values)) != len(values) for values in
           (args.samples, args.features, args.fuzzy_types, args.seeds)):
        parser.error('Grid dimensions and seeds must not contain duplicates')
    if args.plot_only:
        plot(json.loads(args.output.read_text()), args.output.with_suffix('.svg'))
    else:
        run(args)
