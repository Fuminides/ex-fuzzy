"""Offline calibration of the population-evaluation route choice.

By default a fit decides between the scalar and batched evaluators with a short
runtime probe.  The probe is unbiased but costs one generation of whichever
route loses.  Running this campaign once measures the same choice across a grid
of workloads and stores the answers, after which fits look the decision up and
pay nothing.

Both routes compute the same objective bit for bit; the campaign verifies that
on every measured point and refuses to record a profile if it ever fails.  The
stored profile therefore only affects speed, never results.

    python benchmarks/calibrate_population_dispatch.py            # default grid
    python benchmarks/calibrate_population_dispatch.py --quick    # smaller grid
    python benchmarks/calibrate_population_dispatch.py --dry-run  # do not store

The profile lands in ``~/.cache/ex-fuzzy/dispatch_profile.json`` unless
``EX_FUZZY_DISPATCH_PROFILE`` or ``--output`` says otherwise, and is ignored by
fits on machines whose fingerprint does not match the one that measured it.
"""
import argparse
import itertools
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pymoo.core.problem import Problem
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ex_fuzzy'))

import ex_fuzzy.evolutionary_fit as evf          # noqa: E402
import ex_fuzzy.fuzzy_sets as fs                 # noqa: E402
import ex_fuzzy.utils as utils                   # noqa: E402
from ex_fuzzy import _dispatch_profile           # noqa: E402
from ex_fuzzy import _population_fitness         # noqa: E402

DEFAULT_SAMPLES = (100, 200, 400, 800, 1600, 3200)
DEFAULT_RULES = (10, 20, 40)
DEFAULT_FEATURES = (5, 10, 20)
DEFAULT_POPULATION = (30, 50)
QUICK_SAMPLES = (150, 600, 2400)
QUICK_RULES = (20,)
QUICK_FEATURES = (10,)
QUICK_POPULATION = (40,)


def _forced_route(route):
    """Force every eligible generation onto one route for the whole fit."""
    if route == 'scalar':
        return patch.object(evf.FitRuleBase, '_evaluate_elementwise',
                            Problem._evaluate_elementwise)

    def always_batch(self, population):
        probe = _population_fitness._RouteProbe()
        probe.decision = _population_fitness._RouteProbe.BATCH
        return probe

    return patch.object(evf.FitRuleBase, '_route_probe_state', always_batch)


def _outcome(model):
    """The fit results that must match between routes, for the parity check."""
    return (
        float(model.performance),
        np.asarray(model.optimization_result_['X']).tolist(),
        np.asarray(model.rule_base.get_scores()).tolist(),
        np.asarray(model.optimization_result_['algorithm'].pop.get('F')).tolist(),
        int(model.optimization_result_['algorithm'].evaluator.n_eval),
    )


def _time_fit(route, X, y, partitions, rules, antecedents, population,
              generations, seed):
    with _forced_route(route) if route else nullcontext():
        model = evf.BaseFuzzyRulesClassifier(
            nRules=rules, nAnts=antecedents, linguistic_variables=partitions,
            fuzzy_type=fs.FUZZY_SETS.t1)
        start = time.perf_counter()
        model.fit(X, y, n_gen=generations, pop_size=population,
                  random_state=seed)
        elapsed = time.perf_counter() - start
    return elapsed, _outcome(model)


def _measure_point(samples, rules, features, population, args):
    informative = max(2, min(features, features * 3 // 4))
    X, y = make_classification(
        n_samples=samples, n_features=features, n_informative=informative,
        n_redundant=0, n_classes=3, random_state=args.data_seed)
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)

    routes = ('scalar', 'batch') + (('probe',) if args.verify_probe else ())
    timings = {route: [] for route in routes}
    outcomes = {}
    # Alternate the order across repeats so a warming cache or a drifting clock
    # cannot systematically favour whichever route happens to run first.
    for repeat in range(args.repeats):
        order = routes if repeat % 2 == 0 else tuple(reversed(routes))
        for route in order:
            elapsed, outcome = _time_fit(
                None if route == 'probe' else route, X, y, partitions, rules,
                min(args.antecedents, features), population, args.generations,
                args.search_seed)
            timings[route].append(elapsed)
            previous = outcomes.setdefault(route, outcome)
            if previous != outcome:
                raise SystemExit(
                    'route %r was not deterministic at samples=%d rules=%d '
                    'features=%d population=%d' % (route, samples, rules,
                                                   features, population))
    for route in routes[1:]:
        if outcomes[route] != outcomes['scalar']:
            raise SystemExit(
                'route %r disagreed with the scalar oracle at samples=%d '
                'rules=%d features=%d population=%d; refusing to write a '
                'profile' % (route, samples, rules, features, population))

    scalar = float(np.median(timings['scalar']))
    batch = float(np.median(timings['batch']))
    evaluations = args.generations * population
    entry = {
        'samples': samples,
        'rules': rules,
        'features': features,
        'population': population,
        'scalar_seconds': scalar,
        'batch_seconds': batch,
        'scalar_seconds_per_candidate': scalar / evaluations,
        'batch_seconds_per_candidate': batch / evaluations,
        'decision': 'batch' if batch < scalar else 'scalar',
    }
    if args.verify_probe:
        # What the runtime probe costs against a perfectly calibrated choice:
        # its own probing generations plus any route it settles on wrongly.
        entry['probe_seconds'] = float(np.median(timings['probe']))
        entry['probe_regret'] = entry['probe_seconds'] / min(scalar, batch)
    return entry


def run(args):
    if args.quick:
        grid = itertools.product(QUICK_SAMPLES, QUICK_RULES, QUICK_FEATURES,
                                 QUICK_POPULATION)
    else:
        grid = itertools.product(args.samples, args.rules, args.features,
                                 args.population)
    points = [point for point in grid
              if _population_fitness.chunk_size(point[0], point[1], point[2]) >= 2]
    if not points:
        raise SystemExit('no grid point can batch even two candidates')

    entries = []
    print('%-9s %-7s %-9s %-7s %10s %10s %8s  %s' % (
        'samples', 'rules', 'features', 'pop', 'scalar', 'batch', 'ratio',
        'decision'))
    regrets = []
    for index, (samples, rules, features, population) in enumerate(points, 1):
        entry = _measure_point(samples, rules, features, population, args)
        entries.append(entry)
        ratio = entry['scalar_seconds'] / entry['batch_seconds']
        suffix = ('  probe %.4fs (%.2fx best)' % (
            entry['probe_seconds'], entry['probe_regret'])
            if args.verify_probe else '')
        print('%-9d %-7d %-9d %-7d %9.4fs %9.4fs %7.2fx  %-8s%s   [%d/%d]' % (
            samples, rules, features, population, entry['scalar_seconds'],
            entry['batch_seconds'], ratio, entry['decision'], suffix, index,
            len(points)))

        if args.verify_probe:
            regrets.append(entry['probe_regret'])

    batched = sum(1 for entry in entries if entry['decision'] == 'batch')
    print('\n%d of %d workloads favour batching.' % (batched, len(entries)))
    if regrets:
        print('Runtime probe against a perfect choice: median %.2fx, '
              'worst %.2fx.' % (float(np.median(regrets)), max(regrets)))
    if args.dry_run:
        print('Dry run: no profile written.')
        return
    path = _dispatch_profile.save(entries, args.output)
    print('Profile written to %s' % path)
    print('Fits on this machine will now look the route up instead of probing.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, nargs='+', default=DEFAULT_SAMPLES)
    parser.add_argument('--rules', type=int, nargs='+', default=DEFAULT_RULES)
    parser.add_argument('--features', type=int, nargs='+', default=DEFAULT_FEATURES)
    parser.add_argument('--population', type=int, nargs='+',
                        default=DEFAULT_POPULATION)
    parser.add_argument('--antecedents', type=int, default=4)
    parser.add_argument('--generations', type=int, default=15)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--data-seed', type=int, default=42)
    parser.add_argument('--search-seed', type=int, default=7)
    parser.add_argument('--quick', action='store_true',
                        help='measure a small grid instead of the full one')
    parser.add_argument('--verify-probe', action='store_true',
                        help='also time the default runtime probe and report '
                             'what it costs against the better fixed route')
    parser.add_argument('--dry-run', action='store_true',
                        help='measure and report without storing a profile')
    parser.add_argument('--output', default=None,
                        help='profile path (default: the user cache location)')
    run(parser.parse_args())
