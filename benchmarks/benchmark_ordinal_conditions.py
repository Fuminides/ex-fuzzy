"""Small check of ``ordinal_conditions``: the option off and on, on scikit-learn toy data.

Per dataset and arm it reports, over stratified folds and seeds, the macro F1 (classification) or R^2
(regression) on the held-out fold, the number of rules, the mean number of conditions per rule and the fit
time. Partitions are fixed (4 strong trapezoidal terms from the training fold), so the arms differ only in the
conditions the search can write. Arms:

    off      ordinal_conditions=False, the default
    on       ordinal_conditions=True, the same budget (generations x population)
    on-2x    ordinal_conditions=True, twice the generations (the search space is larger)

    python -u benchmarks/benchmark_ordinal_conditions.py --out results.json

It is a sanity check on a few small datasets, not a benchmark of the method.
"""
import argparse
import json
import statistics
import time
import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

from ex_fuzzy import BaseFuzzyRulesClassifier, BaseFuzzyRulesRegressor, utils

ARMS = {'off': (False, 1), 'on': (True, 1), 'on-2x': (True, 2)}


def mean_conditions(rules_list):
    return float(np.mean([sum(code != -1 for code in rule.antecedents) for rule in rules_list])) if rules_list else 0.0


def run_classification(name, data, args, inference, progress):
    X, y = data
    rows = {arm: [] for arm in ARMS}
    for seed in range(args.seeds):
        folds = StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(X, y)
        for fold, (train, test) in enumerate(folds):
            variables = utils.construct_partitions(X[train], n_partitions=args.terms, detect_categorical=False)
            for arm, (ordinal, factor) in ARMS.items():
                options = dict(inference=inference, mixture_search=True, mixture_non_negative=True) \
                    if inference == 'mixture' else {}
                clf = BaseFuzzyRulesClassifier(
                    nRules=args.rules, nAnts=args.ants, linguistic_variables=variables, ordinal_conditions=ordinal,
                    n_gen=args.gens * factor, pop_size=args.pop, random_state=seed, patience=None, **options)
                start = time.perf_counter()
                clf.fit(X[train], y[train])
                seconds = time.perf_counter() - start
                rule_list = clf.rule_base.get_rules()
                rows[arm].append(dict(
                    score=f1_score(y[test], clf.predict(X[test]), average='macro'), rules=len(rule_list),
                    conditions=mean_conditions(rule_list), seconds=seconds,
                    ordinal_rules=int(sum(any(c.operator != 'is' for c in r.conditions(clf.rule_base.antecedents))
                                          for r in rule_list))))
            progress(f'{name}/{inference} seed={seed} fold={fold}')
    return rows


def run_regression(name, data, args, progress):
    X, y = data
    rows = {arm: [] for arm in ARMS}
    for seed in range(args.seeds):
        for fold, (train, test) in enumerate(KFold(args.folds, shuffle=True, random_state=seed).split(X)):
            variables = utils.construct_partitions(X[train], n_partitions=args.terms, detect_categorical=False)
            for arm, (ordinal, factor) in ARMS.items():
                reg = BaseFuzzyRulesRegressor(nRules=args.rules, nAnts=args.ants, linguistic_variables=variables,
                                              ordinal_conditions=ordinal)
                start = time.perf_counter()
                reg.fit(X[train], y[train], n_gen=args.gens * factor, pop_size=args.pop, random_state=seed)
                seconds = time.perf_counter() - start
                rule_list = reg.get_rulebase().get_rules()
                rows[arm].append(dict(
                    score=r2_score(y[test], reg.predict(X[test])), rules=len(rule_list),
                    conditions=mean_conditions(rule_list), seconds=seconds,
                    ordinal_rules=int(sum(any(c.operator != 'is' for c in r.conditions(reg.rule_base.antecedents))
                                          for r in rule_list))))
            progress(f'{name} seed={seed} fold={fold}')
    return rows


def summarize(rows):
    out = {}
    for arm, runs in rows.items():
        out[arm] = {key: (statistics.mean(r[key] for r in runs), statistics.pstdev(r[key] for r in runs))
                    for key in ('score', 'rules', 'conditions', 'seconds', 'ordinal_rules')}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--terms', type=int, default=4)
    parser.add_argument('--rules', type=int, default=20)
    parser.add_argument('--ants', type=int, default=3)
    parser.add_argument('--gens', type=int, default=60)
    parser.add_argument('--pop', type=int, default=50)
    parser.add_argument('--mixture-gens', type=int, default=20, help='generations of the (slow) mixture search arm')
    parser.add_argument('--mixture-seeds', type=int, default=1, help='seeds of the (slow) mixture search arm')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    warnings.simplefilter('ignore')

    classification = {'iris': load_iris(return_X_y=True), 'wine': load_wine(return_X_y=True),
                      'breast_cancer': load_breast_cancer(return_X_y=True)}
    jobs = [('classification', name, 'winning_rule') for name in classification]
    jobs += [('classification', 'iris', 'mixture'), ('classification', 'wine', 'mixture'), ('regression', 'diabetes', None)]
    total = sum(args.folds * (args.mixture_seeds if inference == 'mixture' else args.seeds) for _, _, inference in jobs)
    done = 0

    def progress(message):
        nonlocal done
        done += 1
        print(f'[{done}/{total}] {message}', flush=True)

    results = {}
    for kind, name, inference in jobs:
        if kind == 'regression':
            results[name] = summarize(run_regression(name, load_diabetes(return_X_y=True), args, progress))
        else:
            local = argparse.Namespace(**vars(args))
            if inference == 'mixture':
                local.gens, local.seeds = args.mixture_gens, args.mixture_seeds
            results[f'{name}/{inference}'] = summarize(run_classification(name, classification[name], local, inference, progress))
    print(json.dumps(results, indent=1), flush=True)
    if args.out:
        with open(args.out, 'w') as handle:
            json.dump(results, handle, indent=1)


if __name__ == '__main__':
    main()
