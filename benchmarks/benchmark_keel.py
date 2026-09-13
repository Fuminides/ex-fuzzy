"""Measure Ex-Fuzzy learners against reference baselines on the KEEL collection.

One invocation runs a single (dataset, method) pair through stratified
cross-validation and writes a self-describing JSON result. The grid is meant to
be spread over a cluster, one array task per pair; ``--list-tasks`` prints it.
Aggregate the finished results with ``benchmarks/aggregate_keel.py``.

Every method sees the raw KEEL columns. Baselines use their library defaults;
logistic regression standardizes features inside its own pipeline, fitted on the
training folds only. The Ex-Fuzzy learners use stated, uniform configurations:
a search budget for the genetic learner, and the three FERL operating points of
the fuzzy_greedy_tree paper (compact and medium through FERL, deep through
DeepFERL). Nothing is tuned per dataset, and the figures built
from these results must say so.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keel_datasets import available_datasets, dataset_path, load_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / 'benchmarks' / 'results' / 'keel'
#: Order is the figure's order; keep the Ex-Fuzzy learners first.
METHODS = ('exfuzzy-ga', 'exfuzzy-frc-additive', 'exfuzzy-frc-sufficient',
           'exfuzzy-ferl-compact', 'exfuzzy-ferl-medium', 'exfuzzy-ferl-deep',
           'sklearn-logreg', 'sklearn-tree', 'sklearn-forest', 'sklearn-hgb')
#: Display names. The figure marks Ex-Fuzzy's families by colour and marker, so
#: the names carry no library prefix.
LABELS = {'exfuzzy-ga': 'Genetic Search Rules',
          'exfuzzy-frc-additive': 'Mine+Search, additive',
          'exfuzzy-frc-sufficient': 'Mine+Search, sufficient',
          'exfuzzy-ferl-compact': 'FERL compact',
          'exfuzzy-ferl-medium': 'FERL medium',
          'exfuzzy-ferl-deep': 'FERL deep',
          'sklearn-logreg': 'Logistic regression',
          'sklearn-tree': 'Decision tree',
          'sklearn-forest': 'Random forest',
          'sklearn-hgb': 'Gradient boosting'}
#: Methods from the Ex-Fuzzy library, as opposed to reference baselines.
EXFUZZY_METHODS = frozenset(method for method in METHODS if method.startswith('exfuzzy-'))
#: Methods whose model is not a rule base, so they have no rule count.
RULELESS_METHODS = frozenset({'sklearn-logreg'})

#: FuzzyRulesClassifier configuration, chosen on the 20 development datasets of
#: the KEEL study (see docs/performance/KEEL.md) and shared by both rule modes.
#: Both modes are always reported; additive is the library default.
FRC_CONFIG = dict(max_features=8, n_linguistic_variables='auto', nAnts=3, nRules=None,
                  feature_selection='per_class')
FRC_RULE_MODES = {'exfuzzy-frc-additive': 'additive', 'exfuzzy-frc-sufficient': 'sufficient'}

#: The FERL operating points of the fuzzy_greedy_tree paper (its AAAI tables), as
#: ``(constructor kwargs, fit kwargs)`` for Ex-Fuzzy's native FERL. Compact mirrors
#: that repository's ``fgrt-base`` and medium its ``fgrt-performance`` pipeline
#: configuration, with the pipeline's own defaults (20 rules, depth 5, minimum
#: improvement 0.01, patience 3) written out where a preset leaves them unset,
#: because Ex-Fuzzy's FERL defaults to 15 rules. Deep is a different estimator,
#: :class:`ex_fuzzy.DeepFERL`, and is built separately below.
FERL_PRESETS = {
    'exfuzzy-ferl-compact': (dict(partition='quantile', max_rules=20, max_depth=5,
                                  min_improvement=0.01),
                             dict(patience=3)),
    'exfuzzy-ferl-medium': (dict(partition='quantile', split_mode='learned',
                                 learned_width='bootstrap', max_rules=150, max_depth=12,
                                 min_improvement=0.0),
                            dict(patience=16)),
}

#: Structural budget for the genetic learner: the library's own defaults, so the
#: model stays the small rule base Ex-Fuzzy advertises.
GA_MODEL = dict(nRules=30, nAnts=4)
#: Search budget. The shipped defaults (70 generations, population 30, patience
#: 10) stop early enough to understate the learner: on `vehicle` they reached
#: 0.42 accuracy against 0.62 for the budget below, for a tenth of the time.
#: Cheap on a cluster, so spend it and state it rather than quoting a truncated
#: search. This is a compute budget, not per-dataset tuning: it is identical
#: everywhere, as are the scikit-learn baselines' own defaults.
GA_SEARCH = dict(n_gen=100, pop_size=100, patience=25, min_delta=1e-4)


def _size_exfuzzy_ga(model) -> dict:
    """Count the rules that survived pruning and their antecedent conditions."""
    rule_base = model.rule_base
    conditions = 0
    for rule in rule_base.get_rules():
        conditions += int(sum(1 for antecedent in rule.antecedents if antecedent != -1))
    return dict(rules=len(rule_base.get_rules()), conditions=conditions)


def _size_exfuzzy_ferl(model) -> dict:
    """Count FERL's leaves, matching how a decision tree's rules are counted.

    Node names encode their root path as ``root_F<feature>_L<term>`` segments,
    so the segment count is the rule's antecedent length. Soft inference also
    consults interior nodes; the leaves are what ``max_rules`` bounds.
    """
    leaves = [name for name, node in model.node_dict_access.items()
              if name != 'root' and not node.get('children')]
    return dict(rules=len(leaves), conditions=sum(name.count('_F') for name in leaves))


def _size_frc(model) -> dict:
    """Selected association rules and their total number of conditions."""
    features = model._rules['features'] if model.n_rules_ else []
    return dict(rules=int(model.n_rules_), conditions=int(sum(len(f) for f in features)))


def _size_deep_ferl(model) -> dict:
    """DeepFERL rules are its leaves; a leaf's depth is its condition count."""
    stats = model.get_tree_stats()
    return dict(rules=int(stats['leaves']), conditions=int(stats['total_leaf_depth']))


def _size_logreg(model) -> dict:
    """A linear model has no rules; record its fitted parameter count instead."""
    linear = model[-1]
    return dict(rules=None, conditions=None,
                parameters=int(linear.coef_.size + linear.intercept_.size))


def _size_sklearn_tree(model) -> dict:
    tree = model.tree_
    leaves = int((tree.children_left == -1).sum())
    return dict(rules=leaves, conditions=int(_leaf_depth_total(tree)))


def _size_sklearn_forest(model) -> dict:
    rules = conditions = 0
    for estimator in model.estimators_:
        part = _size_sklearn_tree(estimator)
        rules += part['rules']
        conditions += part['conditions']
    return dict(rules=rules, conditions=conditions)


def _size_sklearn_hgb(model) -> dict:
    """Leaves summed over every boosted tree: one per class per iteration.

    scikit-learn exposes the fitted trees only through the private
    ``_predictors``, whose node arrays flag the leaves and record their depth.
    """
    rules = conditions = 0
    for iteration in model._predictors:
        for predictor in iteration:
            leaves = predictor.nodes['is_leaf'].astype(bool)
            rules += int(leaves.sum())
            conditions += int(predictor.nodes['depth'][leaves].sum())
    return dict(rules=rules, conditions=conditions)


def _leaf_depth_total(tree) -> int:
    """Sum of root-to-leaf depths, i.e. the total antecedent conditions."""
    total = 0
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        left, right = tree.children_left[node], tree.children_right[node]
        if left == -1:
            total += depth
        else:
            stack.append((left, depth + 1))
            stack.append((right, depth + 1))
    return total


def build_method(method: str, seed: int, n_classes: int, y_train=None):
    """Return ``(estimator, fit_kwargs, size_function)`` for one method.

    Baselines keep their library defaults and the Ex-Fuzzy learners take the
    module-level configurations; only the random seed varies by fold.
    ``y_train`` lets gradient boosting drop early stopping where the default
    cannot run at all.
    """
    if method == 'exfuzzy-ga':
        from ex_fuzzy.evolutionary_fit import BaseFuzzyRulesClassifier
        import ex_fuzzy.fuzzy_sets as fs
        model = BaseFuzzyRulesClassifier(fuzzy_type=fs.FUZZY_SETS.t1, **GA_MODEL)
        return model, dict(random_state=seed, **GA_SEARCH), _size_exfuzzy_ga
    if method in FERL_PRESETS:
        from ex_fuzzy.ferl import FERL
        init, fit = FERL_PRESETS[method]
        return FERL(random_state=seed, **init), dict(fit), _size_exfuzzy_ferl
    if method in FRC_RULE_MODES:
        from ex_fuzzy.classifiers import FuzzyRulesClassifier
        model = FuzzyRulesClassifier(random_state=seed, rule_mode=FRC_RULE_MODES[method], **FRC_CONFIG)
        return model, {}, _size_frc
    if method == 'exfuzzy-ferl-deep':
        from ex_fuzzy.ferl_deep import DeepFERL
        return DeepFERL(random_state=seed), {}, _size_deep_ferl
    if method == 'sklearn-logreg':
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        return model, {}, _size_logreg
    if method == 'sklearn-tree':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=seed), {}, _size_sklearn_tree
    if method == 'sklearn-forest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=seed, n_jobs=1), {}, _size_sklearn_forest
    if method == 'sklearn-hgb':
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(random_state=seed)
        # The default early stopping holds out a stratified validation split on
        # large data, which is impossible when a class has a single training
        # member (nursery). Only then does it fall back to boosting without it.
        if y_train is not None and np.unique(y_train, return_counts=True)[1].min() < 2:
            model.set_params(early_stopping=False)
        return model, {}, _size_sklearn_hgb
    raise ValueError(f'Unknown method {method!r}. Known: {", ".join(METHODS)}')


def run_fold(method: str, seed: int, X_train, y_train, X_test, y_test, n_classes: int) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    model, fit_kwargs, measure = build_method(method, seed, n_classes, y_train)
    start = time.perf_counter()
    model.fit(X_train, y_train, **fit_kwargs)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predicted = np.asarray(model.predict(X_test))
    predict_seconds = time.perf_counter() - start
    # Ex-Fuzzy can abstain with -1 when no rule fires; that counts as an error.
    return dict(accuracy=float(accuracy_score(y_test, predicted)),
                balanced_accuracy=float(balanced_accuracy_score(y_test, predicted)),
                macro_f1=float(f1_score(y_test, predicted, average='macro', zero_division=0)),
                unclassified=int(np.sum(predicted < 0)),
                fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                **measure(model))


def method_configuration(method: str) -> dict:
    """The non-default settings a reader needs to reproduce one method."""
    if method == 'exfuzzy-ga':
        return dict(fuzzy_type='t1', **GA_MODEL, **GA_SEARCH)
    if method in FERL_PRESETS:
        init, fit = FERL_PRESETS[method]
        return dict(**init, fit=dict(fit))
    if method in FRC_RULE_MODES:
        return dict(estimator='FuzzyRulesClassifier', rule_mode=FRC_RULE_MODES[method], **FRC_CONFIG)
    if method == 'exfuzzy-ferl-deep':
        return dict(estimator='DeepFERL', library_defaults=True)
    if method == 'sklearn-logreg':
        return dict(pipeline='StandardScaler -> LogisticRegression', max_iter=1000)
    return dict(library_defaults=True)


def run_task(dataset: str, method: str, folds: int, seed: int, root) -> dict:
    from sklearn.model_selection import StratifiedKFold

    data = load_dataset(dataset, root)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    record = dict(method=method, method_label=LABELS[method], folds=folds, seed=seed,
                  configuration=method_configuration(method), **data.summary())
    results = []
    for index, (train, test) in enumerate(splitter.split(data.X, data.y)):
        started = time.perf_counter()
        fold = run_fold(method, seed + index, data.X[train], data.y[train],
                        data.X[test], data.y[test], data.n_classes)
        fold.update(fold_index=index, n_train=len(train), n_test=len(test),
                    wall_seconds=time.perf_counter() - started)
        print(f'{dataset} {method} fold {index}: acc={fold["accuracy"]:.4f} '
              f'rules={fold["rules"]} fit={fold["fit_seconds"]:.1f}s', flush=True)
        results.append(fold)
    record['folds_detail'] = results
    for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'rules', 'conditions',
                'parameters', 'fit_seconds', 'predict_seconds', 'unclassified'):
        values = [fold.get(key) for fold in results]
        if any(value is None for value in values):  # Not defined for this model.
            record[f'mean_{key}'] = record[f'std_{key}'] = None
            continue
        values = [float(value) for value in values]
        record[f'mean_{key}'] = float(np.mean(values))
        record[f'std_{key}'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return record


def environment() -> dict:
    packages = {}
    for name in ('numpy', 'scikit-learn', 'pymoo', 'ex_fuzzy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    return dict(python=platform.python_version(), platform=platform.platform(),
                node=platform.node(), cpu=cpu, packages=packages,
                threads={key: os.environ.get(key) for key in
                         ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', help='KEEL dataset name.')
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--datasets', nargs='*', default=None,
                        help='Restrict --list-tasks to these datasets.')
    parser.add_argument('--methods', nargs='*', default=list(METHODS),
                        help='Restrict --list-tasks to these methods.')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--root', default=None, help='KEEL collection directory.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--list-tasks', action='store_true',
                        help='Print the "<dataset> <method>" grid, one task per line.')
    parser.add_argument('--order', choices=('name', 'size'), default='name',
                        help='Task order. "size" puts the largest files first so '
                             'the longest jobs start first on a busy queue.')
    parser.add_argument('--task-index', type=int, default=None,
                        help='Run the 1-based task at this position in the grid.')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even when the result file already exists.')
    options = parser.parse_args(argv)

    if options.folds < 2:
        parser.error('--folds must be at least 2')
    names = options.datasets or available_datasets(options.root)
    if options.order == 'size':
        names = sorted(names, key=lambda name: -dataset_path(name, options.root).stat().st_size)
    grid = [(dataset, method) for dataset in names for method in options.methods]
    if options.list_tasks:
        for dataset, method in grid:
            print(f'{dataset} {method}')
        return 0
    if options.task_index is not None:
        if not 1 <= options.task_index <= len(grid):
            parser.error(f'--task-index must be within 1..{len(grid)}')
        options.dataset, options.method = grid[options.task_index - 1]
    if not options.dataset or not options.method:
        parser.error('Give --dataset and --method, or --task-index, or --list-tasks')

    options.output_dir.mkdir(parents=True, exist_ok=True)
    destination = options.output_dir / f'{options.dataset}__{options.method}.json'
    if destination.exists() and not options.force:
        print(f'{destination} already exists; pass --force to recompute.')
        return 0

    record = dict(dataset=options.dataset, method=options.method,
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  environment=environment())
    started = time.perf_counter()
    try:
        record.update(run_task(options.dataset, options.method, options.folds,
                               options.seed, options.root))
        record['status'] = 'ok'
    except Exception as error:  # A failed pair must stay visible, not vanish.
        record.update(status='failed', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        print(record['error'], file=sys.stderr, flush=True)
    record['total_seconds'] = time.perf_counter() - started
    # Written through a temporary file so a reader never sees a partial result
    # when several array tasks finish at once on a shared filesystem.
    staging = destination.with_name(f'.{destination.name}.{os.getpid()}')
    staging.write_text(json.dumps(record, indent=2) + '\n')
    staging.replace(destination)
    print(f'wrote {destination} ({record["status"]}, {record["total_seconds"]:.1f}s)')
    return 0 if record['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
