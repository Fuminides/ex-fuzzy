"""Regression rule budget with and without ``unique_antecedents``.

Run from the repository root:
    python benchmarks/benchmark_regression_unique_antecedents.py --seeds 10

Fits the same seeded regressors with ``unique_antecedents`` off and on, for the
additive and sufficient rule modes, and reports per setting the mean number of
distinct antecedent sets, repeated antecedent sets, training R-squared (the GA
objective) and held-out R-squared. Held-out data are a random 30% split. No
timing or threshold assertions: the datasets are small and synthetic.
"""
import argparse
import collections
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.datasets import load_diabetes, make_friedman1
from sklearn.model_selection import train_test_split

# Always benchmark this checkout, even if an older release is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ex_fuzzy import evolutionary_fit_regression as evr

MODES = (("additive", 0.0), ("sufficient", 0.05))


def _dataset(name):
    if name == "diabetes":
        X, y = load_diabetes(return_X_y=True)
    else:
        X, y = make_friedman1(n_samples=1500, n_features=8, noise=1.0, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=0)


def _run(name, rule_mode, tolerance, unique, seed, args):
    X_train, X_test, y_train, y_test = _dataset(name)
    model = evr.BaseFuzzyRulesRegressor(
        nRules=args.n_rules, nAnts=args.n_ants, rule_mode=rule_mode, tolerance=tolerance,
        unique_antecedents=unique,
    ).fit(X_train, y_train, n_gen=args.n_gen, pop_size=args.pop_size, random_state=seed)
    counts = collections.Counter(tuple(rule.antecedents) for rule in model.get_rulebase().rules)
    return dict(
        rules=len(model.get_rulebase().rules),
        distinct=len(counts),
        repeated=sum(count - 1 for count in counts.values()),
        train_r2=model.performance_,
        test_r2=model.score(X_test, y_test),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", default=["diabetes", "friedman1"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-rules", type=int, default=30)
    parser.add_argument("--n-ants", type=int, default=2)
    parser.add_argument("--n-gen", type=int, default=150)
    parser.add_argument("--pop-size", type=int, default=80)
    parser.add_argument("--out", type=Path, default=None, help="optional JSON file for the raw fits")
    args = parser.parse_args()

    raw = []
    header = f"{'dataset':10} {'mode':10} {'unique':6} {'rules':>6} {'distinct':>8} {'repeated':>8} {'train R2':>9} {'test R2':>8}"
    print(header)
    for name in args.datasets:
        for rule_mode, tolerance in MODES:
            for unique in (False, True):
                fits = [_run(name, rule_mode, tolerance, unique, seed, args) for seed in range(args.seeds)]
                raw.append(dict(dataset=name, rule_mode=rule_mode, unique_antecedents=unique, fits=fits))
                mean = {key: float(np.mean([fit[key] for fit in fits])) for key in fits[0]}
                print(f"{name:10} {rule_mode:10} {str(unique):6} {mean['rules']:6.1f} {mean['distinct']:8.1f} "
                      f"{mean['repeated']:8.2f} {mean['train_r2']:9.4f} {mean['test_r2']:8.4f}", flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(dict(args=vars(args) | dict(out=None), runs=raw), indent=1))


if __name__ == "__main__":
    main()
