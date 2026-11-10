"""Fits with ordinal conditions off, recorded on the 3.4.0 release and replayed by test_ordinal_conditions.py.

``python tests/golden_ordinal_default.py <out.json>`` records them with whatever ``ex_fuzzy`` is on the path (the
fixture in ``tests/data`` was recorded on commit 10d811b, before ordinal conditions existed).
"""
import hashlib
import json
import sys
import warnings

import numpy as np
from sklearn.datasets import load_diabetes, load_iris

SEED_FIT = dict(n_gen=12, pop_size=24, random_state=5, patience=None)


def _digest(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def record() -> dict:
    from ex_fuzzy import evolutionary_fit as evf
    from ex_fuzzy import evolutionary_fit_regression as efr
    from ex_fuzzy import persistence, utils
    from ex_fuzzy import fuzzy_sets as fs

    warnings.simplefilter('ignore')
    X, y = load_iris(return_X_y=True)
    out = {}

    def classifier(name, **options):
        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, **options)
        clf.fit(X, y, **SEED_FIT)
        problem_gene = np.asarray(clf.optimization_result_['X'])
        out[name] = dict(
            rules=clf.rule_base.print_rules(True), gene=problem_gene.tolist(),
            predictions=_digest(clf.predict(X).astype(np.int64)), performance=repr(float(np.max(clf.performance))),
            antecedents=[[r.antecedents for r in rb.get_rules()] for rb in clf.rule_base.get_rulebases()])
        return clf

    classifier('optimized_t1')
    classifier('optimized_t2', fuzzy_type=fs.FUZZY_SETS.t2)
    fixed = utils.construct_partitions(X, n_partitions=4)
    clf = classifier('fixed_t1', linguistic_variables=fixed, ds_mode=2)
    classifier('fixed_mixture_search', linguistic_variables=fixed, inference='mixture', mixture_search=True,
               mixture_non_negative=True)
    out['fixed_variables_text'] = persistence.save_fuzzy_variables(clf.rule_base.antecedents)
    out['fixed_rules_roundtrip'] = persistence.load_fuzzy_rules(
        clf.rule_base.print_rules(True), clf.rule_base.antecedents).print_rules(True)

    Xr, yr = load_diabetes(return_X_y=True)
    for name, options in (('regression_crisp', {}), ('regression_fuzzy', dict(consequent_type='fuzzy')),
                          ('regression_sufficient', dict(rule_mode='sufficient', unique_antecedents=True))):
        reg = efr.BaseFuzzyRulesRegressor(nRules=10, nAnts=3, n_linguistic_variables=4, **options)
        reg.fit(Xr, yr, n_gen=12, pop_size=24, random_state=5)
        out[name] = dict(rules=reg.print_rules(True), gene=reg.best_chromosome_.tolist(),
                         predictions=_digest(reg.predict(Xr)), performance=repr(float(reg.performance_)))
    return out


if __name__ == '__main__':
    sys.path.insert(0, '.')
    with open(sys.argv[1], 'w') as handle:
        json.dump(record(), handle, indent=1, sort_keys=True)
