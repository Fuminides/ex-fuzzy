"""The Apriori search and the precomputed pruning reproduce the exhaustive versions exactly."""
from itertools import combinations, product

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification

from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import rule_mining as rm
from ex_fuzzy import rules as rl
from ex_fuzzy import utils


def _exhaustive_rule_search(data, fuzzy_variables, support_threshold, max_depth):
    """The former enumeration of every itemset of every length."""
    list_possible_vars = [[(ix, ax) for ax in range(len(fv))] for ix, fv in enumerate(fuzzy_variables)]
    values = np.asarray(data)
    memberships = [fuzzy_variables[ix](values[:, ix]) for ix in range(values.shape[1])]
    found = []
    for r in range(max_depth):
        for comb in combinations(range(len(list_possible_vars)), r + 1):
            for itemset in product(*[list_possible_vars[x] for x in comb]):
                item_memberships = np.stack([np.asarray(memberships[v][t]) for v, t in itemset])
                if np.mean(np.min(item_memberships, axis=0)) > support_threshold:
                    found.append(itemset)
    return found


def _former_prune(x, y, master, fuzzy_variables, confidence_threshold, lift_threshold):
    """The former pruning, which recomputed memberships per rule and antecedent."""
    x_values = np.asarray(x)
    for ix, rule_base in enumerate(master):
        delete_list = []
        class_samples = x_values[np.equal(y, ix)]
        for jx, rule in enumerate(rule_base):
            real_n = sum(ant != -1 for ant in rule)
            global_array = np.zeros((x_values.shape[0], real_n))
            class_array = np.zeros((class_samples.shape[0], real_n))
            counter = 0
            for zx, antecedent in enumerate(rule):
                if antecedent != -1:
                    global_mem = fuzzy_variables[zx](x_values[:, zx])[antecedent]
                    class_mem = fuzzy_variables[zx](class_samples[:, zx])[antecedent]
                    if global_mem.ndim > 1:
                        global_mem = np.mean(global_mem, axis=1)
                    if class_mem.ndim > 1:
                        class_mem = np.mean(class_mem, axis=1)
                    global_array[:, counter] = global_mem
                    class_array[:, counter] = class_mem
                    counter += 1
            global_support = np.mean(np.min(global_array, axis=1), axis=0)
            class_support = np.mean(np.min(class_array, axis=1), axis=0)
            confidence = class_support / global_support
            lift = confidence / np.mean(np.equal(ix, y))
            if confidence < confidence_threshold or lift < lift_threshold:
                delete_list.append(jx)
        rule_base.remove_rules(delete_list)


def _datasets():
    X, y = load_iris(return_X_y=True)
    yield 'iris', X, y
    X, y = make_classification(n_samples=300, n_features=6, n_informative=4, n_classes=3,
                               n_clusters_per_class=1, random_state=0)
    yield 'synthetic', X, y


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
@pytest.mark.parametrize('threshold,depth', [(0.05, 3), (0.2, 2), (0.01, 4)])
def test_apriori_search_matches_exhaustive_enumeration(kind, threshold, depth):
    for _, X, _ in _datasets():
        variables = utils.construct_partitions(X, kind, detect_categorical=False)
        found = rm.rule_search(X, variables, support_threshold=threshold, max_depth=depth)
        expected = _exhaustive_rule_search(X, variables, threshold, depth)
        assert found == expected


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def test_precomputed_pruning_matches_former_pruning(kind):
    for _, X, y in _datasets():
        variables = utils.construct_partitions(X, kind, detect_categorical=False)
        classes, y_index = np.unique(y, return_inverse=True)
        bases = []
        for label in classes:
            bases.append(rm.mine_rulebase_support(X[y == label], variables, 0.05, 3))
        expected = rl.MasterRuleBase([base.copy() for base in bases])
        actual = rl.MasterRuleBase([base.copy() for base in bases])
        _former_prune(X, y_index, expected, variables, 0.05, 1.05)
        rm.prune_rules_confidence_lift(X, y_index, actual, variables, 0.05, 1.05)
        assert [rule.antecedents for rule in actual.get_rules()] == [rule.antecedents for rule in expected.get_rules()]
        assert actual.get_consequents() == expected.get_consequents()
        assert len(actual.get_rules()) > 0


def test_multiclass_mining_accepts_string_labels():
    X, y = load_iris(return_X_y=True)
    names = np.array(['setosa', 'versicolor', 'virginica'])[y]
    variables = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    with_names = rm.multiclass_mine_rulebase(X, names, variables, 0.05, max_depth=2)
    with_indexes = rm.multiclass_mine_rulebase(X, y, variables, 0.05, max_depth=2)
    assert [rule.antecedents for rule in with_names.get_rules()] == [rule.antecedents for rule in with_indexes.get_rules()]
    assert with_names.get_consequents_names() == list(names[[0, 50, 100]])
