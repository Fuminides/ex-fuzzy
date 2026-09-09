import numpy as np

import rules


def _original_delete(list_rules):
    unique = {}
    for ix, rule in enumerate(list_rules):
        try:
            unique[rule]
        except KeyError:
            unique[rule] = ix
    return [list_rules[x] for x in unique.values()]


def _delete_with_rule_base(list_rules):
    rule_base = object.__new__(rules.RuleBase)
    return rule_base.delete_rule_duplicates(list_rules)


class _CollidingRule(rules.RuleSimple):
    def __hash__(self):
        return 17


def _assert_same_selection(rule_list):
    expected = _original_delete(rule_list)
    actual = _delete_with_rule_base(rule_list)
    assert [id(rule) for rule in actual] == [id(rule) for rule in expected]


def test_duplicate_lookup_preserves_first_occurrence_and_interleaving():
    first = rules.RuleSimple([0, 1], consequent=0)
    duplicate = rules.RuleSimple([0, 1], consequent=0)
    other = rules.RuleSimple([1, 0], consequent=0)
    _assert_same_selection([first, other, duplicate, other, first])


def test_duplicate_lookup_preserves_hash_inconsistent_equal_rules():
    first = rules.RuleSimple([0], consequent=1, modifiers=np.array([1.0]))
    equal_different_hash = rules.RuleSimple([0], consequent=1, modifiers=np.array([2.0]))
    first.score = 0.1
    equal_different_hash.score = 0.9
    first.weight = 0.2
    equal_different_hash.weight = 0.8
    _assert_same_selection([first, equal_different_hash])


def test_duplicate_lookup_preserves_controlled_hash_collisions():
    first = _CollidingRule([0, -1], consequent=0)
    other = _CollidingRule([1, -1], consequent=0)
    duplicate = _CollidingRule([0, -1], consequent=0)
    _assert_same_selection([first, other, duplicate, other])
