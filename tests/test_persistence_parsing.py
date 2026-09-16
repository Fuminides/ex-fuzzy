"""Rule text is parsed on whole keywords and works without consequent headers."""
import numpy as np

from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import persistence
from ex_fuzzy import rules
from ex_fuzzy import utils


def _variables():
    X = np.repeat(np.linspace(0.0, 1.0, 20)[:, None], 3, axis=1)
    variables = utils.construct_partitions(X, fs.FUZZY_SETS.t1, detect_categorical=False)
    for variable, name in zip(variables, ['DISTANCE', 'WITHIN', 'BANDWIDTH']):
        variable.name = name
        variable[2].name = 'Very High'
    return variables


def test_keywords_inside_names_round_trip():
    variables = _variables()
    first = rules.RuleSimple([0, -1, 2])
    second = rules.RuleSimple([-1, 1, -1])
    first.score, first.accuracy = 0.5, 0.75
    second.score, second.accuracy = 0.25, 0.5
    master = rules.MasterRuleBase([rules.RuleBaseT1(variables, [first]),
                                   rules.RuleBaseT1(variables, [second])], ['first', 'second'])
    text = master.print_rules(return_rules=True, bootstrap_results=False)

    loaded = persistence.load_fuzzy_rules(text, variables)
    assert [rule.antecedents for rule in loaded.get_rules()] == [[0, -1, 2], [-1, 1, -1]]
    assert loaded.get_consequents() == [0, 1]
    assert loaded.get_consequents_names() == ['first', 'second']
    assert [rule.score for rule in loaded.get_rules()] == [0.5, 0.25]
    assert [rule.accuracy for rule in loaded.get_rules()] == [0.75, 0.5]


def test_rules_without_headers_form_one_rule_base():
    text = 'IF DISTANCE IS Low AND WITHIN IS Very High (MOD Very) WITH DS 0.4, ACC 0.9\n'
    loaded = persistence.load_fuzzy_rules(text, _variables())
    assert len(loaded) == 1
    rule = loaded.get_rules()[0]
    assert rule.antecedents == [0, 2, -1] and rule.accuracy == 0.9 and rule.score == 0.4
    np.testing.assert_array_equal(rule.modifiers, [1.0, 2.0, 1.0])
    assert loaded.get_consequents_names() == [0]


def test_rules_without_statistics_round_trip():
    variables = _variables()
    with_accuracy = rules.RuleSimple([0, -1, 2])
    with_accuracy.accuracy = 0.75
    plain = rules.RuleSimple([-1, 1, -1])
    master = rules.MasterRuleBase([rules.RuleBaseT1(variables, [with_accuracy]),
                                   rules.RuleBaseT1(variables, [plain])], ['a', 'b'])
    text = master.print_rules(return_rules=True, bootstrap_results=False)
    assert 'WITH ACC 0.75' in text and 'WITH DS' not in text

    loaded = persistence.load_fuzzy_rules(text, variables)
    first, second = loaded.get_rules()
    assert first.antecedents == [0, -1, 2] and first.accuracy == 0.75 and not hasattr(first, 'score')
    assert second.antecedents == [-1, 1, -1] and not hasattr(second, 'score') and not hasattr(second, 'accuracy')
    assert loaded.get_consequents() == [0, 1] and loaded.ds_mode == 0


def test_then_clause_is_ignored_and_statistics_parse_in_any_order():
    variables = _variables()
    text = ('Rules for consequent: x\n'
            '----------------\n'
            'IF DISTANCE IS Low THEN consequent vl is 0\n'
            'IF WITHIN IS Very High WITH WGHT 0.5, DS 0.25 Confidence Interval: [0.1 0.2]\n')
    loaded = persistence.load_fuzzy_rules(text, variables)
    first, second = loaded.get_rules()
    assert first.antecedents == [0, -1, -1] and not hasattr(first, 'score')
    assert second.antecedents == [-1, 2, -1] and second.score == 0.25 and second.weight == 0.5
    assert loaded.ds_mode == 2 and loaded.get_consequents_names() == ['x']
