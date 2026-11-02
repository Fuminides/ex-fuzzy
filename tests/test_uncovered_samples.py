"""Samples that fire no rule: no rule wins them, and they fall back to the training base rates."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score

from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import rules
from ex_fuzzy import temporal
from ex_fuzzy import utils
from ex_fuzzy._fitness import _fitness_cache_scope, score_rulebase, score_rulebase_objects


def _imbalanced(samples=200):
    """Class 2 is the most frequent, so the fallback differs from the class of the first rule."""
    return make_classification(n_samples=samples, n_features=6, n_informative=4, n_redundant=0,
                               n_classes=3, n_clusters_per_class=1, weights=[0.25, 0.25, 0.5],
                               random_state=3)


def _hand_built(allow_unknown):
    """Class 0 has the rule 'x is low', class 1 'x is high' and class 2 none; x = 0.5 fires no rule."""
    x = fs.fuzzyVariable('x', [fs.FS('low', [0, 0, 0.2, 0.4], [0, 1]),
                               fs.FS('high', [0.6, 0.8, 1, 1], [0, 1])])
    bases = [rules.RuleBaseT1([x], [rules.RuleSimple([0])]), rules.RuleBaseT1([x], [rules.RuleSimple([1])]),
             rules.RuleBaseT1([x], [])]
    master = rules.MasterRuleBase(bases, ['a', 'b', 'c'], allow_unknown=allow_unknown)
    X = np.array([[0.0], [0.05], [1.0], [0.95], [0.5], [0.5], [0.5], [0.5], [0.5]])
    y = np.array([0, 0, 1, 1, 2, 2, 2, 2, 0])  # class 2 is the most frequent
    return master, X, y


@pytest.mark.parametrize('allow_unknown, fallback', [(False, 2), (True, -1)])
@pytest.mark.parametrize('oracle', [score_rulebase, score_rulebase_objects])
def test_objective_scores_uncovered_samples_as_the_majority_class(allow_unknown, fallback, oracle):
    master, X, y = _hand_built(allow_unknown)
    score = oracle(master, X, y, 0.0, 0.0, 0.0, max_conditions=2)
    expected = f1_score(y, [0, 0, 1, 1] + [fallback] * 5, labels=[0, 1, 2], average='macro',
                        zero_division=0)
    assert score == pytest.approx(expected, rel=1e-12)
    # Each rule is judged on the samples it wins; the uncovered ones used to count for rule 0 (3 of 7).
    assert [rule.accuracy for rule in master.get_rules()] == [1.0, 1.0]


def _problem(**kwargs):
    X, y = _imbalanced()
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    return evf.FitRuleBase(X, y, kwargs.pop('nRules', 6), kwargs.pop('nAnts', 3), 3,
                           linguistic_variables=partitions, tolerance=0.01, alpha=0.1, beta=0.2, **kwargs)


def _genes(problem, count=30):
    return np.random.default_rng(11).integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                                              size=(count, problem.n_var))


def _oracle_scores(problem, genes, oracle):
    return np.asarray([oracle(problem._construct_ruleBase(gene.copy(), problem.fuzzy_type), problem.X,
                              problem.y, problem.tolerance, problem.alpha_, problem.beta_,
                              problem._precomputed_truth, max_conditions=problem._max_conditions)
                       for gene in genes])


@pytest.mark.parametrize('ds_mode', [0, 1])
@pytest.mark.parametrize('allow_unknown', [False, True])
def test_every_evaluator_scores_uncovered_samples_alike(ds_mode, allow_unknown):
    problem = _problem(ds_mode=ds_mode, allow_unknown=allow_unknown)
    assert problem._label_domain().majority == 2
    genes = _genes(problem)
    decoded = [problem._construct_ruleBase(gene.copy(), problem.fuzzy_type) for gene in genes]
    assert any(np.any(rb.compute_firing_strengths(problem.X).max(axis=1) == 0) for rb in decoded if rb.get_rules())

    expected = _oracle_scores(problem, genes, score_rulebase)
    np.testing.assert_array_equal(_oracle_scores(problem, genes, score_rulebase_objects), expected)
    np.testing.assert_array_equal([problem._array_score(gene.copy()) for gene in genes], expected)
    with _fitness_cache_scope(problem, True):
        batched = problem._population_scores(genes)
    assert batched is not None
    np.testing.assert_array_equal(batched, expected)


@pytest.mark.parametrize('allow_unknown', [False, True])
def test_torch_objective_scores_uncovered_samples_alike(allow_unknown):
    pytest.importorskip('torch')
    from ex_fuzzy import _torch_fitness
    problem = _problem(allow_unknown=allow_unknown)
    genes = _genes(problem)
    scores, declined = _torch_fitness.TorchObjective.build(problem, 'cpu').score(genes)
    assert not declined.any()
    np.testing.assert_array_equal(scores, _oracle_scores(problem, genes, score_rulebase))


def test_decoded_rule_bases_carry_the_training_base_rates():
    problem = _problem()
    master = problem._construct_ruleBase(_genes(problem, 1)[0], problem.fuzzy_type)
    np.testing.assert_array_equal(master.class_priors, np.bincount(problem.y) / len(problem.y))
    assert master._uncovered_consequent() == 2
    copied = master.copy()
    np.testing.assert_array_equal(copied.class_priors, master.class_priors)
    assert copied.class_priors is not master.class_priors
    # Without priors (a rule base built by hand) every class is as frequent: consequent 0.
    assert _hand_built(False)[0]._uncovered_consequent() == 0


@pytest.mark.parametrize('allow_unknown', [False, True])
def test_classifier_predicts_the_base_rates_where_no_rule_fires(allow_unknown):
    X, y = _imbalanced()
    labels = np.array(['ant', 'bee', 'cat'])[y]
    classifier = evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=2, allow_unknown=allow_unknown, n_gen=3,
                                              pop_size=10, random_state=0)
    classifier.fit(X, labels)
    priors = classifier.rule_base.class_priors
    np.testing.assert_array_equal(priors, np.bincount(y) / len(y))

    explained = classifier.explainable_predict(X)
    uncovered = explained.winning_rule == -1
    assert uncovered.any() and not uncovered.all()
    np.testing.assert_array_equal(explained.association_degree[uncovered], 0.0)
    expected = 'Unknown' if allow_unknown else 'cat'
    np.testing.assert_array_equal(explained.prediction[uncovered], expected)
    np.testing.assert_array_equal(classifier.predict(X)[uncovered], expected)
    probabilities = classifier.predict_proba(X)
    np.testing.assert_array_equal(probabilities[uncovered], np.tile(priors, (uncovered.sum(), 1)))
    assert not np.any(np.all(probabilities[~uncovered] == priors, axis=1))


def test_temporal_rule_base_falls_back_to_each_moments_majority():
    X, y = _imbalanced()
    time_moments = np.arange(len(y)) % 2
    variables = [temporal.temporalFuzzyVariable(lv.name, [temporal.temporalFS(fuzzy_set, np.array([1.0, 0.25]))
                                                          for fuzzy_set in lv.linguistic_variables])
                 for lv in utils.construct_partitions(X, fs.FUZZY_SETS.t1)]
    np.random.seed(0)
    model = temporal.TemporalFuzzyRulesClassifier(nRules=4, nAnts=2, linguistic_variables=variables)
    model.fit(X, y, n_gen=2, pop_size=8, time_moments=time_moments)
    rule_base = model.rule_base

    winners, _ = rule_base._winning_rules(X, time_moments)
    uncovered = winners == -1
    assert uncovered.any()
    majority = np.array([mrb._uncovered_consequent() for mrb in rule_base.time_mrule_bases])
    for time, mrb in enumerate(rule_base.time_mrule_bases):
        slice_labels = y[time_moments == time]
        np.testing.assert_array_equal(mrb.class_priors, np.bincount(slice_labels, minlength=3) / len(slice_labels))
    np.testing.assert_array_equal(model.predict(X, time_moments)[uncovered], majority[time_moments[uncovered]])
    rule_base.allow_unknown = True
    np.testing.assert_array_equal(model.predict(X, time_moments)[uncovered], -1)
