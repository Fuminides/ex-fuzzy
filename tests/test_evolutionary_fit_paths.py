"""Tests for the less common configuration and fitting paths of the genetic classifier."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

import _population_fitness as popfit
import evolutionary_backends as backends
import evolutionary_fit as evf
import fuzzy_sets as fs
import rules
import utils
from _fitness import _fitness_cache_scope


@pytest.fixture(scope='module')
def iris():
    return load_iris(return_X_y=True)


def _fixed_problem(nRules=6, nAnts=2, **kwargs):
    X, y = load_iris(return_X_y=True)
    return evf.FitRuleBase(X, y, nRules, nAnts, 3,
                           linguistic_variables=utils.construct_partitions(X, fs.FUZZY_SETS.t1), **kwargs)


def test_constructor_options(iris, capsys):
    X, _ = iris
    names = np.array(['setosa', 'versicolor', 'virginica'])
    assert evf.BaseFuzzyRulesClassifier(class_names=names).classes_names == list(names)
    assert evf.BaseFuzzyRulesClassifier(class_names=['a', 'b']).classes_names == ['a', 'b']

    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    model = evf.BaseFuzzyRulesClassifier(nAnts=6, linguistic_variables=partitions, verbose=True)
    assert model.nAnts == 4
    assert 'Setting nAnts to the number of linguistic variables. (4)' in capsys.readouterr().out

    quiet = evf.BaseFuzzyRulesClassifier(
        nAnts=6, linguistic_variables=partitions, verbose=False
    )
    assert quiet.nAnts == 4

    fallback = evf.BaseFuzzyRulesClassifier(backend='missing', verbose=False)
    assert fallback.backend.name() == 'pymoo'


def test_fit_reports_categorical_variables_and_trims_antecedents(capsys):
    frame = pd.DataFrame({'size': np.linspace(0, 1, 60), 'colour': ['red', 'green', 'blue'] * 20})
    y = (frame['size'] > 0.5).astype(int).to_numpy()
    model = evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=5, verbose=True)

    model.fit(frame, y, n_gen=2, pop_size=6, patience=0)

    output = capsys.readouterr().out
    assert 'Detected categorical variables: [1]' in output
    assert 'Setting nAnts to the number of variables. (2)' in output
    # A non-positive patience runs every generation.
    assert model.n_generations_run_ == 2
    assert model.var_names == ['size', 'colour']


def test_quiet_fit_accepts_per_feature_partition_counts_and_trims_antecedents():
    X = np.column_stack([np.linspace(0, 1, 20), np.linspace(1, 2, 20)])
    y = np.arange(20) % 2
    model = evf.BaseFuzzyRulesClassifier(
        nRules=3, nAnts=4, n_linguistic_variables=[3, 4], verbose=False
    )

    model.fit(X, y, n_gen=1, pop_size=4, random_state=0, patience=None)

    assert model.nAnts == 2
    assert model.n_linguist_variables == [3, 4]


def test_encoding_round_trips_through_the_decoder(iris):
    X, y = iris
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    problem = evf.FitRuleBase(X, y, 5, 4, 3, linguistic_variables=partitions, ds_mode=2)
    rule_lists = [[rules.RuleSimple([0, -1, -1, -1]), rules.RuleSimple([-1, 1, 2, -1])],
                  [rules.RuleSimple([-1, -1, 1, 1])],
                  [rules.RuleSimple([2, -1, -1, 2])]]
    for rule_list, weight in zip(rule_lists, [0.25, 0.5, 0.75]):
        for rule in rule_list:
            rule.weight = weight
    rule_base = rules.MasterRuleBase([rules.RuleBaseT1(partitions, rule_list) for rule_list in rule_lists], ds_mode=2)

    gene = problem.encode_rulebase(rule_base, optimize_lv=False)
    decoded = problem._construct_ruleBase(gene, fs.FUZZY_SETS.t1)
    for original, rebuilt in zip(rule_base, decoded):
        np.testing.assert_array_equal(original.get_rulebase_matrix(), rebuilt.get_rulebase_matrix())
        assert [rule.weight for rule in rebuilt] == [rule.weight for rule in original]
    # The fifth rule slot stays empty.
    assert len(decoded.get_rules()) == 4

    with pytest.raises(NotImplementedError):
        problem.encode_rulebase(rule_base, optimize_lv=True)
    narrow = evf.FitRuleBase(X, y, 5, 2, 3, linguistic_variables=partitions)
    with pytest.raises(ValueError, match='one antecedent slot per feature'):
        narrow.encode_rulebase(rule_base, optimize_lv=False)


def test_fit_starts_from_initial_rules(iris):
    X, y = iris
    first = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2)
    first.fit(X, y, n_gen=3, pop_size=8, random_state=0)

    refined = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2)
    refined.fit(X, y, n_gen=2, pop_size=8, initial_rules=first.rule_base, random_state=0)

    assert refined.nAnts == 4
    assert refined.lvs is first.rule_base[0].antecedents
    assert refined.predict(X).shape == y.shape


def test_checkpoints_are_written_to_the_given_folder(iris, tmp_path):
    X, y = iris
    model = evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=2)
    model.fit(X, y, n_gen=3, pop_size=6, checkpoints=2, checkpoint_path=str(tmp_path), random_state=0, patience=None)

    assert sorted(path.name for path in tmp_path.iterdir()) == ['checkpoint_0', 'checkpoint_2']
    assert (tmp_path / 'checkpoint_0').read_text().startswith('Rules for consequent')


def test_checkpoints_are_skipped_with_non_pymoo_backends(iris, capsys):
    class BackendWithoutCheckpoints:
        def name(self):
            return 'test-backend'

        def optimize(self, **kwargs):
            return backends.PyMooBackend().optimize(**kwargs)

    X, y = iris
    model = evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=2, verbose=True)
    model.backend = BackendWithoutCheckpoints()

    model.fit(X, y, n_gen=2, pop_size=6, checkpoints=1, random_state=0)

    assert 'Checkpoints are not yet supported with test-backend backend' in capsys.readouterr().out
    assert model.n_generations_run_ == 2

    quiet = evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=2, verbose=False)
    quiet.backend = BackendWithoutCheckpoints()
    quiet.fit(X, y, n_gen=1, pop_size=4, checkpoints=1, random_state=0)
    assert quiet.n_generations_run_ == 1


def test_p_value_validation_annotates_the_model(iris, capsys):
    X, y = iris
    np.random.seed(0)
    model = evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=2)
    model.fit(X, y, n_gen=2, pop_size=6, random_state=0, p_value_compute=True, bootstrap_size=5)

    assert 0 < model.p_value_class_structure <= 1
    assert 0 < model.p_value_feature_coalitions <= 1
    model.print_rule_bootstrap_results()
    assert 'Rules for consequent' in capsys.readouterr().out


def test_general_type_2_and_temporal_partitions_cannot_be_optimized(iris):
    X, y = iris
    for fuzzy_type in (fs.FUZZY_SETS.gt2, fs.FUZZY_SETS.temporal):
        with pytest.raises(ValueError, match='precomputed linguistic_variables'):
            evf.FitRuleBase(X, y, 4, 2, 3, fuzzy_type=fuzzy_type)


def test_custom_losses_score_empty_rule_bases_as_zero():
    problem = _fixed_problem(nRules=4)
    problem.fitness_func = lambda *args: 1.0
    gene = np.zeros(problem.n_var, dtype=int)

    gene[2 * 4 * 2:] = -1
    out = {}
    problem._evaluate(gene, out)
    assert out['F'] == 1.0

    gene[2 * 4 * 2:] = 0
    problem._evaluate(gene, out)
    assert out['F'] == 0.0


def test_population_route_guards(monkeypatch):
    problem = _fixed_problem()
    genes = np.random.default_rng(0).integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                                              size=(8, problem.n_var))
    # Outside a fit there is no route probe.
    assert problem._route_probe_state(8) is None

    with _fitness_cache_scope(problem, True):
        assert problem._population_scores(genes.astype(str)) is None

        chunk = popfit.chunk_size(len(problem.X), problem.nRules, problem.X.shape[1])
        assert problem._population_chunks(1) is None
        # A trailing single candidate joins the previous chunk.
        assert problem._population_chunks(chunk + 1) == [slice(0, chunk + 1)]

        monkeypatch.setattr(problem, '_packed_memberships', lambda: None)
        assert problem._population_scores(genes) is None
        assert problem._batched_scores(genes) is None

        class BatchProbe:
            decision = None

            def route(self, count):
                return popfit._RouteProbe.BATCH

        monkeypatch.setattr(problem, '_route_probe_state', lambda population: BatchProbe())
        assert problem._score_fresh(genes, len(genes)) is None


def test_array_score_declines_unsupported_candidates(iris):
    X, y = iris
    general = evf.FitRuleBase(X, y, 4, 2, 3, linguistic_variables=utils.construct_partitions(X, fs.FUZZY_SETS.gt2))
    assert general._array_score(np.zeros(general.n_var, dtype=int)) is None

    problem = _fixed_problem()
    gene = np.zeros(problem.n_var, dtype=int)
    invalid_consequent = gene.copy()
    invalid_consequent[2 * 6 * 2] = 3
    assert problem._array_score(invalid_consequent) is None

    problem.lvs = np.array(problem.lvs, dtype=object)
    assert problem._array_score(gene) is None

    without_memberships = _fixed_problem()
    without_memberships._precomputed_truth = None
    without_memberships._packed_memberships_cache = None
    assert without_memberships._array_score(gene) is None

    optimized = evf.FitRuleBase(X, y, 4, 2, 3)
    assert optimized._packed_memberships() is None


def test_rename_fuzzy_variables_skips_categorical_variables():
    class CategoricalSet:
        def shape(self):
            return 'categorical'

    class CategoricalVariable:
        linguistic_variables = [CategoricalSet()]

        def __getitem__(self, index):
            return self.linguistic_variables[index]

    class RuleBase:
        antecedents = [CategoricalVariable()]

    class MasterRuleBase:
        rule_bases = [RuleBase()]

        def __len__(self):
            return 1

    model = evf.BaseFuzzyRulesClassifier(n_linguistic_variables=3)
    model.rule_base = MasterRuleBase()

    model.rename_fuzzy_variables()

    X = np.linspace(0, 1, 12)[:, None]
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    ordinary = evf.BaseFuzzyRulesClassifier(n_linguistic_variables=3)
    ordinary.n_linguist_variables = [3]
    ordinary.rule_base = rules.MasterRuleBase([rules.RuleBaseT1(partitions, [])])
    ordinary.rename_fuzzy_variables()
    assert partitions[0].linguistic_variable_names() == ['Low', 'Medium', 'High']
