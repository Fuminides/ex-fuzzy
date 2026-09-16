"""Tests for the vectorized Type-1 fuzzy regression optimizer."""

import builtins

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score


from ex_fuzzy import evolutionary_fit_regression as evr
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import rules
from ex_fuzzy import utils


def _partitions(X):
    return utils.construct_partitions(X, fs.FUZZY_SETS.t1, n_partitions=3)


def _unequal_partitions():
    feature_0 = fs.fuzzyVariable(
        "x0",
        [
            fs.FS("low", [0.0, 0.0, 0.3, 0.7], [0.0, 1.0]),
            fs.FS("high", [0.3, 0.7, 1.0, 1.0], [0.0, 1.0]),
        ],
    )
    feature_1 = fs.fuzzyVariable(
        "x1",
        [
            fs.FS("low", [0.0, 0.0, 0.2, 0.5], [0.0, 1.0]),
            fs.FS("medium", [0.2, 0.4, 0.6, 0.8], [0.0, 1.0]),
            fs.FS("high", [0.5, 0.8, 1.0, 1.0], [0.0, 1.0]),
        ],
    )
    return [feature_0, feature_1]


class TestFastPredictionEquivalence:
    def test_decoded_rulebase_matches_fast_path_with_duplicate_rules(self):
        X = np.array([[0.1, 0.2], [0.4, 0.8], [0.9, 0.6]])
        y = np.array([0.0, 5.0, 10.0])
        problem = evr.FitRuleBaseRegression(X, y, 3, 1, _partitions(X))

        # The first two rules deliberately have identical antecedents but distinct
        # consequents. They must remain separate because their repeated firing
        # changes the group's weight relative to the third rule.
        chromosome = np.array([0, 0, 1, 0, 0, 2, 0, 100, 50])
        fast = problem._fast_predict(chromosome)
        decoded = problem._construct_ruleBase(chromosome)

        assert len(decoded) == 3
        np.testing.assert_allclose(decoded.predict(X), fast, rtol=1e-12, atol=1e-12)

    def test_decoded_rulebase_matches_repeated_feature_and_invalid_term_semantics(self):
        X = np.array([[0.1, 0.2], [0.4, 0.8], [0.9, 0.6], [0.7, 0.1]])
        y = np.array([-2.0, 0.0, 4.0, 3.0])
        problem = evr.FitRuleBaseRegression(X, y, 2, 2, _unequal_partitions())

        # Rule 0 selects feature 0 twice; the last slot wins. Rule 1 includes a
        # term that is invalid for feature 0, which is consistently treated as
        # don't-care by both paths.
        chromosome = np.array([0, 0, 1, 0, 0, 1, 1, 2, 20, 80])
        fast = problem._fast_predict(chromosome)
        decoded = problem._construct_ruleBase(chromosome)

        np.testing.assert_allclose(decoded.predict(X), fast, rtol=1e-12, atol=1e-12)

    def test_all_dont_care_uses_mean_fallback_in_both_paths(self):
        X = np.array([[0.1, 0.2], [0.4, 0.8], [0.9, 0.6]])
        y = np.array([1.0, 4.0, 10.0])
        problem = evr.FitRuleBaseRegression(X, y, 2, 1, _partitions(X))
        chromosome = np.array([0, 1, -1, -1, 0, 100])

        decoded = problem._construct_ruleBase(chromosome)
        expected = np.full(X.shape[0], np.mean(y))
        assert len(decoded) == 0
        np.testing.assert_allclose(problem._fast_predict(chromosome), expected)
        np.testing.assert_allclose(decoded.predict(X), expected)

    def test_population_objective_is_negative_full_training_r2(self):
        X = np.array([[0.1, 0.2], [0.4, 0.8], [0.9, 0.6]])
        y = np.array([0.0, 5.0, 10.0])
        problem = evr.FitRuleBaseRegression(X, y, 2, 1, _partitions(X))
        chromosome = np.array([0, 1, 0, 2, 10, 90])
        prediction = problem._fast_predict(chromosome)
        output = {}

        problem._evaluate(chromosome[None, :], output)

        assert output["F"].shape == (1, 1)
        assert output["F"][0, 0] == pytest.approx(-r2_score(y, prediction))

    def test_random_valid_chromosomes_match_decoded_predictions(self):
        rng = np.random.default_rng(19)
        X = rng.uniform(0.0, 1.0, size=(25, 3))
        y = 3.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]
        problem = evr.FitRuleBaseRegression(X, y, 6, 3, _partitions(X))

        for _ in range(40):
            chromosome = rng.integers(problem.xl, problem.xu + 1)
            fast = problem._fast_predict(chromosome)
            decoded = problem._construct_ruleBase(chromosome).predict(X)
            np.testing.assert_allclose(decoded, fast, rtol=1e-12, atol=1e-12)


class TestTorchPopulationEvaluation:
    @pytest.mark.parametrize("consequent_type", ["crisp", "fuzzy"])
    @pytest.mark.parametrize("rule_mode", ["additive", "sufficient"])
    def test_torch_objective_matches_numpy_with_forced_chunks(
        self, consequent_type, rule_mode
    ):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(123)
        X = rng.uniform(-1.0, 1.0, size=(29, 4))
        y = 3.0 * X[:, 0] - 2.0 * X[:, 2] + 0.5
        problem = evr.FitRuleBaseRegression(
            X,
            y,
            nRules=7,
            nAnts=3,
            linguistic_variables=_partitions(X),
            consequent_type=consequent_type,
            rule_mode=rule_mode,
            tolerance=0.05,
        )
        population = rng.integers(
            problem.xl, problem.xu + 1, size=(9, problem.n_var)
        )
        numpy_output = {}
        problem._evaluate(population, numpy_output)

        torch_output = problem._evaluate_torch_population(
            population,
            device="cpu",
            population_batch_size=3,
            sample_batch_size=7,
        )

        assert torch_output.device.type == "cpu"
        np.testing.assert_allclose(
            torch_output.numpy(),
            numpy_output["F"].reshape(-1),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_cuda_objective_matches_numpy(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        rng = np.random.default_rng(321)
        X = rng.uniform(0.0, 1.0, size=(31, 3))
        y = X[:, 0] - 4.0 * X[:, 2]
        problem = evr.FitRuleBaseRegression(
            X, y, 6, 2, _partitions(X), consequent_type="fuzzy"
        )
        population = rng.integers(
            problem.xl, problem.xu + 1, size=(8, problem.n_var)
        )
        numpy_output = {}
        problem._evaluate(population, numpy_output)

        cuda_output = problem._evaluate_torch_population(
            population,
            device="cuda",
            population_batch_size=2,
            sample_batch_size=11,
        )

        assert cuda_output.device.type == "cuda"
        np.testing.assert_allclose(
            cuda_output.cpu().numpy(),
            numpy_output["F"].reshape(-1),
            rtol=2e-5,
            atol=2e-5,
        )


class TestRuleBaseT1Regression:
    def test_empty_rulebase_fallback_and_rule_validation(self):
        X = np.array([[0.1, 0.2], [0.4, 0.8]])
        linguistic_variables = _partitions(X)
        rulebase = evr.RuleBaseT1Regression(
            linguistic_variables, [], np.array([]), y_mean=1.25
        )
        np.testing.assert_allclose(rulebase.predict(X), [1.25, 1.25])
        assert rulebase.fuzzy_type() == fs.FUZZY_SETS.t1

        with pytest.raises(ValueError, match="same length"):
            evr.RuleBaseT1Regression(
                linguistic_variables, [rules.RuleSimple([0, 0])], np.array([])
            )

    def test_print_rules_includes_scalar_consequents(self):
        X = np.array([[0.1, 0.2], [0.4, 0.8]])
        linguistic_variables = _partitions(X)
        rulebase = evr.RuleBaseT1Regression(
            linguistic_variables,
            [rules.RuleSimple([0, -1])],
            np.array([3.25]),
            y_mean=2.0,
        )
        text = rulebase.print_rules(return_rules=True)
        assert "Rule 1: IF" in text
        assert "THEN output = 3.2500" in text


class TestBaseFuzzyRulesRegressor:
    @pytest.fixture
    def dataset(self):
        return make_regression(
            n_samples=50, n_features=3, n_informative=3, noise=0.1, random_state=7
        )

    def test_fit_predict_score_and_public_metadata(self, dataset):
        X, y = dataset
        frame = pd.DataFrame(X, columns=["a", "b", "c"])
        regressor = evr.BaseFuzzyRulesRegressor(nRules=5, nAnts=2)

        returned = regressor.fit(frame, y, n_gen=2, pop_size=8, random_state=7)
        prediction = regressor.predict(frame.iloc[:6])

        assert returned is regressor
        assert prediction.shape == (6,)
        assert np.all(np.isfinite(prediction))
        assert np.isfinite(regressor.score(frame, y))
        assert regressor.n_features_in_ == 3
        assert regressor.n_ants_ == 2
        assert list(regressor.feature_names_in_) == ["a", "b", "c"]
        assert regressor.performance == regressor.performance_
        assert regressor.get_rulebase() is regressor.rule_base
        assert regressor.optimization_result_ is not None

        with pytest.raises(ValueError, match="columns must match"):
            regressor.predict(frame[["b", "a", "c"]])

    def test_estimator_is_cloneable_and_fit_does_not_mutate_nants(self):
        X = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
        y = 2.0 * X[:, 0] + 1.0
        regressor = evr.BaseFuzzyRulesRegressor(nRules=3, nAnts=4)
        cloned = clone(regressor)

        regressor.fit(X, y, n_gen=1, pop_size=6, random_state=3)

        assert cloned.get_params() == evr.BaseFuzzyRulesRegressor(
            nRules=3, nAnts=4
        ).get_params()
        assert regressor.nAnts == 4
        assert regressor.n_ants_ == 1

    def test_predict_before_fit_and_wrong_feature_count_are_rejected(self, dataset):
        X, y = dataset
        regressor = evr.BaseFuzzyRulesRegressor(nRules=3, nAnts=2)
        with pytest.raises(NotFittedError):
            regressor.predict(X)

        regressor.fit(X, y, n_gen=1, pop_size=6, random_state=2)
        with pytest.raises(ValueError, match="expects 3"):
            regressor.predict(X[:, :2])

    def test_type2_and_mismatched_linguistic_variables_are_rejected(self, dataset):
        X, y = dataset
        with pytest.raises(ValueError, match="only Type-1"):
            evr.BaseFuzzyRulesRegressor(fuzzy_type=fs.FUZZY_SETS.t2).fit(
                X, y, n_gen=1, pop_size=4
            )

        with pytest.raises(ValueError, match="one linguistic variable per feature"):
            evr.BaseFuzzyRulesRegressor(
                linguistic_variables=_partitions(X)[:2]
            ).fit(X, y, n_gen=1, pop_size=4)

    def test_constant_target_produces_constant_finite_predictions(self):
        X = np.linspace(0.0, 1.0, 30).reshape(10, 3)
        y = np.full(10, 4.5)
        regressor = evr.BaseFuzzyRulesRegressor(nRules=3, nAnts=2)

        regressor.fit(X, y, n_gen=1, pop_size=6, random_state=5)
        prediction = regressor.predict(X)

        np.testing.assert_allclose(prediction, y)
        assert regressor.score(X, y) == pytest.approx(1.0)

    def test_default_backend_metadata_is_pymoo(self, dataset):
        X, y = dataset
        regressor = evr.BaseFuzzyRulesRegressor(nRules=3, nAnts=2)

        regressor.fit(X, y, n_gen=1, pop_size=6, random_state=5)

        assert regressor.backend == "pymoo"
        assert regressor.backend_ == "pymoo"
        assert regressor.optimization_device_ == "cpu"
        assert regressor.gpu_accelerated_ is False

    def test_evox_backend_trains_regressor(self, dataset):
        torch = pytest.importorskip("torch")
        try:
            __import__("evox")
        except Exception as exc:
            pytest.skip(f"EvoX cannot be imported: {exc}")
        X, y = dataset
        regressor = evr.BaseFuzzyRulesRegressor(
            nRules=5, nAnts=2, backend="evox"
        )

        regressor.fit(X, y, n_gen=2, pop_size=8, random_state=5)

        assert regressor.backend_ == "evox"
        assert regressor.optimization_device_ == (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        assert regressor.gpu_accelerated_ is torch.cuda.is_available()
        assert np.all(np.isfinite(regressor.predict(X[:5])))

    def test_unknown_backend_is_rejected(self, dataset):
        X, y = dataset
        with pytest.raises(ValueError, match="Unknown backend"):
            evr.BaseFuzzyRulesRegressor(
                nRules=3, nAnts=2, backend="not-a-backend"
            ).fit(X, y, n_gen=1, pop_size=6)


def _naive_mamdani(firing, consequent_ix, output_curves, universe, fallback):
    """Reference max-min Mamdani inference, looping exactly as the textbook states it.

    Deliberately does not share the grouping shortcut used by the library, so
    it is an independent check of that optimization.
    """
    predictions = np.zeros(firing.shape[0])
    for sample in range(firing.shape[0]):
        aggregated = np.zeros(universe.shape[0])
        for rule in range(firing.shape[1]):
            clipped = np.minimum(output_curves[consequent_ix[rule]], firing[sample, rule])
            aggregated = np.maximum(aggregated, clipped)
        mass = aggregated.sum()
        predictions[sample] = (
            (universe * aggregated).sum() / mass if mass > 1e-10 else fallback
        )
    return predictions


class TestMamdaniConsequents:
    def _problem(self, X, y, nRules=5, nAnts=2, n_output_lvs=3):
        return evr.FitRuleBaseRegression(
            X, y, nRules, nAnts, _partitions(X),
            consequent_type="fuzzy", n_output_lvs=n_output_lvs,
        )

    def test_fast_path_matches_decoded_rulebase_over_random_chromosomes(self):
        rng = np.random.default_rng(5)
        X = rng.uniform(0.0, 1.0, size=(30, 3))
        y = 2.0 * X[:, 0] - X[:, 1]
        problem = self._problem(X, y, nRules=6, nAnts=3)

        for _ in range(40):
            chromosome = rng.integers(problem.xl, problem.xu + 1)
            fast = problem._fast_predict(chromosome)
            decoded = problem._construct_ruleBase(chromosome).predict(X)
            np.testing.assert_allclose(decoded, fast, rtol=1e-12, atol=1e-12)

    def test_grouped_aggregation_matches_naive_rule_by_rule_inference(self):
        rng = np.random.default_rng(23)
        X = rng.uniform(0.0, 1.0, size=(20, 3))
        y = X[:, 0] + X[:, 2]
        problem = self._problem(X, y, nRules=7, nAnts=2, n_output_lvs=4)
        X_new = rng.uniform(-0.2, 1.2, size=(15, 3))

        for _ in range(25):
            chromosome = rng.integers(problem.xl, problem.xu + 1)
            rulebase = problem._construct_ruleBase(chromosome)
            if len(rulebase) == 0:
                continue
            firing = rulebase.compute_rule_antecedent_memberships(X_new)
            consequent_ix = np.array([int(r.consequent) for r in rulebase.rules])
            expected = _naive_mamdani(
                firing, consequent_ix, rulebase._output_curves,
                rulebase.universe, float(rulebase.y_mean),
            )
            np.testing.assert_allclose(
                rulebase.predict(X_new), expected, rtol=1e-12, atol=1e-12
            )

    def test_decoded_output_sets_are_always_ordered_trapezoids_in_target_range(self):
        rng = np.random.default_rng(31)
        X = rng.uniform(0.0, 1.0, size=(15, 2))
        y = rng.uniform(-4.0, 9.0, size=15)
        problem = self._problem(X, y, n_output_lvs=4)

        for _ in range(30):
            params = problem._decode_output_sets(rng.integers(problem.xl, problem.xu + 1))
            assert params.shape == (4, 4)
            assert np.all(np.diff(params, axis=1) >= 0)
            assert np.all(params >= y.min() - 1e-12)
            assert np.all(params <= y.max() + 1e-12)

    def test_all_dont_care_rules_fall_back_to_the_target_mean(self):
        X = np.array([[0.1, 0.2], [0.4, 0.8], [0.9, 0.6]])
        y = np.array([1.0, 4.0, 10.0])
        problem = evr.FitRuleBaseRegression(
            X, y, 2, 1, _partitions(X), consequent_type="fuzzy", n_output_lvs=2
        )
        chromosome = np.concatenate([[0, 1], [-1, -1], [0, 1], np.full(8, 50)])

        decoded = problem._construct_ruleBase(chromosome)
        expected = np.full(X.shape[0], np.mean(y))
        assert len(decoded) == 0
        np.testing.assert_allclose(problem._fast_predict(chromosome), expected)
        np.testing.assert_allclose(decoded.predict(X), expected)

    def test_fit_produces_linguistic_rules_and_beats_the_mean_baseline(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(0.0, 1.0, size=(120, 2))
        y = 5.0 * X[:, 0] + 2.0 * X[:, 1]
        regressor = evr.BaseFuzzyRulesRegressor(
            nRules=12, nAnts=2, consequent_type="fuzzy", n_output_lvs=4
        )

        regressor.fit(X, y, n_gen=30, pop_size=30, random_state=0)
        text = regressor.print_rules(return_rules=True)

        assert isinstance(regressor.get_rulebase(), evr.RuleBaseT1MamdaniRegression)
        assert "THEN output IS Output_" in text
        assert regressor.score(X, y) > 0.5
        assert np.all(np.isfinite(regressor.predict(X)))

    def test_invalid_mamdani_settings_are_rejected(self):
        X = np.linspace(0.0, 1.0, 30).reshape(15, 2)
        y = X[:, 0] * 3.0

        with pytest.raises(ValueError, match="consequent_type must be"):
            evr.BaseFuzzyRulesRegressor(consequent_type="mamdani").fit(
                X, y, n_gen=1, pop_size=4
            )
        with pytest.raises(ValueError, match="n_output_lvs must be"):
            evr.BaseFuzzyRulesRegressor(consequent_type="fuzzy", n_output_lvs=1).fit(
                X, y, n_gen=1, pop_size=4
            )

    def test_rulebase_rejects_a_rule_naming_a_missing_output_set(self):
        X = np.array([[0.1, 0.2], [0.4, 0.8]])
        linguistic_variables = _partitions(X)
        output_sets = [fs.FS("Output_0", [0.0, 1.0, 2.0, 3.0], [0.0, 3.0])]

        with pytest.raises(ValueError, match="does not exist"):
            evr.RuleBaseT1MamdaniRegression(
                linguistic_variables,
                [rules.RuleSimple([0, -1], consequent=2)],
                output_sets,
                y_mean=1.5,
            )

    def test_estimator_clones_with_the_new_parameters(self):
        regressor = evr.BaseFuzzyRulesRegressor(
            nRules=4, consequent_type="fuzzy", n_output_lvs=5, n_universe_points=51
        )
        params = clone(regressor).get_params()
        assert params["consequent_type"] == "fuzzy"
        assert params["n_output_lvs"] == 5
        assert params["n_universe_points"] == 51


class TestRuleModes:
    def test_sufficient_mode_keeps_only_the_strongest_rule(self):
        firing = np.array([[0.2, 0.9, 0.5], [0.0, 0.0, 0.0], [0.7, 0.1, 0.7]])

        additive = evr._apply_rule_mode(firing, "additive", 0.0)
        sufficient = evr._apply_rule_mode(firing, "sufficient", 0.0)

        np.testing.assert_allclose(additive, firing)
        # Ties resolve to the first index, matching numpy argmax.
        np.testing.assert_allclose(
            sufficient, [[0.0, 0.9, 0.0], [0.0, 0.0, 0.0], [0.7, 0.0, 0.0]]
        )

    def test_tolerance_silences_rules_that_are_too_weak(self):
        firing = np.array([[0.05, 0.02], [0.4, 0.1]])

        result = evr._apply_rule_mode(firing, "sufficient", 0.1)

        np.testing.assert_allclose(result, [[0.0, 0.0], [0.4, 0.0]])

        empty = np.empty((2, 0))
        assert evr._apply_rule_mode(empty, "sufficient", 0.1) is empty

    def test_crisp_sufficient_predictions_come_from_a_single_consequent(self):
        rng = np.random.default_rng(8)
        X = rng.uniform(0.0, 1.0, size=(40, 2))
        y = 3.0 * X[:, 0]
        regressor = evr.BaseFuzzyRulesRegressor(
            nRules=6, nAnts=2, rule_mode="sufficient", tolerance=0.0
        )

        regressor.fit(X, y, n_gen=10, pop_size=10, random_state=1)
        prediction = regressor.predict(X)

        allowed = set(np.round(regressor.rule_base.scalar_consequents, 9))
        allowed.add(round(float(regressor.y_mean_), 9))
        assert set(np.round(prediction, 9)) <= allowed

    def test_weak_firing_falls_back_to_the_target_mean(self):
        X = np.array([[0.1, 0.2], [0.4, 0.8]])
        linguistic_variables = _partitions(X)
        rulebase = evr.RuleBaseT1Regression(
            linguistic_variables,
            [rules.RuleSimple([0, -1])],
            np.array([9.0]),
            y_mean=2.0,
            rule_mode="sufficient",
            tolerance=1.1,  # above any possible membership, so nothing can fire
        )
        np.testing.assert_allclose(rulebase.predict(X), [2.0, 2.0])

    @pytest.mark.parametrize("consequent_type", ["crisp", "fuzzy"])
    def test_fast_path_matches_decoded_rulebase_in_sufficient_mode(self, consequent_type):
        rng = np.random.default_rng(12)
        X = rng.uniform(0.0, 1.0, size=(25, 3))
        y = X[:, 0] - 2.0 * X[:, 2]
        problem = evr.FitRuleBaseRegression(
            X, y, 5, 2, _partitions(X),
            consequent_type=consequent_type, rule_mode="sufficient", tolerance=0.05,
        )

        for _ in range(30):
            chromosome = rng.integers(problem.xl, problem.xu + 1)
            fast = problem._fast_predict(chromosome)
            decoded = problem._construct_ruleBase(chromosome).predict(X)
            np.testing.assert_allclose(decoded, fast, rtol=1e-12, atol=1e-12)

    def test_invalid_rule_mode_and_tolerance_are_rejected(self):
        X = np.linspace(0.0, 1.0, 30).reshape(15, 2)
        y = X[:, 0] * 2.0

        with pytest.raises(ValueError, match="rule_mode must be"):
            evr.BaseFuzzyRulesRegressor(rule_mode="winner").fit(X, y, n_gen=1, pop_size=4)
        with pytest.raises(ValueError, match="tolerance must be non-negative"):
            evr.BaseFuzzyRulesRegressor(tolerance=-0.1).fit(X, y, n_gen=1, pop_size=4)


class TestRegressionValidationAndHelperEdges:
    @pytest.mark.parametrize(
        "value, message",
        [
            (np.array([1.0]), "two-dimensional"),
            (np.empty((0, 1)), "at least one sample"),
            (np.empty((1, 0)), "at least one sample"),
            (np.array([[np.nan]]), "finite numeric"),
        ],
    )
    def test_input_array_validation(self, value, message):
        with pytest.raises(ValueError, match=message):
            evr._as_2d_float_array(value)

    def test_target_array_validation_and_column_flattening(self):
        np.testing.assert_array_equal(
            evr._as_1d_float_array(np.array([[1.0], [2.0]])), [1.0, 2.0]
        )
        invalid = [
            (np.array([[1.0, 2.0]]), None, "one-dimensional"),
            (np.array([]), None, "at least one target"),
            (np.array([1.0]), 2, "inconsistent sample counts"),
            (np.array([np.inf]), None, "finite numeric"),
        ]
        for target, expected_samples, message in invalid:
            with pytest.raises(ValueError, match=message):
                evr._as_1d_float_array(target, expected_samples=expected_samples)

    def test_linguistic_variable_validation(self):
        class Variable:
            def __init__(self, names, fuzzy_type):
                self._names = names
                self._fuzzy_type = fuzzy_type

            def linguistic_variable_names(self):
                return self._names

            def fuzzy_type(self):
                return self._fuzzy_type

        with pytest.raises(ValueError, match="At least one"):
            evr._validate_linguistic_variables([], 0)
        with pytest.raises(ValueError, match="got 0"):
            evr._validate_linguistic_variables(None, 1)
        with pytest.raises(ValueError, match="contains no fuzzy sets"):
            evr._validate_linguistic_variables([Variable([], fs.FUZZY_SETS.t1)], 1)
        with pytest.raises(ValueError, match="only Type-1"):
            evr._validate_linguistic_variables([Variable(["set"], fs.FUZZY_SETS.t2)], 1)

    def test_rulebase_protocol_validation_modifiers_and_printing(self, capsys):
        X = np.array([[0.1, 0.2], [0.4, 0.8]])
        partitions = _partitions(X)
        with pytest.raises(ValueError, match="antecedents; expected"):
            evr.RuleBaseT1Regression(
                partitions, [rules.RuleSimple([0])], np.array([1.0])
            )
        with pytest.raises(ValueError, match="invalid term index"):
            evr.RuleBaseT1Regression(
                partitions, [rules.RuleSimple([99, -1])], np.array([1.0])
            )
        with pytest.raises(ValueError, match="one-dimensional"):
            evr.RuleBaseT1Regression(partitions, [], np.empty((0, 1)))
        with pytest.raises(ValueError, match="finite values"):
            evr.RuleBaseT1Regression(
                partitions, [rules.RuleSimple([0, -1])], np.array([np.nan])
            )

        inactive = rules.RuleSimple([-1, -1])
        modified = rules.RuleSimple([0, -1], modifiers=np.array([2.0, -1.0]))
        rulebase = evr.RuleBaseT1Regression(
            partitions, [inactive, modified], np.array([1.0, 2.0])
        )
        cached = rulebase.compute_antecedents_memberships(X)
        firing = rulebase.compute_rule_antecedent_memberships(X, cached)
        assert np.all(firing[:, 0] == 0.0)
        np.testing.assert_allclose(firing[:, 1], cached[0][0] ** 2)
        with pytest.raises(ValueError, match="one entry per feature"):
            rulebase.compute_rule_antecedent_memberships(X, cached[:1])
        assert rulebase.get_rules() == [inactive, modified]
        assert rulebase[1] is modified
        assert list(rulebase) == [inactive, modified]

        assert rulebase.print_rules() is None
        assert "THEN output" in capsys.readouterr().out

    def test_mamdani_constructor_universe_and_print_edges(self, capsys):
        X = np.array([[0.1, 0.2], [0.4, 0.8]])
        partitions = _partitions(X)
        with pytest.raises(ValueError, match="At least one output"):
            evr.RuleBaseT1MamdaniRegression(partitions, [], [])

        output_set = fs.FS("same", [1.0, 1.0, 1.0, 1.0], [1.0, 1.0])
        with pytest.raises(ValueError, match="at least two"):
            evr.RuleBaseT1MamdaniRegression(
                partitions, [], [output_set], n_universe_points=1
            )
        constant = evr.RuleBaseT1MamdaniRegression(
            partitions,
            [rules.RuleSimple([0, -1], consequent=0)],
            [output_set],
        )
        np.testing.assert_array_equal(constant.universe, [1.0])
        assert constant.print_rules() is None
        assert "THEN output IS same" in capsys.readouterr().out

    def test_problem_parameter_membership_and_chromosome_validation(self):
        X = np.array([[0.0, 0.2], [0.5, 0.7], [1.0, 0.9]])
        y = np.array([0.0, 1.0, 2.0])
        partitions = _partitions(X)
        invalid = [
            ({"nRules": 0, "nAnts": 1}, "nRules"),
            ({"nRules": 1, "nAnts": 0}, "nAnts must"),
            ({"nRules": 1, "nAnts": 3}, "cannot exceed"),
            ({"nRules": 1, "nAnts": 1, "consequent_type": "other"}, "consequent_type"),
            ({"nRules": 1, "nAnts": 1, "consequent_type": "fuzzy", "n_output_lvs": 1}, "n_output_lvs"),
            ({"nRules": 1, "nAnts": 1, "consequent_type": "fuzzy", "n_universe_points": 1}, "n_universe_points"),
            ({"nRules": 1, "nAnts": 1, "y_min": 2.0, "y_max": 1.0}, "y_min"),
        ]
        for kwargs, message in invalid:
            with pytest.raises(ValueError, match=message):
                evr.FitRuleBaseRegression(X, y, linguistic_variables=partitions, **kwargs)

        class BadMembershipVariable:
            def __init__(self, memberships):
                self.memberships = memberships

            def linguistic_variable_names(self):
                return ["only"]

            def fuzzy_type(self):
                return fs.FUZZY_SETS.t1

            def compute_memberships(self, values):
                return self.memberships

        for memberships, message in [
            (np.zeros((1, 2)), "have shape"),
            (np.array([[0.0, np.nan, 1.0]]), "non-finite"),
        ]:
            bad = [BadMembershipVariable(memberships), partitions[1]]
            with pytest.raises(ValueError, match=message):
                evr.FitRuleBaseRegression(X, y, 1, 1, bad)

        problem = evr.FitRuleBaseRegression(X, y, 1, 1, partitions)
        with pytest.raises(ValueError, match="chromosome must have shape"):
            problem._rule_term_matrix(np.zeros(2))
        output = {}
        problem._evaluate(np.zeros(problem.n_var), output)
        assert output["F"].shape == (1, 1)

    def test_torch_error_shape_cache_and_constant_target_paths(self, monkeypatch):
        torch = pytest.importorskip("torch")
        X = np.array([[0.0, 0.2], [0.5, 0.7], [1.0, 0.9]])
        y = np.array([0.0, 1.0, 2.0])
        problem = evr.FitRuleBaseRegression(X, y, 2, 1, _partitions(X))

        chromosome = np.zeros(problem.n_var)
        chromosome[2:4] = -1
        terms = problem._torch_rule_terms(torch.tensor(chromosome)[None, :], torch)
        assert torch.all(terms == problem._dont_care)

        first = problem._torch_tensors(torch.device("cpu"), torch)
        assert problem._torch_tensors(torch.device("cpu"), torch) is first
        assert problem._evaluate_torch_population(
            torch.tensor(chromosome), device="cpu"
        ).shape == (1,)
        with pytest.raises(ValueError, match="population must have shape"):
            problem._evaluate_torch_population(
                torch.zeros((2, problem.n_var - 1)), device="cpu"
            )

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (10_000_000, 20_000_000))
        assert all(size >= 1 for size in problem._torch_chunk_sizes(3, "cuda"))

        def no_memory_info(device):
            raise RuntimeError("unavailable")

        monkeypatch.setattr(torch.cuda, "mem_get_info", no_memory_info)
        assert all(size >= 1 for size in problem._torch_chunk_sizes(3, "cuda"))

        constant = evr.FitRuleBaseRegression(
            X, np.full(3, 4.0), 1, 1, _partitions(X)
        )
        constant_gene = np.zeros(constant.n_var)
        constant._evaluate_torch_population(constant_gene, device="cpu")
        assert constant._fitness(np.zeros(3)) == 0.0

        original_import = builtins.__import__

        def without_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("missing torch")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", without_torch)
        with pytest.raises(ImportError, match="PyTorch is required"):
            problem._evaluate_torch_population(chromosome)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"nRules": 0}, "nRules"),
            ({"nAnts": 0}, "nAnts"),
            ({"n_linguistic_variables": 0}, "n_linguistic_variables"),
            ({"consequent_type": "fuzzy", "n_universe_points": 1}, "n_universe_points"),
        ],
    )
    def test_estimator_parameter_validation(self, kwargs, message):
        X = np.arange(12.0).reshape(6, 2)
        y = np.arange(6.0)
        with pytest.raises(ValueError, match=message):
            evr.BaseFuzzyRulesRegressor(**kwargs).fit(X, y, n_gen=1, pop_size=4)

        valid = evr.BaseFuzzyRulesRegressor(nRules=2, nAnts=1)
        with pytest.raises(ValueError, match="n_gen"):
            valid.fit(X, y, n_gen=0, pop_size=4)
        with pytest.raises(ValueError, match="pop_size"):
            valid.fit(X, y, n_gen=1, pop_size=1)

    def test_verbose_fit_and_invalid_optimizer_result(self, monkeypatch, capsys):
        X = np.linspace(0.0, 1.0, 8)[:, None]
        y = 2.0 * X[:, 0]
        verbose = evr.BaseFuzzyRulesRegressor(nRules=2, nAnts=3, verbose=True)
        verbose.fit(X, y, n_gen=1, pop_size=4, random_state=0)
        output = capsys.readouterr().out
        assert "nAnts exceeds" in output
        assert "Final:" in output

        class InvalidBackend:
            def optimize(self, **kwargs):
                return {"X": None, "F": None}

        monkeypatch.setattr(evr.ev_backends, "get_backend", lambda name: InvalidBackend())
        with pytest.raises(RuntimeError, match="did not return a valid solution"):
            evr.BaseFuzzyRulesRegressor(nRules=2, nAnts=1).fit(
                X, y, n_gen=1, pop_size=4
            )
