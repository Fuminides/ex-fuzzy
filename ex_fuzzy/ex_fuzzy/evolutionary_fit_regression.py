"""Fast evolutionary learning for Type-1 fuzzy regression rule bases.

This module learns zero-order Takagi-Sugeno rules with scalar consequents.  A
prediction is the firing-strength-weighted average of those consequents.  The
linguistic variables are fixed during optimization, which lets the optimizer
precompute all memberships and score chromosomes without constructing Python
rule objects in every fitness evaluation.

The vectorized fitness path and the decoded :class:`RuleBaseT1Regression`
intentionally use the same chromosome semantics.  In particular, duplicate
rules are retained because repeated rules change their relative weight in a
weighted-average rule base.
"""

from typing import Optional

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted

try:
    from . import fuzzy_sets as fs
    from . import rules
    from . import utils
    from . import evolutionary_backends as ev_backends
except ImportError:
    import fuzzy_sets as fs
    import rules
    import utils
    import evolutionary_backends as ev_backends


def _as_2d_float_array(X, *, expected_features: Optional[int] = None) -> np.ndarray:
    """Validate and return a finite two-dimensional floating-point array."""
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array of shape (samples, features).")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must contain at least one sample and one feature.")
    if expected_features is not None and X.shape[1] != expected_features:
        raise ValueError(
            f"X has {X.shape[1]} features, but the fitted regressor expects "
            f"{expected_features}."
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite numeric values.")
    return X


def _as_1d_float_array(y, *, expected_samples: Optional[int] = None) -> np.ndarray:
    """Validate and return a finite one-dimensional floating-point target."""
    y = np.asarray(y, dtype=float)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional target array.")
    if y.size == 0:
        raise ValueError("y must contain at least one target value.")
    if expected_samples is not None and y.shape[0] != expected_samples:
        raise ValueError(
            f"X and y have inconsistent sample counts: {expected_samples} and {y.shape[0]}."
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain only finite numeric values.")
    return y


def _apply_rule_mode(firing: np.ndarray, rule_mode: str, tolerance: float) -> np.ndarray:
    """Return the firing strengths that the selected rule mode lets through.

    ``additive`` keeps every rule.  ``sufficient`` keeps only each sample's
    strongest rule, and only when it clears ``tolerance``; a sample whose best
    rule is too weak ends up with no firing at all and falls back to the mean.
    """
    if rule_mode == "additive":
        return firing
    if firing.shape[1] == 0:
        return firing
    strongest = firing.argmax(axis=1)
    rows = np.arange(firing.shape[0])
    peak = firing[rows, strongest]
    fires = peak > tolerance
    winner = np.zeros_like(firing)
    winner[rows[fires], strongest[fires]] = peak[fires]
    return winner


def _validate_rule_mode(rule_mode: str, tolerance: float) -> None:
    """Validate the rule-firing mode and its tolerance."""
    if rule_mode not in ("additive", "sufficient"):
        raise ValueError("rule_mode must be either 'additive' or 'sufficient'.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")


def _validate_linguistic_variables(
    linguistic_variables: list[fs.fuzzyVariable], n_features: int
) -> None:
    """Validate fixed Type-1 linguistic variables used by the fast scorer."""
    if n_features <= 0:
        raise ValueError("At least one linguistic variable is required.")
    if linguistic_variables is None or len(linguistic_variables) != n_features:
        actual = 0 if linguistic_variables is None else len(linguistic_variables)
        raise ValueError(
            f"Expected one linguistic variable per feature ({n_features}); got {actual}."
        )
    for ix, variable in enumerate(linguistic_variables):
        if len(variable.linguistic_variable_names()) == 0:
            raise ValueError(f"Linguistic variable {ix} contains no fuzzy sets.")
        if variable.fuzzy_type() != fs.FUZZY_SETS.t1:
            raise ValueError("Fast fuzzy regression currently supports only Type-1 fuzzy sets.")


class _RegressionRuleBase:
    """Antecedent machinery shared by the scalar and Mamdani regression bases.

    Subclasses supply the consequent representation and :meth:`inference`.
    """

    def __init__(
        self,
        antecedents: list[fs.fuzzyVariable],
        rule_list: list[rules.RuleSimple],
        y_mean: Optional[float] = None,
        rule_mode: str = "additive",
        tolerance: float = 0.0,
        tnorm=np.prod,
    ) -> None:
        """Store the antecedents and validate every rule against them."""
        _validate_linguistic_variables(antecedents, len(antecedents))
        _validate_rule_mode(rule_mode, tolerance)
        self.antecedents = antecedents
        self.rules = list(rule_list)
        self.y_mean = y_mean
        self.rule_mode = rule_mode
        self.tolerance = float(tolerance)
        self.tnorm = tnorm

        n_features = len(self.antecedents)
        for ix, rule in enumerate(self.rules):
            if len(rule.antecedents) != n_features:
                raise ValueError(
                    f"Rule {ix} has {len(rule.antecedents)} antecedents; expected {n_features}."
                )
            for feature, term in enumerate(rule.antecedents):
                if term < -1 or term >= len(self.antecedents[feature].linguistic_variable_names()):
                    raise ValueError(f"Rule {ix} contains invalid term index {term} for feature {feature}.")

    def fuzzy_type(self) -> fs.FUZZY_SETS:
        """Return the fuzzy-set type used by this rule base."""
        return fs.FUZZY_SETS.t1

    def get_rules(self) -> list[rules.RuleSimple]:
        """Return the rules in chromosome order, including duplicates."""
        return self.rules

    def compute_antecedents_memberships(self, X) -> list[np.ndarray]:
        """Compute every input membership once for the supplied samples."""
        X = _as_2d_float_array(X, expected_features=len(self.antecedents))
        return [variable.compute_memberships(X[:, ix]) for ix, variable in enumerate(self.antecedents)]

    def compute_rule_antecedent_memberships(
        self, X, cached: Optional[list[np.ndarray]] = None
    ) -> np.ndarray:
        """Return the firing strength of each rule for each sample."""
        X = _as_2d_float_array(X, expected_features=len(self.antecedents))
        if cached is None:
            cached = self.compute_antecedents_memberships(X)
        if len(cached) != len(self.antecedents):
            raise ValueError("cached memberships must contain one entry per feature.")

        firing = np.zeros((X.shape[0], len(self.rules)), dtype=float)
        for rule_ix, rule in enumerate(self.rules):
            active = np.flatnonzero(np.asarray(rule.antecedents) >= 0)
            if active.size == 0:
                continue
            membership = np.ones((X.shape[0], active.size), dtype=float)
            for column, feature in enumerate(active):
                term = rule.antecedents[feature]
                values = np.asarray(cached[feature][term], dtype=float)
                modifiers = getattr(rule, "modifiers", None)
                if modifiers is not None and modifiers[feature] not in (-1, 1):
                    values = values ** modifiers[feature]
                membership[:, column] = values
            firing[:, rule_ix] = self.tnorm(membership, axis=1)
        return firing

    def _antecedent_string(self, rule: rules.RuleSimple) -> str:
        """Render the IF part of a rule."""
        return " AND ".join(
            f"{variable.name} IS {variable.linguistic_variable_names()[rule[feature]]}"
            for feature, variable in enumerate(self.antecedents)
            if rule[feature] >= 0
        )

    def predict(self, X) -> np.ndarray:
        """Alias for :meth:`inference`."""
        return self.inference(X)

    def __len__(self) -> int:
        return len(self.rules)

    def __getitem__(self, ix: int) -> rules.RuleSimple:
        return self.rules[ix]

    def __iter__(self):
        return iter(self.rules)


class RuleBaseT1Regression(_RegressionRuleBase):
    """Type-1 fuzzy rule base with scalar consequents.

    Predictions use zero-order Takagi-Sugeno (height) inference.  Samples for
    which no rule fires fall back to the training-target mean.
    """

    def __init__(
        self,
        antecedents: list[fs.fuzzyVariable],
        rule_list: list[rules.RuleSimple],
        scalar_consequents: np.ndarray,
        y_mean: Optional[float] = None,
        rule_mode: str = "additive",
        tolerance: float = 0.0,
        tnorm=np.prod,
    ) -> None:
        """Create a scalar-consequent Type-1 regression rule base."""
        scalar_consequents = np.asarray(scalar_consequents, dtype=float)
        if scalar_consequents.ndim != 1:
            raise ValueError("scalar_consequents must be a one-dimensional array.")
        if len(rule_list) != scalar_consequents.shape[0]:
            raise ValueError("rule_list and scalar_consequents must have the same length.")
        if not np.all(np.isfinite(scalar_consequents)):
            raise ValueError("scalar_consequents must contain only finite values.")

        super().__init__(
            antecedents, rule_list, y_mean=y_mean,
            rule_mode=rule_mode, tolerance=tolerance, tnorm=tnorm,
        )
        self.scalar_consequents = scalar_consequents.copy()
        for ix, rule in enumerate(self.rules):
            rule.consequent = float(self.scalar_consequents[ix])

    def inference(self, X, cached: Optional[list[np.ndarray]] = None) -> np.ndarray:
        """Predict continuous outputs with weighted-average inference."""
        X = _as_2d_float_array(X, expected_features=len(self.antecedents))
        fallback = 0.0 if self.y_mean is None else float(self.y_mean)
        if len(self.rules) == 0:
            return np.full(X.shape[0], fallback, dtype=float)

        firing = _apply_rule_mode(
            self.compute_rule_antecedent_memberships(X, cached), self.rule_mode, self.tolerance
        )
        denominator = np.sum(firing, axis=1)
        prediction = np.full(X.shape[0], fallback, dtype=float)
        covered = denominator > 1e-10
        prediction[covered] = (
            firing[covered] @ self.scalar_consequents
        ) / denominator[covered]
        # A weighted average cannot leave the consequent range; clipping to it
        # removes float division noise so a constant target stays bit-exact.
        if np.any(covered):
            prediction[covered] = np.clip(
                prediction[covered],
                self.scalar_consequents.min(),
                self.scalar_consequents.max(),
            )
        return prediction

    def print_rules(self, return_rules: bool = False):
        """Print or return human-readable scalar-consequent rules."""
        output = ""
        for rule_ix, rule in enumerate(self.rules, start=1):
            output += (
                f"Rule {rule_ix}: IF {self._antecedent_string(rule)} "
                f"THEN output = {float(rule.consequent):.4f}\n"
            )
        if return_rules:
            return output
        print(output)
        return None


class RuleBaseT1MamdaniRegression(_RegressionRuleBase):
    """Type-1 fuzzy rule base whose consequents are output fuzzy sets.

    Each rule names one output set.  Inference clips every consequent by its
    rule's firing strength (min), aggregates the clipped sets with max, and
    defuzzifies the result by taking its centroid over a discretized universe.
    Samples for which nothing fires fall back to the training-target mean.
    """

    def __init__(
        self,
        antecedents: list[fs.fuzzyVariable],
        rule_list: list[rules.RuleSimple],
        output_sets: list[fs.FS],
        y_mean: Optional[float] = None,
        universe: Optional[np.ndarray] = None,
        n_universe_points: int = 101,
        rule_mode: str = "additive",
        tolerance: float = 0.0,
        tnorm=np.prod,
    ) -> None:
        """Create a Mamdani regression rule base over the given output sets."""
        if len(output_sets) == 0:
            raise ValueError("At least one output fuzzy set is required.")
        super().__init__(
            antecedents, rule_list, y_mean=y_mean,
            rule_mode=rule_mode, tolerance=tolerance, tnorm=tnorm,
        )

        for ix, rule in enumerate(self.rules):
            if not 0 <= int(rule.consequent) < len(output_sets):
                raise ValueError(
                    f"Rule {ix} names output set {rule.consequent}, which does not exist."
                )
        self.output_sets = list(output_sets)

        if universe is None:
            if n_universe_points < 2:
                raise ValueError("n_universe_points must be at least two.")
            bounds = np.concatenate([np.asarray(o.membership_parameters, dtype=float)
                                     for o in self.output_sets])
            low, high = float(np.min(bounds)), float(np.max(bounds))
            universe = (
                np.array([low]) if high <= low
                else np.linspace(low, high, int(n_universe_points))
            )
        self.universe = np.asarray(universe, dtype=float)
        # Every consequent curve is fixed, so evaluate the universe only once.
        self._output_curves = np.vstack(
            [np.asarray(o.membership(self.universe), dtype=float) for o in self.output_sets]
        )

    def inference(self, X, cached: Optional[list[np.ndarray]] = None) -> np.ndarray:
        """Predict continuous outputs with max-min Mamdani inference."""
        X = _as_2d_float_array(X, expected_features=len(self.antecedents))
        fallback = 0.0 if self.y_mean is None else float(self.y_mean)
        if len(self.rules) == 0:
            return np.full(X.shape[0], fallback, dtype=float)

        firing = _apply_rule_mode(
            self.compute_rule_antecedent_memberships(X, cached), self.rule_mode, self.tolerance
        )
        consequent_ix = np.array([int(rule.consequent) for rule in self.rules], dtype=int)
        return _mamdani_defuzzify(
            firing, consequent_ix, self._output_curves, self.universe, fallback
        )

    def print_rules(self, return_rules: bool = False):
        """Print or return human-readable linguistic-consequent rules."""
        output = ""
        for rule_ix, rule in enumerate(self.rules, start=1):
            output += (
                f"Rule {rule_ix}: IF {self._antecedent_string(rule)} "
                f"THEN output IS {self.output_sets[int(rule.consequent)].name}\n"
            )
        if return_rules:
            return output
        print(output)
        return None


def _mamdani_defuzzify(
    firing: np.ndarray,
    consequent_ix: np.ndarray,
    output_curves: np.ndarray,
    universe: np.ndarray,
    fallback: float,
) -> np.ndarray:
    """Aggregate clipped consequents with max and return their centroids.

    ``min`` distributes over ``max``, so rules sharing an output set are first
    collapsed to that set's strongest firing.  That keeps the aggregation
    tensor proportional to the number of output sets rather than the number of
    rules, which is what makes the fitness loop affordable.
    """
    n_samples = firing.shape[0]
    n_sets = output_curves.shape[0]

    set_firing = np.zeros((n_samples, n_sets), dtype=float)
    for set_ix in range(n_sets):
        members = consequent_ix == set_ix
        if np.any(members):
            set_firing[:, set_ix] = firing[:, members].max(axis=1)

    aggregated = np.minimum(output_curves[None, :, :], set_firing[:, :, None]).max(axis=1)
    mass = aggregated.sum(axis=1)
    prediction = np.full(n_samples, fallback, dtype=float)
    covered = mass > 1e-10
    prediction[covered] = (aggregated[covered] @ universe) / mass[covered]
    return prediction


class FitRuleBaseRegression(Problem):
    """Pymoo problem for fast optimization of a fixed Type-1 partition.

    The integer chromosome always starts with ``nRules * nAnts`` feature
    indices followed by ``nRules * nAnts`` term indices (``-1`` means don't
    care).  The consequent genes depend on ``consequent_type``:

    ``crisp``
        ``nRules`` scalar consequents encoded into the target range.
    ``fuzzy``
        ``nRules`` output-set indices, then ``4 * n_output_lvs`` trapezoid
        breakpoints.  Each set's four breakpoints are sorted on decoding, so
        every chromosome yields a valid trapezoid.
    """

    def __init__(
        self,
        X,
        y,
        nRules: int,
        nAnts: int,
        linguistic_variables: list[fs.fuzzyVariable],
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        y_mean: Optional[float] = None,
        consequent_type: str = "crisp",
        n_output_lvs: int = 3,
        n_universe_points: int = 101,
        rule_mode: str = "additive",
        tolerance: float = 0.0,
    ) -> None:
        """Initialize the vectorized regression optimization problem."""
        self.X = _as_2d_float_array(X)
        self.y = _as_1d_float_array(y, expected_samples=self.X.shape[0])
        if not isinstance(nRules, (int, np.integer)) or nRules <= 0:
            raise ValueError("nRules must be a positive integer.")
        if not isinstance(nAnts, (int, np.integer)) or nAnts <= 0:
            raise ValueError("nAnts must be a positive integer.")
        if nAnts > self.X.shape[1]:
            raise ValueError("nAnts cannot exceed the number of input features.")
        if consequent_type not in ("crisp", "fuzzy"):
            raise ValueError("consequent_type must be either 'crisp' or 'fuzzy'.")
        if consequent_type == "fuzzy":
            if not isinstance(n_output_lvs, (int, np.integer)) or n_output_lvs < 2:
                raise ValueError("n_output_lvs must be an integer of at least two.")
            if not isinstance(n_universe_points, (int, np.integer)) or n_universe_points < 2:
                raise ValueError("n_universe_points must be an integer of at least two.")
        _validate_rule_mode(rule_mode, tolerance)
        _validate_linguistic_variables(linguistic_variables, self.X.shape[1])
        self.rule_mode = rule_mode
        self.tolerance = float(tolerance)
        self.consequent_type = consequent_type
        self.n_output_lvs = int(n_output_lvs)
        self.n_universe_points = int(n_universe_points)

        self.lvs = linguistic_variables
        self.nRules = int(nRules)
        self.nAnts = int(nAnts)
        self.n_lv = np.array(
            [len(variable.linguistic_variable_names()) for variable in self.lvs], dtype=int
        )
        self.y_min = float(np.min(self.y)) if y_min is None else float(y_min)
        self.y_max = float(np.max(self.y)) if y_max is None else float(y_max)
        self.y_mean = float(np.mean(self.y)) if y_mean is None else float(y_mean)
        if self.y_min > self.y_max:
            raise ValueError("y_min cannot be greater than y_max.")
        self.y_range = self.y_max - self.y_min

        antecedent_slots = self.nRules * self.nAnts
        if self.consequent_type == "fuzzy":
            consequent_bounds = [
                np.tile([0, self.n_output_lvs - 1], (self.nRules, 1)),
                np.tile([0, 100], (4 * self.n_output_lvs, 1)),
            ]
        else:
            consequent_bounds = [np.tile([0, 100], (self.nRules, 1))]
        bounds = np.concatenate(
            [
                np.tile([0, self.X.shape[1] - 1], (antecedent_slots, 1)),
                np.tile([-1, int(np.max(self.n_lv)) - 1], (antecedent_slots, 1)),
                *consequent_bounds,
            ],
            axis=0,
        )
        super().__init__(
            n_var=bounds.shape[0],
            n_obj=1,
            xl=bounds[:, 0],
            xu=bounds[:, 1],
            vtype=int,
        )
        self._precompute()

    def _precompute(self) -> None:
        """Precompute the dense membership tensor shared by all chromosomes."""
        n_samples, n_features = self.X.shape
        max_terms = int(np.max(self.n_lv))
        self._dont_care = max_terms

        membership = np.zeros((n_samples, n_features, max_terms + 1), dtype=float)
        membership[:, :, self._dont_care] = 1.0
        for feature, variable in enumerate(self.lvs):
            values = np.asarray(variable.compute_memberships(self.X[:, feature]), dtype=float)
            expected_shape = (self.n_lv[feature], n_samples)
            if values.shape != expected_shape:
                raise ValueError(
                    f"Memberships for feature {feature} have shape {values.shape}; "
                    f"expected {expected_shape}."
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Memberships for feature {feature} contain non-finite values.")
            membership[:, feature, : self.n_lv[feature]] = values.T
        self._membership_array = membership
        self._feature_gather_index = np.broadcast_to(
            np.arange(n_features)[None, :], (self.nRules, n_features)
        )
        self._target_ss = float(np.sum((self.y - self.y_mean) ** 2))
        # Output sets are scaled into the target range, so the range is exactly
        # the universe the centroid needs to integrate over.
        self._universe = (
            np.array([self.y_min])
            if self.y_range == 0
            else np.linspace(self.y_min, self.y_max, self.n_universe_points)
        )
        self._torch_tensor_cache = {}

    def _rule_term_matrix(self, chromosome: np.ndarray) -> np.ndarray:
        """Decode antecedent genes into a rule-by-feature term matrix."""
        chromosome = np.rint(np.asarray(chromosome)).astype(int)
        if chromosome.ndim != 1 or chromosome.shape[0] != self.n_var:
            raise ValueError(f"chromosome must have shape ({self.n_var},).")

        n_rules, n_ants, n_features = self.nRules, self.nAnts, self.X.shape[1]
        slot_count = n_rules * n_ants
        chosen_features = np.clip(
            chromosome[:slot_count].reshape(n_rules, n_ants), 0, n_features - 1
        )
        chosen_terms = chromosome[slot_count : 2 * slot_count].reshape(n_rules, n_ants)
        valid = (chosen_terms >= 0) & (chosen_terms < self.n_lv[chosen_features])

        rule_terms = np.full((n_rules, n_features), self._dont_care, dtype=int)
        rows = np.arange(n_rules)
        for slot in range(n_ants):
            valid_rows = valid[:, slot]
            rule_terms[rows[valid_rows], chosen_features[valid_rows, slot]] = chosen_terms[
                valid_rows, slot
            ]
        return rule_terms

    def _decode_consequents(self, chromosome: np.ndarray) -> np.ndarray:
        """Decode consequent genes into the target range."""
        start = 2 * self.nRules * self.nAnts
        encoded = np.clip(np.rint(chromosome[start : start + self.nRules]), 0, 100)
        if self.y_range == 0:
            return np.full(self.nRules, self.y_min, dtype=float)
        return self.y_min + (encoded / 100.0) * self.y_range

    def _decode_consequent_indices(self, chromosome: np.ndarray) -> np.ndarray:
        """Decode which output set each rule points at."""
        start = 2 * self.nRules * self.nAnts
        encoded = np.rint(chromosome[start : start + self.nRules]).astype(int)
        return np.clip(encoded, 0, self.n_output_lvs - 1)

    def _decode_output_sets(self, chromosome: np.ndarray) -> np.ndarray:
        """Decode the output trapezoids, sorted so each one is well formed."""
        start = 2 * self.nRules * self.nAnts + self.nRules
        encoded = np.clip(
            np.rint(chromosome[start : start + 4 * self.n_output_lvs]), 0, 100
        ).reshape(self.n_output_lvs, 4)
        return self.y_min + (np.sort(encoded, axis=1) / 100.0) * self.y_range

    def _decode_output_curves(self, chromosome: np.ndarray) -> np.ndarray:
        """Evaluate every decoded output set across the universe."""
        params = self._decode_output_sets(chromosome)
        return np.vstack(
            [
                np.asarray(fs.trapezoidal_membership(self._universe, params[ix]), dtype=float)
                for ix in range(self.n_output_lvs)
            ]
        )

    def _fast_predict(self, chromosome: np.ndarray) -> np.ndarray:
        """Predict all training samples without constructing rule objects."""
        chromosome = np.rint(np.asarray(chromosome)).astype(int)
        rule_terms = self._rule_term_matrix(chromosome)
        active_rules = np.any(rule_terms != self._dont_care, axis=1)

        gathered = self._membership_array[:, self._feature_gather_index, rule_terms]
        firing = np.prod(gathered, axis=2)
        firing[:, ~active_rules] = 0.0
        firing = _apply_rule_mode(firing, self.rule_mode, self.tolerance)

        if self.consequent_type == "fuzzy":
            # Zeroed rules contribute min(curve, 0) = 0, so the max aggregation
            # already ignores them exactly as dropping them would.
            return _mamdani_defuzzify(
                firing,
                self._decode_consequent_indices(chromosome),
                self._decode_output_curves(chromosome),
                self._universe,
                self.y_mean,
            )

        consequents = self._decode_consequents(chromosome)
        denominator = np.sum(firing, axis=1)
        prediction = np.full(self.X.shape[0], self.y_mean, dtype=float)
        covered = denominator > 1e-10
        prediction[covered] = (firing[covered] @ consequents) / denominator[covered]
        if np.any(covered):
            active_consequents = consequents[active_rules]
            prediction[covered] = np.clip(
                prediction[covered], active_consequents.min(), active_consequents.max()
            )
        return prediction

    def _fitness(self, prediction: np.ndarray) -> float:
        """Return full-training R-squared for a prediction vector."""
        if self._target_ss <= 0:
            return 1.0 if np.allclose(prediction, self.y) else 0.0
        residual_ss = float(np.sum((self.y - prediction) ** 2))
        return 1.0 - residual_ss / self._target_ss

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Score a Pymoo population using the vectorized rule-inference path."""
        population = np.asarray(X)
        if population.ndim == 1:
            population = population.reshape(1, -1)
        fitness = np.array(
            [self._fitness(self._fast_predict(individual)) for individual in population],
            dtype=float,
        )
        out["F"] = -fitness[:, None]

    def _torch_chunk_sizes(self, population_size: int, device) -> tuple[int, int]:
        """Choose conservative population/sample chunks for GPU evaluation."""
        import torch

        n_samples, n_features = self.X.shape
        # The largest temporary tensors are antecedent memberships and, for
        # Mamdani inference, clipped output curves.  Include a generous factor
        # for reductions and allocator workspace.
        floats_per_population_sample = self.nRules * (n_features + 2)
        if self.consequent_type == "fuzzy":
            floats_per_population_sample += self.n_output_lvs * (
                self.n_universe_points + 1
            )
        bytes_per_population_sample = max(1, 12 * floats_per_population_sample)

        if torch.device(device).type == "cuda" and torch.cuda.is_available():
            try:
                free_memory, _ = torch.cuda.mem_get_info(device)
                memory_budget = int(free_memory * 0.35)
            except Exception:
                memory_budget = 256 * 1024**2
        else:
            memory_budget = 512 * 1024**2

        max_pairs = max(1, memory_budget // bytes_per_population_sample)
        population_batch = max(1, min(population_size, max_pairs // max(1, n_samples)))
        sample_batch = max(1, min(n_samples, max_pairs // population_batch))
        return population_batch, sample_batch

    def _torch_rule_terms(self, population, torch):
        """Decode a population into rule/feature term matrices on the GPU."""
        population = torch.round(population).long()
        population_size = population.shape[0]
        n_features = self.X.shape[1]
        slot_count = self.nRules * self.nAnts
        chosen_features = population[:, :slot_count].reshape(
            population_size, self.nRules, self.nAnts
        ).clamp(0, n_features - 1)
        chosen_terms = population[:, slot_count : 2 * slot_count].reshape(
            population_size, self.nRules, self.nAnts
        )
        n_lv = torch.as_tensor(self.n_lv, dtype=torch.long, device=population.device)
        valid = (chosen_terms >= 0) & (chosen_terms < n_lv[chosen_features])

        rule_terms = torch.full(
            (population_size, self.nRules, n_features),
            self._dont_care,
            dtype=torch.long,
            device=population.device,
        )
        # Assign slots in chromosome order so repeated features preserve the
        # NumPy scorer's documented "last valid slot wins" behavior.
        for slot in range(self.nAnts):
            valid_rows, valid_rules = torch.where(valid[:, :, slot])
            if valid_rows.numel() == 0:
                continue
            features = chosen_features[valid_rows, valid_rules, slot]
            rule_terms[valid_rows, valid_rules, features] = chosen_terms[
                valid_rows, valid_rules, slot
            ]
        return rule_terms

    def _torch_tensors(self, device, torch):
        """Cache immutable training tensors on each evaluation device."""
        cache_key = str(device)
        cached = self._torch_tensor_cache.get(cache_key)
        if cached is None:
            cached = {
                "membership": torch.as_tensor(
                    self._membership_array, dtype=torch.float32, device=device
                ),
                "target": torch.as_tensor(
                    self.y, dtype=torch.float32, device=device
                ),
                "universe": torch.as_tensor(
                    self._universe, dtype=torch.float32, device=device
                ),
                "feature_indices": torch.arange(
                    self.X.shape[1], device=device
                ).view(1, 1, -1),
            }
            self._torch_tensor_cache[cache_key] = cached
        return cached

    def _torch_output_curves(self, population, universe, torch):
        """Decode Mamdani output trapezoids entirely with PyTorch."""
        population_size = population.shape[0]
        start = 2 * self.nRules * self.nAnts + self.nRules
        encoded = torch.round(
            population[:, start : start + 4 * self.n_output_lvs]
        ).clamp(0, 100).reshape(population_size, self.n_output_lvs, 4)
        params = self.y_min + (
            torch.sort(encoded, dim=2).values.float() / 100.0
        ) * self.y_range
        a, b, c, d = params.unbind(dim=2)
        singleton = a == d
        epsilon = 10e-5
        adjusted_a = torch.where((b == a) & ~singleton, a - epsilon, a)
        adjusted_d = torch.where((c == d) & ~singleton, d + epsilon, d)
        points = universe.view(1, 1, -1)
        rising = (points - adjusted_a.unsqueeze(2)) / (
            b - adjusted_a
        ).unsqueeze(2)
        falling = (adjusted_d.unsqueeze(2) - points) / (
            adjusted_d - c
        ).unsqueeze(2)
        curves = torch.minimum(rising, falling).clamp(0.0, 1.0)
        singleton_curves = (points == a.unsqueeze(2)).float()
        return torch.where(singleton.unsqueeze(2), singleton_curves, curves)

    def _evaluate_torch_population(
        self,
        population,
        device="cuda",
        population_batch_size: Optional[int] = None,
        sample_batch_size: Optional[int] = None,
    ):
        """Return negative R-squared for a population using batched PyTorch.

        All chromosome decoding, membership lookup, fuzzy inference, and
        objective computation happen on ``device``.  Both the population and
        samples are chunked, so the same implementation works on modest GPUs
        and large datasets without materializing the full four-dimensional
        inference tensor at once.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for EvoX regression. "
                "Install with: pip install ex-fuzzy[evox]"
            ) from exc

        device = torch.device(device)
        if not isinstance(population, torch.Tensor):
            population = torch.as_tensor(population, device=device)
        population = population.to(device=device)
        if population.ndim == 1:
            population = population.unsqueeze(0)
        if population.ndim != 2 or population.shape[1] != self.n_var:
            raise ValueError(
                f"population must have shape (population_size, {self.n_var})."
            )

        cached = self._torch_tensors(device, torch)
        automatic_population_batch, automatic_sample_batch = self._torch_chunk_sizes(
            population.shape[0], device
        )
        population_batch_size = (
            automatic_population_batch
            if population_batch_size is None
            else max(1, int(population_batch_size))
        )
        sample_batch_size = (
            automatic_sample_batch
            if sample_batch_size is None
            else max(1, int(sample_batch_size))
        )

        membership = cached["membership"]
        target = cached["target"]
        universe = cached["universe"]
        feature_indices = cached["feature_indices"]
        objective_batches = []

        for population_start in range(0, population.shape[0], population_batch_size):
            population_end = min(
                population_start + population_batch_size, population.shape[0]
            )
            batch = torch.round(population[population_start:population_end]).long()
            batch_size = batch.shape[0]
            rule_terms = self._torch_rule_terms(batch, torch)
            active_rules = torch.any(rule_terms != self._dont_care, dim=2)
            expanded_features = feature_indices.expand(
                batch_size, self.nRules, self.X.shape[1]
            )

            consequent_start = 2 * self.nRules * self.nAnts
            if self.consequent_type == "fuzzy":
                consequent_indices = torch.round(
                    batch[:, consequent_start : consequent_start + self.nRules]
                ).long().clamp(0, self.n_output_lvs - 1)
                output_curves = self._torch_output_curves(batch, universe, torch)
            else:
                encoded = torch.round(
                    batch[:, consequent_start : consequent_start + self.nRules]
                ).clamp(0, 100).float()
                consequents = self.y_min + (encoded / 100.0) * self.y_range
                active_min = torch.where(
                    active_rules, consequents, torch.full_like(consequents, torch.inf)
                ).min(dim=1).values
                active_max = torch.where(
                    active_rules, consequents, torch.full_like(consequents, -torch.inf)
                ).max(dim=1).values

            residual_ss = torch.zeros(batch_size, dtype=torch.float32, device=device)
            constant_target_match = torch.ones(batch_size, dtype=torch.bool, device=device)
            for sample_start in range(0, self.X.shape[0], sample_batch_size):
                sample_end = min(sample_start + sample_batch_size, self.X.shape[0])
                membership_slice = membership[sample_start:sample_end].permute(1, 2, 0)
                gathered = membership_slice[expanded_features, rule_terms]
                firing = gathered.prod(dim=2).permute(0, 2, 1)
                firing = torch.where(active_rules[:, None, :], firing, 0.0)

                if self.rule_mode == "sufficient":
                    peak, strongest = firing.max(dim=2, keepdim=True)
                    winning_strength = torch.where(peak > self.tolerance, peak, 0.0)
                    selected = torch.zeros_like(firing)
                    selected.scatter_(2, strongest, winning_strength)
                    firing = selected

                if self.consequent_type == "fuzzy":
                    set_firing = []
                    for output_set in range(self.n_output_lvs):
                        members = consequent_indices == output_set
                        set_firing.append(
                            torch.where(members[:, None, :], firing, 0.0).amax(dim=2)
                        )
                    set_firing = torch.stack(set_firing, dim=2)
                    aggregated = torch.minimum(
                        output_curves[:, None, :, :], set_firing[:, :, :, None]
                    ).amax(dim=2)
                    mass = aggregated.sum(dim=2)
                    numerator = (aggregated * universe.view(1, 1, -1)).sum(dim=2)
                    prediction = torch.where(
                        mass > 1e-10,
                        numerator / mass.clamp_min(1e-10),
                        self.y_mean,
                    )
                else:
                    denominator = firing.sum(dim=2)
                    numerator = torch.einsum("bsr,br->bs", firing, consequents)
                    prediction = torch.where(
                        denominator > 1e-10,
                        numerator / denominator.clamp_min(1e-10),
                        self.y_mean,
                    )
                    clipped = torch.minimum(
                        torch.maximum(prediction, active_min[:, None]),
                        active_max[:, None],
                    )
                    prediction = torch.where(denominator > 1e-10, clipped, prediction)

                expected = target[sample_start:sample_end].unsqueeze(0)
                residual_ss += ((prediction - expected) ** 2).sum(dim=1)
                constant_target_match &= torch.isclose(
                    prediction, expected, rtol=1e-5, atol=1e-8
                ).all(dim=1)

            if self._target_ss <= 0:
                r_squared = constant_target_match.float()
            else:
                r_squared = 1.0 - residual_ss / float(self._target_ss)
            objective_batches.append(-r_squared)

        return torch.cat(objective_batches)

    def _construct_ruleBase(self, chromosome: np.ndarray) -> _RegressionRuleBase:
        """Decode a chromosome into a prediction-equivalent concrete rule base."""
        chromosome = np.rint(np.asarray(chromosome)).astype(int)
        rule_terms = self._rule_term_matrix(chromosome)
        active_rules = np.any(rule_terms != self._dont_care, axis=1)
        decoded_terms = np.where(rule_terms == self._dont_care, -1, rule_terms)

        if self.consequent_type == "fuzzy":
            params = self._decode_output_sets(chromosome)
            consequent_ix = self._decode_consequent_indices(chromosome)
            output_sets = [
                fs.FS(f"Output_{ix}", params[ix].tolist(), [self.y_min, self.y_max])
                for ix in range(self.n_output_lvs)
            ]
            return RuleBaseT1MamdaniRegression(
                self.lvs,
                [
                    rules.RuleSimple(decoded_terms[ix].tolist(), consequent=int(consequent_ix[ix]))
                    for ix in range(self.nRules)
                    if active_rules[ix]
                ],
                output_sets,
                y_mean=self.y_mean,
                universe=self._universe,
                rule_mode=self.rule_mode,
                tolerance=self.tolerance,
            )

        consequents = self._decode_consequents(chromosome)
        rule_list = [
            rules.RuleSimple(decoded_terms[ix].tolist(), consequent=0)
            for ix in range(self.nRules)
            if active_rules[ix]
        ]
        active_consequents = consequents[active_rules]
        return RuleBaseT1Regression(
            self.lvs,
            rule_list,
            active_consequents,
            y_mean=self.y_mean,
            rule_mode=self.rule_mode,
            tolerance=self.tolerance,
        )


class BaseFuzzyRulesRegressor(RegressorMixin, BaseEstimator):
    """Scikit-learn-compatible fast Type-1 fuzzy rules regressor.

    ``consequent_type`` selects how a rule states its output:

    ``crisp``
        Zero-order Takagi-Sugeno.  Each rule carries a number and predictions
        are the firing-strength-weighted average of those numbers.
    ``fuzzy``
        Mamdani.  Each rule names an output fuzzy set whose shape is evolved
        alongside the rules; predictions are the centroid of the clipped and
        aggregated consequents.  Rules read fully linguistically
        (``THEN output IS Output_2``) at some cost in numeric resolution.

    ``rule_mode`` selects how many rules speak per sample: ``additive`` lets
    every rule contribute, while ``sufficient`` keeps only the single
    strongest rule and falls back to the target mean when even that one fires
    below ``tolerance``.

    ``backend="pymoo"`` uses the vectorized NumPy objective on the CPU.
    ``backend="evox"`` evolves populations with EvoX and evaluates complete
    population batches with PyTorch on CUDA when available.
    """

    def __init__(
        self,
        nRules: int = 30,
        nAnts: int = 4,
        n_linguistic_variables: int = 3,
        fuzzy_type: fs.FUZZY_SETS = fs.FUZZY_SETS.t1,
        linguistic_variables: Optional[list[fs.fuzzyVariable]] = None,
        consequent_type: str = "crisp",
        n_output_lvs: int = 3,
        n_universe_points: int = 101,
        rule_mode: str = "additive",
        tolerance: float = 0.0,
        verbose: bool = False,
        backend: str = "pymoo",
    ) -> None:
        """Configure the regressor; optimization is performed in :meth:`fit`."""
        self.nRules = nRules
        self.nAnts = nAnts
        self.n_linguistic_variables = n_linguistic_variables
        self.fuzzy_type = fuzzy_type
        self.linguistic_variables = linguistic_variables
        self.consequent_type = consequent_type
        self.n_output_lvs = n_output_lvs
        self.n_universe_points = n_universe_points
        self.rule_mode = rule_mode
        self.tolerance = tolerance
        self.verbose = verbose
        self.backend = backend

    def _validate_parameters(self) -> None:
        """Validate estimator hyperparameters without mutating them."""
        if not isinstance(self.nRules, (int, np.integer)) or self.nRules <= 0:
            raise ValueError("nRules must be a positive integer.")
        if not isinstance(self.nAnts, (int, np.integer)) or self.nAnts <= 0:
            raise ValueError("nAnts must be a positive integer.")
        if (
            not isinstance(self.n_linguistic_variables, (int, np.integer))
            or self.n_linguistic_variables <= 0
        ):
            raise ValueError("n_linguistic_variables must be a positive integer.")
        if self.fuzzy_type != fs.FUZZY_SETS.t1:
            raise ValueError("Fast fuzzy regression currently supports only Type-1 fuzzy sets.")
        if self.consequent_type not in ("crisp", "fuzzy"):
            raise ValueError("consequent_type must be either 'crisp' or 'fuzzy'.")
        _validate_rule_mode(self.rule_mode, self.tolerance)
        if self.consequent_type == "fuzzy":
            if not isinstance(self.n_output_lvs, (int, np.integer)) or self.n_output_lvs < 2:
                raise ValueError("n_output_lvs must be an integer of at least two.")
            if (
                not isinstance(self.n_universe_points, (int, np.integer))
                or self.n_universe_points < 2
            ):
                raise ValueError("n_universe_points must be an integer of at least two.")

    def fit(
        self,
        X,
        y,
        n_gen: int = 50,
        pop_size: int = 50,
        random_state: int = 42,
    ):
        """Fit scalar consequents and rule antecedents with a genetic algorithm."""
        self._validate_parameters()
        if not isinstance(n_gen, (int, np.integer)) or n_gen <= 0:
            raise ValueError("n_gen must be a positive integer.")
        if not isinstance(pop_size, (int, np.integer)) or pop_size <= 1:
            raise ValueError("pop_size must be an integer greater than one.")

        if isinstance(X, pd.DataFrame):
            feature_names = np.asarray(X.columns, dtype=object)
        else:
            feature_names = None
        X_values = _as_2d_float_array(X)
        y_values = _as_1d_float_array(y, expected_samples=X_values.shape[0])

        self.n_features_in_ = X_values.shape[1]
        if feature_names is not None:
            self.feature_names_in_ = feature_names
        self.n_ants_ = min(int(self.nAnts), self.n_features_in_)
        if self.verbose and self.n_ants_ != self.nAnts:
            print(
                "Warning: nAnts exceeds the number of features; "
                f"using {self.n_ants_} antecedent slots."
            )

        if self.linguistic_variables is None:
            linguistic_variables = utils.construct_partitions(
                X_values,
                fz_type_studied=self.fuzzy_type,
                n_partitions=int(self.n_linguistic_variables),
            )
            if feature_names is not None:
                for variable, name in zip(linguistic_variables, feature_names):
                    variable.name = str(name)
        else:
            linguistic_variables = self.linguistic_variables
        _validate_linguistic_variables(linguistic_variables, self.n_features_in_)

        self.y_min_ = float(np.min(y_values))
        self.y_max_ = float(np.max(y_values))
        self.y_mean_ = float(np.mean(y_values))
        problem = FitRuleBaseRegression(
            X_values,
            y_values,
            nRules=int(self.nRules),
            nAnts=self.n_ants_,
            linguistic_variables=linguistic_variables,
            y_min=self.y_min_,
            y_max=self.y_max_,
            y_mean=self.y_mean_,
            consequent_type=self.consequent_type,
            n_output_lvs=int(self.n_output_lvs),
            n_universe_points=int(self.n_universe_points),
            rule_mode=self.rule_mode,
            tolerance=float(self.tolerance),
        )

        backend = ev_backends.get_backend(self.backend)
        result = backend.optimize(
            problem=problem,
            n_gen=int(n_gen),
            pop_size=int(pop_size),
            random_state=random_state,
            verbose=self.verbose,
            var_prob=0.9,
            sbx_eta=3.0,
            mutation_eta=7.0,
            tournament_size=3,
            patience=None,
        )
        if result.get("X") is None or result.get("F") is None:
            raise RuntimeError("The evolutionary optimizer did not return a valid solution.")

        self.optimization_result_ = result
        self.backend_ = backend.name()
        self.optimization_device_ = result.get("device", "cpu")
        self.gpu_accelerated_ = bool(result.get("gpu_accelerated", False))
        self.n_generations_run_ = int(result.get("n_gen_run", n_gen))
        self.stopped_early_ = bool(result.get("stopped_early", False))
        self.best_chromosome_ = np.rint(np.asarray(result["X"])).astype(int)
        self.performance_ = float(-np.asarray(result["F"]).reshape(-1)[0])
        self.performance = self.performance_
        self.rule_base = problem._construct_ruleBase(self.best_chromosome_)
        self.lvs = self.rule_base.antecedents
        self.is_fitted_ = True

        if self.verbose:
            print(f"Final: {len(self.rule_base)} rules, fitness={self.performance_:.4f}")
        return self

    def predict(self, X) -> np.ndarray:
        """Predict a continuous target for each sample."""
        check_is_fitted(self, "is_fitted_")
        if isinstance(X, pd.DataFrame) and hasattr(self, "feature_names_in_"):
            supplied_names = np.asarray(X.columns, dtype=object)
            if not np.array_equal(supplied_names, self.feature_names_in_):
                raise ValueError("DataFrame columns must match those seen during fit, in the same order.")
        X_values = _as_2d_float_array(X, expected_features=self.n_features_in_)
        return self.rule_base.predict(X_values)

    def score(self, X, y) -> float:
        """Return the coefficient of determination of the predictions."""
        prediction = self.predict(X)
        target = _as_1d_float_array(y, expected_samples=prediction.shape[0])
        return float(r2_score(target, prediction))

    def print_rules(self, return_rules: bool = False):
        """Print or return the fitted rule base in IF-THEN form."""
        check_is_fitted(self, "is_fitted_")
        return self.rule_base.print_rules(return_rules=return_rules)

    def get_rulebase(self) -> RuleBaseT1Regression:
        """Return the fitted scalar-consequent rule base."""
        check_is_fitted(self, "is_fitted_")
        return self.rule_base
