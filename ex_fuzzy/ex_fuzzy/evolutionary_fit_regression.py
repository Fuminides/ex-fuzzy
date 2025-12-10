"""
Evolutionary Optimization for Fuzzy Regression Rule Base Learning

This module extends the evolutionary_fit module to support regression problems
with numeric outputs. It implements genetic algorithm-based optimization for
learning fuzzy rule bases where rules output numeric values instead of classes.

Main Components:
    - FitRuleBaseRegression: Optimization problem class for regression
    - BaseFuzzyRulesRegressor: Main regressor interface (like BaseFuzzyRulesClassifier)
    - Fitness functions using MSE/RMSE instead of classification metrics
    - Support for numeric consequents in rules

Key Features:
    - Numeric rule consequents (output is a real number)
    - MSE/RMSE-based fitness evaluation
    - Support for Type-1, Type-2 fuzzy systems
    - Parallel evaluation support
    - Both PyMoo and EvoX backends supported
    - Memory-efficient batch processing for large datasets
"""

import numpy as np
import pandas as pd
from typing import Callable

from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pymoo.core.problem import Problem
from pymoo.core.variable import Integer, Real
from pymoo.parallelization.starmap import StarmapParallelization
from multiprocessing.pool import ThreadPool

try:
    from . import evolutionary_backends as ev_backends
    from . import fuzzy_sets as fs
    from . import rules
    from . import eval_rules as evr
    from .evolutionary_fit import FitRuleBase  # Reuse most functionality
except ImportError:
    import evolutionary_backends as ev_backends
    import fuzzy_sets as fs
    import rules
    import eval_rules as evr
    from evolutionary_fit import FitRuleBase


class FitRuleBaseRegression(FitRuleBase):
    '''
    Class to model regression problems for fuzzy rule bases using evolutionary strategies.
    Rules have numeric consequents instead of class labels.
    
    Extends FitRuleBase but overrides evaluation functions to use regression metrics (MSE/RMSE)
    instead of classification metrics (MCC).
    '''

    def __init__(self, X: np.array, y: np.array, nRules: int, nAnts: int, y_range: tuple = None,
                 thread_runner: StarmapParallelization = None, linguistic_variables: list[fs.fuzzyVariable] = None,
                 n_linguistic_variables: int = 3, fuzzy_type=fs.FUZZY_SETS.t1, domain: list = None,
                 categorical_mask: np.array = None, tolerance: float = 0.01, alpha: float = 0.0, beta: float = 0.0,
                 rule_mode: str = 'additive', backend_name: str = 'pymoo') -> None:
        '''
        Constructor method for regression problem.

        :param X: np array or pandas dataframe samples x features.
        :param y: np vector containing the target values (continuous). vector sample
        :param nRules: number of rules to optimize.
        :param nAnts: max number of antecedents to use.
        :param y_range: tuple (min, max) for the output range. If None, computed from data.
        :param linguistic_variables: list of linguistic variables precomputed. If given, the rest of conflicting arguments are ignored.
        :param n_linguistic_variables: number of linguistic variables per antecedent.
        :param fuzzy_type: Define the fuzzy set or fuzzy set extension used as linguistic variable.
        :param domain: list with the upper and lower domains of each input variable. If None (as default) it will establish the empirical min/max as the limits.
        :param tolerance: float. Tolerance for rule firing (used only in 'sufficient' mode).
        :param alpha: float. Weight for the rulebase size term in the fitness function. (Penalizes number of rules)
        :param beta: float. Weight for the average rule size term in the fitness function.
        :param rule_mode: str. 'additive' (all rules contribute) or 'sufficient' (only rules above tolerance).
        :param backend_name: str. Backend to use for evolutionary optimization ('pymoo' or 'evox')
        '''
        
        # Store target values
        try:
            self.var_names = list(X.columns)
            self.X = X.values
        except AttributeError:
            self.X = X
            self.var_names = [str(ix) for ix in range(X.shape[1])]

        self.y = y
        self.nRules = nRules
        self.nAnts = nAnts
        self.nCons = 1  # This is fixed to MISO rules
        self.rule_mode = rule_mode  # 'additive' or 'sufficient'
        
        # Determine output range
        if y_range is not None:
            self.y_min, self.y_max = y_range
        else:
            self.y_min = np.min(y)
            self.y_max = np.max(y)
        
        self.y_range = self.y_max - self.y_min
        self.tolerance = tolerance
        self.alpha_ = alpha
        self.beta_ = beta
        self.backend_name = backend_name
        
        # Initialize fuzzy variables (reuse parent class logic)
        if categorical_mask is None:
            self.categorical_mask = np.zeros(X.shape[1])
            categorical_mask = self.categorical_mask

        if linguistic_variables is not None:
            self._init_precomputed_vl(linguistic_variables, X)
        else:
            if isinstance(n_linguistic_variables, int):
                n_linguistic_variables = [n_linguistic_variables] * self.X.shape[1]
            self._init_optimize_vl(
                fuzzy_type=fuzzy_type, n_linguist_variables=n_linguistic_variables, 
                categorical_variables=categorical_mask, domain=domain, X=X)

        if self.domain is None:
            # Compute min/max bounds
            if np.all([np.issubdtype(self.X[:, ix].dtype, np.number) for ix in range(self.X.shape[1])]):
                self.min_bounds = np.min(self.X, axis=0)
                self.max_bounds = np.max(self.X, axis=0)
            else:
                self.min_bounds = np.zeros(self.X.shape[1])
                self.max_bounds = np.zeros(self.X.shape[1])
                for ix in range(self.X.shape[1]):
                    if np.issubdtype(self.X[:, ix].dtype, np.number):
                        self.min_bounds[ix] = np.min(self.X[:, ix])
                        self.max_bounds[ix] = np.max(self.X[:, ix])
                    else:
                        self.min_bounds[ix] = 0
                        self.max_bounds[ix] = len(np.unique(self.X[:, ix][~pd.isna(self.X[:, ix])]))
        else:
            self.min_bounds, self.max_bounds = self.domain

        self.antecedents_referencial = [np.linspace(
            self.min_bounds[ix], self.max_bounds[ix], 100) for ix in range(self.X.shape[1])]

        # Build variable bounds (antecedents same as classification)
        possible_antecedent_bounds = np.array(
            [[0, self.X.shape[1] - 1]] * self.nAnts * self.nRules)
        vl_antecedent_bounds = np.array(
            [[-1, self.n_lv_possible[ax] - 1] for ax in range(self.nAnts)] * self.nRules)
        antecedent_bounds = np.concatenate(
            (possible_antecedent_bounds, vl_antecedent_bounds))
        vars_antecedent = {ix: Integer(
            bounds=antecedent_bounds[ix]) for ix in range(len(antecedent_bounds))}
        aux_counter = len(vars_antecedent)

        # Membership function parameters (if optimizing)
        if self.lvs is None:
            self.feature_domain_bounds = np.array(
                [[0, 99] for ix in range(self.X.shape[1])])
            if self.fuzzy_type == fs.FUZZY_SETS.t1:
                correct_size = [(self.n_lv_possible[ixx] - 1) * 4 + 3 for ixx in range(len(self.n_lv_possible))]
            elif self.fuzzy_type == fs.FUZZY_SETS.t2:
                correct_size = [(self.n_lv_possible[ixx] - 1) * 6 + 2 for ixx in range(len(self.n_lv_possible))]
            membership_bounds = np.concatenate(
                [[self.feature_domain_bounds[ixx]] * correct_size[ixx] for ixx in range(len(self.n_lv_possible))])

            vars_memberships = {
                aux_counter + ix: Integer(bounds=membership_bounds[ix]) for ix in range(len(membership_bounds))}
            aux_counter += len(vars_memberships)

        # REGRESSION: Consequents are real numbers, not class indices
        # We normalize consequents to [0, 99] range for the optimizer
        final_consequent_bounds = np.array(
            [[0, 99]] * self.nRules)
        vars_consequent = {aux_counter + ix: Integer(
            bounds=final_consequent_bounds[ix]) for ix in range(len(final_consequent_bounds))}

        # Build final variables dict
        if self.lvs is None:
            vars = {key: val for d in [
                vars_antecedent, vars_memberships, vars_consequent] for key, val in d.items()}
            varbound = np.concatenate(
                (antecedent_bounds, membership_bounds, final_consequent_bounds), axis=0)
        else:
            vars = {key: val for d in [vars_antecedent,
                                       vars_consequent] for key, val in d.items()}
            varbound = np.concatenate(
                (antecedent_bounds, final_consequent_bounds), axis=0)

        # No weights needed for additive regression
        nVar = len(varbound)
        self.single_gen_size = nVar

        # Initialize Problem parent
        if thread_runner is not None:
            Problem.__init__(
                self,
                vars=vars,
                n_var=nVar,
                n_obj=1,
                elementwise=True,
                vtype=int,
                xl=varbound[:, 0],
                xu=varbound[:, 1],
                elementwise_runner=thread_runner)
        else:
            Problem.__init__(
                self,
                vars=vars,
                n_var=nVar,
                n_obj=1,
                elementwise=True,
                vtype=int,
                xl=varbound[:, 0],
                xu=varbound[:, 1])

    def _construct_ruleBase_regression(self, x: np.array, fuzzy_type: fs.FUZZY_SETS, **kwargs) -> rules.MasterRuleBase:
        '''
        Construct a rule base with numeric consequents for regression.
        
        Similar to parent's _construct_ruleBase but consequents are real numbers.
        
        :param x: gene array
        :param fuzzy_type: fuzzy set type
        :return: MasterRuleBase with numeric consequents
        '''
        
        rule_list = []  # Single list for regression (no classes)
        
        mf_size = 4 if fuzzy_type == fs.FUZZY_SETS.t1 else 6
        
        # Calculate pointer to consequents (no weights in additive regression)
        if self.lvs is None:
            if fuzzy_type == fs.FUZZY_SETS.t1:
                fourth_pointer = 2 * self.nAnts * self.nRules + \
                    len(self.n_lv_possible) * 3 + sum(np.array(self.n_lv_possible) - 1) * 4
            elif fuzzy_type == fs.FUZZY_SETS.t2:
                fourth_pointer = 2 * self.nAnts * self.nRules + \
                    len(self.n_lv_possible) * 2 + sum(np.array(self.n_lv_possible) - 1) * mf_size
        else:
            fourth_pointer = 2 * self.nAnts * self.nRules

        # Handle mixed data types
        min_domain = np.zeros(self.X.shape[1])
        max_domain = np.zeros(self.X.shape[1])

        for ix in range(self.X.shape[1]):
            if np.issubdtype(self.X[:, ix].dtype, np.number):
                min_domain[ix] = np.nanmin(self.X[:, ix])
                max_domain[ix] = np.nanmax(self.X[:, ix])
            else:
                min_domain[ix] = 0
                max_domain[ix] = len(np.unique(self.X[:, ix][~pd.isna(self.X[:, ix])]))

        range_domain = np.zeros((self.X.shape[1],))
        for ix in range(self.X.shape[1]):
            try:
                range_domain[ix] = max_domain[ix] - min_domain[ix]
            except TypeError:
                pass

        # Convert gene to int array
        try:
            x = np.array(list(x.values())).astype(int)
        except AttributeError:
            x = x.astype(int)

        # Reconstruct rules
        aux_pointer = 0
        for i0 in range(self.nRules):
            first_pointer = i0 * self.nAnts
            chosen_ants = x[first_pointer:first_pointer + self.nAnts]

            second_pointer = (i0 * self.nAnts) + (self.nAnts * self.nRules)
            antecedent_parameters = x[second_pointer:second_pointer + self.nAnts]

            init_rule_antecedents = np.zeros((self.X.shape[1],)) - 1  # -1 is don't care

            for jx, ant in enumerate(chosen_ants):
                if self.lvs is not None:
                    antecedent_parameters[jx] = min(antecedent_parameters[jx], len(self.lvs[ant]) - 1)
                else:
                    antecedent_parameters[jx] = min(antecedent_parameters[jx], self.n_lv_possible[ant] - 1)

                init_rule_antecedents[ant] = antecedent_parameters[jx]

            # REGRESSION: Consequent is a real number (denormalize from [0, 99] to [y_min, y_max])
            consequent_normalized = x[fourth_pointer + aux_pointer]
            consequent_value = self.y_min + (consequent_normalized / 99.0) * self.y_range
            aux_pointer += 1

            # Create rule if it has at least one antecedent (no weights for additive regression)
            if np.any(init_rule_antecedents != -1):
                rs_instance = rules.RuleSimple(init_rule_antecedents, consequent_value, None)
                rule_list.append(rs_instance)

        # Decode membership functions if needed
        if self.lvs is None:
            antecedents_raw = self._decode_membership_functions(x, fuzzy_type)

            # Normalize to data domain
            antecedents = []
            for fuzzy_variable, fv_raw in enumerate(antecedents_raw):
                if self.categorical_boolean_mask is not None and self.categorical_boolean_mask[fuzzy_variable]:
                    antecedents.append(fv_raw)
                else:
                    lv_FS = [lv.membership_parameters for lv in fv_raw.linguistic_variables]
                    min_lv = np.min(np.array(lv_FS))
                    max_lv = np.max(np.array(lv_FS))

                    linguistic_variables = []
                    for lx, relevant_lv in enumerate(lv_FS):
                        relevant_lv = (relevant_lv - min_lv) / (max_lv - min_lv) * range_domain[fuzzy_variable] + \
                                      min_domain[fuzzy_variable]
                        if fuzzy_type == fs.FUZZY_SETS.t1:
                            proper_FS = fs.FS(self.vl_names[fuzzy_variable][lx], relevant_lv,
                                             (min_domain[fuzzy_variable], max_domain[fuzzy_variable]))
                        elif fuzzy_type == fs.FUZZY_SETS.t2:
                            proper_FS = fs.IVFS(self.vl_names[fuzzy_variable][lx], relevant_lv[0], relevant_lv[1],
                                               (min_domain[fuzzy_variable], max_domain[fuzzy_variable]))
                        linguistic_variables.append(proper_FS)

                    linguistic_variable = fs.fuzzyVariable(self.var_names[fuzzy_variable], linguistic_variables)
                    antecedents.append(linguistic_variable)
        else:
            try:
                antecedents = self.lvs[kwargs['time_moment']]
            except:
                antecedents = self.lvs

        # Create rule base
        if fuzzy_type == fs.FUZZY_SETS.t1:
            rule_base = rules.RuleBaseT1(antecedents, rule_list)
        elif fuzzy_type == fs.FUZZY_SETS.t2:
            rule_base = rules.RuleBaseT2(antecedents, rule_list)
        elif fuzzy_type == fs.FUZZY_SETS.gt2:
            rule_base = rules.RuleBaseGT2(antecedents, rule_list)

        # For regression, we use a single "class" (output is numeric, no ds_mode for additive regression)
        res = rules.MasterRuleBase([rule_base], ['output'], allow_unknown=False)

        return res

    def _evaluate_numpy_fast_regression(self, x: np.array, y: np.array, fuzzy_type: fs.FUZZY_SETS, **kwargs) -> float:
        '''
        Fast vectorized evaluation for regression using RMSE as fitness.
        
        Similar to classification version but predicts continuous values and uses MSE/RMSE.
        
        :param x: gene array
        :param y: target values (continuous)
        :param fuzzy_type: fuzzy set type
        :return: negative RMSE (to be minimized; negated so higher is better)
        '''
        
        mf_size = 4 if fuzzy_type == fs.FUZZY_SETS.t1 else 6

        # Calculate pointer to consequents (no weights in additive regression)
        if self.lvs is None:
            if fuzzy_type == fs.FUZZY_SETS.t1:
                fourth_pointer = 2 * self.nAnts * self.nRules + \
                    len(self.n_lv_possible) * 3 + sum(np.array(self.n_lv_possible) - 1) * 4
            elif fuzzy_type == fs.FUZZY_SETS.t2:
                fourth_pointer = 2 * self.nAnts * self.nRules + \
                    len(self.n_lv_possible) * 2 + sum(np.array(self.n_lv_possible) - 1) * mf_size
        else:
            fourth_pointer = 2 * self.nAnts * self.nRules

        # Get precomputed memberships
        if self.lvs is None:
            antecedents = self._decode_membership_functions(x, fuzzy_type)
            precomputed_antecedent_memberships = rules.compute_antecedents_memberships(antecedents, self.X)
        else:
            precomputed_antecedent_memberships = self._precomputed_truth

        # Convert x to int array
        try:
            x = np.array(list(x.values())).astype(int)
        except AttributeError:
            x = x.astype(int)

        # Extract gene segments
        n_samples = self.X.shape[0]
        n_features = self.X.shape[1]

        chosen_ants = x[:self.nAnts * self.nRules].reshape(self.nRules, self.nAnts)
        ant_params = x[self.nAnts * self.nRules:2 * self.nAnts * self.nRules].reshape(self.nRules, self.nAnts)

        # Clamp parameters
        for feat_idx in range(n_features):
            mask = chosen_ants == feat_idx
            if self.lvs is not None:
                max_param = len(self.lvs[feat_idx]) - 1
            else:
                max_param = self.n_lv_possible[feat_idx] - 1
            ant_params = np.where(mask, np.minimum(ant_params, max_param), ant_params)

        # Build membership array
        max_lvars = max(self.n_lv_possible)
        membership_array = np.zeros((n_samples, n_features, max_lvars))
        for feat_idx in range(n_features):
            feat_memberships = precomputed_antecedent_memberships[feat_idx]
            n_lvars = feat_memberships.shape[0]
            membership_array[:, feat_idx, :n_lvars] = feat_memberships.T

        # Create indicator matrix
        indicators = np.zeros((n_features, max_lvars, self.nRules, self.nAnts))
        rule_indices = np.arange(self.nRules)[:, None]
        ant_indices = np.arange(self.nAnts)[None, :]
        indicators[chosen_ants, ant_params, rule_indices, ant_indices] = 1.0

        # Compute memberships
        ant_memberships = np.einsum('sfl,flra->sra', membership_array, indicators)
        rule_memberships = np.prod(ant_memberships, axis=2)

        # Get consequents (denormalize from [0, 99] to [y_min, y_max])
        rule_consequents_normalized = x[fourth_pointer:fourth_pointer + self.nRules]
        rule_consequents = self.y_min + (rule_consequents_normalized / 99.0) * self.y_range

        # ADDITIVE REGRESSION: All rules contribute weighted by their membership
        # predicted_value = sum(membership * consequent) / sum(membership)
        numerator = np.sum(rule_memberships * rule_consequents[np.newaxis, :], axis=1)
        denominator = np.sum(rule_memberships, axis=1) + 1e-10  # avoid division by zero
        predicted_values = numerator / denominator

        # Compute RMSE
        mse = mean_squared_error(y, predicted_values)
        rmse = np.sqrt(mse)

        # Return negative RMSE (so higher fitness is better)
        # Normalize by y_range to make it scale-independent
        normalized_rmse = rmse / (self.y_range + 1e-10)

        return -normalized_rmse  # Negative because pymoo minimizes

    def _evaluate(self, x: np.array, out: dict, *args, **kwargs):
        '''
        Evaluation function for regression problem.
        '''
        score = self._evaluate_numpy_fast_regression(x, self.y, self.fuzzy_type)
        out["F"] = -score  # Pymoo minimizes, we want to maximize (minimize -score)

    def fitness_func(self, ruleBase: rules.RuleBase, X: np.array, y: np.array, tolerance: float,
                    alpha: float = 0.0, beta: float = 0.0, precomputed_truth: np.array = None) -> float:
        '''
        Fitness function for regression using RMSE.
        
        :param ruleBase: RuleBase object
        :param X: input samples
        :param y: target values
        :param tolerance: tolerance for rule evaluation
        :param alpha: weight for rule size penalty
        :param beta: weight for average antecedents penalty
        :param precomputed_truth: precomputed membership values
        :return: fitness value (higher is better)
        '''
        if precomputed_truth is None:
            precomputed_truth = rules.compute_antecedents_memberships(ruleBase.antecedents, X)

        # Predict using the rule base (mode depends on rule_mode parameter)
        predictions = []
        for i in range(X.shape[0]):
            sample_memberships = []
            sample_consequents = []

            for rule in ruleBase.get_rules():
                # Compute rule membership for this sample
                membership = 1.0
                for ant_idx, ant_val in enumerate(rule.antecedents):
                    if ant_val >= 0:  # not don't care
                        ant_val = int(ant_val)
                        membership *= precomputed_truth[ant_idx][ant_val][i]

                # Filter by tolerance if using sufficient rules
                if self.rule_mode == 'sufficient':
                    if membership > tolerance:
                        sample_memberships.append(membership)
                        sample_consequents.append(rule.consequent)
                else:  # additive mode (default)
                    sample_memberships.append(membership)
                    sample_consequents.append(rule.consequent)

            if len(sample_memberships) > 0:
                # Weighted average
                pred = np.sum(np.array(sample_memberships) * np.array(sample_consequents)) / (np.sum(sample_memberships) + 1e-10)
            else:
                # No rules (can happen with sufficient mode), use mean
                pred = np.mean(y)

            predictions.append(pred)

        predictions = np.array(predictions)

        # Compute RMSE
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        normalized_rmse = rmse / (self.y_range + 1e-10)

        # Base score (negative RMSE, normalized)
        score_acc = -normalized_rmse

        # Penalize complexity
        ruleBase.purge_rules(tolerance)
        if len(ruleBase.get_rules()) > 0:
            avg_antecedents = np.mean([np.sum(r.antecedents >= 0) for r in ruleBase.get_rules()])
            score_rules_size = -(avg_antecedents / self.nAnts)
            score_nrules = -(len(ruleBase.get_rules()) / self.nRules)

            score = score_acc + score_rules_size * alpha + score_nrules * beta
        else:
            score = -1.0  # Very bad score if no rules

        return score


class BaseFuzzyRulesRegressor(RegressorMixin):
    '''
    Fuzzy rule-based regressor with evolutionary optimization.
    
    Similar interface to BaseFuzzyRulesClassifier but for regression problems.
    Rules output numeric values instead of class labels.
    '''

    def __init__(self, nRules: int = 30, nAnts: int = 4, fuzzy_type: fs.FUZZY_SETS = fs.FUZZY_SETS.t1,
                 tolerance: float = 0.0, y_range: tuple = None, n_linguistic_variables: list[int] | int = 3,
                 verbose=False, linguistic_variables: list[fs.fuzzyVariable] = None,
                 categorical_mask: list[int] = None, domain: list[float] = None,
                 precomputed_rules: rules.MasterRuleBase = None, runner: int = 1,
                 rule_mode: str = 'additive', backend: str = 'pymoo') -> None:
        '''
        Initialize fuzzy rules regressor.

        :param nRules: number of rules to optimize
        :param nAnts: max number of antecedents per rule
        :param fuzzy_type: type of fuzzy set (t1, t2, etc.)
        :param tolerance: tolerance for rule evaluation
        :param y_range: tuple (min, max) for output range. If None, computed from data
        :param n_linguistic_variables: number of linguistic variables per feature
        :param verbose: print optimization progress
        :param linguistic_variables: precomputed fuzzy variables
        :param categorical_mask: mask for categorical features
        :param domain: domain for each input feature
        :param precomputed_rules: precomputed rule base
        :param runner: number of threads for parallel evaluation
        :param rule_mode: 'additive' (all rules contribute) or 'sufficient' (only rules above tolerance)
        :param backend: evolutionary backend ('pymoo' or 'evox')
        '''
        self.nRules = nRules
        self.nAnts = nAnts
        self.fuzzy_type = fuzzy_type
        self.tolerance = tolerance  # Used in 'sufficient' mode, ignored in 'additive' mode
        self.rule_mode = rule_mode  # 'additive' or 'sufficient'
        self.y_range = y_range
        self.n_linguist_variables = n_linguistic_variables
        self.verbose = verbose
        self.lvs = linguistic_variables
        self.categorical_mask = categorical_mask
        self.domain = domain
        self.precomputed_rules = precomputed_rules
        self.custom_loss = None
        self.alpha_ = 0.0
        self.beta_ = 0.0

        # Initialize evolutionary backend
        try:
            self.backend = ev_backends.get_backend(backend)
            if verbose:
                print(f"Using evolutionary backend: {self.backend.name()}")
        except ValueError as e:
            if verbose:
                print(f"Warning: {e}. Falling back to pymoo backend.")
            self.backend = ev_backends.get_backend('pymoo')

        # Thread runner
        if runner > 1:
            pool = ThreadPool(runner)
            self.thread_runner = StarmapParallelization(pool.starmap)
        else:
            self.thread_runner = None

        if precomputed_rules is not None:
            self.rule_base = precomputed_rules

    def fit(self, X: np.array, y: np.array, n_gen: int = 70, pop_size: int = 30,
            random_state: int = 33, var_prob: float = 0.3, sbx_eta: float = 3.0,
            mutation_eta: float = 7.0, tournament_size: int = 3) -> None:
        '''
        Fit the regressor to training data.

        :param X: input features (n_samples, n_features)
        :param y: target values (n_samples,)
        :param n_gen: number of generations
        :param pop_size: population size
        :param random_state: random seed
        :param var_prob: crossover probability
        :param sbx_eta: SBX eta parameter
        :param mutation_eta: mutation eta parameter
        :param tournament_size: tournament selection size
        '''
        if isinstance(X, pd.DataFrame):
            lvs_names = list(X.columns)
            X = X.values
        else:
            lvs_names = [str(ix) for ix in range(X.shape[1])]

        # Determine output range if not provided
        if self.y_range is None:
            y_min, y_max = np.min(y), np.max(y)
        else:
            y_min, y_max = self.y_range

        # Create optimization problem
        problem = FitRuleBaseRegression(
            X=X, y=y, nRules=self.nRules, nAnts=self.nAnts,
            y_range=(y_min, y_max), thread_runner=self.thread_runner,
            linguistic_variables=self.lvs,
            n_linguistic_variables=self.n_linguist_variables,
            fuzzy_type=self.fuzzy_type, domain=self.domain,
            categorical_mask=self.categorical_mask,
            tolerance=self.tolerance, alpha=self.alpha_, beta=self.beta_,
            rule_mode=self.rule_mode, backend_name=self.backend.name()
        )

        # Run optimization using selected backend
        result = self.backend.optimize(
            problem=problem,
            n_gen=n_gen,
            pop_size=pop_size,
            random_state=random_state,
            var_prob=var_prob,
            sbx_eta=sbx_eta,
            mutation_eta=mutation_eta,
            tournament_size=tournament_size,
            verbose=self.verbose
        )

        # Extract best solution
        if isinstance(result, dict):
            best_gene = result['X']
        elif hasattr(result, 'X'):
            best_gene = result.X
        else:
            best_gene = result[0]

        # Construct final rule base
        self.rule_base = problem._construct_ruleBase_regression(best_gene, self.fuzzy_type)
        self.lvs = self.rule_base.rule_bases[0].antecedents if self.lvs is None else self.lvs
        
        # Store the scalar y_range and bounds for predictions
        self.y_range_scalar = problem.y_range
        self.y_min = problem.y_min
        self.y_max = problem.y_max
        
        # ADDITIVE REGRESSION: Keep all rules (no filtering by tolerance)
        # All rules contribute to the final prediction weighted by their membership

        if self.verbose:
            print(f"Optimization complete. Final rule base has {len(self.rule_base.get_rules())} rules.")

    def predict(self, X: np.array) -> np.array:
        '''
        Predict target values for input samples.

        :param X: input features (n_samples, n_features)
        :return: predicted values (n_samples,)
        '''
        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = []
        precomputed_truth = rules.compute_antecedents_memberships(self.rule_base.antecedents, X)

        for i in range(X.shape[0]):
            sample_memberships = []
            sample_consequents = []

            for rule in self.rule_base.get_rules():
                # Compute rule membership
                membership = 1.0
                for ant_idx, ant_val in enumerate(rule.antecedents):
                    if ant_val >= 0:  # not don't care
                        ant_val = int(ant_val)
                        membership *= precomputed_truth[ant_idx][ant_val][i]

                # Filter by rule_mode: 'sufficient' uses tolerance, 'additive' includes all
                if self.rule_mode == 'sufficient':
                    if membership > self.tolerance:
                        sample_memberships.append(membership)
                        sample_consequents.append(rule.consequent)
                else:  # additive mode (default)
                    sample_memberships.append(membership)
                    sample_consequents.append(rule.consequent)

            if len(sample_memberships) > 0:
                # Weighted average
                pred = np.sum(np.array(sample_memberships) * np.array(sample_consequents)) / (np.sum(sample_memberships) + 1e-10)
            else:
                # No rules fired (can happen with sufficient mode), use midpoint
                pred = (self.y_min + self.y_max) / 2

            predictions.append(pred)

        return np.array(predictions)

    def score(self, X: np.array, y: np.array) -> float:
        '''
        Compute R² score on test data.

        :param X: input features
        :param y: true target values
        :return: R² score
        '''
        predictions = self.predict(X)
        return r2_score(y, predictions)

    def __str__(self) -> str:
        '''
        String representation showing the regression rules.
        '''
        if not hasattr(self, 'rule_base') or self.rule_base is None:
            return "BaseFuzzyRulesRegressor (not fitted)"
        
        output = f"BaseFuzzyRulesRegressor with {len(self.rule_base.get_rules())} rules (mode: {self.rule_mode})\n"
        output += "=" * 80 + "\n"
        output += self.rule_base.print_rules_regression(return_rules=True, output_name='output')
        return output

    def __repr__(self) -> str:
        '''
        Developer-friendly representation.
        '''
        if not hasattr(self, 'rule_base') or self.rule_base is None:
            return f"BaseFuzzyRulesRegressor(nRules={self.nRules}, nAnts={self.nAnts}, rule_mode='{self.rule_mode}', not fitted)"
        
        return f"BaseFuzzyRulesRegressor(nRules={len(self.rule_base.get_rules())}, nAnts={self.nAnts}, rule_mode='{self.rule_mode}', fuzzy_type={self.fuzzy_type})"
