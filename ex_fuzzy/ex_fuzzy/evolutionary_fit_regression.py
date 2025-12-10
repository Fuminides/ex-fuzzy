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
                 rule_mode: str = 'additive', backend_name: str = 'pymoo',
                 consequent_type: str = 'crisp', output_fuzzy_sets: list[fs.FS] = None,
                 n_output_linguistic_variables: int = 3, universe_points: int = 100) -> None:
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
        :param consequent_type: str. 'crisp' (numeric outputs) or 'fuzzy' (fuzzy set outputs with defuzzification).
        :param output_fuzzy_sets: list of fuzzy sets for output (Mamdani). If None, will be evolved.
        :param n_output_linguistic_variables: int. Number of output linguistic variables if not precomputed.
        :param universe_points: int. Number of discretization points for fuzzy inference.
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
        
        # Mamdani inference parameters
        self.consequent_type = consequent_type  # 'crisp' or 'fuzzy'
        self.output_fuzzy_sets = output_fuzzy_sets
        self.n_output_lvs = n_output_linguistic_variables
        self.universe_points = universe_points
        
        # Create discretized universe for Mamdani inference
        if consequent_type == 'fuzzy':
            self.universe = np.linspace(self.y_min, self.y_max, universe_points)
        
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

        # CONSEQUENTS: Different encoding for crisp vs fuzzy
        if self.consequent_type == 'crisp':
            # Crisp: Consequents are real numbers normalized to [0, 99]
            final_consequent_bounds = np.array([[0, 99]] * self.nRules)
            vars_consequent = {aux_counter + ix: Integer(
                bounds=final_consequent_bounds[ix]) for ix in range(len(final_consequent_bounds))}
            aux_counter += len(vars_consequent)
            
            # Build final variables dict (no output MFs)
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
        else:
            # Mamdani: Consequents are indices to output fuzzy sets
            if self.output_fuzzy_sets is not None:
                # Precomputed output fuzzy sets
                n_output_fs = len(self.output_fuzzy_sets)
            else:
                # Will evolve output fuzzy sets
                n_output_fs = self.n_output_lvs
            
            # Consequent indices
            consequent_index_bounds = np.array([[0, n_output_fs - 1]] * self.nRules)
            vars_consequent = {aux_counter + ix: Integer(
                bounds=consequent_index_bounds[ix]) for ix in range(len(consequent_index_bounds))}
            aux_counter += len(vars_consequent)
            
            # Output membership functions (if not precomputed)
            if self.output_fuzzy_sets is None:
                # Encode output fuzzy sets similar to input fuzzy sets
                if fuzzy_type == fs.FUZZY_SETS.t1:
                    output_mf_size = (n_output_fs - 1) * 4 + 3
                elif fuzzy_type == fs.FUZZY_SETS.t2:
                    output_mf_size = (n_output_fs - 1) * 6 + 2
                
                output_mf_bounds = np.array([[0, 99]] * output_mf_size)
                vars_output_mf = {aux_counter + ix: Integer(
                    bounds=output_mf_bounds[ix]) for ix in range(len(output_mf_bounds))}
                
                # Build final variables dict with output MFs
                if self.lvs is None:
                    vars = {key: val for d in [
                        vars_antecedent, vars_memberships, vars_consequent, vars_output_mf] for key, val in d.items()}
                    varbound = np.concatenate(
                        (antecedent_bounds, membership_bounds, consequent_index_bounds, output_mf_bounds), axis=0)
                else:
                    vars = {key: val for d in [vars_antecedent,
                                               vars_consequent, vars_output_mf] for key, val in d.items()}
                    varbound = np.concatenate(
                        (antecedent_bounds, consequent_index_bounds, output_mf_bounds), axis=0)
            else:
                # Precomputed output fuzzy sets
                if self.lvs is None:
                    vars = {key: val for d in [
                        vars_antecedent, vars_memberships, vars_consequent] for key, val in d.items()}
                    varbound = np.concatenate(
                        (antecedent_bounds, membership_bounds, consequent_index_bounds), axis=0)
                else:
                    vars = {key: val for d in [vars_antecedent,
                                               vars_consequent] for key, val in d.items()}
                    varbound = np.concatenate(
                        (antecedent_bounds, consequent_index_bounds), axis=0)

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

            # CONSEQUENT: Depends on consequent_type
            if self.consequent_type == 'crisp':
                # Crisp: Consequent is a real number (denormalize from [0, 99] to [y_min, y_max])
                consequent_normalized = x[fourth_pointer + aux_pointer]
                consequent_value = self.y_min + (consequent_normalized / 99.0) * self.y_range
                aux_pointer += 1
            else:
                # Mamdani: Consequent is an index to output fuzzy set
                consequent_value = int(x[fourth_pointer + aux_pointer])
                aux_pointer += 1

            # Create rule if it has at least one antecedent
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

        # Decode output fuzzy sets for Mamdani inference (if needed)
        if self.consequent_type == 'fuzzy' and self.output_fuzzy_sets is None:
            # Output fuzzy sets are encoded in the gene
            output_fs_pointer = fourth_pointer + self.nRules
            output_fs_genes = x[output_fs_pointer:]
            
            # Decode similar to input fuzzy sets
            if fuzzy_type == fs.FUZZY_SETS.t1:
                output_fs_size = (self.n_output_lvs - 1) * 4 + 3
            elif fuzzy_type == fs.FUZZY_SETS.t2:
                output_fs_size = (self.n_output_lvs - 1) * 6 + 2
            
            # Decode output membership functions
            decoded_output_fs = self._decode_output_membership_functions(output_fs_genes, fuzzy_type)
            
            # Normalize to output domain [y_min, y_max]
            # Flatten all parameters to find global min/max
            all_params = []
            for lv in decoded_output_fs:
                params = np.array(lv.membership_parameters).flatten()
                all_params.extend(params)
            min_lv = np.min(all_params)
            max_lv = np.max(all_params)
            
            output_fuzzy_sets = []
            for lx, output_fs in enumerate(decoded_output_fs):
                relevant_lv = np.array(output_fs.membership_parameters)
                relevant_lv = (relevant_lv - min_lv) / (max_lv - min_lv) * self.y_range + self.y_min
                if fuzzy_type == fs.FUZZY_SETS.t1:
                    proper_FS = fs.FS(f'Output_{lx}', relevant_lv, (self.y_min, self.y_max))
                elif fuzzy_type == fs.FUZZY_SETS.t2:
                    proper_FS = fs.IVFS(f'Output_{lx}', relevant_lv[0], relevant_lv[1], 
                                       (self.y_min, self.y_max))
                output_fuzzy_sets.append(proper_FS)
        elif self.consequent_type == 'fuzzy':
            # Use precomputed output fuzzy sets
            output_fuzzy_sets = self.output_fuzzy_sets
        else:
            output_fuzzy_sets = None

        # Create rule base
        if fuzzy_type == fs.FUZZY_SETS.t1:
            rule_base = rules.RuleBaseT1(antecedents, rule_list)
        elif fuzzy_type == fs.FUZZY_SETS.t2:
            rule_base = rules.RuleBaseT2(antecedents, rule_list)
        elif fuzzy_type == fs.FUZZY_SETS.gt2:
            rule_base = rules.RuleBaseGT2(antecedents, rule_list)

        # Store output fuzzy sets in rule base for Mamdani inference
        if output_fuzzy_sets is not None:
            rule_base.output_fuzzy_sets = output_fuzzy_sets

        # For regression, we use a single "class" (output is numeric)
        res = rules.MasterRuleBase([rule_base], ['output'], allow_unknown=False)

        return res

    def _decode_output_membership_functions(self, output_genes: np.array, fuzzy_type: fs.FUZZY_SETS) -> list[fs.FS]:
        '''
        Decode output fuzzy sets from gene array.
        
        :param output_genes: gene segment containing output FS parameters
        :param fuzzy_type: type of fuzzy set
        :return: list of output fuzzy sets
        '''
        output_fuzzy_sets = []
        
        if fuzzy_type == fs.FUZZY_SETS.t1:
            # Decode T1 fuzzy sets
            pointer = 0
            for lv_idx in range(self.n_output_lvs):
                if lv_idx == 0:
                    # First FS: triangular (3 parameters)
                    params_tri = output_genes[pointer:pointer + 3]
                    pointer += 3
                    # Convert triangular to trapezoidal: [a, b, c] → [a, b, b, c]
                    params = [params_tri[0], params_tri[1], params_tri[1], params_tri[2]]
                else:
                    # Subsequent FSs: trapezoidal (4 parameters)
                    params = output_genes[pointer:pointer + 4]
                    pointer += 4
                
                # Parameters are in [0, 99] range, will be normalized later
                fs_instance = fs.FS(f'Output_{lv_idx}', params)
                output_fuzzy_sets.append(fs_instance)
                
        elif fuzzy_type == fs.FUZZY_SETS.t2:
            # Decode T2 fuzzy sets (interval type-2)
            pointer = 0
            for lv_idx in range(self.n_output_lvs):
                if lv_idx == 0:
                    # First FS: 2 parameters
                    params_lower = output_genes[pointer:pointer + 1]
                    params_upper = output_genes[pointer + 1:pointer + 2]
                    pointer += 2
                else:
                    # Subsequent FSs: 6 parameters
                    params_lower = output_genes[pointer:pointer + 3]
                    params_upper = output_genes[pointer + 3:pointer + 6]
                    pointer += 6
                
                fs_instance = fs.IVFS(f'Output_{lv_idx}', params_lower, params_upper)
                output_fuzzy_sets.append(fs_instance)
        
        return output_fuzzy_sets

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

        # Get consequents - different handling for crisp vs fuzzy
        if self.consequent_type == 'crisp':
            # Crisp: Denormalize from [0, 99] to [y_min, y_max]
            rule_consequents_normalized = x[fourth_pointer:fourth_pointer + self.nRules]
            rule_consequents = self.y_min + (rule_consequents_normalized / 99.0) * self.y_range

            # Weighted average
            numerator = np.sum(rule_memberships * rule_consequents[np.newaxis, :], axis=1)
            denominator = np.sum(rule_memberships, axis=1) + 1e-10
            predicted_values = numerator / denominator
        else:
            # Mamdani: Fuzzy inference with defuzzification
            # Get consequent indices
            rule_consequent_indices = x[fourth_pointer:fourth_pointer + self.nRules].astype(int)
            
            # Decode output fuzzy sets if needed
            if self.output_fuzzy_sets is None:
                output_fs_pointer = fourth_pointer + self.nRules
                output_fs_genes = x[output_fs_pointer:]
                output_fuzzy_sets = self._decode_output_membership_functions(output_fs_genes, fuzzy_type)
                
                # Normalize to output domain
                # Flatten all parameters to find global min/max
                all_params = []
                for lv in output_fuzzy_sets:
                    params = np.array(lv.membership_parameters).flatten()
                    all_params.extend(params)
                min_lv = np.min(all_params)
                max_lv = np.max(all_params)
                
                normalized_output_fs = []
                for lx, output_fs in enumerate(output_fuzzy_sets):
                    relevant_lv = np.array(output_fs.membership_parameters)
                    relevant_lv = (relevant_lv - min_lv) / (max_lv - min_lv) * self.y_range + self.y_min
                    if fuzzy_type == fs.FUZZY_SETS.t1:
                        proper_FS = fs.FS(f'Output_{lx}', relevant_lv, (self.y_min, self.y_max))
                    normalized_output_fs.append(proper_FS)
                output_fuzzy_sets = normalized_output_fs
            else:
                output_fuzzy_sets = self.output_fuzzy_sets
            
            # Perform Mamdani inference for each sample
            predicted_values = self._mamdani_inference_vectorized(
                rule_memberships, rule_consequent_indices, output_fuzzy_sets)

        # Compute RMSE
        mse = mean_squared_error(y, predicted_values)
        rmse = np.sqrt(mse)

        # Return negative RMSE (so higher fitness is better)
        # Normalize by y_range to make it scale-independent
        normalized_rmse = rmse / (self.y_range + 1e-10)

        return -normalized_rmse  # Negative because pymoo minimizes

    def _mamdani_inference_vectorized(self, rule_memberships: np.array, rule_consequent_indices: np.array,
                                     output_fuzzy_sets: list[fs.FS]) -> np.array:
        '''
        Vectorized Mamdani fuzzy inference with centroid defuzzification.
        
        :param rule_memberships: array of shape (n_samples, n_rules) with rule firing strengths
        :param rule_consequent_indices: array of shape (n_rules,) with output FS indices for each rule
        :param output_fuzzy_sets: list of output fuzzy sets
        :return: defuzzified predictions array of shape (n_samples,)
        '''
        n_samples = rule_memberships.shape[0]
        n_rules = rule_memberships.shape[1]
        
        # Discretized universe
        universe = self.universe
        n_points = len(universe)
        
        # Compute membership values for all output fuzzy sets at all universe points
        output_memberships = np.zeros((len(output_fuzzy_sets), n_points))
        for fs_idx, output_fs in enumerate(output_fuzzy_sets):
            output_memberships[fs_idx, :] = output_fs.membership(universe)
        
        # For each sample, aggregate clipped fuzzy sets and defuzzify
        predictions = np.zeros(n_samples)
        for sample_idx in range(n_samples):
            # Initialize aggregated output membership (start with zeros)
            aggregated = np.zeros(n_points)
            
            # For each rule, clip the consequent fuzzy set by the rule's firing strength
            for rule_idx in range(n_rules):
                firing_strength = rule_memberships[sample_idx, rule_idx]
                consequent_idx = rule_consequent_indices[rule_idx]
                
                # Clip: multiply output FS memberships by firing strength
                clipped = np.minimum(output_memberships[consequent_idx, :], firing_strength)
                
                # Aggregate: MAX operation (union)
                aggregated = np.maximum(aggregated, clipped)
            
            # Defuzzify using centroid method
            if np.sum(aggregated) > 1e-10:
                predictions[sample_idx] = np.sum(universe * aggregated) / np.sum(aggregated)
            else:
                # No rules fired, use midpoint
                predictions[sample_idx] = (self.y_min + self.y_max) / 2
        
        return predictions

    def _evaluate_torch_fast_regression(self, x, y, fuzzy_type: fs.FUZZY_SETS, device='cuda', return_predictions=False):
        '''
        PyTorch GPU-accelerated evaluation for regression.
        
        :param x: gene tensor (can be numpy array or torch tensor)
        :param y: target values tensor (can be numpy array or torch tensor)
        :param fuzzy_type: enum type for fuzzy set type
        :param device: device to run computation on ('cuda' or 'cpu')
        :param return_predictions: if True, return predictions; if False, return R² score
        :return: R² score (as Python float) or prediction tensor
        '''
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for GPU acceleration. Install with: pip install torch")
        
        # Convert inputs to torch tensors if needed
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).to(device)
        else:
            x = x.to(device)
            
        if not isinstance(y, torch.Tensor):
            y_torch = torch.from_numpy(y).float().to(device)
        else:
            y_torch = y.float().to(device)
        
        # Convert training data to torch if not already done
        if not hasattr(self, 'X_torch_reg') or self.X_torch_reg is None:
            self.X_torch_reg = torch.from_numpy(self.X).float().to(device)
        
        mf_size = 4 if fuzzy_type == fs.FUZZY_SETS.t1 else 6
        
        # Calculate pointers
        if self.lvs is None:
            if fuzzy_type == fs.FUZZY_SETS.t1:
                fourth_pointer = 2 * self.nAnts * self.nRules + \
                    len(self.n_lv_possible) * 3 + sum(np.array(self.n_lv_possible) - 1) * 4
            elif fuzzy_type == fs.FUZZY_SETS.t2:
                fourth_pointer = 2 * self.nAnts * self.nRules + \
                    len(self.n_lv_possible) * 2 + sum(np.array(self.n_lv_possible) - 1) * mf_size
        else:
            fourth_pointer = 2 * self.nAnts * self.nRules
        
        # Get precomputed memberships and convert to torch
        if self.lvs is None:
            # Convert to numpy for decoding (membership decoding is complex, keep in numpy)
            x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
            antecedents = self._decode_membership_functions(x_np, fuzzy_type)
            precomputed_antecedent_memberships_np = rules.compute_antecedents_memberships(antecedents, self.X)
            precomputed_antecedent_memberships = [
                torch.from_numpy(ant_mems).float().to(device) 
                for ant_mems in precomputed_antecedent_memberships_np
            ]
        else:
            if not hasattr(self, '_precomputed_truth_torch_reg') or self._precomputed_truth_torch_reg is None:
                self._precomputed_truth_torch_reg = [
                    torch.from_numpy(ant_mems).float().to(device) 
                    for ant_mems in self._precomputed_truth
                ]
            precomputed_antecedent_memberships = self._precomputed_truth_torch_reg
        
        # Ensure x is integer type
        x = x.long()
        
        n_samples = self.X.shape[0]
        n_features = self.X.shape[1]
        
        # Extract gene segments
        chosen_ants = x[:self.nAnts * self.nRules].reshape(self.nRules, self.nAnts)
        ant_params = x[self.nAnts * self.nRules:2 * self.nAnts * self.nRules].reshape(self.nRules, self.nAnts)
        
        # Clamp parameters to valid range
        for ant_idx in range(n_features):
            mask = chosen_ants == ant_idx
            if self.lvs is not None:
                max_param = len(self.lvs[ant_idx]) - 1
            else:
                max_param = self.n_lv_possible[ant_idx] - 1
            ant_params = torch.where(mask, torch.clamp(ant_params, max=max_param), ant_params)
        
        # Compute rule memberships (vectorized)
        rule_memberships = torch.ones((n_samples, self.nRules), device=device)
        for rule_idx in range(self.nRules):
            for ant_idx in range(self.nAnts):
                feature = chosen_ants[rule_idx, ant_idx].item()
                param = ant_params[rule_idx, ant_idx].item()
                rule_memberships[:, rule_idx] *= precomputed_antecedent_memberships[feature][param, :]
        
        # Get consequents and handle crisp vs fuzzy
        if self.consequent_type == 'crisp':
            # Crisp consequents: denormalize from [0, 99] to [y_min, y_max]
            rule_consequents_normalized = x[fourth_pointer:fourth_pointer + self.nRules].float()
            rule_consequents = self.y_min + (rule_consequents_normalized / 99.0) * (self.y_max - self.y_min)
            
            if self.rule_mode == 'additive':
                # Weighted average
                numerator = torch.sum(rule_memberships * rule_consequents.unsqueeze(0), dim=1)
                denominator = torch.sum(rule_memberships, dim=1) + 1e-10
                predictions = numerator / denominator
            else:  # sufficient mode
                # Winner-takes-all
                max_memberships, max_indices = torch.max(rule_memberships, dim=1)
                predictions = rule_consequents[max_indices]
                # Set to mean for samples with no active rules
                predictions = torch.where(max_memberships > self.tolerance, predictions, torch.mean(y_torch))
        else:
            # Fuzzy consequents: Mamdani inference (currently not GPU-optimized, fall back to numpy)
            x_np = x.cpu().numpy()
            predictions_np = self._mamdani_inference_vectorized(x_np, fourth_pointer, fuzzy_type)
            predictions = torch.from_numpy(predictions_np).float().to(device)
        
        if return_predictions:
            return predictions
        else:
            # Compute R² score
            ss_res = torch.sum((y_torch - predictions) ** 2)
            ss_tot = torch.sum((y_torch - torch.mean(y_torch)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            return r2.item()

    def has_torch_eval(self) -> bool:
        '''
        Check if PyTorch evaluation is available.
        
        :return: True if torch is available, False otherwise
        '''
        try:
            import torch
            return True
        except ImportError:
            return False

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

        # Predict using the rule base - different for crisp vs fuzzy consequents
        if self.consequent_type == 'crisp':
            # Crisp inference
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
        else:
            # Mamdani inference
            # Compute rule memberships for all samples
            n_samples = X.shape[0]
            n_rules = len(ruleBase.get_rules())
            rule_memberships = np.zeros((n_samples, n_rules))
            rule_consequent_indices = np.zeros(n_rules, dtype=int)
            
            for rule_idx, rule in enumerate(ruleBase.get_rules()):
                rule_consequent_indices[rule_idx] = int(rule.consequent)
                
                for sample_idx in range(n_samples):
                    membership = 1.0
                    for ant_idx, ant_val in enumerate(rule.antecedents):
                        if ant_val >= 0:
                            ant_val = int(ant_val)
                            membership *= precomputed_truth[ant_idx][ant_val][sample_idx]
                    rule_memberships[sample_idx, rule_idx] = membership
            
            # Get output fuzzy sets from rule base
            output_fuzzy_sets = ruleBase.output_fuzzy_sets if hasattr(ruleBase, 'output_fuzzy_sets') else self.output_fuzzy_sets
            
            # Perform Mamdani inference
            predictions = self._mamdani_inference_vectorized(
                rule_memberships, rule_consequent_indices, output_fuzzy_sets)

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
                 rule_mode: str = 'additive', backend: str = 'pymoo',
                 consequent_type: str = 'crisp', output_fuzzy_sets: list[fs.FS] = None,
                 n_output_linguistic_variables: int = 3, universe_points: int = 100) -> None:
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
        :param consequent_type: 'crisp' (numeric outputs) or 'fuzzy' (fuzzy set outputs with defuzzification)
        :param output_fuzzy_sets: precomputed output fuzzy sets for Mamdani inference. If None, will be evolved
        :param n_output_linguistic_variables: number of output linguistic variables (if not precomputed)
        :param universe_points: number of discretization points for fuzzy inference
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
        
        # Mamdani fuzzy inference parameters
        self.consequent_type = consequent_type  # 'crisp' or 'fuzzy'
        self.output_fuzzy_sets = output_fuzzy_sets  # Precomputed output FSs
        self.n_output_linguistic_variables = n_output_linguistic_variables
        self.universe_points = universe_points  # Discretization for Mamdani

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
            consequent_type=self.consequent_type,
            output_fuzzy_sets=self.output_fuzzy_sets,
            n_output_linguistic_variables=self.n_output_linguistic_variables,
            universe_points=self.universe_points,
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

        precomputed_truth = rules.compute_antecedents_memberships(self.rule_base.antecedents, X)

        if self.consequent_type == 'crisp':
            # Crisp inference
            predictions = []
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
        else:
            # Mamdani inference
            n_samples = X.shape[0]
            n_rules = len(self.rule_base.get_rules())
            rule_memberships = np.zeros((n_samples, n_rules))
            rule_consequent_indices = np.zeros(n_rules, dtype=int)
            
            for rule_idx, rule in enumerate(self.rule_base.get_rules()):
                rule_consequent_indices[rule_idx] = int(rule.consequent)
                
                for sample_idx in range(n_samples):
                    membership = 1.0
                    for ant_idx, ant_val in enumerate(rule.antecedents):
                        if ant_val >= 0:
                            ant_val = int(ant_val)
                            membership *= precomputed_truth[ant_idx][ant_val][sample_idx]
                    rule_memberships[sample_idx, rule_idx] = membership
            
            # Get output fuzzy sets
            rule_base_obj = self.rule_base.rule_bases[0]
            output_fuzzy_sets = rule_base_obj.output_fuzzy_sets if hasattr(rule_base_obj, 'output_fuzzy_sets') else self.output_fuzzy_sets
            
            # Create discretized universe
            universe = np.linspace(self.y_min, self.y_max, self.universe_points)
            
            # Perform Mamdani inference
            predictions = self._mamdani_inference_predict(
                rule_memberships, rule_consequent_indices, output_fuzzy_sets, universe)
            
            return predictions

    def _mamdani_inference_predict(self, rule_memberships: np.array, rule_consequent_indices: np.array,
                                   output_fuzzy_sets: list[fs.FS], universe: np.array) -> np.array:
        '''
        Mamdani fuzzy inference with centroid defuzzification for prediction.
        
        :param rule_memberships: array of shape (n_samples, n_rules) with rule firing strengths
        :param rule_consequent_indices: array of shape (n_rules,) with output FS indices for each rule
        :param output_fuzzy_sets: list of output fuzzy sets
        :param universe: discretized universe of discourse
        :return: defuzzified predictions array of shape (n_samples,)
        '''
        n_samples = rule_memberships.shape[0]
        n_rules = rule_memberships.shape[1]
        n_points = len(universe)
        
        # Compute membership values for all output fuzzy sets at all universe points
        output_memberships = np.zeros((len(output_fuzzy_sets), n_points))
        for fs_idx, output_fs in enumerate(output_fuzzy_sets):
            output_memberships[fs_idx, :] = output_fs.membership(universe)
        
        # For each sample, aggregate clipped fuzzy sets and defuzzify
        predictions = np.zeros(n_samples)
        for sample_idx in range(n_samples):
            # Initialize aggregated output membership (start with zeros)
            aggregated = np.zeros(n_points)
            
            # For each rule, clip the consequent fuzzy set by the rule's firing strength
            for rule_idx in range(n_rules):
                firing_strength = rule_memberships[sample_idx, rule_idx]
                consequent_idx = rule_consequent_indices[rule_idx]
                
                # Clip: multiply output FS memberships by firing strength
                clipped = np.minimum(output_memberships[consequent_idx, :], firing_strength)
                
                # Aggregate: MAX operation (union)
                aggregated = np.maximum(aggregated, clipped)
            
            # Defuzzify using centroid method
            if np.sum(aggregated) > 1e-10:
                predictions[sample_idx] = np.sum(universe * aggregated) / np.sum(aggregated)
            else:
                # No rules fired, use midpoint
                predictions[sample_idx] = (self.y_min + self.y_max) / 2
        
        return predictions

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
        
        output = f"BaseFuzzyRulesRegressor with {len(self.rule_base.get_rules())} rules "
        output += f"(mode: {self.rule_mode}, consequent: {self.consequent_type})\n"
        output += "=" * 80 + "\n"
        output += self.rule_base.print_rules_regression(return_rules=True, output_name='output')
        return output

    def __repr__(self) -> str:
        '''
        Developer-friendly representation.
        '''
        if not hasattr(self, 'rule_base') or self.rule_base is None:
            return f"BaseFuzzyRulesRegressor(nRules={self.nRules}, nAnts={self.nAnts}, rule_mode='{self.rule_mode}', consequent_type='{self.consequent_type}', not fitted)"
        
        return f"BaseFuzzyRulesRegressor(nRules={len(self.rule_base.get_rules())}, nAnts={self.nAnts}, rule_mode='{self.rule_mode}', consequent_type='{self.consequent_type}', fuzzy_type={self.fuzzy_type})"
