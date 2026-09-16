"""
Evolutionary Search for Fuzzy Rule Selection

This module provides optimization classes for selecting optimal subsets of fuzzy rules
from a candidate pool using evolutionary algorithms. It's designed for scenarios where
you have a pre-generated set of candidate rules and want to find the best combination
for your classification problem.

Main Components:
    - ExploreRuleBases: Genetic algorithm-based rule selection from candidate pools
    - Fitness evaluation for rule subset quality
    - Integration with pymoo optimization framework

Use Cases:
    - Rule subset selection from large candidate pools
    - Ensemble creation from pre-computed rules
    - Feature selection at the rule level
    - Optimizing rule combinations for specific datasets
"""

import numpy as np
import pandas as pd
from typing import Optional, Any

# Import necessary modules
from ._problem import Integer, Problem
from . import fuzzy_sets as fs
from . import rules
from . import eval_rules as evr


class ExploreRuleBases(Problem):
    """
    Evolutionary search for optimal rule subset selection.
    
    This class formulates the rule selection problem as a pymoo optimization problem,
    where the goal is to find the best subset of rules from a candidate pool that
    maximizes classification performance while maintaining rule base simplicity.
    
    The genetic algorithm explores different combinations of candidate rules to build
    an optimal rule base for the given dataset.
    
    Attributes:
        X (np.array): Training data samples (n_samples, n_features)
        y (np.array): Training labels (n_samples,)
        nRules (int): Number of rules to select from candidates
        n_classes (int): Number of target classes
        candidate_rules (rules.MasterRuleBase): Pool of candidate rules
        tolerance (float): Tolerance for rule evaluation
        fuzzy_type (fs.FUZZY_SETS): Type of fuzzy sets used
    
    Example:
        >>> # Assuming you have candidate_rules already generated
        >>> problem = ExploreRuleBases(
        ...     X=X_train, y=y_train, nRules=20, 
        ...     n_classes=3, candidate_rules=candidate_pool
        ... )
        >>> # Use with pymoo optimizer
        >>> from pymoo.algorithms.soo.nonconvex.ga import GA
        >>> from pymoo.optimize import minimize
        >>> from ex_fuzzy.evolutionary_backends import as_pymoo_problem
        >>> algorithm = GA(pop_size=50)
        >>> res = minimize(as_pymoo_problem(problem), algorithm, ('n_gen', 100))
    """

    def __init__(self, X: np.array, y: np.array, nRules: int, n_classes: int,
                 candidate_rules: rules.MasterRuleBase, thread_runner: Optional[Any]=None,
                 tolerance:float = 0.01) -> None:
        """
        Initialize the rule selection optimization problem.

        Args:
            X: np array or pandas dataframe samples x features.
            y: np vector containing the target classes. vector sample
            nRules: number of rules to select from the candidate pool
            n_classes: number of classes in the problem.
            candidate_rules: MasterRuleBase object containing candidate rules.
            thread_runner: Optional elementwise runner, such as ``StarmapParallelization``
            tolerance: float. Tolerance for the size evaluation.
        """
        try:
            self.var_names = list(X.columns)
            self.X = X.values
        except AttributeError:
            self.X = X
            self.var_names = [str(ix) for ix in range(X.shape[1])]

        self.tolerance = tolerance
        self.fuzzy_type = candidate_rules.fuzzy_type()
        self.y = y
        self.nCons = 1  # This is fixed to MISO rules.
        self.n_classes = n_classes
        self.candidate_rules = candidate_rules
        self.nRules = nRules
        self._precomputed_truth = rules.compute_antecedents_memberships(
            candidate_rules.get_antecedents(), X
        )

        self.fuzzy_type = self.candidate_rules[0].antecedents[0].fuzzy_type()

        self.min_bounds = np.min(self.X, axis=0)
        self.max_bounds = np.max(self.X, axis=0)

        nTotalRules = len(self.candidate_rules.get_rules())
        # Each gene position selects one rule from the candidate pool
        vars = {ix: Integer(bounds=[0, nTotalRules - 1]) for ix in range(self.nRules)}
        varbound = np.array([[0, nTotalRules- 1]] * self.nRules)

        nVar = len(vars.keys())
        if thread_runner is not None:
            super().__init__(
                vars=vars,
                n_var=nVar,
                n_obj=1,
                elementwise=True,
                vtype=int,
                xl=varbound[:, 0],
                xu=varbound[:, 1],
                elementwise_runner=thread_runner)
        else:
            super().__init__(
                vars=vars,
                n_var=nVar,
                n_obj=1,
                elementwise=True,
                vtype=int,
                xl=varbound[:, 0],
                xu=varbound[:, 1])


    def _construct_ruleBase(self, x: np.array, fuzzy_type: fs.FUZZY_SETS, 
                           ds_mode:int=0, allow_unknown:bool=False) -> rules.MasterRuleBase:
        """
        Construct a rule base from selected candidate rules.

        Args:
            x: Gene representing selected rule indices from the candidate pool
            fuzzy_type: FUZZY_SET enum type in fuzzy_sets module
            ds_mode: int. Mode for the dominance score. 0: normal dominance score,
                1: rules without weights, 2: weights optimized for each rule
            allow_unknown: if True, allows unknown class in classification

        Returns:
            MasterRuleBase object with selected rules
        """
        x = x.astype(int)
        # Each gene position names one rule of the candidate pool, in class-major order.
        total_rules = self.candidate_rules.get_rules()
        rule_consequents = sum([[ix] * len(rule_base) for ix, rule_base in enumerate(self.candidate_rules)], [])
        antecedents = self.candidate_rules[0].antecedents

        # One rule base per consequent, empty when no selected rule belongs to it, so
        # that every rule base stays aligned with its consequent.
        rule_bases = []
        for consequent in range(len(self.candidate_rules)):
            rule_list = [total_rules[val] for val in x if rule_consequents[val] == consequent]
            if fuzzy_type == fs.FUZZY_SETS.t1:
                rule_bases.append(rules.RuleBaseT1(antecedents, rule_list))
            elif fuzzy_type == fs.FUZZY_SETS.t2:
                rule_bases.append(rules.RuleBaseT2(antecedents, rule_list))
            else:
                rule_bases.append(rules.RuleBaseGT2(antecedents, rule_list))

        return rules.MasterRuleBase(rule_bases, list(range(len(self.candidate_rules))),
                                    ds_mode=ds_mode, allow_unknown=allow_unknown)


    def _evaluate(self, x: np.array, out: dict, *args, **kwargs):
        """
        Evaluate the fitness of a rule subset selection.

        Args:
            x: Gene representing selected rule indices
            out: dict where the F field is the fitness (to be minimized)
        """
        ruleBase = self._construct_ruleBase(x, self.fuzzy_type)
        score = self.fitness_func(
            ruleBase, self.X, self.y, self.tolerance,
            precomputed_truth=self._precomputed_truth
        )
        out["F"] = 1 - score
    
    
    def fitness_func(self, ruleBase: rules.RuleBase, X:np.array, y:np.array, 
                    tolerance:float, alpha:float=0.0, beta:float=0.0, 
                    precomputed_truth=None) -> float:
        """
        Compute fitness for a rule base.
        
        Fitness is computed as a weighted combination of:
        - Classification accuracy
        - Rule size complexity (optional, controlled by alpha)
        - Number of rules (optional, controlled by beta)

        Args:
            ruleBase: RuleBase object to evaluate
            X: Training samples (n_samples, n_features)
            y: Training labels (n_samples,)
            tolerance: Tolerance for size evaluation
            alpha: Weight for rule size complexity penalty (default: 0.0)
            beta: Weight for number of rules penalty (default: 0.0)
            precomputed_truth: Precomputed membership values (optional)

        Returns:
            Fitness score (higher is better)
        """
        ev_object = evr.evalRuleBase(ruleBase, X, y, precomputed_truth=precomputed_truth)
        ev_object.add_rule_weights()

        score_acc = ev_object.classification_eval()
        score_rules_size = ev_object.size_antecedents_eval(tolerance)
        score_nrules = ev_object.effective_rulesize_eval(tolerance)

        score = score_acc + score_rules_size * alpha + score_nrules * beta

        return score
