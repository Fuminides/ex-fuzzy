"""
Fuzzy Rule Mining Module for Ex-Fuzzy Library

This module provides comprehensive fuzzy rule mining capabilities for extracting meaningful
rules from datasets using support-based itemset mining algorithms. It implements efficient
algorithms for discovering frequent fuzzy patterns and generating fuzzy association rules.

Main Components:
    - Itemset mining: Discovery of frequent fuzzy itemsets using support thresholds
    - Rule generation: Conversion of frequent itemsets into fuzzy association rules
    - Support calculation: Efficient computation of fuzzy support measures
    - Rule filtering: Quality-based filtering of discovered rules
    - Integration with evolutionary algorithms: Compatibility with genetic optimization

Key Features:
    - Support-based fuzzy itemset mining with configurable thresholds
    - Efficient combination generation for large datasets
    - Integration with fuzzy variable definitions
    - Support for both Type-1 and Type-2 fuzzy systems
    - Scalable algorithms for large rule bases
    - Direct integration with evolutionary optimization workflows

The module is designed to work seamlessly with the evolutionary_fit module to provide
a complete pipeline from rule discovery to rule optimization, enabling the creation
of high-quality fuzzy rule-based classifiers from data.
"""
import typing
from itertools import product, combinations

import pandas as pd
import numpy as np

from . import rules as rl
from . import fuzzy_sets as fs
from . import utils


def _as_array(data) -> np.ndarray:
    """Return the sample matrix of a DataFrame or array-like as a NumPy array."""
    return data.values if hasattr(data, 'values') else np.asarray(data)


def _generate_combinations(lists: list, k: int) -> typing.Iterator:
    """
    Generate all the combinations between elements of different lists of length k without repeting elements of the same list.

    Args:
        lists: list of lists.
        k: integer with the length of the combinations.

    Returns:
        a list with all the combinations.
    """
    # Get all combinations of elements for k
    all_combs = combinations(np.arange(len(lists)), k)
    
    # For those elements, get the cartesian product between them
    for comb in all_combs:
        selected_lists = [lists[x] for x in comb]
        all_combinations = product(*selected_lists)

        # Add them to the global combination list
        yield all_combinations


def rule_search(data: pd.DataFrame, fuzzy_variables: dict[fs.fuzzyVariable], support_threshold:float=0.05, max_depth:int=None) -> list:
    """
    Computes the apriori algorithm for the given dataframe and threshold the support.
    
    The min t-norm support of an itemset cannot exceed the support of any of its
    subsets, so only the extensions of frequent itemsets are evaluated (Apriori).
    The itemsets found, and their order, are those of the exhaustive enumeration.

    Args:
        data: Dataframe of shape: samples x features
        fuzzy_variables: list with the fuzzy variable of each feature.
        support_threshold: minimum support to consider frequent an itemset.
        max_depth: maximum number of items per itemset. Defaults to the number of features.

    Returns:
        all the frequent itemsets as a list of tuples of (feature, linguistic label) pairs.
    """
    values = _as_array(data)
    memberships = [fuzzy_variables[ix](values[:, ix]) for ix in range(values.shape[1])]

    if max_depth is None:
        max_depth = data.shape[1]

    def support(itemset):
        # Minimum t-norm per sample, then the mean over the samples and any
        # interval or alpha-cut axes of Type-2 memberships.
        item_memberships = np.stack([np.asarray(memberships[item_var][item_vl]) for item_var, item_vl in itemset])
        return np.mean(np.min(item_memberships, axis=0))

    def enumeration_order(itemset):
        # The exhaustive enumeration lists itemsets by their features, then by their labels.
        return tuple(feature for feature, _ in itemset), tuple(label for _, label in itemset)

    items = [(ix, ax) for ix, fuzzy_variable in enumerate(fuzzy_variables) for ax in range(len(fuzzy_variable))]
    frequent = [(item,) for item in items if support((item,)) > support_threshold]
    freq_itemsets = list(frequent)
    for _ in range(2, max_depth + 1):
        known = set(frequent)
        candidates = {itemset + (item,) for itemset in frequent for item in items if item[0] > itemset[-1][0]}
        frequent = [candidate for candidate in sorted(candidates, key=enumeration_order)
                    if all(candidate[:index] + candidate[index + 1:] in known for index in range(len(candidate)))
                    and support(candidate) > support_threshold]
        if not frequent:
            break
        freq_itemsets.extend(frequent)

    return freq_itemsets


def generate_rules_from_itemsets(itemsets:list, nAnts:int) -> list[rl.RuleSimple]:
    """
    Given a list of itemsets, it creates the rules for each one and returns a list of rules containing them.

    Args:
        itemsets: list of tuple (antecedent, linguistic variable value)
        nAnts: number of possible antecedents.

    Returns:
        the rules for ech itemset.
    """
    rules = []
    for itemset in itemsets:
        template = np.ones((nAnts, )) * -1
        for ant, vl in itemset:
            template[ant] = vl
        
        rule = rl.RuleSimple(list(template))
        rules.append(rule)

    return rules


def mine_rulebase_support(x: pd.DataFrame, fuzzy_variables:list[fs.fuzzyVariable], support_threshold:float=0.05, max_depth:int=3) -> rl.RuleBase:
    """
    Search the data for associations that are frequent given a list of fuzzy variables for each antecedent.

    Args:
        x: the data to mine. Dims: samples x features.
        fuzzy_variables: list of the fuzzy variables for each of the input variables.
        support_threshold: minimum threshold to decide if prune or not the rule.
        max_depth: maximum number of antecedents per rule.

    Returns:
        a rulebase object with the rules denoted as good.
    """
    
    freq_itemsets = rule_search(x, fuzzy_variables, support_threshold, max_depth)
    rule_list = generate_rules_from_itemsets(freq_itemsets, len(fuzzy_variables))

    fuzzy_type = fuzzy_variables[0].fs_type

    if fuzzy_type == fs.FUZZY_SETS.t1:
        rule_base = rl.RuleBaseT1(fuzzy_variables, rule_list)
    elif fuzzy_type == fs.FUZZY_SETS.t2:
        rule_base = rl.RuleBaseT2(fuzzy_variables, rule_list)
    else:
        rule_base = rl.RuleBaseGT2(fuzzy_variables, rule_list)

    return rule_base


def prune_rules_confidence_lift(x: pd.DataFrame, y:np.array, rules: rl.MasterRuleBase, fuzzy_variables: list[fs.fuzzyVariable], confidence_threshold:float=0.5, 
                                lift_threshold:float=1.05):
    """
    Removes the rules from the rule base that do not meet a minimum value for confidence and lift measures.

    Confidence is the ratio of rules that have a particular antecedent and consequent, and those that only have the antecedent.
    Lift is ratio between confidence and expected confidence, which is the percentage of class samples in the original data.

    Args:
        x: data to mine. samples x features.
        y: class vector, as indexes of the rule bases in the master rule base (0 for the first rule base, and so on).
        rules: MasterRuleBase object with the rules to prune.
        fuzzy_variables: a list of the fuzzy variables per antecedent.
        confidence_threshold: minimum confidence required to the rules.
        lift_threshold: minimum lift required to the rules.
    """
    x_values = _as_array(x)
    y = np.asarray(y)
    # Memberships of every sample to every linguistic label, computed once per
    # variable instead of once per rule and antecedent. Fuzzy variables are
    # aligned with the columns, so antecedent zx reads column zx. Interval and
    # alpha-cut memberships are summarised by their mean.
    memberships = []
    for zx, fuzzy_variable in enumerate(fuzzy_variables):
        variable_memberships = np.asarray(fuzzy_variable(x_values[:, zx]))
        if variable_memberships.ndim > 2:
            variable_memberships = np.mean(variable_memberships, axis=tuple(range(2, variable_memberships.ndim)))
        memberships.append(variable_memberships)

    for ix, rule_base in enumerate(rules):
        delete_list = []
        relevant_class = ix
        class_samples = np.equal(y, relevant_class)
        class_prior = np.mean(class_samples)

        for jx, rule in enumerate(rule_base):
            antecedents = [(zx, antecedent) for zx, antecedent in enumerate(rule) if antecedent != -1]
            global_membership_array = np.stack(
                [memberships[zx][antecedent] for zx, antecedent in antecedents], axis=1)
            class_samples_membership_array = global_membership_array[class_samples]

            # Compute rule confidence
            global_support = np.mean(np.min(global_membership_array, axis=1), axis=0)
            class_support = np.mean(np.min(class_samples_membership_array, axis=1), axis=0)

            rule_confidence = class_support / global_support
            rule_lift = rule_confidence / class_prior

            if rule_confidence < confidence_threshold or rule_lift < lift_threshold:
                delete_list.append(jx)

        rule_base.remove_rules(delete_list)


def simple_mine_rulebase(x: pd.DataFrame, fuzzy_type:fs.FUZZY_SETS=fs.FUZZY_SETS.t1, support_threshold:float=0.05, max_depth:int=3) -> rl.RuleBase:
    """
    Search the data for associations that are frequent. Computes the fuzzy variables using a 3 label partition (low, medium, high).

    Args:
        x: the data to mine. Dims: samples x features.
        fuzzy_type: fuzzy type to use.
        support_threshold: minimum threshold to decide if prune or not the rule.
        max_depth: maximum number of antecedents per rule.

    Returns:
        a rulebase object with the rules denoted as good.
    """
    
    precomputed_partitions = utils.construct_partitions(x, fuzzy_type)
    return mine_rulebase_support(x, precomputed_partitions, support_threshold, max_depth)


def multiclass_mine_rulebase(x: pd.DataFrame, y: np.array, fuzzy_variables:list[fs.fuzzyVariable], support_threshold:float=0.05, max_depth:int=3,
                             confidence_threshold:float=0.05, lift_threshold:float=1.05) -> rl.MasterRuleBase:
    """
    Search the data for associations that are frequent and have good confidence/lift values given a list of fuzzy variables for each antecedent. Computes a different ruleBase for each 
    class and then uses them to form a MasterRuleBase.

    Args:
        x: the data to mine. Dims: samples x features.
        fuzzy_variables: list of the fuzzy variables for each of the input variables.
        support_threshold: minimum threshold to decide if prune or not the rule.
        max_depth: maximum number of antecedents per rule.
        confidence_threshold: minimum confidence value.
        lift_threshold:

    Returns:
        a rulebase object with the rules denoted as good.
    """
    unique_classes, y_index = np.unique(y, return_inverse=True)
    rulebases = []
    for yclass in unique_classes:
        selected_samples = np.equal(yclass, y)
        selected_x = _as_array(x)[selected_samples]

        rulebase = mine_rulebase_support(selected_x, fuzzy_variables, support_threshold, max_depth)
        rulebases.append(rulebase)

    master_rulebase = rl.MasterRuleBase(rulebases, list(map(str, unique_classes)))
    # The pruning compares the labels with the rule base indexes, so it gets
    # the labels encoded in the same order the rule bases were built.
    prune_rules_confidence_lift(x, y_index, master_rulebase, fuzzy_variables, confidence_threshold, lift_threshold)
    return master_rulebase


def simple_multiclass_mine_rulebase(x: pd.DataFrame, y: np.array, fuzzy_type:fs.FUZZY_SETS, support_threshold:float=0.05, max_depth:int=3,
                                    confidence_threshold:float=0.5, lift_threshold:float=1.1) -> rl.MasterRuleBase:
    """
    Search the data for associations that are frequent and have good confidence/lift values given a list of fuzzy variables for each antecedent. 
    Computes a different ruleBase for each class and then uses them to form a MasterRuleBase.

    Computes the fuzzy variables using a 3 label partition (low, medium, high).

    Args:
        x: the data to mine. Dims: samples x features.
        fuzzy_type: fuzzy type to use.
        support_threshold: minimum threshold to decide if prune or not the rule.
        max_depth: maximum number of antecedents per rule.

    Returns:
        a rulebase object with the rules denoted as good.
    """
    precomputed_partitions = utils.construct_partitions(x, fuzzy_type)
    return multiclass_mine_rulebase(x, y, precomputed_partitions, support_threshold, max_depth, 
                                    confidence_threshold=confidence_threshold, lift_threshold=lift_threshold)





