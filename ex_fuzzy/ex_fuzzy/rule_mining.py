"""
Module to perform rule mining in a pandas dataframe or numpy array. The methods use the support of the different itemsets to look for good 
rule candidates. It can be used then by a Genetic optimizator from evolutionary_fit module to search the optimal combination of them.

"""
import typing
from itertools import product, combinations

import pandas as pd
import numpy as np

try:
    from . import rules as rl
    from . import fuzzy_sets as fs
    from . import maintenance as mnt
    from . import utils
except ImportError:
    import utils
    import rules as rl
    import fuzzy_sets as fs
    import maintenance as mnt


def _generate_combinations(lists: list, k: int) -> typing.Iterator:
    '''
    Generate all the combinations between elements of different lists of length k without repeting elements of the same list.

    :param lists: list of lists.
    :param k: integer with the length of the combinations.
    :return: a list with all the combinations.
    '''
    # Get all combinations of elements for k
    all_combs = combinations(np.arange(len(lists)), k)
    
    # For those elements, get the cartesian product between them
    for comb in all_combs:
        selected_lists = [lists[x] for x in comb]
        all_combinations = product(*selected_lists)

        # Add them to the global combination list
        yield all_combinations


def rule_search(data: pd.DataFrame, fuzzy_variables: dict[fs.fuzzyVariable], support_threshold:float=0.05, max_depth:int=None) -> list:
    '''
    Computes the apriori algorithm for the given dataframe and threshold the support.
    
    :param data: Dataframe of shape: samples x features
    :param fuzzy variables: dict that maps each feature name with a fuzzy variable.
    :param support_threshold: minimum support to consider frequent an itemset.
    :return: all the frequent itemsets as a list.
    '''
    n_linguist_variables = len(fuzzy_variables[0])
    list_possible_vars = []
    for ix in range(len(fuzzy_variables)):
        list_possible_vars.append([(ix, ax) for ax in range(n_linguist_variables)])

    memberships = [fuzzy_variables[ix](data.iloc[:, ix].values) for ix in range(data.shape[1])]
    freq_itemsets = []

    if max_depth is None:
        max_depth = data.shape[1]
    
    # For all possible lengths
    for r in range(max_depth):
        all_r_combs = _generate_combinations(list_possible_vars, r+1)

        # Iterate through the possible itemsets
        for itemsets in all_r_combs:
            for ix, itemset in enumerate(itemsets):
                relevant_memberships = []
                for item in itemset:
                    item_var, item_vl = item
                    relevant_memberships.append([memberships[item_var][item_vl]])
                
                array_membership = np.array(relevant_memberships).T[:,0,:]
                support = np.mean(np.min(array_membership, axis=1))
                if fuzzy_variables[0].fuzzy_type == fs.FUZZY_SETS.t2 or fuzzy_variables[0].fuzzy_type == fs.FUZZY_SETS.gt2:
                    support = np.mean(support, axis=1)

                if support > support_threshold:
                    freq_itemsets.append(itemset) 

    return freq_itemsets


def generate_rules_from_itemsets(itemsets:list, nAnts:int) -> list[rl.RuleSimple]:
    '''
    Given a list of itemsets, it creates the rules for each one and returns a list of rules containing them.

    :param itemsets: list of tuple (antecedent, linguistic variable value) 
    :param nAnts: number of possible antecedents.
    :return: the rules for ech itemset.
    '''
    rules = []
    for itemset in itemsets:
        template = np.ones((nAnts, )) * -1
        for ant, vl in itemset:
            template[ant] = vl
        
        rule = rl.RuleSimple(list(template))
        rules.append(rule)

    return rules


def mine_rulebase_support(x: pd.DataFrame, fuzzy_variables:list[fs.fuzzyVariable], support_threshold:float=0.05, max_depth:int=3) -> rl.RuleBase:
    '''
    Search the data for associations that are frequent given a list of fuzzy variables for each antecedent.

    :param x: the data to mine. Dims: samples x features.
    :param fuzzy_variables: list of the fuzzy variables for each of the input variables.
    :param support_threshold: minimum threshold to decide if prune or not the rule.
    :param max_depth: maximum number of antecedents per rule.
    :return: a rulebase object with the rules denoted as good.
    '''
    
    if mnt.save_usage_flag:
            mnt.usage_data[mnt.usage_categories.RuleMining]['mine_rulebase'] += 1
            
    freq_itemsets = rule_search(x, fuzzy_variables, support_threshold, max_depth)
    rule_list = generate_rules_from_itemsets(freq_itemsets, len(fuzzy_variables))

    fuzzy_type = fuzzy_variables[0].fs_type

    if fuzzy_type == fs.FUZZY_SETS.t1:
        rule_base = rl.RuleBaseT1(fuzzy_variables, rule_list)
    elif fuzzy_type == fs.FUZZY_SETS.t2:
        rule_base = rl.RuleBaseT2(fuzzy_variables, rule_list)
    elif fuzzy_type == fs.FUZZY_SETS.gt2:
        rule_base = rl.RuleBaseGT2(fuzzy_variables, rule_list)
    
    return rule_base


def prune_rules_confidence_lift(x: pd.DataFrame, y:np.array, rules: rl.MasterRuleBase, fuzzy_variables: list[fs.fuzzyVariable], confidence_threshold:float=0.5, 
                                lift_threshold:float=1.05):
    '''
    Removes the rules from the rule base that do not meet a minimum value for confidence and lift measures.

    Confidence is the ratio of rules that have a particular antecedent and consequent, and those that only have the antecedent.
    Lift is ratio between confidence and expected confidence, which is the percentage of class samples in the original data.

    :param x: data to mine. samples x features.
    :param y: class vector.
    :param rules: MasterRuleBase object with the rules to prune.
    :param fuzzy_variables: a list of the fuzzy variables per antecedent.
    :param confidence_threshold: minimum confidence required to the rules.
    :param lift_threshold: minimum lift required to the rules.
    '''
    for ix, rule_base in enumerate(rules):
        delete_list = []
        relevant_class = ix
        relevant_class_samples = x.loc[np.equal(y, relevant_class), :]

        for jx, rule in enumerate(rule_base):
            real_nAnts = sum([ant != -1 for ant in rule])
            global_membership_array = np.zeros((x.shape[0], real_nAnts))
            class_samples_membership_array = np.zeros((relevant_class_samples.shape[0], real_nAnts))
            ant_counter = 0
            for zx, antecedent in enumerate(rule):
                if antecedent != -1:
                    global_membership_array[:, ant_counter] = fuzzy_variables[zx](x[fuzzy_variables[zx].name])[antecedent]
                    class_samples_membership_array[:, ant_counter] = fuzzy_variables[zx](relevant_class_samples[fuzzy_variables[zx].name])[antecedent]
                    ant_counter += 1

            # Compute rule confidence
            global_support = np.mean(np.min(global_membership_array, axis=1), axis=0)
            class_support = np.mean(np.min(class_samples_membership_array, axis=1), axis=0)
            if fuzzy_variables[0].fuzzy_type == fs.FUZZY_SETS.t2 or fuzzy_variables[0].fuzzy_type == fs.FUZZY_SETS.gt2:
                    global_support = np.mean(global_support, axis=1)
                    class_support = np.mean(class_support, axis=1)

            rule_confidence = class_support / global_support
            rule_lift = rule_confidence / np.mean(np.equal(relevant_class, y))

            if rule_confidence < confidence_threshold or rule_lift < lift_threshold:
                delete_list.append(jx)

        rule_base.remove_rules(delete_list)


def simple_mine_rulebase(x: pd.DataFrame, fuzzy_type:fs.FUZZY_SETS=fs.FUZZY_SETS.t1, support_threshold:float=0.05, max_depth:int=3) -> rl.RuleBase:
    '''
    Search the data for associations that are frequent. Computes the fuzzy variables using a 3 label partition (low, medium, high).

    :param x: the data to mine. Dims: samples x features.
    :param fuzzy_type: fuzzy type to use.
    :param support_threshold: minimum threshold to decide if prune or not the rule.
    :param max_depth: maximum number of antecedents per rule.
    :return: a rulebase object with the rules denoted as good.
    '''
    
    precomputed_partitions = utils.construct_partitions(x, fuzzy_type)
    return mine_rulebase_support(x, precomputed_partitions, support_threshold, max_depth)


def multiclass_mine_rulebase(x: pd.DataFrame, y: np.array, fuzzy_variables:list[fs.fuzzyVariable], support_threshold:float=0.05, max_depth:int=3,
                             confidence_threshold:float=0.05, lift_threshold:float=1.05) -> rl.MasterRuleBase:
    '''
    Search the data for associations that are frequent and have good confidence/lift values given a list of fuzzy variables for each antecedent. Computes a different ruleBase for each 
    class and then uses them to form a MasterRuleBase.

    :param x: the data to mine. Dims: samples x features.
    :param fuzzy_variables: list of the fuzzy variables for each of the input variables.
    :param support_threshold: minimum threshold to decide if prune or not the rule.
    :param max_depth: maximum number of antecedents per rule.
    :param confidence_threshold: minimum confidence value.
    :param lift_threshold: 
    :return: a rulebase object with the rules denoted as good.
    '''
    unique_classes = np.unique(y)
    rulebases = []
    for yclass in unique_classes:
        selected_samples = np.equal(yclass, y) 
        selected_x = x.loc[selected_samples, :]

        rulebase = mine_rulebase_support(selected_x, fuzzy_variables, support_threshold, max_depth)
        rulebases.append(rulebase)

    master_rulebase = rl.MasterRuleBase(rulebases, list(map(str, unique_classes)))
    prune_rules_confidence_lift(x, y, master_rulebase, fuzzy_variables, confidence_threshold, lift_threshold)
    return master_rulebase


def simple_multiclass_mine_rulebase(x: pd.DataFrame, y: np.array, fuzzy_type:fs.FUZZY_SETS, support_threshold:float=0.05, max_depth:int=3,
                                    confidence_threshold:float=0.5, lift_threshold:float=1.1) -> rl.MasterRuleBase:
    '''
    Search the data for associations that are frequent and have good confidence/lift values given a list of fuzzy variables for each antecedent. 
    Computes a different ruleBase for each class and then uses them to form a MasterRuleBase.

    Computes the fuzzy variables using a 3 label partition (low, medium, high).

    :param x: the data to mine. Dims: samples x features.
    :param fuzzy_type: fuzzy type to use.
    :param support_threshold: minimum threshold to decide if prune or not the rule.
    :param max_depth: maximum number of antecedents per rule.
    :return: a rulebase object with the rules denoted as good.
    '''
    precomputed_partitions = utils.construct_partitions(x, fuzzy_type)
    return multiclass_mine_rulebase(x, y, precomputed_partitions, support_threshold, max_depth, 
                                    confidence_threshold=confidence_threshold, lift_threshold=lift_threshold)





