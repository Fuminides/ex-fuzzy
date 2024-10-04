'''
Module that contains the bootraping tests to evaluate the performance of the fuzzy rules and fuzzy rule based classifiers.


'''

import numpy as np
import pandas as pd

from sklearn.base import ClassifierMixin

try:
    from . import rules
    from . import fuzzy_sets as fs
except:
    import rules
    import fuzzy_sets as fs


def generate_bootstrap_samples(X: np.array, y: np.array, n_samples:int):
    '''
    Generates bootstrap samples from the given data.

    :param X: data to sample.
    :param y: labels to sample.
    :param n_samples: number of samples to generate.
    :return: a list of tuples with the sampled data and labels.
    '''
    samples = []
    for i in range(n_samples):
        idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
        samples.append((X[idx], y[idx]))
    
    return samples


def generate_random_rules(nAnts: int, nRules: int, n_linguistic_variables: int, class_names: np.array):
    '''
    Generates random rules for the given parameters.

    :param nAnts: number of ants in the classifier.
    :param nRules: number of rules in the classifier.
    :param n_linguistic_variables: number of linguistic variables in the classifier.
    :param class_names: classes in the classifier.
    :return: a list of tuples with the antecedents and consequents of the rules.
    '''
    rules = []
    for i in range(nRules):
        antecedents = []
        for j in range(nAnts):
            antecedents.append(np.random.randint(-1, n_linguistic_variables, size=1)[0])
        
        consequent = np.random.choice(class_names)
        rules.append((antecedents, consequent))
    
    return rules


def membership_randomrule(rule, X: np.array, fuzzy_variables: list[fs.fuzzyVariable]):
    '''
    Computes the membership of the given data to the rule.

    :param rule: rule to evaluate.
    :param X: data to evaluate.
    :param fuzzy_variables: fuzzy variables of the classifier.
    :return: the membership of the data to the rule.
    '''
    if fuzzy_variables[0].fuzzy_type() == fs.FUZZY_SETS.t1:
        membership = np.zeros(X.shape[0]) + 1
    elif fuzzy_variables[0].fuzzy_type() == fs.FUZZY_SETS.t2:
        membership = np.zeros((X.shape[0], 2)) + 1
    elif fuzzy_variables[0].fuzzy_type() == fs.FUZZY_SETS.gt2:
        membership = np.zeros((X.shape[0], len(fuzzy_variables[0][0].alpha_cuts), 2)) + 1

    else:
        raise ValueError('Fuzzy type not supported')
    
    for i in range(len(rule[0])):
        if rule[0][i] != -1:
            membership = membership * fuzzy_variables[i][rule[0][i]].membership(X[:, i])
    
    return membership


def quality_metric_rule(rule, X: np.array, y: np.array, fuzzy_variables: list[fs.fuzzyVariable]):
    '''
    Computes the quality of the rule as the percentage of samples of the consequent class that have more membership to the rule than to the other classes.

    :param rule: rule to evaluate.
    :param X: data to evaluate.
    :param y: labels to evaluate.
    :param fuzzy_variables: fuzzy variables of the classifier.
    :param class_names: classes in the classifier.
    :return: the cross entropy of the rule.
    '''
    membership = membership_randomrule(rule, X, fuzzy_variables)
    quality_metric = 0
    class_names = np.unique(y)

    class_samples = np.where(y == rule[1])[0]
    if fuzzy_variables[0].fuzzy_type() == fs.FUZZY_SETS.t1:
        membership_class = np.mean(membership[class_samples])
    elif fuzzy_variables[0].fuzzy_type() == fs.FUZZY_SETS.t2:
        membership_class = np.mean(membership[class_samples], axis=0).mean()
    elif fuzzy_variables[0].fuzzy_type() == fs.FUZZY_SETS.gt2:
        membership_class = fuzzy_variables[0][0].alpha_reduction(membership[class_samples])
        membership_class = membership_class.mean(axis=1).mean()

    non_class_samples = np.where(y != rule[1])[0]
    if fuzzy_variables[0].fuzzy_type() == fs.FUZZY_SETS.t1:
        membership_non_class = np.mean(membership[non_class_samples])
    elif fuzzy_variables[0].fuzzy_type() == fs.FUZZY_SETS.t2:
        membership_non_class = np.mean(membership[non_class_samples], axis=0).mean()
    elif fuzzy_variables[0].fuzzy_type() == fs.FUZZY_SETS.gt2:
        membership_non_class = fuzzy_variables[0][0].alpha_reduction(membership[non_class_samples])
        membership_non_class = membership_non_class.mean(axis=1).mean()


    quality_metric = membership_class - membership_non_class

    return quality_metric 


def generate_null_distribution(classifier, X: np.array, y: np.array, n_samples:int, nRules: int):
    '''
    Generates the null distribution of the classifier.

    :param classifier: classifier to evaluate.
    :param X: data to evaluate.
    :param y: labels to evaluate.
    :param n_samples: number of samples to generate.
    :param nRules: number of rules to bootstrap.
    :return: a list with the cross entropy of the rules in the null distribution.
    '''
    null_distribution = []
    nAnts = X.shape[1]
    class_names = np.unique(y)
    lvs = [fv for fv in classifier.lvs] if isinstance(classifier, ClassifierMixin) else [fv for fv in classifier[0].antecedents]
    
    n_linguistic_variables = [len(fv) for fv in lvs]
    
    for i in range(n_samples):
        rules = generate_random_rules(nAnts, nRules, n_linguistic_variables, class_names)
        qualities_metric = []
        for rule in rules:
            qualities_metric.append(quality_metric_rule(rule, X, y, lvs))
        null_distribution.append(qualities_metric)
    
    return np.array(null_distribution)


def _aux_compute_rule_p_value(rule, X: np.array, y: np.array, fuzzy_variables: list[fs.fuzzyVariable], null_distribution: np.array):
    '''
    Computes the p-value of the rule.

    :param rule: rule to evaluate.
    :param X: data to evaluate.
    :param y: labels to evaluate.
    :param fuzzy_variables: fuzzy variables of the classifier.
    :param null_distribution: null distribution of the classifier.
    :return: the p-value of the rule.
    '''
    quality_metric = quality_metric_rule(rule, X, y, fuzzy_variables)
    # Select randomly one column per row
    chosen_columns = np.random.randint(0, null_distribution.shape[1], size=(null_distribution.shape[0], 1))
    chosen_population = null_distribution[np.arange(null_distribution.shape[0]), chosen_columns.flatten()]
    p_value = np.mean(quality_metric < chosen_population)

    return p_value


def compute_rule_p_value(classifier, X: np.array, y: np.array, nSamples: int=100, nRules: int=10):
    '''
    Computes the p-value of the rules in the classifier.

    :param classifier: classifier to evaluate.
    :param X: data to evaluate.
    :param y: labels to evaluate.
    :return: a list with the p-value of the rules.
    '''
    null_distribution = generate_null_distribution(classifier, X, y, nSamples, nRules)
    p_values = []

    if isinstance(classifier, ClassifierMixin):
        crules = classifier.rule_base.get_rules()
        lvs = classifier.lvs
        rconsequents = classifier.rule_base.get_consequents()
    else:
        crules = classifier.get_rules()
        lvs = classifier[0].antecedents
        rconsequents = classifier.get_consequents()
    
    for ix, rule in enumerate(crules):
        p_values.append(_aux_compute_rule_p_value([rule, rconsequents[ix]], X, y, lvs, null_distribution))
    
    return np.array(p_values)
