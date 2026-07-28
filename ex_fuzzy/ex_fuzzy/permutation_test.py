'''
Module to implement different boostrapping tests to evaluate the performance of the fuzzy rules and fuzzy rule based classifiers.

@authors:
    - Javier Fumanal Idocin
    
@references:
    - Permutation Tests for studying Classifier Performance, Ojala, J., Garriga, G. (2010) JMLR 
'''

import numpy as np
import pandas as pd

from sklearn.base import ClassifierMixin

try:
    from . import rules
except:
    import rules

class classifierWrapper(ClassifierMixin):

    def __init__(self, classifier: rules.MasterRuleBase, nrule:int):
        '''
        Creates a classifier wrapper for the given rule in a rule-based classifier.

        :param classifier: rule-based classifier to evaluate.
        :param nrule: rule to evaluate.
        '''
        self.classifier = classifier
        self.nrule = nrule
    

    def predict(self, X: np.array):
        '''
        Predicts the labels of the given data using the rule-based classifier.

        :param X: data to evaluate.
        :return: predicted labels.
        '''
        winning_rule_preds, _winning_association_degrees = self.classifier._winning_rules(X)

        return winning_rule_preds == self.nrule


def _aux_ova_rule_classifier(classifier: rules.MasterRuleBase, nrule:int, labels: np.array):
    '''
    Formats the data to the OVA format and creates a classifier wrapper for the given rule in a rule-based classifier.

    :param classifier: rule-based classifier to evaluate.
    :param nrule: rule to evaluate.
    :param labels: labels to evaluate.
    :return: an object of the sklearn classifier class.
    '''
    ruleOVA_classifier = classifierWrapper(classifier, nrule)
    rules_consequents = classifier.get_consequents()

    new_labels = labels == rules_consequents[nrule]

    return ruleOVA_classifier, new_labels
    

def permutation_p_compute(error_rates: np.array, og_error_rate: float):
    """
    Function to compute the p-value of a permutation test.

    :param true_values: list of true values of the test.
    :param error_rates: list of error rates of the test.
    :param og_error_rate: original error rate of the test.
    :return: p-value of the permutation test.
    """
    sum_random_victories = error_rates < og_error_rate
    p_value = (np.sum(sum_random_victories)+1) / (len(error_rates)+1)

    return p_value


def estimate_og_error_rate(classifier, X: np.array, labels: np.array, r:int=10):
    '''
    Function to estimate the original error rate of a classifier.
    
    :param classifier: classifier to evaluate.
    :param X: data to evaluate.
    :param labels: labels to evaluate.
    :return: original error rate of the classifier.
    '''
    og_error_rates = []
    for _ in range(r): 
        og_error_rate0 = np.mean(classifier.predict(X) != labels)
        og_error_rates.append(og_error_rate0)

    og_error_rate = np.mean(og_error_rates)

    return og_error_rate


def permutation_labels_test(classifier, X, labels, k=100, r=10):
    '''
    Function to perform a permutation test to evaluate the performance of a classifier.
    
    :param classifier: classifier to evaluate.
    :param X: data to evaluate.
    :param labels: labels to evaluate.
    :param k: number of permutations to perform.
    :param r: number of repetitions to estimate the original error rate.
    :return: p-value of the permutation test.
    '''
    error_rates = []
    og_error_rate = estimate_og_error_rate(classifier, X, labels, r=r)
    og_predictions = classifier.predict(X)

    for i in range(k):
        permuted_labels = np.random.permutation(labels)
        permuted_error_rate = np.mean(permuted_labels != og_predictions)
        error_rates.append(permuted_error_rate)

    
    p_value = permutation_p_compute(np.array(error_rates), og_error_rate)

    return p_value


def permute_columns_class_test(classifier, X:np.array, labels:np.array, k:int=100, r:int=10):
    '''
    Function to perform a permutation test to evaluate the performance of a classifier.
    
    :param classifier: classifier to evaluate.
    :param X: data to evaluate.
    :param labels: labels to evaluate.
    :param k: number of permutations to perform.
    :param r: number of repetitions to estimate the original error rate.
    :return: p-value of the permutation test.
    '''
    error_rates = []
    og_error_rate = estimate_og_error_rate(classifier, X, labels, r=r)
    classes = np.unique(labels)

    for i in range(k):
        permuted_X = X.copy()
        for clas in classes:
            premuted_x_clas = permuted_X[labels == clas]
            premuted_x_clas = np.random.permutation(premuted_x_clas)
            permuted_X[labels == clas] = premuted_x_clas
            
        permuted_error_rate = np.mean(classifier.predict(permuted_X) != labels)
        error_rates.append(permuted_error_rate)

    p_value = permutation_p_compute(np.array(error_rates), og_error_rate)

    return p_value


def rulewise_label_permutation_test(classifier: rules.MasterRuleBase, X: np.array, labels: np.array, k:int=100, r:int=10):
    '''
    Function to perform a permutation test to evaluate the performance of a classifier.
    
    :param classifier: classifier to evaluate.
    :param X: data to evaluate.
    :param labels: labels to evaluate.
    :param k: number of permutations to perform.
    :param r: number of repetitions to estimate the original error rate.
    :return: p-value of the permutation test.
    '''
    for ix_rule, rule in enumerate(classifier.get_rules()):
        error_rates = []

        ruleOVA_classifier, new_labels = _aux_ova_rule_classifier(classifier, ix_rule, labels)
        og_error_rate = estimate_og_error_rate(ruleOVA_classifier, X, new_labels, r=r)
        og_predictions = ruleOVA_classifier.predict(X)

        for i in range(k):
            permuted_labels = np.random.permutation(new_labels)
            permuted_error_rate = np.mean(permuted_labels != og_predictions)
            error_rates.append(permuted_error_rate)

    
        p_value = permutation_p_compute(np.array(error_rates), og_error_rate)
        rule.p_value_class_structure = p_value


def rulewise_column_permutation_test(classifier: rules.MasterRuleBase, X: np.array, labels: np.array, k:int=100, r:int=10):
    '''
    Function to perform a permutation test to evaluate the performance of a classifier.
    
    :param classifier: classifier to evaluate.
    :param X: data to evaluate.
    :param labels: labels to evaluate.
    :param k: number of permutations to perform.
    :param r: number of repetitions to estimate the original error rate.
    :return: p-value of the permutation test.
    '''
    for ix_rule, rule in enumerate(classifier.get_rules()):
        error_rates = []

        ruleOVA_classifier, new_labels = _aux_ova_rule_classifier(classifier, ix_rule, labels)
        og_error_rate = estimate_og_error_rate(ruleOVA_classifier, X, new_labels, r=r)
        classes = np.unique(labels)

        for i in range(k):
            permuted_X = X.copy()
            clas = classifier.get_consequents()[ix_rule]
            premuted_x_clas = permuted_X[new_labels == clas]
            premuted_x_clas = np.random.permutation(premuted_x_clas)
            permuted_X[new_labels == clas] = premuted_x_clas
                
            permuted_error_rate = np.mean(ruleOVA_classifier.predict(permuted_X) != new_labels)
            error_rates.append(permuted_error_rate)

        p_value = permutation_p_compute(np.array(error_rates), og_error_rate)
        rule.p_value_feature_coalitions = p_value

    