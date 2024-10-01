'''
Module to implement different boostrapping tests to evaluate the performance of the fuzzy rules and fuzzy rule based classifiers.

@authors:
    - Javier Fumanal Idocin
    
@references:
    - Permutation Tests for studying Classifier Performance, Ojala, J., Garriga, G. (2010) JMLR 
'''

import numpy as np
import pandas as pd

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
