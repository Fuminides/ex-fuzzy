"""
Created on Thu Jan  7 09:35:55 2021
All rights reserved

@author: Javier Fumanal Idocin - University of Essex
@author: Javier Andreu-Perez - University of Essex


This is a the source file that contains a demo for a tip computation example, where a diferent set of T1-FS are used to compute
a t1 reasoning approach.

We also show the GA to optimize the rules obtained in classification.

"""
import sys

# In case you run this without installing the package, you need to add the path to the package

# This is for launching from root folder path
sys.path.append('./ex_fuzzy/')
sys.path.append('./ex_fuzzy/ex_fuzzy/')

# This is for launching from Demos folder
sys.path.append('../ex_fuzzy/')
sys.path.append('../ex_fuzzy/ex_fuzzy/')

import numpy as np
import ex_fuzzy.rules as rules
import ex_fuzzy.eval_rules as evr
import matplotlib.pyplot as plt


epsilon =  [0, 10E-3, 50E-3, 10E-2, 50E-2]


def new_loss(ruleBase, X, y, tolerance, alpha, beta, precomputed_truth) -> float:

        '''
        Fitness function for the optimization problem.
        :param ruleBase: RuleBase object
        :param X: array of train samples. X shape = (n_samples, n_features)
        :param y: array of train labels. y shape = (n_samples,)
        :param tolerance: float. Tolerance for the size evaluation.
        :return: float. Fitness value.
        '''
        # FOR JESS: explanation of what are we doing here.
        relevant_pattern = [-1, -1, 1, -1] # This is the pattern we want to check. In this case, we want to check if sepal width low is taken into account for class 1 (Does not make particular sense, but is just an example)
        # The structure of the pattern is as follows:
        # [sepal length, sepal width, petal length, petal width] 
        # The values are the indexes of the fuzzy sets we want to check. -1 means that we dont care about that feature.
        # For example, if we want to check if the rule is of the form:
        # IF sepal length is low AND sepal width is medium AND petal length is high AND petal width is low THEN class 1
        # We would set the pattern to [0, 1, 2, 0] (assuming that the fuzzy sets are ordered as low, medium, high)

        relevant_class = 1 # This is the class we want to check if the rule exists

        # We want to check if the desired rule is part of the rule base
        # Lets say we want to see if sepal width is contained in class 1
        # We will get the 
        def check_rule(masterRuleBase, target_class, pattern):
            # Check if the rule base contains the desired rule
            relevant_ruleBase = masterRuleBase[target_class]

            for rule in relevant_ruleBase:
                if rule.antecedents == pattern:
                    return True
                
            return False
        
        from ex_fuzzy.utils import mcc_loss
        # Compute the accuracy of the model
        std_mcc = mcc_loss(ruleBase, X, y, tolerance, alpha, beta, 0.0, precomputed_truth)
        relevant_pattern_found = check_rule(ruleBase, relevant_class, relevant_pattern)

        return std_mcc + 0.1 * relevant_pattern_found # We add a small penalty if the rule is not found



import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split

import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.evolutionary_fit as GA
import ex_fuzzy.utils as  utils
import ex_fuzzy.eval_tools as eval_tools

try:
    n_gen = int(sys.argv[1])
    n_pop = int(sys.argv[2])
except:
    n_gen = 50
    n_pop = 30
    nRules = 4
    nAnts = 4
    vl = 3
    tolerance = 0.0001
    fz_type_studied = fs.FUZZY_SETS.t2


# import some data to play with
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X = np.concatenate((X_train, X_test), axis=0)
precomputed_partitions = utils.construct_partitions(X, fz_type_studied)

fl_classifier = GA.BaseFuzzyRulesClassifier(nRules=nRules, linguistic_variables=precomputed_partitions, nAnts=nAnts,
                                            n_linguistic_variables=vl, fuzzy_type=fz_type_studied, verbose=False, tolerance=tolerance)
fl_classifier.customized_loss(new_loss)
fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=n_pop)

eval_tools.eval_fuzzy_model(fl_classifier, X_train, y_train, X_test, y_test, 
                        plot_rules=False, print_rules=True, plot_partitions=False)

print('Done!')
