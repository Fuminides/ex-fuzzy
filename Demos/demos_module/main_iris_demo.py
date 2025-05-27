"""
Created on Thu Jan  7 09:35:55 2021
All rights reserved

@author: Javier Fumanal Idocin - University of Essex

This is a the source file that contains a demo for an Iris classification example using a FRBC where we will also show different possible ways to print the rules, precompute the partitions, etc.

"""


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import sys


# In case you run this without installing the package, you need to add the path to the package
# This is for launching from root folder path
sys.path.append('./ex_fuzzy/')
sys.path.append('./ex_fuzzy/ex_fuzzy/')
# This is for launching from Demos folder
sys.path.append('../ex_fuzzy/')
sys.path.append('../ex_fuzzy/ex_fuzzy/')


import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.evolutionary_fit as GA
import ex_fuzzy.utils as  utils
import ex_fuzzy.eval_tools as eval_tools
import ex_fuzzy.pattern_stability as pattern_stability

runner = 1 # 1: single thread, 2+: corresponding multi-thread

# GA parameters
n_gen = 30
n_pop = 30

# FRBC parameters
nRules = 15 # Number of maximum rules
nAnts = 3 # Number of maximum antecedents per rule
vl = 3 # Number of linguistic variables
tolerance = 0.001 # Minimum dominance score to accept a rule
fz_type_studied = fs.FUZZY_SETS.t1 # Fuzzy set type


iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
class_names = iris.target_names

# Compute the fuzzy partitions using n linguistic variables
precomputed_partitions_vl = utils.construct_partitions(X, fz_type_studied, n_partitions=vl, shape='triangular')
valid = utils.validate_partitions(X, precomputed_partitions_vl, verbose=True)
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# We create a FRBC with the precomputed partitions or None (will optimize them as well in that case) and the specified fuzzy set type
fl_classifier = GA.BaseFuzzyRulesClassifier(nRules=nRules, linguistic_variables=precomputed_partitions_vl, nAnts=nAnts, class_names=class_names, n_linguistic_variables=vl, fuzzy_type=fz_type_studied, verbose=True, tolerance=tolerance, runner=runner, allow_unknown=True, ds_mode=0)

# fl_classifier.customized_loss(utils.mcc_loss) Use this to change the loss function, but be sure to look at the API first
fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=n_pop, checkpoints=0, random_state=0, p_value_compute=False)

# We evaluate the fuzzy model, this will print the rules, the accuracy, the Matthew's correlation coefficient, etc.
fuzzy_evaluator = eval_tools.FuzzyEvaluator(fl_classifier)
str_rules = fuzzy_evaluator.eval_fuzzy_model(X_train, y_train, X_test, y_test, 
                        plot_rules=False, print_rules=True, plot_partitions=True, return_rules=True, bootstrap_results_print=False)

# print(vis_rules.rules_to_latex(fl_classifier.rule_base)) # Do this to print the rules in latex format

# Save the rules as a plain text file
#with open('rules_iris_t2.txt', 'w') as f:
#    f.write(str_rules)

# Compute the explainable predictions: this will return the predictions, the winning rules, the winning association degrees, and the confidence interval of the certainty of the predictions
y_pred, winning_rules, winning_association_degrees, conf_interval_certainty = fl_classifier.explainable_predict(X_test, out_class_names=True)

print('Done!')