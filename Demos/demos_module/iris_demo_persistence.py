"""
Created on Thu Jan  7 09:35:55 2021
All rights reserved

@author: Javier Fumanal Idocin - University of Essex


This is a the source file that contains a demo for a tip computation example, where a diferent set of T1-FS are used to compute
a t1 reasoning approach.

We also show the GA to optimize the rules obtained in classification.

"""

import pandas as pd

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
import ex_fuzzy.persistence as persistence
import ex_fuzzy.vis_rules as vis_rules

# Import some data to play with
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
n_linguistic_variables = 3

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

fz_type_studied = fs.FUZZY_SETS.t1

# Compute the fuzzy partitions using 3 quartiles
precomputed_partitions = utils.construct_partitions(X, fz_type_studied, n_partitions=n_linguistic_variables)

fl_classifier = GA.BaseFuzzyRulesClassifier(nRules=10, linguistic_variables=precomputed_partitions, nAnts=3, 
                                            n_linguistic_variables=n_linguistic_variables, fuzzy_type=fz_type_studied, 
                                            verbose=True, tolerance=0.01, runner=1, ds_mode=0, fuzzy_modifiers=False)

fl_classifier.fit(X_train, y_train, n_gen=20)

fl_evaluator = eval_tools.FuzzyEvaluator(fl_classifier)
str_rules = fl_evaluator.eval_fuzzy_model(X_train, y_train, X_test, y_test, 
                        plot_rules=False, print_rules=True, plot_partitions=True, return_rules=True)

# Save rules to a plain text file
with open('iris_rules.txt', 'w') as f:
    f.write(str_rules)

# Save the fuzzy partitions to a plain text file
with open('iris_partitions.txt', 'w') as f:
    str_partitions = persistence.save_fuzzy_variables(precomputed_partitions)
    f.write(str_partitions)

# Load rules from a plain text file
with open('iris_rules.txt', 'r') as f:
    str_rules = f.read()
# Load partitions from a plain text file (follows a fomart)
with open('iris_partitions.txt', 'r') as f:
    loaded_partitions = persistence.load_fuzzy_variables(f.read())

# Persistence of the rules example
mrule_base = persistence.load_fuzzy_rules(str_rules, loaded_partitions)

fl_classifier2 = GA.BaseFuzzyRulesClassifier(precomputed_rules=mrule_base, ds_mode=2, allow_unknown=False)
# fl_classifier2.load_master_rule_base(mrule_base) # (Another possibility)

fl_evaluator = eval_tools.FuzzyEvaluator(fl_classifier2)

str_rules = fl_evaluator.eval_fuzzy_model(X_train, y_train, X_test, y_test, 
                        plot_rules=False, print_rules=True, plot_partitions=False, return_rules=True)
                        
print('Done')