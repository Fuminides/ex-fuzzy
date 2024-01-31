"""
Created on Thu Jan  7 09:35:55 2021
All rights reserved

@author: Javier Fumanal Idocin - University of Essex
@author: Javier Andreu-Perez - University of Essex


This is a the source file that contains a demo for a tip computation example, where a diferent set of T1-FS are used to compute
a t1 reasoning approach.

We also show the GA to optimize the rules obtained in classification.

"""
import pandas as pd
import numpy as np

import sys

import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.evolutionary_fit as GA
import ex_fuzzy.utils as utils
import ex_fuzzy.eval_tools as eval_tools

# In case you run this without installing the package, you need to add the path to the package

# This is for launching from root folder path
sys.path.append('./ex_fuzzy/')
sys.path.append('./ex_fuzzy/ex_fuzzy/')

# This is for launching from Demos folder
sys.path.append('../ex_fuzzy/')
sys.path.append('../ex_fuzzy/ex_fuzzy/')


def load_occupancy(path='./Demos/occupancy_data/'):
    train_data = pd.read_csv(path + 'datatraining.txt', index_col=0)
    X_train = train_data[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
    y_train = np.squeeze(train_data[['Occupancy']].values)

    test_data = pd.read_csv(path + 'datatest2.txt', index_col=0)
    X_test = test_data[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
    y_test = np.squeeze(test_data[['Occupancy']].values)

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_occupancy()

try:
    n_gen = int(sys.argv[1])
    pop_size = int(sys.argv[2])
    nRules = int(sys.argv[2])
    nAnts = int(sys.argv[3])
except:
    n_gen = 50
    pop_size = 30
    nRules = 10
    nAnts = 3

fz_type_studied = fs.FUZZY_SETS.t1
X = pd.concat([X_train, X_test])
precomputed_partitions = utils.construct_partitions(pd.concat([X_train, X_test]), fz_type_studied)
min_bounds = np.min(X, axis=0).values
max_bounds = np.max(X, axis=0).values
domain = [min_bounds, max_bounds]

fl_classifier = GA.BaseFuzzyRulesClassifier(nRules=nRules, nAnts=nAnts, 
    linguistic_variables=precomputed_partitions, n_linguist_variables=3, 
    fuzzy_type=fz_type_studied, verbose=True, tolerance=0.01, domain=domain)

fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=pop_size)

eval_tools.eval_fuzzy_model(fl_classifier, X_train, y_train, X_test, y_test, 
                        plot_rules=True, print_rules=True, plot_partitions=True)

print('Done')