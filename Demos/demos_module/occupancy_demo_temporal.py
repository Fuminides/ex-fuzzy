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

import utils
import fuzzy_sets as t2
import evolutionary_fit as GA
import pandas as pd
import eval_tools
import temporal
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split


def load_occupancy(path='./Demos/occupancy_data/', random_mixing=True):
    train_data = pd.read_csv(path + 'datatraining.txt', index_col=0)
    X_train = train_data[['date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
    y_train = np.squeeze(train_data[['Occupancy']].values)

    test_data = pd.read_csv(path + 'datatest.txt', index_col=0)
    X_test = test_data[['date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
    y_test = np.squeeze(test_data[['Occupancy']].values)

    if random_mixing:

        X_total = pd.concat([X_train, X_test])
        y_total = np.concatenate([y_train, y_test])

        X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.33, random_state=0)

    return X_train, y_train, X_test, y_test, X_total, y_total


_, _, _, _, X_total, y_total = load_occupancy()
X_total_array = np.array(X_total.drop(columns=['date']))


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

fz_type_studied = t2.FUZZY_SETS.temporal
fz_type_inside = t2.FUZZY_SETS.t1
precomputed_partitions = utils.construct_partitions(X_total.drop(columns=['date']), fz_type_inside)

cut_point_morning0 = '00:00:00'
cut_point_morning1 = '10:00:00'
cut_points_morning = [cut_point_morning0, cut_point_morning1]
cut_point_daytime0 = '11:00:00'
cut_point_daytime1 = '19:00:00'
cut_points_daytime = [cut_point_daytime0, cut_point_daytime1]
cut_point_evening0 = '20:00:00'
cut_point_evening1 = '23:00:00'
cutpoints_evening = [cut_point_evening0, cut_point_evening1]


temporal_boolean_markers = utils.temporal_cuts(X_total, cutpoints=[cut_points_morning, cut_points_daytime, cutpoints_evening], time_resolution='hour')
time_moments = np.array([utils.assign_time(a, temporal_boolean_markers) for a in range(X_total.shape[0])])
partitions, partition_markers = utils.temporal_assemble(X_total, y_total, temporal_moments=temporal_boolean_markers)
X_train, X_test, y_train, y_test = partitions
train_markers, test_markers = partition_markers

train_time_moments = np.array([utils.assign_time(a, train_markers) for a in range(X_train.shape[0])])
test_time_moments = np.array([utils.assign_time(a, test_markers) for a in range(X_test.shape[0])])

temp_partitions = utils.create_tempVariables(X_total_array, time_moments, precomputed_partitions) 

X_train = X_train.drop(columns=['date'])
X_test = X_test.drop(columns=['date'])
fl_classifier = temporal.TemporalFuzzyRulesClassifier(nRules=nRules, nAnts=nAnts, 
    linguistic_variables=temp_partitions, n_linguist_variables=3,
    fuzzy_type=fz_type_studied, verbose=True, tolerance=0.001, n_class=2)
fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=pop_size, time_moments=train_time_moments, checkpoints=0)

temporal.eval_temporal_fuzzy_model(fl_classifier, X_train, y_train, X_test, y_test, 
                            time_moments=train_time_moments, test_time_moments=test_time_moments,
                            plot_rules=False, print_rules=True, plot_partitions=False)


print('Done')