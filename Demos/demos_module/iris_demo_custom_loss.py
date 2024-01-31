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


def new_loss(ruleBase: rules.RuleBase, X:np.array, y:np.array, tolerance:float, alpha:float=0.99, beta:float=0.0125, gamma:float=0.0125) -> float:

        '''
        Fitness function for the optimization problem.
        :param ruleBase: RuleBase object
        :param X: array of train samples. X shape = (n_samples, n_features)
        :param y: array of train labels. y shape = (n_samples,)
        :param tolerance: float. Tolerance for the size evaluation.
        :return: float. Fitness value.
        '''
        def subloss(ruleBase1, X1, y1, epsilon_val):

            X1 = X1 + epsilon_val * np.random.uniform(-1, 1, X1.shape)
            ev_object = evr.evalRuleBase(ruleBase1, X1, y1)
            ev_object.add_rule_weights()

            score_acc = ev_object.classification_eval()
            score_size = ev_object.effective_rulesize_eval(tolerance)
            beta = 1 - alpha

            score = score_acc * alpha + score_size * beta
        
            return score
        
        epsilon_list =  [0, 10E-3, 50E-3, 10E-2, 50E-2]
        weights = np.array([1 / len(epsilon_list)] * len(epsilon_list))**2
        weights = weights / np.sum(weights)

        score_pondered = 0
        for epsilon, weight in zip(epsilon_list, weights):
            score = subloss(ruleBase, X, y, epsilon)
            score_pondered += score * weight
        
        return score_pondered



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

def load_occupancy(path='./demos/occupancy_data/'):
    train_data = pd.read_csv(path + 'datatraining.txt', index_col=0)
    X_train = train_data[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
    y_train = np.squeeze(train_data[['Occupancy']].values)

    test_data = pd.read_csv(path + 'datatest2.txt', index_col=0)
    X_test = test_data[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
    y_test = np.squeeze(test_data[['Occupancy']].values)

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_occupancy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X = np.concatenate((X_train, X_test), axis=0)
precomputed_partitions = utils.construct_partitions(X, fz_type_studied)

# Standard loss experiments
fl_classifier = GA.BaseFuzzyRulesClassifier(nRules=nRules, linguistic_variables=precomputed_partitions, nAnts=nAnts,
                                            n_linguist_variables=vl, fuzzy_type=fz_type_studied, verbose=False, tolerance=tolerance)
fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=n_pop)

eval_tools.eval_fuzzy_model(fl_classifier, X_train, y_train, X_test, y_test, 
                        plot_rules=False, print_rules=True, plot_partitions=False)

og_accs = []
for eps in epsilon:
    X1 = X_test + eps * np.random.uniform(-1, 1, X_test.shape)
    og_accs.append(np.mean(np.equal(fl_classifier.predict(X1), y_test)))


fl_classifier = GA.BaseFuzzyRulesClassifier(nRules=nRules, linguistic_variables=precomputed_partitions, nAnts=nAnts,
                                            n_linguist_variables=vl, fuzzy_type=fz_type_studied, verbose=False, tolerance=tolerance)
fl_classifier.customized_loss(new_loss)
fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=n_pop)

eval_tools.eval_fuzzy_model(fl_classifier, X_train, y_train, X_test, y_test, 
                        plot_rules=False, print_rules=True, plot_partitions=False)


accs = []
for eps in epsilon:
    X1 = X_test + eps * np.random.uniform(-1, 1, X_test.shape)
    accs.append(np.mean(np.equal(fl_classifier.predict(X1), y_test)))


plt.figure()
plt.plot(epsilon, og_accs)
plt.plot(epsilon, accs)
plt.ylim(0, 1)
plt.legend(['Original Fitness', 'Epsilon Fitness'])
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epsilon')
plt.savefig('iris_epsilon_t2.pdf')
print('Done')