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

n_gen = 20
n_pop = 50
    
nRules = 15
nAnts = 2
tolerance = 0.1

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
class_names = iris.target_names
y_class_names = [class_names[i] for i in y]
fz_type_studied = fs.FUZZY_SETS.t1
vl = 3
# Compute the fuzzy partitions using n linguistic variables
precomputed_partitions_vl = utils.construct_partitions(X, fz_type_studied, n_partitions=vl)

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y_class_names, test_size=0.33, random_state=0)
pts = pattern_stability.pattern_stabilizer(X_train, y_train, nRules=nRules, nAnts=nAnts, fuzzy_type=fz_type_studied, tolerance=tolerance, class_names=None, n_linguistic_variables=vl, verbose=True, linguistic_variables=precomputed_partitions_vl, runner=runner)
pts.stability_report(20)
pts.pie_chart_class(0)
pts.pie_chart_class(1)
pts.pie_chart_class(2)
pts.pie_chart_var(0)
pts.pie_chart_var(1)
pts.pie_chart_var(2)
print('Done!')