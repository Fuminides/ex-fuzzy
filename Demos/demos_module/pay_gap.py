# %%
import pandas as pd
import random
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

from sklearn import datasets
from sklearn.model_selection import train_test_split
# %%
# Data taken from: https://www.kaggle.com/datasets/nilimajauhari/glassdoor-analyze-gender-pay-gap
df = pd.read_csv('./Demos/paygap data/pay_gap.csv')
df.head()

# %%
X = df.drop(columns=['Gender'])
y = df['Gender']
# %%

# Factorize all object-type variables
import numpy as np

categorical_mask = np.zeros(X.shape[1], dtype=int)
for i, column in enumerate(X.columns):
    if X[column].dtype == 'object':
        _, unique_classes = pd.factorize(X[column])
        categorical_mask[i] = len(unique_classes)
        print(f"Column '{column}' unique classes: {unique_classes.tolist()}")


random.seed(2024)

# %%
fz_type_studied = fs.FUZZY_SETS.t1  # T1 fuzzy sets
n_linguistic_variables = 3  # Define the number of linguistic variables
precomputed_partitions = utils.construct_partitions(X, fz_type_studied, n_partitions=n_linguistic_variables, categorical_mask=categorical_mask)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# %%
n_gen = 50
n_pop = 30
n_rules = 20

# Train the fuzzy rules classifier
fl_classifier = GA.BaseFuzzyRulesClassifier(nRules=n_rules, 
                                           linguistic_variables=precomputed_partitions,
                                            #linguistic_variables = None,
                                           nAnts=3, 
                                           n_linguistic_variables=n_linguistic_variables, 
                                           fuzzy_type=fz_type_studied, 
                                           verbose=True, 
                                           tolerance=0.01, 
                                           runner=1, 
                                           ds_mode=1,
                                            #allow_unknown=True,
                                           fuzzy_modifiers=False)

fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=n_pop)

rule_base = fl_classifier.get_rulebase()
fl_evaluator = eval_tools.FuzzyEvaluator(fl_classifier)
str_rules = fl_evaluator.eval_fuzzy_model(X_train, y_train, X_test, y_test, 
                        plot_rules=False, print_rules=True, plot_partitions=False, return_rules=False)
