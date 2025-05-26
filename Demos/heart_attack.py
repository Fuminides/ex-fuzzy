# %%
import pandas as pd
import numpy as np
import random
import sys

sys.path.append('./ex_fuzzy/')
sys.path.append('../ex_fuzzy/')

from sklearn import datasets
from sklearn.model_selection import train_test_split

import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.evolutionary_fit as GA
import ex_fuzzy.utils as utils
import ex_fuzzy.eval_tools as eval_tools

df = pd.read_csv('./Demos/paygap data/heart_attack.csv')
df

# %%
X = df.drop(columns=['heart_attack'])
y = df['heart_attack']

missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
df['alcohol_consumption'] = df['alcohol_consumption'].fillna('Unknown')

# %%
fz_type_studied = fs.FUZZY_SETS.t1  # T1 fuzzy sets
n_linguistic_variables = 3  # Define the number of linguistic variables
n_gen = 30
n_pop = 30
n_rules = 20
categorical_mask = np.zeros(X.shape[1], dtype=int)
for i, column in enumerate(X.columns):
    if X[column].dtype == 'object':
        _, unique_classes = pd.factorize(X[column])
        categorical_mask[i] = len(unique_classes)
        print(f"Column '{i, column}' unique classes: {unique_classes.tolist()}")


precomputed_partitions = utils.construct_partitions(X, fz_type_studied, n_partitions=n_linguistic_variables, categorical_mask=categorical_mask)


