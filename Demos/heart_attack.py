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
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
df['alcohol_consumption'] = df['alcohol_consumption'].fillna('Unknown')

X = df.drop(columns=['heart_attack'])
y = df['heart_attack']

# %%
fz_type_studied = fs.FUZZY_SETS.t1  # T1 fuzzy sets
n_linguistic_variables = 3  # Define the number of linguistic variables
n_gen = 30
n_pop = 30
n_rules = 20

# The categorical variables are detected from the data: the string ones, and the
# numeric flags that only take a handful of values. construct_partitions does this
# on its own, so pass a mask only when you want to decide it yourself.
categorical_mask = utils.detect_categorical_mask(X)
for i, column in enumerate(X.columns):
    if categorical_mask[i] > 0:
        unique_classes = sorted(X[column].dropna().unique().tolist())
        print(f"Column '{i, column}' unique classes: {unique_classes}")


precomputed_partitions = utils.construct_partitions(X, fz_type_studied, n_partitions=n_linguistic_variables, categorical_mask=categorical_mask)


