"""
Quick demo showing the simplest way to view regression rules: print(regressor)
"""
import sys
sys.path.insert(0, './ex_fuzzy/ex_fuzzy/')

import numpy as np
from evolutionary_fit_regression import BaseFuzzyRulesRegressor
import fuzzy_sets as fs

# Simple synthetic data
np.random.seed(42)
X = np.random.rand(50, 2) * 10
y = 2*X[:, 0] + 3*X[:, 1] + np.random.randn(50)

# Create linguistic variables
low = fs.FS("LOW", [0, 0, 3, 4])
med = fs.FS("MEDIUM", [3, 4.5, 5.5, 7])
high = fs.FS("HIGH", [6, 7, 10, 10])

linguistic_vars = [
    fs.fuzzyVariable("x1", [low, med, high]),
    fs.fuzzyVariable("x2", [low, med, high])
]

# Train regressor
regressor = BaseFuzzyRulesRegressor(
    nRules=5,
    nAnts=2,
    linguistic_variables=linguistic_vars,
    verbose=False
)

regressor.fit(X, y, n_gen=15, pop_size=20, random_state=42)

print("================================================================================")
print("SIMPLEST WAY TO VIEW RULES")
print("================================================================================")
print()
print("Just use: print(regressor)")
print()
print(regressor)
print()
print("That's it! No manual iteration, no formatting needed.")
print("================================================================================")
