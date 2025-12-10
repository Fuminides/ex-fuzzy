"""
Comprehensive demo: Train a fuzzy regressor and view rules with print(regressor)
"""
import sys
sys.path.insert(0, './ex_fuzzy/ex_fuzzy/')

import numpy as np
from evolutionary_fit_regression import BaseFuzzyRulesRegressor
import fuzzy_sets as fs

print("=" * 80)
print("DEMO: Complete Workflow with Simple Rule Printing")
print("=" * 80)
print()

# 1. Generate data
print("1. Generating synthetic data...")
np.random.seed(123)
X = np.random.rand(100, 3) * 20  # 3 features, range 0-20
y = 5*X[:, 0] - 2*X[:, 1] + 3*X[:, 2] + np.random.randn(100) * 5
print(f"   Data: y ≈ 5*x1 - 2*x2 + 3*x3 + noise")
print(f"   Samples: {len(X)}, Features: {X.shape[1]}")
print()

# 2. Create linguistic variables
print("2. Creating linguistic variables (LOW, MEDIUM, HIGH)...")
linguistic_vars = []
for i in range(3):
    low = fs.FS("LOW", [0, 0, 6, 8])
    med = fs.FS("MEDIUM", [6, 9, 11, 14])
    high = fs.FS("HIGH", [12, 14, 20, 20])
    fv = fs.fuzzyVariable(f"Feature_{i+1}", [low, med, high])
    linguistic_vars.append(fv)
print(f"   Created {len(linguistic_vars)} fuzzy variables")
print()

# 3. Create regressor (before fitting)
print("3. Creating regressor...")
regressor = BaseFuzzyRulesRegressor(
    nRules=10,
    nAnts=2,
    linguistic_variables=linguistic_vars,
    fuzzy_type=fs.FUZZY_SETS.t1,
    rule_mode='additive',
    verbose=False
)
print("   Before fitting:")
print(f"   {regressor}")
print()

# 4. Train
print("4. Training regressor...")
regressor.fit(X, y, n_gen=25, pop_size=40, random_state=123)
print("   Training complete!")
print()

# 5. View rules - THE SIMPLE WAY!
print("=" * 80)
print("5. VIEWING THE LEARNED RULES - Just use print(regressor)!")
print("=" * 80)
print()
print(regressor)

# 6. Make predictions
print()
print("=" * 80)
print("6. Making predictions...")
print("=" * 80)
print()
test_points = np.array([
    [5, 10, 15],   # x1=LOW, x2=MEDIUM, x3=HIGH
    [15, 5, 10],   # x1=HIGH, x2=LOW, x3=MEDIUM
    [10, 10, 10],  # x1=MEDIUM, x2=MEDIUM, x3=MEDIUM
])

predictions = regressor.predict(test_points)

for i, (point, pred) in enumerate(zip(test_points, predictions), 1):
    true_val = 5*point[0] - 2*point[1] + 3*point[2]
    print(f"Test point {i}: x1={point[0]:.1f}, x2={point[1]:.1f}, x3={point[2]:.1f}")
    print(f"  True value:      {true_val:.2f}")
    print(f"  Predicted value: {pred:.2f}")
    print(f"  Error:           {abs(true_val - pred):.2f}")
    print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("✓ To view rules, just use: print(regressor)")
print("✓ Rules are automatically formatted in human-readable form")
print("✓ Shows rule mode (additive/sufficient)")
print("✓ Works even before fitting (shows 'not fitted')")
print("✓ No manual iteration or formatting needed!")
print()
print("For more options, see PRINTING_RULES.md")
print("=" * 80)
