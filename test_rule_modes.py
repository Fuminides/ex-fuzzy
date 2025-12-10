"""
Test and compare 'additive' vs 'sufficient' rule modes
"""
import sys
sys.path.insert(0, './ex_fuzzy/ex_fuzzy/')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from evolutionary_fit_regression import BaseFuzzyRulesRegressor
import fuzzy_sets as fs

print("=" * 70)
print("COMPARING ADDITIVE vs SUFFICIENT RULE MODES")
print("=" * 70)
print()

# Generate simple dataset
np.random.seed(42)
X = np.linspace(-2, 2, 100).reshape(-1, 1)
y = 3 * X.flatten()**2 - 2 * X.flatten() + 1 + np.random.normal(0, 0.5, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Dataset: {len(X)} samples")
print(f"True function: y = 3*x² - 2*x + 1 (with noise)")
print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
print()

# Create linguistic variables
X_min, X_max = X_train.min(), X_train.max()
X_range = X_max - X_min

low = fs.FS("LOW", [X_min, X_min, X_min + 0.3*X_range, X_min + 0.4*X_range])
med = fs.FS("MEDIUM", [X_min + 0.3*X_range, X_min + 0.5*X_range, 
                        X_min + 0.5*X_range, X_min + 0.7*X_range])
high = fs.FS("HIGH", [X_min + 0.6*X_range, X_min + 0.7*X_range, X_max, X_max])
fv = fs.fuzzyVariable("X", [low, med, high])

# Test ADDITIVE mode
print("1. Training with ADDITIVE rules (default)")
print("   - All rules contribute to every prediction")
print("   - Tolerance parameter ignored")
print()

regressor_additive = BaseFuzzyRulesRegressor(
    nRules=5,
    nAnts=1,
    linguistic_variables=[fv],
    fuzzy_type=fs.FUZZY_SETS.t1,
    tolerance=0.01,  # This will be ignored
    rule_mode='additive',  # All rules contribute
    verbose=False,
    backend='pymoo'
)

regressor_additive.fit(X_train, y_train, n_gen=30, pop_size=20)

train_score_add = regressor_additive.score(X_train, y_train)
test_score_add = regressor_additive.score(X_test, y_test)

print(f"   Train R²: {train_score_add:.4f}")
print(f"   Test R²:  {test_score_add:.4f}")
print(f"   Rules:    {len(regressor_additive.rule_base.get_rules())}")
print()

# Test SUFFICIENT mode
print("2. Training with SUFFICIENT rules")
print("   - Only rules with membership > tolerance fire")
print("   - Some predictions may have no firing rules")
print()

regressor_sufficient = BaseFuzzyRulesRegressor(
    nRules=5,
    nAnts=1,
    linguistic_variables=[fv],
    fuzzy_type=fs.FUZZY_SETS.t1,
    tolerance=0.1,  # Rules must exceed this to fire
    rule_mode='sufficient',  # Only strong rules
    verbose=False,
    backend='pymoo'
)

regressor_sufficient.fit(X_train, y_train, n_gen=30, pop_size=20)

train_score_suff = regressor_sufficient.score(X_train, y_train)
test_score_suff = regressor_sufficient.score(X_test, y_test)

print(f"   Train R²: {train_score_suff:.4f}")
print(f"   Test R²:  {test_score_suff:.4f}")
print(f"   Rules:    {len(regressor_sufficient.rule_base.get_rules())}")
print()

# Compare predictions on specific points
print("3. Comparison on specific test points")
print("-" * 70)

test_points = np.array([[-1.5], [0.0], [1.5]])

for test_x in test_points:
    pred_add = regressor_additive.predict(test_x.reshape(1, -1))[0]
    pred_suff = regressor_sufficient.predict(test_x.reshape(1, -1))[0]
    true_val = 3 * test_x[0]**2 - 2 * test_x[0] + 1
    
    print(f"\nX = {test_x[0]:5.2f}")
    print(f"  True value:     {true_val:7.2f}")
    print(f"  Additive pred:  {pred_add:7.2f}  (error: {abs(pred_add - true_val):5.2f})")
    print(f"  Sufficient pred:{pred_suff:7.2f}  (error: {abs(pred_suff - true_val):5.2f})")
    
    # Check which rules fire for sufficient mode
    precomputed = [fv.compute_memberships(test_x)]
    firing_count = 0
    for rule in regressor_sufficient.rule_base.get_rules():
        ant_val = int(rule.antecedents[0])
        if ant_val >= 0:
            membership = precomputed[0][ant_val][0]
            if membership > regressor_sufficient.tolerance:
                firing_count += 1
    
    print(f"  Sufficient mode: {firing_count}/{len(regressor_sufficient.rule_base.get_rules())} rules fired")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("ADDITIVE mode:")
print("  ✓ All rules always contribute")
print("  ✓ Smoother predictions")
print("  ✓ No dead zones")
print("  ✓ Better for continuous functions")
print(f"  → R² score: {test_score_add:.4f}")
print()
print("SUFFICIENT mode:")
print("  ✓ Only strong rules contribute")
print("  ✓ Can ignore weak/irrelevant rules")
print("  ✓ May have dead zones (fallback to default)")
print("  ✓ More similar to classification behavior")
print(f"  → R² score: {test_score_suff:.4f}")
print()
print("Recommendation: Use 'additive' (default) for most regression tasks")
print("=" * 70)
