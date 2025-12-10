"""
Demonstration of additive fuzzy regression
Shows how all rules contribute to the prediction
"""
import sys
sys.path.insert(0, './ex_fuzzy/ex_fuzzy/')

import numpy as np
from evolutionary_fit_regression import BaseFuzzyRulesRegressor
import fuzzy_sets as fs

print("=" * 70)
print("ADDITIVE FUZZY REGRESSION DEMONSTRATION")
print("=" * 70)
print()

# Simple dataset: y = 2*x
np.random.seed(42)
X = np.linspace(-1, 1, 50).reshape(-1, 1)
y = 2 * X.flatten() + np.random.normal(0, 0.1, 50)

print(f"Dataset: {len(X)} samples")
print(f"True relationship: y = 2*x (with noise)")
print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
print()

# Create simple linguistic variables: LOW, MEDIUM, HIGH
X_min, X_max = X.min(), X.max()
X_range = X_max - X_min

low = fs.FS("LOW", [X_min, X_min, X_min + 0.3*X_range, X_min + 0.4*X_range])
med = fs.FS("MEDIUM", [X_min + 0.3*X_range, X_min + 0.5*X_range, 
                        X_min + 0.5*X_range, X_min + 0.7*X_range])
high = fs.FS("HIGH", [X_min + 0.6*X_range, X_min + 0.7*X_range, X_max, X_max])

fv = fs.fuzzyVariable("X", [low, med, high])

print("Linguistic variables:")
print("  LOW: covers negative values")
print("  MEDIUM: covers near-zero values")
print("  HIGH: covers positive values")
print()

# Train regressor with few rules to see additive effect clearly
print("Training regressor with 3 rules...")
regressor = BaseFuzzyRulesRegressor(
    nRules=3,
    nAnts=1,
    linguistic_variables=[fv],
    fuzzy_type=fs.FUZZY_SETS.t1,
    verbose=False,
    backend='pymoo'
)

regressor.fit(X, y, n_gen=30, pop_size=20)

print(f"Trained successfully with {len(regressor.rule_base.get_rules())} rules")
print()

# Show learned rules
print("Learned Rules:")
print("-" * 70)
for i, rule in enumerate(regressor.rule_base.get_rules(), 1):
    ant_val = int(rule.antecedents[0])
    if ant_val >= 0:
        lv_name = fv.linguistic_variable_names()[ant_val]
        print(f"Rule {i}: IF X is {lv_name:8s} THEN y = {rule.consequent:6.2f}")
    else:
        print(f"Rule {i}: IF X is DON'T CARE THEN y = {rule.consequent:6.2f}")
print()

# Demonstrate additive behavior on specific points
print("Additive Prediction Example:")
print("-" * 70)
test_points = np.array([[-0.8], [0.0], [0.8]])

for test_x in test_points:
    # Compute memberships for this point
    precomputed = [fv.compute_memberships(test_x)]
    
    print(f"\nTest point: X = {test_x[0]:.2f}")
    print(f"  Memberships to linguistic variables:")
    for lv_idx, lv_name in enumerate(fv.linguistic_variable_names()):
        membership = precomputed[0][lv_idx][0]
        print(f"    {lv_name:8s}: {membership:.3f}")
    
    # Show contribution of each rule
    print(f"  Rule contributions:")
    total_weight = 0
    weighted_sum = 0
    
    for rule_idx, rule in enumerate(regressor.rule_base.get_rules(), 1):
        ant_val = int(rule.antecedents[0])
        if ant_val >= 0:
            membership = precomputed[0][ant_val][0]
        else:
            membership = 1.0  # Don't care
        
        contribution = membership * rule.consequent
        total_weight += membership
        weighted_sum += contribution
        
        print(f"    Rule {rule_idx}: membership={membership:.3f} × consequent={rule.consequent:6.2f} = {contribution:7.3f}")
    
    prediction = weighted_sum / total_weight if total_weight > 0 else 0
    true_value = 2 * test_x[0]
    
    print(f"  Prediction: {weighted_sum:.3f} / {total_weight:.3f} = {prediction:.3f}")
    print(f"  True value: {true_value:.3f}")
    print(f"  Error: {abs(prediction - true_value):.3f}")

print()
print("=" * 70)
print("Key observations:")
print("  1. ALL rules contribute to every prediction")
print("  2. Rules with higher membership have more influence")
print("  3. No rules are filtered out by tolerance")
print("  4. The prediction is a smooth weighted average")
print("=" * 70)
