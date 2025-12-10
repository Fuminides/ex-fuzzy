"""
Demo showing different ways to print regression rules
"""
import sys
sys.path.insert(0, './ex_fuzzy/ex_fuzzy/')

import numpy as np
from evolutionary_fit_regression import BaseFuzzyRulesRegressor
import fuzzy_sets as fs

print("=" * 80)
print("DEMO: Printing Regression Rules")
print("=" * 80)
print()

# Generate simple synthetic data
np.random.seed(42)
X = np.random.rand(100, 3) * 10  # 3 features, values 0-10
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100) * 0.5

# Create linguistic variables
linguistic_vars = []
for i in range(3):
    low = fs.FS(f"LOW", [0, 0, 3, 4])
    med = fs.FS(f"MEDIUM", [3, 4.5, 5.5, 7])
    high = fs.FS(f"HIGH", [6, 7, 10, 10])
    fv = fs.fuzzyVariable(f"x{i+1}", [low, med, high])
    linguistic_vars.append(fv)

# Train regressor
regressor = BaseFuzzyRulesRegressor(
    nRules=8,
    nAnts=2,
    linguistic_variables=linguistic_vars,
    fuzzy_type=fs.FUZZY_SETS.t1,
    rule_mode='additive',
    verbose=False
)

print("Training fuzzy regressor on synthetic data...")
print(f"  Data: y = 2*x1 + 3*x2 - x3 + noise")
print(f"  Samples: {len(X)}")
print(f"  Features: {X.shape[1]}")
print()

regressor.fit(X, y, n_gen=20, pop_size=30, random_state=42)

print()
print("-" * 80)
print("METHOD 1: Simply print(regressor) - Most Convenient!")
print("-" * 80)
print()
print(regressor)

print()
print("-" * 80)
print("METHOD 2: Direct method call (original)")
print("-" * 80)
print()
regressor.rule_base.print_rules_regression()

print()
print("-" * 80)
print("METHOD 3: Custom output name")
print("-" * 80)
print()
regressor.rule_base.print_rules_regression(output_name='predicted_value')

print()
print("-" * 80)
print("METHOD 4: With membership function parameters")
print("-" * 80)
print()
regressor.rule_base.print_rules_regression(
    output_name='y', 
    show_memberships=True
)

print()
print("-" * 80)
print("METHOD 5: Return as string for further processing")
print("-" * 80)
print()
rules_string = regressor.rule_base.print_rules_regression(
    return_rules=True, 
    output_name='target'
)
print("Returned string (first 300 characters):")
print(rules_string[:300] + "...")
print()
print(f"Total rules: {len(rules_string.strip().split(chr(10)))}")

print()
print("=" * 80)
print("USAGE NOTES")
print("=" * 80)
print()
print("SIMPLEST WAY: Just use print(regressor)!")
print("  → Automatically shows all rules in readable format")
print()
print("For more control, use print_rules_regression method:")
print("  • return_rules: Get rules as string instead of printing")
print("  • output_name: Customize the output variable name")
print("  • show_memberships: Show membership function parameters")
print()
print("Example usage:")
print("  # Simple")
print("  print(regressor)")
print()
print("  # Advanced")
print("  regressor.rule_base.print_rules_regression(")
print("      output_name='house_price',")
print("      show_memberships=False")
print("  )")
print()
print("=" * 80)
