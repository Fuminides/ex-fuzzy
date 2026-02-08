"""
Quick test to verify EvoX backend works with the current implementation.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

try:
    from ex_fuzzy.ex_fuzzy import evolutionary_fit as GA_RB
    from ex_fuzzy.ex_fuzzy import evolutionary_backends
except ImportError:
    import sys
    sys.path.insert(0, './ex_fuzzy')
    from ex_fuzzy import evolutionary_fit as GA_RB
    from ex_fuzzy import evolutionary_backends

print("Testing EvoX backend implementation...")
print("=" * 60)

# Check available backends
available = evolutionary_backends.list_available_backends()
print(f"Available backends: {available}")

if 'evox' not in available:
    print("\n⚠️  EvoX backend not available.")
    print("Install with: pip install 'evox[jax]'")
    exit(0)

# Load data
print("\nLoading Iris dataset...")
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Test EvoX backend
print("\n" + "=" * 60)
print("Testing EvoX Backend")
print("=" * 60)

try:
    clf = GA_RB.BaseFuzzyRulesClassifier(
        nRules=10,
        nAnts=3,
        n_linguistic_variables=3,
        verbose=True,
        backend='evox'
    )
    
    print("\n✅ Classifier created successfully")
    print(f"Backend: {clf.backend.name()}")
    
    print("\nTraining...")
    clf.fit(X_train, y_train, n_gen=10, pop_size=20)
    
    print("\n✅ Training completed successfully")
    
    # Test prediction
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Number of rules: {len(clf.rule_base.get_rules())}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
