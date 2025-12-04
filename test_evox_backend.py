"""
Example script demonstrating EvoX backend support in ex-fuzzy.

This script shows how to use both pymoo (CPU) and evox (GPU-accelerated) 
backends for evolutionary optimization of fuzzy rule-based systems.

Requirements:
    - Basic: pip install ex-fuzzy
    - For EvoX support: pip install ex-fuzzy[evox]
"""

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Import ex-fuzzy components
try:
    from ex_fuzzy.ex_fuzzy import evolutionary_fit as GA_RB
    from ex_fuzzy.ex_fuzzy import evolutionary_backends
except ImportError:
    import sys
    sys.path.insert(0, './ex_fuzzy')
    from ex_fuzzy import evolutionary_fit as GA_RB
    from ex_fuzzy import evolutionary_backends


def test_backend(backend_name: str, X_train, y_train, X_test, y_test):
    """
    Test a specific backend with a classification problem.
    
    Args:
        backend_name: Name of the backend ('pymoo' or 'evox')
        X_train, y_train: Training data
        X_test, y_test: Test data
    
    Returns:
        dict with timing and performance metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing {backend_name.upper()} backend")
    print(f"{'='*60}")
    
    try:
        # Create classifier with specified backend
        fl_classifier = GA_RB.BaseFuzzyRulesClassifier(
            nRules=10,
            nAnts=3,
            n_linguistic_variables=3,
            verbose=True,
            backend=backend_name
        )
        
        # Measure training time
        start_time = time.time()
        
        # Train the classifier
        fl_classifier.fit(
            X_train, 
            y_train, 
            n_gen=20,  # Reduced for faster testing
            pop_size=30
        )
        
        training_time = time.time() - start_time
        
        # Evaluate on test set
        y_pred = fl_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{backend_name.upper()} Results:")
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Test accuracy: {accuracy:.4f}")
        print(f"  Number of rules: {len(fl_classifier.rule_base.get_rules())}")
        
        return {
            'backend': backend_name,
            'training_time': training_time,
            'accuracy': accuracy,
            'n_rules': len(fl_classifier.rule_base.get_rules()),
            'success': True
        }
        
    except Exception as e:
        print(f"\nError with {backend_name} backend: {e}")
        return {
            'backend': backend_name,
            'success': False,
            'error': str(e)
        }


def main():
    """Main function to compare backends."""
    
    print("Ex-Fuzzy Backend Comparison Demo")
    print("=" * 60)
    
    # Check available backends
    available_backends = evolutionary_backends.list_available_backends()
    print(f"\nAvailable backends: {available_backends}")
    
    # Load dataset
    print("\nLoading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    
    # Test each available backend
    results = []
    
    for backend in available_backends:
        result = test_backend(backend, X_train, y_train, X_test, y_test)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) > 0:
        print("\nBackend Performance Comparison:")
        print(f"{'Backend':<15} {'Time (s)':<12} {'Accuracy':<12} {'Rules':<10}")
        print("-" * 60)
        
        for result in successful_results:
            print(f"{result['backend']:<15} "
                  f"{result['training_time']:<12.2f} "
                  f"{result['accuracy']:<12.4f} "
                  f"{result['n_rules']:<10}")
        
        # Speed comparison
        if len(successful_results) > 1:
            fastest = min(successful_results, key=lambda x: x['training_time'])
            print(f"\nFastest backend: {fastest['backend']}")
            
            for result in successful_results:
                if result['backend'] != fastest['backend']:
                    speedup = result['training_time'] / fastest['training_time']
                    print(f"  {fastest['backend']} is {speedup:.2f}x faster than {result['backend']}")
    
    # Failed backends
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print("\nFailed backends:")
        for result in failed_results:
            print(f"  {result['backend']}: {result['error']}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    
    # Provide installation instructions if evox is not available
    if 'evox' not in available_backends:
        print("\nTo enable EvoX backend (GPU acceleration):")
        print("  pip install ex-fuzzy[evox]")
        print("\nOr directly:")
        print("  pip install 'evox[jax]'")


if __name__ == "__main__":
    main()
