"""
EvoX Backend Demo for ex-fuzzy - Local Version

This script demonstrates the new EvoX backend for ex-fuzzy, which provides 
GPU-accelerated evolutionary computation for fuzzy rule learning.

Usage:
    python demo_evox_local.py

Requirements:
    - ex-fuzzy
    - evox (for GPU acceleration)
    - torch (EvoX uses PyTorch)
    - scikit-learn
    - numpy
    - pandas
    - matplotlib
"""

# ======================================================================
# CONFIGURATION - Modify these parameters to customize the experiments
# ======================================================================
#
# All parameters for the demo are centralized here for easy modification.
# Simply change the values below and run the script to see different results.
#
# Example modifications:
#   - Increase nRules for more complex rule bases (e.g., 20, 30)
#   - Increase n_gen for longer optimization (e.g., 50, 100)
#   - Adjust sbx_eta and mutation_eta for different exploration vs exploitation
# ======================================================================

# Fuzzy Classifier Parameters
CONFIG = {
    'nRules': 40,                    # Number of fuzzy rules
    'nAnts': 3,                      # Number of antecedents per rule
    'n_linguistic_variables': 3,     # Number of linguistic variables per feature
    
    # Training Parameters
    'n_gen': 30,                     # Number of generations
    'pop_size': 60,                  # Population size
    'random_state': 42,              # Random seed for reproducibility
    
    # Genetic Algorithm Parameters
    'sbx_eta': 20.0,                 # SBX crossover distribution index
    'mutation_eta': 20.0,            # Polynomial mutation distribution index
    'var_prob': 0.3,                 # Crossover probability
    
    # Dataset Parameters
    'test_size': 0.3,                # Proportion of test set
    'large_dataset_samples': 100000,   # Samples for large dataset test
    'large_dataset_features': 8,     # Features for large dataset test
}

# ======================================================================
# END CONFIGURATION
# ======================================================================

import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available. Visualizations will be skipped.")

from ex_fuzzy import evolutionary_fit as GA_RB
from ex_fuzzy import evolutionary_backends
import ex_fuzzy


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_backends():
    """Check available backends and hardware."""
    print_section("CHECKING AVAILABLE BACKENDS AND HARDWARE")
    
    # Check available backends
    print("\nAvailable backends:")
    available = evolutionary_backends.list_available_backends()
    for backend in available:
        print(f"  ✓ {backend}")
    
    # Check if GPU is available for EvoX
    if 'evox' in available:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"\n🎉 GPU detected! EvoX will use GPU acceleration.")
                print(f"   Device: {torch.cuda.get_device_name(0)}")
            else:
                print("\n⚠️  No GPU detected. EvoX will run on CPU.")
        except:
            print("⚠️  Could not check PyTorch GPU availability")
    else:
        print("\n⚠️  EvoX not available. Only pymoo backend will be tested.")
        print("   To enable EvoX: pip install evox torch")
    
    return available


def load_and_prepare_data():
    """Load and prepare the Iris dataset."""
    print_section("LOADING AND PREPARING DATASET")
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'], stratify=y
    )
    
    print("\nDataset Information:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Classes: {list(class_names)}")
    print(f"\nFeatures: {list(feature_names)}")
    
    return X_train, X_test, y_train, y_test, feature_names, class_names


def test_pymoo_backend(X_train, y_train, X_test, y_test, class_names):
    """Test the traditional pymoo backend."""
    print_section("TESTING PYMOO BACKEND (CPU)")
    fv_partitions = ex_fuzzy.utils.construct_partitions(X_train, n_partitions=CONFIG['n_linguistic_variables'])
    # Create classifier with pymoo backend
    clf_pymoo = GA_RB.BaseFuzzyRulesClassifier(
        nRules=CONFIG['nRules'],
        nAnts=CONFIG['nAnts'],
        n_linguistic_variables=CONFIG['n_linguistic_variables'],
        verbose=True,
        backend='pymoo',  # Explicitly specify pymoo
        linguistic_variables=fv_partitions
    )
    
    
    # Train
    print("\nTraining with PyMoo backend...")
    start_time = time.time()
    
    clf_pymoo.fit(
        X_train,
        y_train,
        n_gen=CONFIG['n_gen'],
        pop_size=CONFIG['pop_size'],
        random_state=CONFIG['random_state']
    )
    
    pymoo_time = time.time() - start_time
    
    # Evaluate
    y_pred_pymoo = clf_pymoo.predict(X_test)
    pymoo_accuracy = accuracy_score(y_test, y_pred_pymoo)
    pymoo_n_rules = len(clf_pymoo.rule_base.get_rules())
    
    print(f"\n{'=' * 70}")
    print("PYMOO RESULTS:")
    print(f"{'=' * 70}")
    print(f"  Training time: {pymoo_time:.2f} seconds")
    print(f"  Test accuracy: {pymoo_accuracy:.4f}")
    print(f"  Number of rules: {pymoo_n_rules}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_pymoo, target_names=class_names))
    
    # Display learned rules
    print("\nLearned Fuzzy Rules (PyMoo):")
    print("=" * 70)
    rule_str = clf_pymoo.rule_base.print_rules(return_rules=True)
    print(rule_str)
    
    return {
        'time': pymoo_time,
        'accuracy': pymoo_accuracy,
        'n_rules': pymoo_n_rules,
        'classifier': clf_pymoo
    }


def test_evox_backend(X_train, y_train, X_test, y_test, class_names):
    """Test the EvoX backend with GPU acceleration."""
    print_section("TESTING EVOX BACKEND (GPU-ACCELERATED)")
    
    fv_partitions = ex_fuzzy.utils.construct_partitions(X_train, n_partitions=CONFIG['n_linguistic_variables'])
    # Create classifier with evox backend
    clf_evox = GA_RB.BaseFuzzyRulesClassifier(
        nRules=CONFIG['nRules'],
        nAnts=CONFIG['nAnts'],
        n_linguistic_variables=CONFIG['n_linguistic_variables'],
        verbose=True,
        backend='evox',  # Use EvoX backend
        linguistic_variables=fv_partitions
    )
    
    # Train
    print("\nTraining with EvoX backend...")
    start_time = time.time()
    
    clf_evox.fit(
        X_train,
        y_train,
        n_gen=CONFIG['n_gen'],
        pop_size=CONFIG['pop_size'],
        random_state=CONFIG['random_state'],
        sbx_eta=CONFIG['sbx_eta'],
        mutation_eta=CONFIG['mutation_eta']
    )
    
    evox_time = time.time() - start_time
    
    # Evaluate
    y_pred_evox = clf_evox.predict(X_test)
    evox_accuracy = accuracy_score(y_test, y_pred_evox)
    evox_n_rules = len(clf_evox.rule_base.get_rules())
    
    print(f"\n{'=' * 70}")
    print("EVOX RESULTS:")
    print(f"{'=' * 70}")
    print(f"  Training time: {evox_time:.2f}s")
    print(f"  Test accuracy: {evox_accuracy:.4f}")
    print(f"  Number of rules: {evox_n_rules}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_evox, target_names=class_names))
    
    # Display learned rules
    print("\nLearned Fuzzy Rules (EvoX):")
    print("=" * 70)
    rule_str = clf_evox.rule_base.print_rules(return_rules=True)
    print(rule_str)
    
    return {
        'time': evox_time,
        'accuracy': evox_accuracy,
        'n_rules': evox_n_rules,
        'classifier': clf_evox
    }


def compare_results(pymoo_results, evox_results=None):
    """Compare results from both backends."""
    print_section("PERFORMANCE COMPARISON")
    
    # Create comparison table
    comparison_data = {
        'Backend': ['PyMoo'],
        'Time (s)': [pymoo_results['time']],
        'Accuracy': [pymoo_results['accuracy']],
        'Rules': [pymoo_results['n_rules']]
    }
    
    if evox_results is not None:
        comparison_data['Backend'].append('EvoX')
        comparison_data['Time (s)'].append(evox_results['time'])
        comparison_data['Accuracy'].append(evox_results['accuracy'])
        comparison_data['Rules'].append(evox_results['n_rules'])
    
    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + df_comparison.to_string(index=False))
    
    # Calculate speedup if both backends are available
    if evox_results is not None:
        speedup = pymoo_results['time'] / evox_results['time']
        print(f"\n📊 Speedup: EvoX is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than PyMoo")
        
        # Visualize comparison if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            visualize_comparison(comparison_data)
    else:
        print("\n(Install EvoX to see performance comparison)")


def visualize_comparison(comparison_data):
    """Create visualization of backend comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Training time comparison
    axes[0].bar(comparison_data['Backend'], comparison_data['Time (s)'], 
                color=['#3498db', '#e74c3c'])
    axes[0].set_ylabel('Time (seconds)')
    axes[0].set_title('Training Time Comparison')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Accuracy comparison
    axes[1].bar(comparison_data['Backend'], comparison_data['Accuracy'], 
                color=['#3498db', '#e74c3c'])
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Test Accuracy Comparison')
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backend_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Comparison chart saved as 'backend_comparison.png'")
    plt.show()



def test_large_dataset(available_backends):
    """Test with a larger synthetic dataset."""
    print_section("TESTING WITH LARGER SYNTHETIC DATASET")
    
    # Create a larger synthetic dataset
    X_large, y_large = make_classification(
        n_samples=CONFIG['large_dataset_samples'],
        n_features=CONFIG['large_dataset_features'],
        n_informative=max(2, CONFIG['large_dataset_features'] - 2),
        n_redundant=min(2, CONFIG['large_dataset_features'] // 4),
        n_classes=3,
        n_clusters_per_class=2,
        random_state=CONFIG['random_state']
    )
    
    X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(
        X_large, y_large, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'], stratify=y_large
    )
    
    print(f"\nLarge dataset:")
    print(f"  Training samples: {len(X_train_large)}")
    print(f"  Test samples: {len(X_test_large)}")
    print(f"  Features: {X_large.shape[1]}")
    
    results_large = {}
    fv_partitions = ex_fuzzy.utils.construct_partitions(X_train_large, n_partitions=CONFIG['n_linguistic_variables'])
    # Test PyMoo
    print(f"\n{'─' * 70}")
    print("Testing PyMoo on large dataset...")
    clf_pymoo_large = GA_RB.BaseFuzzyRulesClassifier(
        nRules=CONFIG['nRules'], nAnts=CONFIG['nAnts'], n_linguistic_variables=CONFIG['n_linguistic_variables'], 
        verbose=False, backend='pymoo', linguistic_variables=fv_partitions
    )
    start = time.time()
    clf_pymoo_large.fit(X_train_large, y_train_large, n_gen=CONFIG['n_gen'], pop_size=CONFIG['pop_size'])
    results_large['pymoo_time'] = time.time() - start
    results_large['pymoo_acc'] = accuracy_score(
        y_test_large, clf_pymoo_large.predict(X_test_large)
    )
    print(f"PyMoo - Time: {results_large['pymoo_time']:.2f}s, "
          f"Accuracy: {results_large['pymoo_acc']:.4f}")
    
    # Test EvoX if available
    if 'evox' in available_backends:
        print(f"\n{'─' * 70}")
        print("Testing EvoX on large dataset...")
        clf_evox_large = GA_RB.BaseFuzzyRulesClassifier(
            nRules=CONFIG['nRules'], nAnts=CONFIG['nAnts'], n_linguistic_variables=CONFIG['n_linguistic_variables'], 
            verbose=False, backend='evox', linguistic_variables=fv_partitions
        )
        start = time.time()
        clf_evox_large.fit(X_train_large, y_train_large, n_gen=CONFIG['n_gen'], pop_size=CONFIG['pop_size'], 
                          sbx_eta=CONFIG['sbx_eta'], mutation_eta=CONFIG['mutation_eta'])
        results_large['evox_time'] = time.time() - start
        results_large['evox_acc'] = accuracy_score(
            y_test_large, clf_evox_large.predict(X_test_large)
        )
        print(f"EvoX - Time: {results_large['evox_time']:.2f}s, "
              f"Accuracy: {results_large['evox_acc']:.4f}")
        
        speedup_large = results_large['pymoo_time'] / results_large['evox_time']
        print(f"\n📊 Large dataset speedup: {speedup_large:.2f}x")
    else:
        print("\n⚠️  EvoX not available for large dataset test")
    
    print(f"\n{'=' * 70}")


def print_summary():
    """Print summary and recommendations."""
    print_section("SUMMARY")
    
    summary_text = """
### Key Takeaways:

1. ✅ PyMoo backend works exactly as before - Full backward compatibility maintained
2. ✅ EvoX backend provides GPU acceleration - Potentially faster training on larger datasets
3. ✅ Easy to switch backends - Just change the `backend` parameter
4. ✅ Default behavior unchanged - Existing code continues to work

### When to use each backend:

- **PyMoo**: Small datasets, CPU-only environments, checkpoint support needed
- **EvoX**: Large datasets, GPU available, need faster training

### Installation:

```bash
# Basic (pymoo only)
pip install ex-fuzzy

# With EvoX support (now uses PyTorch)
pip install evox torch
```

### Additional Resources:

- ex-fuzzy Documentation: https://github.com/Fuminides/ex-fuzzy
- EvoX Library: https://github.com/EMI-Group/evox
- PyMoo Library: https://pymoo.org/
"""
    print(summary_text)


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("EvoX Backend Demo for ex-fuzzy (Local Version)")
    print("=" * 70)
    
    # Display current configuration
    print("\n📝 Current Configuration:")
    print("   Fuzzy Rules: {} rules with {} antecedents".format(CONFIG['nRules'], CONFIG['nAnts']))
    print("   Linguistic Variables: {}".format(CONFIG['n_linguistic_variables']))
    print("   Training: {} generations, population size {}".format(CONFIG['n_gen'], CONFIG['pop_size']))
    print("   GA Parameters: SBX η={}, Mutation η={}".format(CONFIG['sbx_eta'], CONFIG['mutation_eta']))
    print("   (Modify CONFIG at the top of the script to change these)")
    
    # Check backends and hardware
    available_backends = check_backends()
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names, class_names = load_and_prepare_data()
    
    # Test PyMoo backend
    
    pymoo_results = test_pymoo_backend(X_train, y_train, X_test, y_test, class_names)
    
    # Test EvoX backend if available
    evox_results = None
    if 'evox' in available_backends:
        evox_results = test_evox_backend(X_train, y_train, X_test, y_test, class_names)
    else:
        print("\n  EvoX backend not available. Skipping EvoX test.")
        print("   To enable EvoX: pip install evox torch")
    
    # Compare results
    compare_results(pymoo_results, evox_results)
    
    # Test with larger dataset
    test_large_dataset(['evox'])

    # Print summary
    print_summary()
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
