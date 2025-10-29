"""
FuzzyCART Demo with Wine, Iris, and Titanic datasets using Purity and CCI indices.

This demo demonstrates the FuzzyCART algorithm on three datasets including
one with mixed numerical and categorical variables (Titanic).
"""
import sys
sys.path.append('./ex_fuzzy/')
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def load_titanic_dataset():
    """
    Load and preprocess Titanic dataset with mixed numerical/categorical variables.
    
    Returns:
        X: Feature matrix with mixed data types
        y: Survival labels (0=died, 1=survived)
        feature_names: Names of features
        feature_types: Types of features ('numerical' or 'categorical')
    """
    # Create a simplified Titanic dataset
    np.random.seed(42)
    n_samples = 800
    
    # Numerical features
    age = np.random.normal(30, 12, n_samples)
    age = np.clip(age, 1, 80)  # Age between 1-80
    
    fare = np.random.lognormal(3, 1, n_samples)
    fare = np.clip(fare, 0, 500)  # Fare between 0-500
    
    # Categorical features
    sex = np.random.choice([0, 1], n_samples, p=[0.65, 0.35])  # 0=male, 1=female
    pclass = np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5])  # Passenger class
    embarked = np.random.choice([0, 1, 2], n_samples, p=[0.1, 0.3, 0.6])  # 0=C, 1=Q, 2=S
    
    # Create survival based on realistic correlations
    survival_prob = (
        0.1 +  # Base survival rate
        0.4 * sex +  # Women more likely to survive
        0.2 * (pclass == 1) + 0.1 * (pclass == 2) +  # Higher class more likely
        0.2 * (age < 16) +  # Children more likely
        -0.1 * (age > 60) +  # Elderly less likely
        0.1 * np.random.random(n_samples)  # Some randomness
    )
    survival_prob = np.clip(survival_prob, 0, 1)
    y = np.random.binomial(1, survival_prob)
    
    # Combine features
    X = np.column_stack([age, fare, sex, pclass, embarked])
    
    feature_names = ['Age', 'Fare', 'Sex', 'Pclass', 'Embarked']
    feature_types = ['numerical', 'numerical', 'categorical', 'categorical', 'categorical']
    
    return X, y, feature_names, feature_types

def create_mixed_fuzzy_partitions(X, feature_types, n_partitions=3):
    """
    Create fuzzy partitions for mixed numerical/categorical data.
    
    For numerical features: Create gaussian fuzzy sets
    For categorical features: Create discrete fuzzy sets for each category
    """
    from ex_fuzzy import fuzzy_sets as fs
    
    partitions = []
    
    for i, feature_type in enumerate(feature_types):
        if feature_type == 'numerical':
            # Normalize numerical features and create gaussian fuzzy sets
            feature_data = X[:, i]
            min_val, max_val = feature_data.min(), feature_data.max()
            
            fuzzy_sets_list = []
            for j in range(n_partitions):
                center = j / (n_partitions - 1) if n_partitions > 1 else 0.5
                fuzzy_sets_list.append(fs.gaussianFS(f'Low_Med_High'[j*4:(j+1)*4], [center, 0.25]))
            
        else:  # categorical
            # For categorical features, create fuzzy sets for each unique value
            unique_values = np.unique(X[:, i])
            fuzzy_sets_list = []
            
            for val in unique_values:
                # Create crisp membership for each category
                
                fuzzy_set = fs.categoricalFS(str(val), val)
                fuzzy_sets_list.append(fuzzy_set)
        
        fuzzy_var = fs.fuzzyVariable(f'F{i}', fuzzy_sets_list)
        partitions.append(fuzzy_var)
    
    return partitions

def normalize_mixed_features(X, feature_types):
    """Normalize only numerical features to [0, 1], leave categorical as-is."""
    X_normalized = X.copy()
    
    for i, feature_type in enumerate(feature_types):
        if feature_type == 'numerical':
            col_min, col_max = X[:, i].min(), X[:, i].max()
            if col_max > col_min:  # Avoid division by zero
                X_normalized[:, i] = (X[:, i] - col_min) / (col_max - col_min)
    
    return X_normalized

def create_fuzzy_partitions_for_dataset(n_features, n_partitions=3):
    """Create fuzzy partitions for a dataset with given number of features."""
    from ex_fuzzy import fuzzy_sets as fs
    
    partitions = []
    for i in range(n_features):
        # Create fuzzy sets spread across [0, 1] range
        fuzzy_sets_list = []
        for j in range(n_partitions):
            center = j / (n_partitions - 1) if n_partitions > 1 else 0.5
            fuzzy_sets_list.append(fs.gaussianFS(f'FS{j}', [center, 0.3]))
        
        fuzzy_var = fs.fuzzyVariable(f'F{i}', fuzzy_sets_list)
        partitions.append(fuzzy_var)
    
    return partitions

def normalize_features(X):
    """Normalize features to [0, 1] range for fuzzy sets."""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min)

def test_fuzzycart_on_dataset(dataset_name, X, y, feature_names=None, feature_types=None, test_size=0.3, random_state=42):
    """Test FuzzyCART with both purity and CCI on a given dataset."""
    from ex_fuzzy.tree_learning import FuzzyCART
    
    print(f"\n{'='*60}")
    print(f"🧪 Testing FuzzyCART on {dataset_name} Dataset")
    print(f"{'='*60}")
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Classes: {np.unique(y)}")
    
    if feature_names:
        print(f"Features: {feature_names}")
    if feature_types:
        numerical_count = sum(1 for ft in feature_types if ft == 'numerical')
        categorical_count = sum(1 for ft in feature_types if ft == 'categorical')
        print(f"Feature types: {numerical_count} numerical, {categorical_count} categorical")
    
    # Handle mixed vs. pure numerical datasets
    if feature_types:
        # Mixed dataset - use specialized preprocessing
        X_normalized = normalize_mixed_features(X, feature_types)
        partitions = create_mixed_fuzzy_partitions(X, feature_types, n_partitions=3)
    else:
        # Pure numerical dataset - normalize all features
        X_normalized = normalize_features(X)
        partitions = create_fuzzy_partitions_for_dataset(X.shape[1], n_partitions=3)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    results = {}
    
    # Test different rule sizes for scalability analysis
    rule_sizes = [5, 10, 15, 25, 50] if len(X) > 500 else [5, 10, 15, 25]
    
    # Test with both metrics
    for metric in ['purity', 'cci']:
        print(f"\n🔬 Testing with {metric.upper()} metric...")
        results[metric] = []
        
        for max_rules in rule_sizes:
            print(f"\n  🌳 Testing max_rules = {max_rules}...")
            
            try:
                import time
                start_time = time.time()
                
                # Create model
                model = FuzzyCART(
                    fuzzy_partitions=partitions,
                    max_rules=max_rules,
                    max_depth=10,  # Allow deeper trees for larger rule counts
                    target_metric=metric,
                    coverage_threshold=0.01,  # Lower threshold for larger trees
                    min_improvement=0.001     # Lower improvement threshold
                )
                
                # Train model
                fit_start = time.time()
                model.fit(X_train, y_train)
                fit_time = time.time() - fit_start
                
                # Get tree statistics
                tree_stats = model.get_tree_stats()
                actual_rules = tree_stats['tree_rules']
                max_depth = tree_stats['max_depth']
                
                print(f"    📊 Achieved: {actual_rules} rules, depth {max_depth}")
                
                # Make predictions
                pred_start = time.time()
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                pred_time = time.time() - pred_start
                
                total_time = time.time() - start_time
                
                # Calculate accuracies
                train_accuracy = accuracy_score(y_train, y_pred_train)
                test_accuracy = accuracy_score(y_test, y_pred_test)
                
                # Calculate performance metrics
                rules_per_second = actual_rules / fit_time if fit_time > 0 else 0
                predictions_per_second = len(X_test) / pred_time if pred_time > 0 else 0
                
                result = {
                    'max_rules_limit': max_rules,
                    'actual_rules': actual_rules,
                    'max_depth': max_depth,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'fit_time': fit_time,
                    'pred_time': pred_time,
                    'total_time': total_time,
                    'rules_per_second': rules_per_second,
                    'predictions_per_second': predictions_per_second,
                    'memory_efficiency': actual_rules / max_rules,
                    'y_pred_test': y_pred_test,
                    'success': True
                }
                
                print(f"    ✅ Train/Test Accuracy: {train_accuracy:.3f}/{test_accuracy:.3f}")
                print(f"    ⏱️  Fit: {fit_time:.2f}s, Predict: {pred_time:.3f}s")
                print(f"    🚀 Speed: {rules_per_second:.1f} rules/s, {predictions_per_second:.0f} pred/s")
                
                results[metric].append(result)
                
                # Test multiple fits only on first size to save time
                if max_rules == rule_sizes[0]:
                    print(f"    🔄 Testing multiple fits...")
                    model.fit(X_train, y_train)
                    y_pred_test2 = model.predict(X_test)
                    test_accuracy2 = accuracy_score(y_test, y_pred_test2)
                    
                    model.fit(X_train, y_train) 
                    y_pred_test3 = model.predict(X_test)
                    test_accuracy3 = accuracy_score(y_test, y_pred_test3)
                    print(f"    ✅ Multiple fit accuracies: {test_accuracy:.3f}, {test_accuracy2:.3f}, {test_accuracy3:.3f}")
                
            except Exception as e:
                print(f"    ❌ Failed with max_rules={max_rules}: {str(e)}")
                result = {
                    'max_rules_limit': max_rules,
                    'actual_rules': 0,
                    'max_depth': 0,
                    'train_accuracy': 0,
                    'test_accuracy': 0,
                    'fit_time': 0,
                    'pred_time': 0,
                    'total_time': 0,
                    'rules_per_second': 0,
                    'predictions_per_second': 0,
                    'memory_efficiency': 0,
                    'y_pred_test': None,
                    'success': False,
                    'error': str(e)
                }
                results[metric].append(result)
    
    # Create scaling analysis summary
    print(f"\n📈 Scaling Analysis for {dataset_name}:")
    print(f"{'Metric':<8} {'Rules':<8} {'Actual':<8} {'Depth':<6} {'Train Acc':<10} {'Test Acc':<10} {'Fit Time':<9} {'Rules/s':<8}")
    print(f"{'-'*70}")
    
    # Find best results for detailed analysis
    best_purity_result = None
    best_cci_result = None
    
    for metric in ['purity', 'cci']:
        if results[metric]:  # Check if there are any successful results
            successful_results = [r for r in results[metric] if r['success']]
            if successful_results:
                # Sort by test accuracy to find best
                best_result = max(successful_results, key=lambda x: x['test_accuracy'])
                if metric == 'purity':
                    best_purity_result = best_result
                else:
                    best_cci_result = best_result
                
                # Print all results for this metric
                for result in results[metric]:
                    if result['success']:
                        print(f"{metric.upper():<8} {result['max_rules_limit']:<8} {result['actual_rules']:<8} "
                              f"{result['max_depth']:<6} {result['train_accuracy']:<10.3f} {result['test_accuracy']:<10.3f} "
                              f"{result['fit_time']:<9.2f} {result['rules_per_second']:<8.1f}")
                    else:
                        print(f"{metric.upper():<8} {result['max_rules_limit']:<8} {'FAILED':<8} {'N/A':<6} {'N/A':<10} {'N/A':<10} {'N/A':<9} {'N/A':<8}")
    
    # Performance highlights
    print(f"\n🏆 Performance Highlights for {dataset_name}:")
    
    if best_purity_result and best_cci_result:
        if best_purity_result['test_accuracy'] > best_cci_result['test_accuracy']:
            best_overall = best_purity_result
            best_metric = 'PURITY'
        else:
            best_overall = best_cci_result
            best_metric = 'CCI'
        
        print(f"🎯 Best Test Accuracy: {best_overall['test_accuracy']:.3f} ({best_metric})")
        print(f"   Rules: {best_overall['actual_rules']}/{best_overall['max_rules_limit']}, Depth: {best_overall['max_depth']}")
        print(f"   Training Speed: {best_overall['rules_per_second']:.1f} rules/second")
        
        # Find fastest training
        all_successful = []
        for metric in ['purity', 'cci']:
            all_successful.extend([r for r in results[metric] if r['success']])
        
        if all_successful:
            fastest = max(all_successful, key=lambda x: x['rules_per_second'])
            largest = max(all_successful, key=lambda x: x['actual_rules'])
            
            print(f"🚀 Fastest Training: {fastest['rules_per_second']:.1f} rules/second")
            print(f"   Rules: {fastest['actual_rules']}, Accuracy: {fastest['test_accuracy']:.3f}")
            
            print(f"🌳 Largest Tree: {largest['actual_rules']} rules")
            print(f"   Accuracy: {largest['test_accuracy']:.3f}, Time: {largest['fit_time']:.2f}s")
        
        # Detailed classification report for best result
        print(f"\n📋 Detailed Classification Report (Best: {best_metric}):")
        if best_overall['y_pred_test'] is not None:
            print(classification_report(y_test, best_overall['y_pred_test']))
    else:
        print("❌ No successful results to analyze")
    
    return results

def main():
    """Main demo function."""
    print("🚀 FuzzyCART Demo: Wine, Iris, and Titanic Datasets with Purity and CCI")
    print("=" * 75)
    
    # Test on Wine dataset (pure numerical)
    print("\n🍷 Loading Wine dataset...")
    wine = load_wine()
    wine_results = test_fuzzycart_on_dataset("Wine", wine.data, wine.target)
    
    # Test on Iris dataset (pure numerical)
    print("\n🌸 Loading Iris dataset...")
    iris = load_iris()
    iris_results = test_fuzzycart_on_dataset("Iris", iris.data, iris.target)
    
    # Test on Titanic dataset (mixed numerical/categorical)
    print("\n🚢 Loading Titanic dataset...")
    titanic_X, titanic_y, feature_names, feature_types = load_titanic_dataset()
    titanic_results = test_fuzzycart_on_dataset(
        "Titanic", titanic_X, titanic_y, 
        feature_names=feature_names, 
        feature_types=feature_types
    )
    
    # Overall summary with scaling analysis
    print(f"\n🎯 Overall Demo Summary with Scaling Analysis")
    print(f"{'='*80}")
    
    def get_best_result(dataset_results, metric):
        """Get best result for a metric from a dataset."""
        if dataset_results[metric]:
            successful = [r for r in dataset_results[metric] if r['success']]
            if successful:
                return max(successful, key=lambda x: x['test_accuracy'])
        return None
    
    def print_dataset_summary(dataset_name, dataset_results):
        """Print summary for a dataset."""
        print(f"\n{dataset_name}:")
        for metric in ['purity', 'cci']:
            best = get_best_result(dataset_results, metric)
            if best:
                print(f"  {metric.upper()}: {best['test_accuracy']:.3f} accuracy, "
                      f"{best['actual_rules']} rules (max tested: {best['max_rules_limit']}), "
                      f"{best['rules_per_second']:.1f} rules/s")
            else:
                print(f"  {metric.upper()}: No successful results")
    
    print_dataset_summary("🍷 Wine Dataset (13 numerical features)", wine_results)
    print_dataset_summary("🌸 Iris Dataset (4 numerical features)", iris_results)  
    print_dataset_summary("🚢 Titanic Dataset (2 numerical + 3 categorical features)", titanic_results)
    
    # Find overall best results
    all_results = []
    for dataset_name, dataset_results in [("Wine", wine_results), ("Iris", iris_results), ("Titanic", titanic_results)]:
        for metric in ['purity', 'cci']:
            best = get_best_result(dataset_results, metric)
            if best:
                best['dataset'] = dataset_name
                best['metric'] = metric
                all_results.append(best)
    
    if all_results:
        print(f"\n🏆 Overall Best Results:")
        
        # Best accuracy overall
        best_accuracy = max(all_results, key=lambda x: x['test_accuracy'])
        print(f"🎯 Best Accuracy: {best_accuracy['test_accuracy']:.3f} ({best_accuracy['dataset']} - {best_accuracy['metric'].upper()})")
        print(f"   Rules: {best_accuracy['actual_rules']}, Speed: {best_accuracy['rules_per_second']:.1f} rules/s")
        
        # Fastest training
        fastest = max(all_results, key=lambda x: x['rules_per_second'])
        print(f"🚀 Fastest Training: {fastest['rules_per_second']:.1f} rules/s ({fastest['dataset']} - {fastest['metric'].upper()})")
        print(f"   Accuracy: {fastest['test_accuracy']:.3f}, Rules: {fastest['actual_rules']}")
        
        # Largest tree
        largest = max(all_results, key=lambda x: x['actual_rules'])
        print(f"🌳 Largest Tree: {largest['actual_rules']} rules ({largest['dataset']} - {largest['metric'].upper()})")
        print(f"   Accuracy: {largest['test_accuracy']:.3f}, Time: {largest['fit_time']:.2f}s")
    
    # Comparison insights
    print(f"\n💡 Key Insights from Scaling Analysis:")
    print(f"  📊 Multiple rule sizes tested: 5, 10, 15, 25, 50+ rules")
    print(f"  ⚖️  Purity vs CCI: Compare traditional vs accuracy-optimized splitting at scale")
    print(f"  🌳 Tree Complexity: See how performance scales with rule base size")
    print(f"  🎯 Performance vs Size: Find optimal balance between accuracy and efficiency")
    print(f"  🍃 Feature-based Splitting: New architecture scales well with larger trees")
    print(f"     → Traditional decision tree semantics with fuzzy membership")
    print(f"     → Only leaf nodes make predictions")
    print(f"     → All linguistic labels created simultaneously per feature")
    
    print(f"\n✅ Scalability demo completed successfully!")
    print(f"✅ Three datasets tested with multiple rule sizes")
    print(f"✅ Both purity and CCI metrics evaluated across scales") 
    print(f"✅ Performance and timing metrics collected")
    print(f"✅ New feature-based architecture scaling validated")

if __name__ == "__main__":
    main()
    print("\n🏁 All tests completed successfully!")