"""
Test for the cache/node naming issue that causes 'Node name root_F1_L1 already exists' error.

This test checks if the caching mechanism is properly reset between fits.
"""

def test_cache_reset_between_fits():
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from ex_fuzzy.ex_fuzzy.tree_learning import FuzzyCART
    from ex_fuzzy.ex_fuzzy import fuzzy_sets as fs
    import numpy as np
    
    print("✅ Imports successful")
    
    # Create simple fuzzy partitions
    partitions = []
    for i in range(2):  # 2 features
        fuzzy_var = fs.fuzzyVariable(f'F{i}', [
            fs.gaussianFS('L', [0.0, 0.5]),   # Low
            fs.gaussianFS('M', [0.5, 0.5]),   # Medium
            fs.gaussianFS('H', [1.0, 0.5])    # High
        ])
        partitions.append(fuzzy_var)
    
    print("✅ Fuzzy partitions created")
    
    # Create test data
    np.random.seed(42)
    X = np.random.rand(50, 2)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    
    print("✅ Test data created")
    
    # Test multiple fits on the same instance
    model = FuzzyCART(fuzzy_partitions=partitions, max_rules=5, max_depth=3, target_metric='purity')
    
    print("🔄 Testing first fit...")
    model.fit(X, y)
    print("✅ First fit successful")
    
    print("🔄 Testing second fit (this should not fail)...")
    model.fit(X, y)  # This should NOT raise "Node name already exists"
    print("✅ Second fit successful")
    
    print("🔄 Testing third fit...")
    model.fit(X, y)
    print("✅ Third fit successful")
    
    # Test with different data
    print("🔄 Testing fit with different data...")
    X2 = np.random.rand(30, 2) 
    y2 = (X2[:, 0] > 0.5).astype(int)
    model.fit(X2, y2)
    print("✅ Fit with different data successful")
    
    # Test explicit cache clearing
    print("🧹 Testing explicit cache clearing...")
    model.clear_all_caches()
    model.fit(X, y)
    print("✅ Fit after cache clearing successful")
    
    print("\n🎉 All cache reset tests passed!")
    print("✅ No 'Node name already exists' errors")
    return True
    


def test_multiple_model_instances():
    """Test that multiple model instances don't interfere with each other."""
    print("\n🧪 Testing multiple model instances...")
    
    from ex_fuzzy.ex_fuzzy.tree_learning import FuzzyCART
    from ex_fuzzy.ex_fuzzy import fuzzy_sets as fs
    import numpy as np
    
    # Create partitions
    partitions = []
    for i in range(2):
        fuzzy_var = fs.fuzzyVariable(f'F{i}', [
            fs.gaussianFS('l', [0.0, 0.5]),
            fs.gaussianFS('h', [1.0, 0.5])
        ])
        partitions.append(fuzzy_var)
    
    # Create test data
    np.random.seed(123)
    X = np.random.rand(30, 2)
    y = (X[:, 0] > 0.5).astype(int)
    target_metric = 'purity'
    # Create multiple instances
    model1 = FuzzyCART(fuzzy_partitions=partitions, max_rules=3, target_metric=target_metric)
    model2 = FuzzyCART(fuzzy_partitions=partitions, max_rules=3, target_metric=target_metric)
    model3 = FuzzyCART(fuzzy_partitions=partitions, max_rules=3, target_metric=target_metric)
    
    print("✅ Created 3 model instances")
    
    # Fit all models
    print("🔄 Fitting model 1...")
    model1.fit(X, y)
    
    print("🔄 Fitting model 2...")
    model2.fit(X, y)
    
    print("🔄 Fitting model 3...")
    model3.fit(X, y)
    
    print("✅ All models fitted successfully")
    
    # Refit models
    print("🔄 Refitting all models...")
    model1.fit(X, y)
    model2.fit(X, y)
    model3.fit(X, y)
    
    print("✅ All models refitted successfully")
    print("✅ No interference between model instances")
    
    return True
        
 

def main():
    """Run all cache-related tests."""
    print("🚀 Cache Reset and Node Naming Tests")
    print("=" * 50)
    
    test1_success = test_cache_reset_between_fits()
    test2_success = test_multiple_model_instances()
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed!")
        print("✅ Cache reset mechanism working correctly")
        print("✅ No 'Node name already exists' errors")
    else:
        print("\n❌ Some tests failed")
        print("🔧 Cache reset mechanism needs fixing")

if __name__ == "__main__":
    main()