"""
Test GPU-accelerated regression with EvoX backend.
"""
import sys
sys.path.insert(0, './ex_fuzzy/ex_fuzzy/')

from evolutionary_fit_regression import BaseFuzzyRulesRegressor
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import time

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(5000, 5) * 10  # Larger dataset for GPU benefit
y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] - 0.8 * X[:, 4] + np.random.randn(5000) * 2

print("="*80)
print("GPU-Accelerated Fuzzy Regression Test")
print("="*80)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print()

# Test 1: PyMoo backend (CPU)
print("Test 1: PyMoo Backend (CPU)")
print("-"*80)
regressor_pymoo = BaseFuzzyRulesRegressor(
    nRules=10,
    nAnts=2,
    consequent_type='crisp',
    rule_mode='additive',
    n_linguistic_variables=3
)

start_time = time.time()
regressor_pymoo.fit(X, y, n_gen=30, pop_size=50, random_state=42)
pymoo_time = time.time() - start_time

y_pred_pymoo = regressor_pymoo.predict(X)
r2_pymoo = r2_score(y, y_pred_pymoo)
rmse_pymoo = np.sqrt(mean_squared_error(y, y_pred_pymoo))

print(f"Training time: {pymoo_time:.2f}s")
print(f"R² score: {r2_pymoo:.4f}")
print(f"RMSE: {rmse_pymoo:.4f}")
print()

# Test 2: EvoX backend (GPU)
print("Test 2: EvoX Backend (GPU)")
print("-"*80)
try:
    regressor_evox = BaseFuzzyRulesRegressor(
        nRules=10,
        nAnts=2,
        consequent_type='crisp',
        rule_mode='additive',
        n_linguistic_variables=3,
        backend='evox'  # Use EvoX backend
    )

    start_time = time.time()
    regressor_evox.fit(X, y, n_gen=30, pop_size=50, random_state=42)
    evox_time = time.time() - start_time

    y_pred_evox = regressor_evox.predict(X)
    r2_evox = r2_score(y, y_pred_evox)
    rmse_evox = np.sqrt(mean_squared_error(y, y_pred_evox))

    print(f"Training time: {evox_time:.2f}s")
    print(f"R² score: {r2_evox:.4f}")
    print(f"RMSE: {rmse_evox:.4f}")
    print()
    
    # Speedup
    print("="*80)
    print("Performance Comparison")
    print("="*80)
    speedup = pymoo_time / evox_time
    print(f"PyMoo time:  {pymoo_time:.2f}s")
    print(f"EvoX time:   {evox_time:.2f}s")
    print(f"Speedup:     {speedup:.2f}x")
    print()
    
    # Show rules from GPU-trained model
    print("Learned Rules (GPU-trained):")
    print("-"*80)
    print(regressor_evox)
    
    print()
    print("✅ GPU acceleration test PASSED!")
    
except Exception as e:
    print(f"❌ EvoX backend test failed: {e}")
    import traceback
    traceback.print_exc()
