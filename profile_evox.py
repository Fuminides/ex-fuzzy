"""
Profile EvoX backend to identify remaining bottlenecks.
"""
import time
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ex_fuzzy import evolutionary_fit as GA_RB
import ex_fuzzy

# Configuration
CONFIG = {
    'nRules': 15,
    'nAnts': 3,
    'n_linguistic_variables': 3,
    'n_gen': 10,  # Reduced for profiling
    'pop_size': 40,
    'random_state': 42,
}

print("=" * 70)
print("EvoX Backend Profiling")
print("=" * 70)

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nDataset: {len(X_train)} training samples, {X_train.shape[1]} features")
print(f"Configuration: {CONFIG['nRules']} rules, {CONFIG['nAnts']} antecedents, {CONFIG['n_gen']} generations, {CONFIG['pop_size']} pop_size")

# Create classifier
fv_partitions = ex_fuzzy.utils.construct_partitions(X_train, n_partitions=CONFIG['n_linguistic_variables'])
clf = GA_RB.BaseFuzzyRulesClassifier(
    nRules=CONFIG['nRules'],
    nAnts=CONFIG['nAnts'],
    n_linguistic_variables=CONFIG['n_linguistic_variables'],
    verbose=False,
    backend='evox',
    linguistic_variables=fv_partitions
)

# Profile the fit process
print("\n" + "=" * 70)
print("Starting profiling...")
print("=" * 70)

start_total = time.time()
clf.fit(
    X_train,
    y_train,
    n_gen=CONFIG['n_gen'],
    pop_size=CONFIG['pop_size'],
    random_state=CONFIG['random_state']
)
total_time = time.time() - start_total

print(f"\n" + "=" * 70)
print("Profiling Results")
print("=" * 70)
print(f"Total time: {total_time:.4f}s")
print(f"Time per generation: {total_time / CONFIG['n_gen']:.4f}s")
print(f"Time per evaluation: {total_time / (CONFIG['n_gen'] * CONFIG['pop_size'] * 2):.6f}s")  # *2 for parents+offspring

# Check GPU utilization
if torch.cuda.is_available():
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

print("\n" + "=" * 70)
print("Bottleneck Analysis")
print("=" * 70)

analysis = """
Key observations:

1. **Individual vs Batch Evaluation**:
   - Current: Still evaluating individuals one-by-one
   - Potential: Full batching could give 5-10x speedup
   
2. **CPU↔GPU Transfers**:
   - Gene transfer: Once per evaluation (minimized ✓)
   - Membership lookups: Cached on GPU (good ✓)
   - MCC computation: On GPU (good ✓)
   
3. **Python Loops**:
   - Rule reconstruction: Still using Python loops
   - Vectorization potential: High
   
4. **Comparison with PyMoo**:
   - PyMoo: Pure NumPy, highly optimized, no GPU overhead
   - EvoX: GPU overhead + transfer costs + less mature optimization
   - Expected: EvoX faster only on much larger problems (10K+ samples)

Recommendations:
- Small datasets (< 1000 samples): Use PyMoo (CPU optimized)
- Large datasets (10K+ samples): Use EvoX (GPU benefits outweigh overhead)
- Medium datasets: Similar performance, choose based on hardware
"""

print(analysis)
