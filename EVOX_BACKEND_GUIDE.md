# EvoX Backend Support for ex-fuzzy

This guide explains how to use the new EvoX backend for GPU-accelerated evolutionary optimization in ex-fuzzy.

## Overview

Ex-fuzzy now supports multiple evolutionary computation backends:

- **pymoo** (default): CPU-based optimization using the pymoo library
- **evox**: GPU-accelerated optimization using EvoX with JAX

## Installation

### Basic Installation (pymoo only)
```bash
pip install ex-fuzzy
```

### With EvoX Support (GPU acceleration)
```bash
pip install ex-fuzzy[evox]
```

Or install EvoX separately:
```bash
pip install 'evox[jax]'
```

## Usage

### Using the Default pymoo Backend

```python
from ex_fuzzy.ex_fuzzy import evolutionary_fit as GA_RB

# Create classifier (pymoo is default)
clf = GA_RB.BaseFuzzyRulesClassifier(
    nRules=30,
    nAnts=4,
    backend='pymoo'  # Optional, this is the default
)

# Train
clf.fit(X_train, y_train, n_gen=50, pop_size=50)

# Predict
predictions = clf.predict(X_test)
```

### Using the EvoX Backend (GPU)

```python
from ex_fuzzy.ex_fuzzy import evolutionary_fit as GA_RB

# Create classifier with EvoX backend
clf = GA_RB.BaseFuzzyRulesClassifier(
    nRules=30,
    nAnts=4,
    backend='evox'  # Use GPU-accelerated backend
)

# Train (automatically uses GPU if available)
clf.fit(X_train, y_train, n_gen=50, pop_size=50)

# Predict
predictions = clf.predict(X_test)
```

### Checking Available Backends

```python
from ex_fuzzy.ex_fuzzy import evolutionary_backends

# List all available backends
available = evolutionary_backends.list_available_backends()
print(f"Available backends: {available}")

# Check if a specific backend is available
try:
    backend = evolutionary_backends.get_backend('evox')
    print(f"EvoX backend is available: {backend.name()}")
except ValueError as e:
    print(f"EvoX backend not available: {e}")
```

## Performance Comparison

### When to Use Each Backend

**Use pymoo (CPU) when:**
- You don't have a GPU available
- Working with small datasets (< 1000 samples)
- You need checkpoint functionality
- You prefer stability and wide compatibility

**Use EvoX (GPU) when:**
- You have a CUDA-compatible GPU
- Working with large datasets (> 1000 samples)
- You need faster training times
- You're optimizing many rules or complex fuzzy systems

### Expected Performance

On typical classification tasks:
- **Small datasets** (< 500 samples): Similar performance, pymoo may be slightly faster
- **Medium datasets** (500-5000 samples): EvoX typically 2-5x faster
- **Large datasets** (> 5000 samples): EvoX can be 5-10x faster or more

*Note: Actual speedup depends on your hardware, problem complexity, and population size.*

## Features and Limitations

### Supported Features

Both backends support:
- ✅ All fuzzy set types (Type-1, Type-2, GT2)
- ✅ Custom loss functions
- ✅ Linguistic variable optimization
- ✅ Precomputed linguistic variables
- ✅ Multi-objective optimization parameters (alpha, beta)
- ✅ Categorical variables
- ✅ Fuzzy modifiers

### Backend-Specific Limitations

**EvoX backend:**
- ❌ Checkpoint functionality not yet supported
- ✅ Requires JAX and EvoX installation
- ✅ Best performance with GPU

**pymoo backend:**
- ✅ Full checkpoint support
- ✅ No additional dependencies
- ❌ CPU-only (no GPU acceleration)

## Algorithm Parameters

Both backends use similar genetic algorithm parameters:

```python
clf.fit(
    X_train, y_train,
    n_gen=50,              # Number of generations
    pop_size=50,           # Population size
    var_prob=0.3,          # Crossover probability
    sbx_eta=3.0,           # SBX crossover eta (pymoo) / 20.0 (evox recommended)
    mutation_eta=7.0,      # Polynomial mutation eta (pymoo) / 20.0 (evox recommended)
    tournament_size=3,     # Tournament selection size
    random_state=42        # Random seed
)
```

### Parameter Tuning Notes

- **EvoX typically works better with higher eta values** (15-25) for both crossover and mutation
- **pymoo typically works better with lower eta values** (3-10)
- These are just guidelines; experiment with your specific problem

## Example: Complete Comparison

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ex_fuzzy.ex_fuzzy import evolutionary_fit as GA_RB
import time

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Test both backends
for backend in ['pymoo', 'evox']:
    try:
        print(f"\nTesting {backend} backend...")
        
        clf = GA_RB.BaseFuzzyRulesClassifier(
            nRules=15,
            nAnts=3,
            n_linguistic_variables=3,
            verbose=True,
            backend=backend
        )
        
        start = time.time()
        clf.fit(X_train, y_train, n_gen=30, pop_size=40)
        train_time = time.time() - start
        
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        
        print(f"{backend} - Time: {train_time:.2f}s, Accuracy: {accuracy:.4f}")
        
    except ValueError as e:
        print(f"{backend} not available: {e}")
```

## Troubleshooting

### EvoX Backend Not Available

```python
# Error: Backend 'evox' is not available
```

**Solution:** Install EvoX with JAX support
```bash
pip install 'evox[jax]'
# or
pip install ex-fuzzy[evox]
```

### GPU Not Detected

If EvoX falls back to CPU:
```
EvoX backend using CPU (GPU not available)
```

**Check JAX GPU installation:**
```python
import jax
print(jax.devices())  # Should show GPU devices
```

**Install CUDA-enabled JAX:**
```bash
# For CUDA 12
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 11
pip install -U "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Slower Than Expected

If EvoX is not faster than pymoo:
1. Ensure you have a GPU and JAX is using it
2. Try larger population sizes (EvoX scales better)
3. Increase the number of generations
4. Check for other processes using the GPU

## Contributing

If you encounter issues or have suggestions for improving the backend system:
1. Check existing issues on GitHub
2. Create a new issue with details about your problem
3. Include system information (OS, GPU, JAX version, EvoX version)

## Future Improvements

Planned enhancements:
- [ ] Checkpoint support for EvoX backend
- [ ] Multi-GPU support for EvoX
- [ ] Additional optimization algorithms (PSO, DE, CMA-ES)
- [ ] Automatic backend selection based on problem size
- [ ] Mixed precision training for EvoX

## References

- **pymoo**: [https://pymoo.org/](https://pymoo.org/)
- **EvoX**: [https://github.com/EMI-Group/evox](https://github.com/EMI-Group/evox)
- **JAX**: [https://github.com/google/jax](https://github.com/google/jax)
