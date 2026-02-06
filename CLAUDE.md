# CLAUDE.md - AI Assistant Guide for Ex-Fuzzy

## Project Overview

**Ex-Fuzzy** is an explainable AI library for fuzzy logic programming in Python. It provides interpretable machine learning models using fuzzy association rules with genetic algorithm optimization.

- **Version**: 2.1.5
- **License**: AGPL v3
- **Python**: 3.7+
- **Main Branch**: `main`
- **Current Branch**: `evox-support` (GPU backend support)

## Quick Commands

```bash
# Install from source
pip install -e .

# Install with GPU support
pip install -e ".[evox]"

# Run tests
pytest tests/

# Run specific test
pytest tests/test_fuzzy_sets_comprehensive.py -v
```

## Directory Structure

```
ex-fuzzy/
├── ex_fuzzy/ex_fuzzy/          # Main package
│   ├── fuzzy_sets.py           # Fuzzy set classes (FS, IVFS, fuzzyVariable)
│   ├── rules.py                # Rule classes & inference engine
│   ├── evolutionary_fit.py     # GA-based rule optimization (main classifier)
│   ├── evolutionary_backends.py # Backend abstraction (pymoo/evox)
│   ├── classifiers.py          # High-level classifiers (RuleMineClassifier)
│   ├── rule_mining.py          # Association rule mining (Apriori)
│   ├── utils.py                # Utilities & partition construction
│   ├── eval_tools.py           # Model evaluation
│   ├── eval_rules.py           # Rule quality metrics
│   ├── vis_rules.py            # Visualization (NetworkX)
│   ├── bootstrapping_test.py   # Bootstrap analysis
│   ├── pattern_stability.py    # Pattern stability metrics
│   ├── permutation_test.py     # Permutation testing
│   ├── persistence.py          # Model save/load
│   ├── temporal.py             # Temporal fuzzy sets
│   ├── centroid.py             # Centroid computation
│   ├── cognitive_maps.py       # Fuzzy Cognitive Maps
│   ├── tree_learning.py        # Fuzzy decision trees
│   ├── tree_learning_new/      # New modular tree implementation
│   └── conformal.py            # Conformal prediction for uncertainty quantification
├── tests/                      # Pytest test suite
├── Demos/                      # Example notebooks
└── docs/                       # Sphinx documentation
```

## Key Classes

### Core Fuzzy Logic
- `FUZZY_SETS` enum: `T1`, `T2`, `GT2` fuzzy set types
- `FS`: Type-1 fuzzy sets (trapezoidal, triangular, gaussian)
- `IVFS`: Interval-Valued Type-2 fuzzy sets
- `fuzzyVariable`: Container for linguistic variables
- `RuleSimple`: Individual rule representation
- `RuleBaseT1`, `RuleBaseT2`: Type-specific rule collections
- `MasterRuleBase`: Multi-class rule container

### Classifiers
- `BaseFuzzyRulesClassifier`: Main GA-optimized classifier (scikit-learn compatible)
- `RuleMineClassifier`: Two-stage mine-then-optimize classifier
- `ConformalFuzzyClassifier`: Conformal prediction wrapper with coverage guarantees

### Backends
- `PyMooBackend`: CPU-based optimization (default, supports checkpoints)
- `EvoXBackend`: GPU-accelerated with JAX (no checkpoints yet)

## Coding Conventions

### Naming
- Classes: `PascalCase` (e.g., `BaseFuzzyRulesClassifier`)
- Functions: `snake_case` (e.g., `compute_memberships`)
- Constants: `UPPER_CASE` (e.g., `FUZZY_SETS`)
- Private: Leading underscore (e.g., `_get_torch()`)

### Imports
```python
# Conditional imports for dual-mode execution
try:
    from . import module
except ImportError:
    import module

# Lazy imports for optional dependencies
def _get_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None
```

### Docstrings
Google-style with Args, Returns, Examples sections.

### Type Hints
Use throughout: `def function(x: np.array, params: list[float]) -> np.array:`

## Dependencies

### Core (Required)
- numpy, matplotlib, pymoo, pandas, scikit-learn

### Optional
- networkx (visualization)
- torch (GPU tensors)
- evox, jax, jaxlib (GPU optimization)

## Testing

Tests use pytest with fixtures in `tests/conftest.py`:
- `iris_dataset`, `binary_dataset`, `regression_dataset`
- `sample_fuzzy_sets`, `sample_fuzzy_variables`, `sample_rules`
- `fuzzy_type` (parametrized T1/T2)

Helper functions:
- `assert_fuzzy_set_properties()`: Validate fuzzy set behavior
- `assert_classifier_properties()`: Validate classifier training

## Common Workflows

### Train a Classifier
```python
from ex_fuzzy import BaseFuzzyRulesClassifier

clf = BaseFuzzyRulesClassifier(
    nRules=20, nAnts=4,
    backend='pymoo'  # or 'evox' for GPU
)
clf.fit(X_train, y_train, n_gen=50, pop_size=50)
predictions = clf.predict(X_test)
```

### Mine Rules
```python
from ex_fuzzy import rule_mining, utils

fuzzy_vars = utils.construct_partitions(X_train)
rules = rule_mining.multiclass_mine_rulebase(
    X_train, y_train, fuzzy_vars,
    support_threshold=0.1, max_depth=3
)
```

### Backend Selection
```python
# PyMoo (default, CPU, supports checkpoints)
clf = BaseFuzzyRulesClassifier(backend='pymoo')

# EvoX (GPU, faster for large datasets)
clf = BaseFuzzyRulesClassifier(backend='evox')
```

### Conformal Prediction
```python
from ex_fuzzy.conformal import ConformalFuzzyClassifier, evaluate_conformal_coverage

# Wrap existing classifier
conf_clf = ConformalFuzzyClassifier(trained_clf)
conf_clf.calibrate(X_cal, y_cal)
pred_sets = conf_clf.predict_set(X_test, alpha=0.1)  # 90% coverage

# Or create from scratch
conf_clf = ConformalFuzzyClassifier(nRules=20, nAnts=4)
conf_clf.fit(X_train, y_train, X_cal, y_cal, n_gen=50)

# Get rule-wise explanations
results = conf_clf.predict_set_with_rules(X_test, alpha=0.1)

# Evaluate coverage
metrics = evaluate_conformal_coverage(conf_clf, X_test, y_test, alpha=0.1)
```

## Important Notes for AI Assistants

1. **Package Path**: The actual Python package is at `ex_fuzzy/ex_fuzzy/`, not just `ex_fuzzy/`

2. **Backend System**: Two backends exist (pymoo/evox) with different capabilities:
   - PyMoo: CPU, checkpoints, default
   - EvoX: GPU, no checkpoints, faster for large data

3. **Scikit-learn Compatible**: Classifiers inherit from `ClassifierMixin` and use standard `fit()`/`predict()` API

4. **Fuzzy Set Types**: Three types supported - T1 (Type-1), T2 (Interval Type-2), GT2 (General Type-2)

5. **Error Handling**: Provide helpful messages with installation instructions when optional dependencies are missing

6. **Relative Imports**: Modules use relative imports; some have fallbacks for direct execution

7. **Lazy Loading**: Optional dependencies (torch, evox) are imported on demand

8. **Key Parameters for fit()**:
   - `n_gen`: Number of generations (30-100 typical)
   - `pop_size`: Population size (30-100 typical)
   - `checkpoints`: Checkpoint frequency (pymoo only)

9. **Current Development**: The `evox-support` branch adds GPU acceleration via EvoX/JAX

10. **No Interactive Git**: Never use `-i` flags with git commands (e.g., `git rebase -i`)

## File Locations for Common Tasks

| Task | File(s) |
|------|---------|
| Add fuzzy set type | `fuzzy_sets.py` |
| Modify inference | `rules.py` |
| Change optimization | `evolutionary_fit.py`, `evolutionary_backends.py` |
| Add classifier | `classifiers.py` |
| Update rule mining | `rule_mining.py` |
| Add evaluation metric | `eval_tools.py`, `eval_rules.py` |
| Modify visualization | `vis_rules.py` |
| Add statistical test | `bootstrapping_test.py`, `permutation_test.py` |
| Update persistence | `persistence.py` |
| Add conformal prediction | `conformal.py` |
