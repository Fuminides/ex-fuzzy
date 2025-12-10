# Fuzzy Regression Implementation Guide

## Overview

This document describes the implementation of fuzzy regression with numeric rule outputs in the ex-fuzzy library. This allows fuzzy rule-based systems to perform regression tasks (predicting continuous values) in addition to classification.

## Architecture

### Key Classes

1. **FitRuleBaseRegression** (`evolutionary_fit_regression.py`)
   - Extends `FitRuleBase` for regression problems
   - Handles gene encoding/decoding with numeric consequents
   - Implements RMSE-based fitness evaluation
   - Supports both PyMoo (CPU) and EvoX (GPU) backends

2. **BaseFuzzyRulesRegressor** (`evolutionary_fit_regression.py`)
   - Implements `sklearn.base.RegressorMixin` interface
   - Provides `fit()`, `predict()`, and `score()` methods
   - Compatible with sklearn pipelines and cross-validation

### Gene Encoding

Genes encode fuzzy rules with the following structure:

```
[antecedents | antecedent_params | membership_functions* | consequents | weights*]
```

- **antecedents**: Feature indices for each rule's antecedents
- **antecedent_params**: Linguistic variable indices for each antecedent
- **membership_functions**: Parameters for fuzzy sets (only if not precomputed)
- **consequents**: Numeric outputs normalized to [0, 99]
- **weights**: Rule weights (only if ds_mode == 2)

*Optional segments

### Numeric Consequents

Unlike classification where consequents are class labels, regression uses numeric values:

1. **Normalization**: Consequents are stored as integers in [0, 99] in the gene
2. **Denormalization**: During rule construction, they're mapped to [y_min, y_max]
   ```python
   consequent_value = y_min + (normalized / 99.0) * (y_max - y_min)
   ```

### Prediction Method: Configurable Rule Modes

The regressor supports two rule firing modes controlled by the `rule_mode` parameter:

#### Additive Mode (Default)

**`rule_mode='additive'`** - All rules contribute to every prediction (Takagi-Sugeno style):

```python
prediction = sum(membership_i * consequent_i) / sum(membership_i)
```

Process:
1. Compute membership degree for ALL rules
2. Calculate weighted average using membership as weights
3. NO tolerance filtering - all rules contribute
4. No "dead zones" where predictions fail

**Advantages:**
- More stable predictions (all rules contribute)
- Prevents "no rules firing" problem
- Standard approach in fuzzy regression literature
- Smoother output surface
- Better gradient for optimization

#### Sufficient Mode

**`rule_mode='sufficient'`** - Only rules above tolerance fire (similar to classification):

```python
# Only rules with membership > tolerance contribute
active_rules = [rule for rule in rules if membership(rule) > tolerance]
prediction = sum(membership_i * consequent_i) / sum(membership_i)  # over active rules only
```

Process:
1. Compute membership degree for all rules
2. Filter rules where membership > tolerance
3. Calculate weighted average using only firing rules
4. If no rules fire, use fallback value (midpoint of range)

**When to use:**
- Want to explicitly filter weak rules
- Sparse or irregular data patterns
- More interpretable (fewer active rules per prediction)
- Similar behavior to classification systems

### Fitness Function

Fitness is based on Root Mean Squared Error (RMSE):

```python
rmse = sqrt(mean_squared_error(y_true, y_predicted))
normalized_rmse = rmse / (y_max - y_min)
fitness = -normalized_rmse  # Negative for minimization
```

Additional penalties can be added for rule complexity:
- `alpha`: Penalty for average number of antecedents per rule
- `beta`: Penalty for total number of rules

## Usage

### Basic Example

```python
from ex_fuzzy.evolutionary_fit_regression import BaseFuzzyRulesRegressor
import ex_fuzzy.fuzzy_sets as fs
import numpy as np

# Prepare data
X_train, y_train = ...  # Your training data

# Create linguistic variables for each feature
X_min = np.min(X_train, axis=0)
X_max = np.max(X_train, axis=0)

linguistic_vars = []
for feat_idx in range(X_train.shape[1]):
    feat_min, feat_max = X_min[feat_idx], X_max[feat_idx]
    feat_range = feat_max - feat_min
    
    # Create LOW, MEDIUM, HIGH fuzzy sets
    low = fs.FS("LOW", [feat_min, feat_min, 
                        feat_min + 0.4*feat_range, feat_min + 0.5*feat_range])
    med = fs.FS("MEDIUM", [feat_min + 0.3*feat_range, feat_min + 0.5*feat_range,
                            feat_min + 0.5*feat_range, feat_min + 0.7*feat_range])
    high = fs.FS("HIGH", [feat_min + 0.5*feat_range, feat_min + 0.6*feat_range,
                          feat_max, feat_max])
    
    fv = fs.fuzzyVariable(f"Feature_{feat_idx}", [low, med, high])
    linguistic_vars.append(fv)

# Create regressor
regressor = BaseFuzzyRulesRegressor(
    nRules=15,              # Number of rules to evolve
    nAnts=3,                # Antecedents per rule
    linguistic_variables=linguistic_vars,  # Precomputed fuzzy sets
    fuzzy_type=fs.FUZZY_SETS.t1,
    tolerance=0.001,        # Used only in 'sufficient' mode
    rule_mode='additive',   # 'additive' or 'sufficient'
    verbose=True,
    backend='pymoo'         # or 'evox' for GPU
)

# Train
regressor.fit(X_train, y_train, n_gen=50, pop_size=80)

# Predict
y_pred = regressor.predict(X_test)

# Evaluate
r2 = regressor.score(X_test, y_test)
print(f"R² score: {r2:.4f}")
```

### Hyperparameters

**Model Parameters:**
- `nRules`: Number of fuzzy rules (10-30 typical)
- `nAnts`: Antecedents per rule (2 to n_features)
- `linguistic_variables`: Precomputed fuzzy variables (recommended)
- `n_linguistic_variables`: Number of LVs per feature if not precomputed
- `tolerance`: Minimum membership for rule firing (only used in `'sufficient'` mode)
- `rule_mode`: **'additive'** (default, all rules contribute) or **'sufficient'** (only rules > tolerance)
- `fuzzy_type`: Type 1 (`fs.FUZZY_SETS.t1`) or Type 2 fuzzy sets
- `backend`: 'pymoo' for CPU or 'evox' for GPU acceleration

**Optimization Parameters:**
- `n_gen`: Generations for evolution (30-100)
- `pop_size`: Population size (40-100)
- `sbx_eta`: Crossover distribution index (15-30)
- `mutation_eta`: Mutation distribution index (15-30)
- `var_prob`: Crossover probability (0.9-1.0)

## Important Notes

### Precomputed Linguistic Variables

**Always use precomputed linguistic variables** for regression. Evolving membership functions alongside rules creates a very large search space that is difficult to optimize.

❌ **Don't do this:**
```python
regressor = BaseFuzzyRulesRegressor(
    nRules=15,
    nAnts=3,
    n_linguistic_variables=3,  # Will try to evolve MFs
    ...
)
```

✅ **Do this instead:**
```python
linguistic_vars = create_linguistic_variables(X_train)  # See example above
regressor = BaseFuzzyRulesRegressor(
    nRules=15,
    nAnts=3,
    linguistic_variables=linguistic_vars,  # Use precomputed
    ...
)
```

### Performance Considerations

1. **Start small**: Begin with fewer rules (10-15) and increase if needed
2. **More generations**: Regression often needs 50-100 generations
3. **Larger populations**: Use pop_size=80-100 for better exploration
4. **GPU acceleration**: Use `backend='evox'` for large datasets (>1000 samples)
5. **Feature scaling**: Consider normalizing features to similar ranges

### Interpreting Rules

Learned rules can be inspected for interpretability:

```python
for i, rule in enumerate(regressor.rule_base.get_rules(), 1):
    print(f"Rule {i}:")
    print(f"  IF", end="")
    for feat_idx, ant in enumerate(rule.antecedents):
        if ant >= 0:  # Not don't-care
            lv_name = linguistic_vars[feat_idx].linguistic_variable_names()[int(ant)]
            print(f" Feature_{feat_idx} is {lv_name} AND", end="")
    print(f"\n  THEN output = {rule.consequent:.2f}")
```

## Comparison with Classification

| Aspect | Classification | Regression |
|--------|---------------|------------|
| Consequent type | Class label (integer) | Numeric value (float) |
| Fitness metric | MCC, Accuracy | RMSE, MSE |
| Prediction | Weighted voting | Weighted average |
| Output range | Fixed classes | [y_min, y_max] |
| Evaluation | Confusion matrix | R², RMSE, MAE |
| Rule firing | Sufficient (tolerance) | Additive (all rules) |
| Weights | Optional (ds_mode) | Not used |

## Future Work

### Fuzzy Set Consequents (Not Yet Implemented)

The second configuration will allow rules to output fuzzy sets instead of numeric values:

```
Rule: IF x1 is LOW AND x2 is HIGH THEN y is "MEDIUM_HIGH"
```

This requires:
1. Defuzzification method (centroid, bisector, etc.)
2. Fuzzy set encoding in genes
3. Modified fitness function
4. Updated prediction logic

## Testing

Run the test suite:

```bash
python test_regression.py
```

Expected output:
- Test 1: Synthetic data regression (should pass with R² > -0.5)
- Test 2: Function approximation y = x₁² + 2x₂ + 5 (should achieve R² > 0)

## Troubleshooting

**Problem**: R² is negative or very low

**Solutions**:
1. Increase `n_gen` to 50-100
2. Increase `pop_size` to 80-100
3. Use precomputed linguistic variables
4. Check feature scaling
5. Increase `nRules`
6. Reduce `tolerance` to 0.001

**Problem**: Fitness not improving

**Causes**:
- Not using precomputed linguistic variables (most common)
- Search space too large
- Poor initialization

**Problem**: Slow optimization

**Solutions**:
- Use `backend='evox'` with GPU
- Reduce `pop_size` or `n_gen`
- Use fewer `nRules`
- Simplify linguistic variables (fewer LVs per feature)

## Files

- `ex_fuzzy/ex_fuzzy/evolutionary_fit_regression.py`: Main implementation
- `test_regression.py`: Test suite
- `test_precomputed_lvs.py`: Example with precomputed LVs
- `REGRESSION_GUIDE.md`: This document

## References

- Original classification implementation: `evolutionary_fit.py`
- Fuzzy sets: `fuzzy_sets.py`
- Rules: `rules.py`
- Backend implementations: `evolutionary_backends.py`
