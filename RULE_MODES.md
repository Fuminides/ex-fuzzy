# Rule Mode Parameter: Additive vs Sufficient

## Overview

The fuzzy regression implementation now supports two rule firing modes via the `rule_mode` parameter:
- **'additive'** (default): All rules contribute to predictions
- **'sufficient'**: Only rules above tolerance threshold fire

## Usage

```python
from ex_fuzzy.evolutionary_fit_regression import BaseFuzzyRulesRegressor

# Additive mode (default) - recommended for most cases
regressor = BaseFuzzyRulesRegressor(
    nRules=15,
    nAnts=3,
    rule_mode='additive',  # All rules contribute
    tolerance=0.001,       # Ignored in additive mode
    ...
)

# Sufficient mode - for filtering weak rules
regressor = BaseFuzzyRulesRegressor(
    nRules=15,
    nAnts=3,
    rule_mode='sufficient',  # Only strong rules
    tolerance=0.1,          # Rules must exceed this to fire
    ...
)
```

## Comparison

| Feature | Additive | Sufficient |
|---------|----------|------------|
| **Rules used** | All rules | Only rules with membership > tolerance |
| **Tolerance** | Ignored | Required (e.g., 0.1) |
| **Predictions** | Always produces output | May have no firing rules → fallback |
| **Smoothness** | Very smooth | Can have discontinuities |
| **Dead zones** | None | Possible (when no rules fire) |
| **Interpretability** | All rules active | Fewer active rules per prediction |
| **Use case** | Continuous functions | Sparse/irregular patterns |
| **Similar to** | Takagi-Sugeno systems | Classification systems |

## When to Use Each

### Use Additive (Default)

✅ Most regression tasks
✅ Continuous smooth functions  
✅ When you want stable predictions
✅ When you have good linguistic variable coverage
✅ Following fuzzy regression literature standards

### Use Sufficient

✅ When weak rules add noise
✅ Sparse or irregular data patterns
✅ When interpretability requires clear "firing" vs "not firing"
✅ Mimicking classification behavior
✅ When tolerance filtering is domain-appropriate

## Example Results

From `test_rule_modes.py` on y = 3x² - 2x + 1:

```
Test point: X = 1.50
  True value:        4.75
  
  Additive mode:
    - Prediction:     0.00  (error: 4.75)
    - All 2 rules contributed
  
  Sufficient mode (tolerance=0.1):
    - Prediction:     8.27  (error: 3.52)
    - 0/2 rules fired → used fallback value
```

## Implementation Details

Both modes implemented in:
- `FitRuleBaseRegression._evaluate_numpy_fast_regression()` - fitness evaluation
- `FitRuleBaseRegression.fitness_func()` - fitness function
- `BaseFuzzyRulesRegressor.predict()` - prediction

The `rule_mode` parameter is passed through the entire chain:
```
BaseFuzzyRulesRegressor(rule_mode='additive')
  → fit()
    → FitRuleBaseRegression(rule_mode='additive')
      → fitness_func() uses rule_mode
      → predict() uses rule_mode
```

## Code Examples

See:
- `test_rule_modes.py` - Complete comparison of both modes
- `demo_additive_regression.py` - Demonstrates additive behavior
- `test_regression.py` - Uses additive mode (default)

## Recommendation

**Use `rule_mode='additive'` (default) for most regression tasks.** This is the standard approach in fuzzy regression literature and provides more stable, smooth predictions without dead zones.

Only switch to `rule_mode='sufficient'` if you have specific reasons to filter weak rules (e.g., domain knowledge suggests weak memberships should be ignored, or you need behavior similar to classification systems).
