"""
Real-world regression demo using California Housing dataset
Predicting median house value based on location and demographics
"""
import sys
sys.path.insert(0, './ex_fuzzy/ex_fuzzy/')

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from evolutionary_fit_regression import BaseFuzzyRulesRegressor
import fuzzy_sets as fs
import time

print("=" * 80)
print("REAL-WORLD REGRESSION DEMO: CALIFORNIA HOUSING DATASET")
print("=" * 80)
print()

# Load California Housing dataset
print("Loading California Housing dataset...")
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target  # Median house value in $100,000s

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target: Median house value (in $100k)")
print()
print("Features:")
for i, name in enumerate(X.columns):
    print(f"  {i+1}. {name:15s} - Range: [{X[name].min():.2f}, {X[name].max():.2f}]")
print()

# Take a subset for faster training
print("Using a subset for faster training (2000 samples)...")
X_subset, _, y_subset, _ = train_test_split(X, y, train_size=2000, random_state=42, stratify=None)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.3, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}] (${y_train.min()*100:.0f}k - ${y_train.max()*100:.0f}k)")
print()

# Normalize features for better fuzzy set definition
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# Create linguistic variables for each feature
print("Creating linguistic variables...")
print("  Each feature gets 3 fuzzy sets: LOW, MEDIUM, HIGH")

linguistic_vars = []
for feat_idx in range(X_train_scaled.shape[1]):
    feat_name = X_train_scaled.columns[feat_idx]
    feat_values = X_train_scaled.iloc[:, feat_idx]
    feat_min = feat_values.min()
    feat_max = feat_values.max()
    feat_range = feat_max - feat_min
    
    # Create 3 overlapping fuzzy sets
    low = fs.FS("LOW", [feat_min, feat_min, 
                        feat_min + 0.35*feat_range, feat_min + 0.45*feat_range])
    med = fs.FS("MEDIUM", [feat_min + 0.25*feat_range, feat_min + 0.45*feat_range,
                           feat_min + 0.55*feat_range, feat_min + 0.75*feat_range])
    high = fs.FS("HIGH", [feat_min + 0.55*feat_range, feat_min + 0.65*feat_range,
                          feat_max, feat_max])
    
    fv = fs.fuzzyVariable(feat_name, [low, med, high])
    linguistic_vars.append(fv)

print(f"  Created {len(linguistic_vars)} fuzzy variables")
print()

# Train fuzzy regressor
print("-" * 80)
print("1. FUZZY REGRESSION (Additive Rules)")
print("-" * 80)

start_time = time.time()

fuzzy_regressor = BaseFuzzyRulesRegressor(
    nRules=20,              # Number of rules
    nAnts=4,                # Use 4 out of 8 features per rule
    linguistic_variables=linguistic_vars,
    fuzzy_type=fs.FUZZY_SETS.t1,
    tolerance=0.001,        # Not used in additive mode
    rule_mode='additive',   # All rules contribute
    verbose=True,
    backend='pymoo'
)

print("Training fuzzy regressor...")
fuzzy_regressor.fit(
    X_train_scaled.values, 
    y_train.values,
    n_gen=40,
    pop_size=60,
    random_state=42
)

train_time = time.time() - start_time

# Predictions
y_pred_train_fuzzy = fuzzy_regressor.predict(X_train_scaled.values)
y_pred_test_fuzzy = fuzzy_regressor.predict(X_test_scaled.values)

# Metrics
train_r2_fuzzy = r2_score(y_train, y_pred_train_fuzzy)
test_r2_fuzzy = r2_score(y_test, y_pred_test_fuzzy)
train_rmse_fuzzy = np.sqrt(mean_squared_error(y_train, y_pred_train_fuzzy))
test_rmse_fuzzy = np.sqrt(mean_squared_error(y_test, y_pred_test_fuzzy))
train_mae_fuzzy = mean_absolute_error(y_train, y_pred_train_fuzzy)
test_mae_fuzzy = mean_absolute_error(y_test, y_pred_test_fuzzy)

print(f"\nResults:")
print(f"  Training time: {train_time:.2f}s")
print(f"  Train R²:      {train_r2_fuzzy:.4f}")
print(f"  Test R²:       {test_r2_fuzzy:.4f}")
print(f"  Train RMSE:    ${train_rmse_fuzzy*100:.2f}k")
print(f"  Test RMSE:     ${test_rmse_fuzzy*100:.2f}k")
print(f"  Train MAE:     ${train_mae_fuzzy*100:.2f}k")
print(f"  Test MAE:      ${test_mae_fuzzy*100:.2f}k")
print(f"  Rules:         {len(fuzzy_regressor.rule_base.get_rules())}")
print()

# Show example rules using the built-in print method
print("Example Rules (showing first 10):")
print()
all_rules = fuzzy_regressor.rule_base.print_rules_regression(
    return_rules=True, 
    output_name='house_value (in $100k)'
)
# Print first 10 rules
rules_list = all_rules.strip().split('\n')
for rule_str in rules_list[:10]:
    print(f"  {rule_str}")
print()
if len(rules_list) > 10:
    print(f"  ... and {len(rules_list) - 10} more rules")
print()
print()

# Compare with traditional methods
print("-" * 80)
print("2. LINEAR REGRESSION (Baseline)")
print("-" * 80)

start_time = time.time()
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
train_time_lr = time.time() - start_time

y_pred_train_lr = lr.predict(X_train_scaled)
y_pred_test_lr = lr.predict(X_test_scaled)

train_r2_lr = r2_score(y_train, y_pred_train_lr)
test_r2_lr = r2_score(y_test, y_pred_test_lr)
train_rmse_lr = np.sqrt(mean_squared_error(y_train, y_pred_train_lr))
test_rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_test_lr))

print(f"Training time: {train_time_lr:.2f}s")
print(f"Train R²:      {train_r2_lr:.4f}")
print(f"Test R²:       {test_r2_lr:.4f}")
print(f"Train RMSE:    ${train_rmse_lr*100:.2f}k")
print(f"Test RMSE:     ${test_rmse_lr*100:.2f}k")
print()

print("-" * 80)
print("3. RANDOM FOREST (Strong Baseline)")
print("-" * 80)

start_time = time.time()
rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
train_time_rf = time.time() - start_time

y_pred_train_rf = rf.predict(X_train_scaled)
y_pred_test_rf = rf.predict(X_test_scaled)

train_r2_rf = r2_score(y_train, y_pred_train_rf)
test_r2_rf = r2_score(y_test, y_pred_test_rf)
train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_pred_train_rf))
test_rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_test_rf))

print(f"Training time: {train_time_rf:.2f}s")
print(f"Train R²:      {train_r2_rf:.4f}")
print(f"Test R²:       {test_r2_rf:.4f}")
print(f"Train RMSE:    ${train_rmse_rf*100:.2f}k")
print(f"Test RMSE:     ${test_rmse_rf*100:.2f}k")
print()

# Summary comparison
print("=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print()
print(f"{'Method':<20} {'Train R²':>10} {'Test R²':>10} {'Test RMSE':>12} {'Time (s)':>10} {'Interpretable':>15}")
print("-" * 80)
print(f"{'Fuzzy Regression':<20} {train_r2_fuzzy:>10.4f} {test_r2_fuzzy:>10.4f} ${test_rmse_fuzzy*100:>10.2f}k {train_time:>10.2f} {'Yes (rules)':>15}")
print(f"{'Linear Regression':<20} {train_r2_lr:>10.4f} {test_r2_lr:>10.4f} ${test_rmse_lr*100:>10.2f}k {train_time_lr:>10.2f} {'Partial':>15}")
print(f"{'Random Forest':<20} {train_r2_rf:>10.4f} {test_r2_rf:>10.4f} ${test_rmse_rf*100:>10.2f}k {train_time_rf:>10.2f} {'No (black-box)':>15}")
print()

# Prediction examples
print("=" * 80)
print("SAMPLE PREDICTIONS")
print("=" * 80)
print()
print("Comparing predictions on 5 test samples:")
print()

sample_indices = np.random.choice(len(X_test), 5, replace=False)
for idx in sample_indices:
    true_val = y_test.iloc[idx]
    fuzzy_pred = y_pred_test_fuzzy[idx]
    lr_pred = y_pred_test_lr[idx]
    rf_pred = y_pred_test_rf[idx]
    
    print(f"Sample {idx}:")
    print(f"  Actual:        ${true_val*100:.2f}k")
    print(f"  Fuzzy:         ${fuzzy_pred*100:.2f}k  (error: ${abs(fuzzy_pred-true_val)*100:.2f}k)")
    print(f"  Linear:        ${lr_pred*100:.2f}k  (error: ${abs(lr_pred-true_val)*100:.2f}k)")
    print(f"  Random Forest: ${rf_pred*100:.2f}k  (error: ${abs(rf_pred-true_val)*100:.2f}k)")
    print()

print("=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print()
print("✓ Fuzzy regression provides INTERPRETABLE rules")
print(f"  - {len(fuzzy_regressor.rule_base.get_rules())} human-readable IF-THEN rules")
print(f"  - Rules use linguistic terms (LOW, MEDIUM, HIGH)")
print(f"  - Can explain why a prediction was made")
print()
print("✓ Competitive performance with traditional methods")
print(f"  - Test R² = {test_r2_fuzzy:.4f} (compare to Linear: {test_r2_lr:.4f}, RF: {test_r2_rf:.4f})")
print(f"  - RMSE within ${abs(test_rmse_fuzzy-test_rmse_rf)*100:.2f}k of Random Forest")
print()
print("✓ Additive rule mode ensures stable predictions")
print("  - All 20 rules contribute to every prediction")
print("  - No dead zones or undefined regions")
print("  - Smooth prediction surface")
print()
print("Use case: When you need BOTH accuracy AND interpretability")
print("Examples: Healthcare, finance, policy-making, expert systems")
print("=" * 80)
