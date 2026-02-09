"""
Conformal learning demo for ex-fuzzy.

This demo trains a fuzzy classifier on Iris, calibrates conformal prediction,
and reports coverage and prediction-set behavior.
"""

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

# Resolve repository paths and prioritize local source over site-packages.
ROOT_DIR = Path(__file__).resolve().parents[2]
PKG_DIR = ROOT_DIR / "ex_fuzzy"
INNER_PKG_DIR = PKG_DIR / "ex_fuzzy"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PKG_DIR))
sys.path.insert(0, str(INNER_PKG_DIR))

import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.utils as utils
try:
    from ex_fuzzy.conformal import ConformalFuzzyClassifier, evaluate_conformal_coverage
except (ModuleNotFoundError, ImportError):
    from ex_fuzzy import conformal as conformal_module

    ConformalFuzzyClassifier = conformal_module.ConformalFuzzyClassifier
    evaluate_conformal_coverage = conformal_module.evaluate_conformal_coverage


# Training parameters
N_GEN = 20
POP_SIZE = 30

# Model parameters
N_RULES = 15
N_ANTS = 3
N_LV = 3
FUZZY_TYPE = fs.FUZZY_SETS.t1
TOLERANCE = 0.01
ALPHA = 0.10


def main():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # Split into train/calibration/test sets.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.30, random_state=0, stratify=y_train_full
    )

    # Build fuzzy partitions from the training split.
    precomputed_partitions = utils.construct_partitions(
        X_train, FUZZY_TYPE, n_partitions=N_LV
    )

    # Fit + calibrate conformal model.
    conf_clf = ConformalFuzzyClassifier(
        nRules=N_RULES,
        nAnts=N_ANTS,
        fuzzy_type=FUZZY_TYPE,
        tolerance=TOLERANCE,
        linguistic_variables=precomputed_partitions,
        verbose=False,
    )
    conf_clf.fit(
        X_train,
        y_train,
        X_cal,
        y_cal,
        n_gen=N_GEN,
        pop_size=POP_SIZE,
        random_state=0,
    )

    metrics = evaluate_conformal_coverage(conf_clf, X_test, y_test, alpha=ALPHA)
    pred_sets = conf_clf.predict_set(X_test, alpha=ALPHA)
    explained = conf_clf.predict_set_with_rules(X_test[:3], alpha=ALPHA)

    print("Conformal learning demo (Iris)")
    print(f"Expected coverage: {metrics['expected_coverage']:.3f}")
    print(f"Empirical coverage: {metrics['coverage']:.3f}")
    print(f"Average set size: {metrics['avg_set_size']:.3f}")
    print(f"Singleton sets: {metrics['singleton_sets']:.3f}")
    print(f"Empty sets: {metrics['empty_sets']:.3f}")
    coverage_by_class = {int(k): float(v) for k, v in metrics["coverage_by_class"].items()}
    print(f"Coverage by class: {coverage_by_class}")

    print("\nFirst 5 prediction sets:")
    for i, pred_set in enumerate(pred_sets[:5]):
        print(f"Sample {i}: {pred_set}")

    print("\nRule-aware conformal explanations (first 3 samples):")
    for i, item in enumerate(explained):
        print(f"Sample {i} prediction set: {item['prediction_set']}")
        class_p_values = {int(k): float(v) for k, v in item["class_p_values"].items()}
        print(f"Sample {i} class p-values: {class_p_values}")
        top_rules = item["rule_contributions"][:3]
        for contrib in top_rules:
            print(
                "  Rule {rule_index} -> class {class_}, strength={strength:.3f}, "
                "confidence={confidence:.3f}".format(
                    rule_index=contrib["rule_index"],
                    class_=contrib["class"],
                    strength=contrib["firing_strength"],
                    confidence=contrib["rule_confidence"],
                )
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
