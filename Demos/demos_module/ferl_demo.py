"""Train FERL and inspect its native evidential predictions on Iris."""

from pathlib import Path
import sys

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Prefer this checkout when the demo is run without an editable installation.
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from ex_fuzzy import FERL


def main():
    dataset = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.25,
        random_state=0,
        stratify=dataset.target,
    )

    model = FERL(
        max_rules=10,
        max_depth=5,
        min_improvement=0.0,
        random_state=0,
    )
    model.fit(X_train, y_train, patience=5)

    predictions = model.predict(X_test)
    betp, belief, plausibility, ignorance = model.predict_credal(X_test)
    prediction_sets = model.predict_set(X_test)

    print("FERL Iris demo")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"Rules: {model.n_rules()}")
    print(f"Mean ignorance: {ignorance.mean():.3f}")
    print(f"Mean prediction-set size: {prediction_sets.sum(axis=1).mean():.2f}")

    print("\nFirst five evidential predictions:")
    for row in range(5):
        labels = dataset.target_names[model.classes_[prediction_sets[row]]]
        print(
            f"sample={row}, predicted={dataset.target_names[predictions[row]]}, "
            f"set={labels.tolist()}, ignorance={ignorance[row]:.3f}, "
            f"pignistic={np.round(betp[row], 3).tolist()}, "
            f"belief={np.round(belief[row], 3).tolist()}, "
            f"plausibility={np.round(plausibility[row], 3).tolist()}"
        )

    print("\nLearned rule tree:")
    model.print_tree()


if __name__ == "__main__":
    main()
