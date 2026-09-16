"""FERL predicts string labels of any length, exactly as it predicts their integer codes."""
import numpy as np
from sklearn.datasets import load_iris

from ex_fuzzy import FERL

NAMES = np.array(['setosa', 'versicolor', 'virginica'])


def test_string_labels_are_predicted_whole():
    X, y = load_iris(return_X_y=True)
    named = FERL(max_rules=8, random_state=0).fit(X, NAMES[y])
    coded = FERL(max_rules=8, random_state=0).fit(X, y)

    prediction = named.predict(X)
    assert set(prediction) <= set(NAMES)          # no truncated labels such as 'versic'
    np.testing.assert_array_equal(prediction, NAMES[coded.predict(X)])
    np.testing.assert_array_equal(named.predict_set(X), coded.predict_set(X))
    assert named.score(X, NAMES[y]) == coded.score(X, y)
