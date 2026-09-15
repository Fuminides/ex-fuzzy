============
FERL Example
============

This example trains a compact FERL classifier and inspects both point and
native evidential predictions.

.. code-block:: python

   import numpy as np
   from ex_fuzzy import FERL
   from sklearn.datasets import load_iris
   from sklearn.metrics import accuracy_score
   from sklearn.model_selection import train_test_split

   iris = load_iris(as_frame=True)
   X_train, X_test, y_train, y_test = train_test_split(
       iris.data,
       iris.target,
       test_size=0.25,
       random_state=0,
       stratify=iris.target,
   )

   ferl = FERL(
       max_rules=10,
       max_depth=5,
       min_improvement=0.0,
       random_state=0,
   )
   ferl.fit(X_train, y_train, patience=5)

   predictions = ferl.predict(X_test)
   betp, belief, plausibility, ignorance = ferl.predict_credal(X_test)
   prediction_sets = ferl.predict_set(X_test)

   print(f"accuracy={accuracy_score(y_test, predictions):.3f}")
   print(f"rules={ferl.n_rules()}")
   print(f"mean ignorance={ignorance.mean():.3f}")
   print(f"mean set size={prediction_sets.sum(axis=1).mean():.2f}")

   for row in range(5):
       labels = iris.target_names[ferl.classes_[prediction_sets[row]]]
       print(
           row,
           labels.tolist(),
           np.round(betp[row], 3),
           round(float(ignorance[row]), 3),
       )

   ferl.print_tree()

The equivalent runnable script is
``Demos/demos_module/ferl_demo.py`` in the source distribution.
