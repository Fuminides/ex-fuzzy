This folder contains a series of demos to try different features of the ex-fuzzy library. These are presented in two different formats: jupyter notebooks and python modules, which are stored under the demos_module folder. You dont need to install the library to execute them.

The list of demos is the following:

1. iris_demo: shows a simple classification example. It shows how to train a classifier, how to save checkpoints, how to show the rules in latex tabular format and to save them into a text file.
2. iris_demo_custom_loss: a classification example where the predefined loss is changed by other function.
3. iris_demo_persistence: a classification example where the rules are saved into a file and then imported for another classifier.
4. precandidate_rules_demo: a classification example where we first fit a fuzzy classifier as usual, and then, we look for the optimal subset of those rules.
5. regression_demo: an example of a regression problem using inerval-type 2 fuzzy sets.
6. occupancy_demo_temporal: an example of the use of temporal fuzzy sets.
7. iris_demo_advanced_classifiers: in this example we show an the different training procedures that can be found in classifiers.py file.
8. conformal_learning_demo:
   - Notebook: Demos/conformal_learning_demo.ipynb
   - Python module: Demos/demos_module/conformal_learning_demo.py
   - Description: trains a fuzzy classifier with split conformal calibration and reports coverage, prediction-set size, and rule-aware uncertainty outputs.

Conformal learning support in this repository

The repository includes end-to-end support for conformal learning on top of fuzzy classifiers:
- Core implementation: ex_fuzzy/ex_fuzzy/conformal.py
- User guide: docs/source/user-guide/conformal-learning.rst
- API docs: docs/source/api/conformal.rst
- Example demos:
  - Demos/conformal_learning_demo.ipynb
  - Demos/demos_module/conformal_learning_demo.py

How it works:
1. Train a fuzzy classifier as usual.
2. Calibrate on a held-out calibration set to estimate nonconformity distributions.
3. Predict set-valued outputs with target coverage 1 - alpha using predict_set().
4. Use predict_set_with_rules() for rule-level explainable uncertainty.
