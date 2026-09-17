# Ex-Fuzzy demos

Seven notebooks walk through the library, from a first classifier to
uncertainty quantification. They are executed with their outputs stored, so
they read well on GitHub, and every one of them runs in well under a minute
on a laptop. Open them with Jupyter after `pip install ex-fuzzy` (or
`pip install -e .` from a checkout), or read them online.

| Notebook | What it shows |
| --- | --- |
| [01 Getting started](01_getting_started.ipynb) | Fit, score, read the rules, probabilities, per-sample explanations, partition plots. |
| [02 Scikit-learn integration](02_scikit_learn_integration.ipynb) | Titanic data with categorical columns and missing values: a `Pipeline` with imputation, cross-validation, `GridSearchCV`. |
| [03 Rules and partitions](03_rules_and_partitions.ipynb) | Fuzzy sets and variables by hand, handmade rules, fixed versus optimised partitions, Type-2 sets, validation, inference modes, LaTeX export, saving and loading. |
| [04 Controlling the search](04_controlling_the_search.ipynb) | Budget and early stopping, custom objectives, checkpoints, mined candidate rules, and a comparison of every classifier on the Wine data. |
| [05 Regression](05_regression.ipynb) | `BaseFuzzyRulesRegressor` with crisp and Mamdani consequents on California housing, then inference by hand. |
| [06 Uncertainty](06_uncertainty.ipynb) | Conformal prediction sets with coverage evaluation and rule-level explanations, next to FERL and DeepFERL evidential outputs. |
| [07 Robustness](07_robustness.ipynb) | Pattern stability over repeated fits, and permutation and bootstrap validation of a fitted rule base. |

The Titanic and California housing notebooks download their data through
scikit-learn on the first run and cache it in your home directory; nothing
else leaves the repository.

`evox_backend_demo.py` compares the PyMoo and EvoX backends and reports
whether EvoX and CUDA are available; run it as `python Demos/evox_backend_demo.py`.

To refresh the stored outputs after changing the library, run
`python Demos/run_notebooks.py` from the repository root.
