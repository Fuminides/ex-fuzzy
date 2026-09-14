# Ex-Fuzzy demos

Run the examples from the repository root after installing the project and its
demo dependencies. Python versions live in `demos_module/`; several examples
also have notebooks in this directory.

## Curated examples

| Example | Python | Notebook | Purpose |
| --- | --- | --- | --- |
| Iris classification | `demos_module/main_iris_demo.py` | `iris_demo.ipynb` | Train and evaluate a basic fuzzy classifier. |
| Advanced classifiers | `demos_module/iris_demo_advanced_classifiers.py` | `iris_demo_advanced_classifiers.ipynb` | Compare the higher-level classifier workflows. |
| Custom loss | `demos_module/iris_demo_custom_loss.py` | `iris_demo_custom_loss.ipynb` | Supply a custom optimization objective. |
| Persistence | `demos_module/iris_demo_persistence.py` | `iris_demo_persistence.ipynb` | Save and restore fuzzy partitions and rules. |
| Precandidate rules | `demos_module/precandidate_rules_demo.py` | `precandidate_rules_demo.ipynb` | Optimize a subset of candidate rules. |
| Regression | `demos_module/regression_demo.py` | `regression_demo.ipynb` | Fit an interval type-2 fuzzy regressor. |
| Temporal occupancy | `demos_module/occupancy_demo_temporal.py` | `occupancy_demo_temporal.ipynb` | Model time-dependent occupancy with temporal fuzzy sets. |
| Conformal learning | `demos_module/conformal_learning_demo.py` | `conformal_learning_demo.ipynb` | Produce calibrated prediction sets and rule-level uncertainty. |
| Pattern stability | `demos_module/iris_demo_stability_report.py` | — | Generate a stability report for repeated rule fits. |
| FERL | `demos_module/ferl_demo.py` | — | Train an evidential fuzzy rule-tree classifier. |
| EvoX backend | `evox_backend_demo.py` | — | Exercise the optional GPU-oriented backend and compare it with PyMoo. |

For example:

```bash
python Demos/demos_module/main_iris_demo.py 2 10
python Demos/evox_backend_demo.py
```

The EvoX example automatically reports whether EvoX and CUDA are available.
Install the optional backend with `pip install -e '.[evox]'` when you want to
exercise GPU optimization.

## Conformal learning

The conformal examples train a fuzzy classifier, calibrate it on a held-out
set, and call `predict_set()` or `predict_set_with_rules()` to produce
set-valued predictions at the requested coverage. See the
[user guide](../docs/source/user-guide/conformal-learning.rst) and
[API reference](../docs/source/api/conformal.rst) for details.
