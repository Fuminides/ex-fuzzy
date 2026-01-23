===============
Troubleshooting
===============

Common issues and quick fixes when working with Ex-Fuzzy.

Related Guides
==============

- :doc:`../user-guide/recipes`
- :doc:`../user-guide/results-io`
- :doc:`../user-guide/validation-visualization`

No Rules Learned
================

Symptoms:
- Rule list is empty after training.
- `print_rules` outputs nothing.

Try:
- Increase `nRules` or `nAnts` and re-train.
- Check that your features are numeric and have variance.
- Use fewer classes or rebalance data if classes are extremely imbalanced.

Plots Not Showing or Empty
==========================

Symptoms:
- Plot windows do not appear.
- Saved figures are blank.

Try:
- Ensure `matplotlib` is installed and you are running in an environment that supports plotting.
- Call `plt.savefig(...)` before `plt.show()` or `plt.close()`.
- For headless environments, set a non-interactive backend (e.g., `Agg`).

Rule Network Export Missing
===========================

Symptoms:
- `export_path` creates no files.
- `visualize_rulebase` throws errors about layout.

Try:
- Ensure the directory exists and is writable.
- Install Graphviz and PyGraphviz for improved layout; otherwise, it falls back to Kamada-Kawai.
- If the rule base is very small, the network may collapse to no edges.

Class Label Mismatch
====================

Symptoms:
- Metrics fail with label errors.
- Predictions appear shifted.

Try:
- Ensure training and test labels use the same encoding.
- If labels are strings, confirm that `classes_names` matches label order.
- Avoid mixing integer and string labels in the same dataset.

Persistence Load Errors
=======================

Symptoms:
- `load_fuzzy_rules` raises `ValueError` or `IndexError`.
- Loaded rules do not match variables.

Try:
- Load variables first using `load_fuzzy_variables`.
- Ensure the variable list matches the rule text (same names and order).
- Do not edit the saved text format unless necessary.

Next Steps
==========

- Follow :doc:`../user-guide/results-io` for save/export examples
- Use :doc:`../user-guide/recipes` to verify a known-good workflow
