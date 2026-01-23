==============
Project Layout
==============

This page suggests a simple, consistent folder layout for projects using Ex-Fuzzy.
It keeps raw data, trained artifacts, and exported visuals separate.

Related Guides
==============

- :doc:`../user-guide/results-io`
- :doc:`../user-guide/recipes`
- :doc:`../user-guide/troubleshooting`

Recommended Structure
=====================

.. code-block:: text

    project/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── notebooks/
    ├── reports/
    │   ├── figures/
    │   ├── rule_graphs/
    │   ├── rules.txt
    │   ├── variables.txt
    │   └── metrics.csv
    ├── src/
    └── README.md

Where to Save What
==================

- `reports/rules.txt`: Text output from `print_rules` or `print_rules_regression`.
- `reports/variables.txt`: Output from `persistence.save_fuzzy_variables`.
- `reports/rule_graphs/`: GEXF exports from `visualize_rulebase`.
- `reports/figures/`: Saved Matplotlib figures (partitions, stability plots, etc.).
- `reports/metrics.csv`: Scalar metrics from `FuzzyEvaluator.get_metric`.

Minimal Save Helpers
====================

.. code-block:: python

    from pathlib import Path
    from ex_fuzzy import persistence

    reports = Path("reports")
    reports.mkdir(exist_ok=True)

    (reports / "rules.txt").write_text(
        classifier.rule_base.print_rules(return_rules=True)
    )
    (reports / "variables.txt").write_text(
        persistence.save_fuzzy_variables(classifier.lvs)
    )

Next Steps
==========

- Export plots and rule graphs in :doc:`../user-guide/results-io`
- Use :doc:`../user-guide/recipes` for full workflows
