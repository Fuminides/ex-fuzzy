==============================
Results, Exports, and Storage
==============================

This page shows practical ways to visualize results and save artifacts produced by Ex-Fuzzy
models. It focuses on saving plots, rule text, fuzzy variables, and metrics in simple files
you can version or share.

Related Guides
==============

- :doc:`../user-guide/recipes`
- :doc:`../user-guide/project-layout`
- :doc:`../user-guide/validation-visualization`

Quick Recipe
============

.. code-block:: python

    from pathlib import Path
    import pandas as pd
    from ex_fuzzy import eval_tools, persistence, vis_rules
    import matplotlib.pyplot as plt

    reports_dir = Path("reports")
    graphs_dir = reports_dir / "rule_graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    evaluator = eval_tools.FuzzyEvaluator(classifier)

    metrics = {
        "accuracy_train": evaluator.get_metric("accuracy_score", X_train, y_train),
        "accuracy_test": evaluator.get_metric("accuracy_score", X_test, y_test),
        "mcc_train": evaluator.get_metric("matthews_corrcoef", X_train, y_train),
        "mcc_test": evaluator.get_metric("matthews_corrcoef", X_test, y_test),
    }
    pd.Series(metrics).to_csv(reports_dir / "metrics.csv")

    rules_text = classifier.rule_base.print_rules(return_rules=True)
    (reports_dir / "rules.txt").write_text(rules_text)

    variables_text = persistence.save_fuzzy_variables(classifier.lvs)
    (reports_dir / "variables.txt").write_text(variables_text)

    for ix, fv in enumerate(classifier.lvs):
        vis_rules.plot_fuzzy_variable(fv)
        plt.savefig(reports_dir / f"fuzzy_variable_{ix}.png", dpi=300, bbox_inches="tight")
        plt.close()

    vis_rules.visualize_rulebase(classifier.rule_base, export_path=str(graphs_dir))

Save Evaluation Output
======================

``FuzzyEvaluator`` exposes metric helpers that return scalars you can store in CSV/JSON.
For a compact summary, call ``get_metric`` for the metrics you care about and persist them
with your data tooling (pandas, csv, or json).

.. code-block:: python

    evaluator = eval_tools.FuzzyEvaluator(classifier)
    accuracy = evaluator.get_metric("accuracy_score", X_test, y_test)
    f1 = evaluator.get_metric("f1_score", X_test, y_test, average="weighted")

    pd.DataFrame([{"accuracy": accuracy, "f1_weighted": f1}]).to_csv(
        "reports/metrics.csv", index=False
    )

Export Rule Visualizations
==========================

To export rule networks, use ``vis_rules.visualize_rulebase`` or the ``export_path``
argument in ``FuzzyEvaluator.eval_fuzzy_model``. The export creates GEXF files that can
be opened in tools like Gephi.

.. code-block:: python

    evaluator.eval_fuzzy_model(
        X_train, y_train, X_test, y_test,
        plot_rules=True,
        export_path="reports/rule_graphs"
    )

If you want static images of fuzzy variables, call ``plot_fuzzy_variable`` and save the
figure with Matplotlib:

.. code-block:: python

    for ix, fv in enumerate(classifier.lvs):
        vis_rules.plot_fuzzy_variable(fv)
        plt.savefig(f"reports/fuzzy_variable_{ix}.png", dpi=300, bbox_inches="tight")
        plt.close()

Persist Rules and Variables
===========================

Rule text can be obtained from the rule base and stored in a plain text file. Fuzzy
variables are serialized with the persistence module. These files can be reloaded later
via :doc:`../persistence`.

.. code-block:: python

    rules_text = classifier.rule_base.print_rules(return_rules=True)
    Path("reports/rules.txt").write_text(rules_text)

    variables_text = persistence.save_fuzzy_variables(classifier.lvs)
    Path("reports/variables.txt").write_text(variables_text)

For regression models, use ``print_rules_regression`` to format numeric consequents:

.. code-block:: python

    rules_text = regressor.rule_base.print_rules_regression(
        return_rules=True, output_name="target"
    )
    Path("reports/rules_regression.txt").write_text(rules_text)

See Also
========

- :doc:`../user-guide/validation-visualization`
- :doc:`../persistence`
- :doc:`../api/eval_tools`
- :doc:`../api/index`

Next Steps
==========

- Continue with :doc:`../user-guide/recipes` for end-to-end workflows
- Review :doc:`../user-guide/project-layout` for suggested output folders
- If exports fail, see :doc:`../user-guide/troubleshooting`
