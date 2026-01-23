=======
Recipes
=======

This page provides short, copy-paste workflows for common tasks.

Related Guides
==============

- :doc:`../user-guide/results-io`
- :doc:`../user-guide/project-layout`
- :doc:`../user-guide/troubleshooting`

Train, Evaluate, Save, Reload
=============================

.. code-block:: python

    from pathlib import Path
    import pandas as pd
    from ex_fuzzy import eval_tools, persistence, evolutionary_fit as evf
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    classifier = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=4)
    classifier.fit(X_train, y_train, n_gen=50, pop_size=30)

    evaluator = eval_tools.FuzzyEvaluator(classifier)
    metrics = {
        "accuracy": evaluator.get_metric("accuracy_score", X_test, y_test),
        "mcc": evaluator.get_metric("matthews_corrcoef", X_test, y_test),
    }
    pd.Series(metrics).to_csv("reports/metrics.csv")

    Path("reports/rules.txt").write_text(
        classifier.rule_base.print_rules(return_rules=True)
    )

    Path("reports/variables.txt").write_text(
        persistence.save_fuzzy_variables(classifier.lvs)
    )

    # Later: reload rules and variables
    fuzzy_variables = persistence.load_fuzzy_variables(
        Path("reports/variables.txt").read_text()
    )
    rule_base = persistence.load_fuzzy_rules(
        Path("reports/rules.txt").read_text(), fuzzy_variables
    )
    classifier.load_master_rule_base(rule_base)

Quick Rule Visualization Export
===============================

.. code-block:: python

    from ex_fuzzy import eval_tools

    evaluator = eval_tools.FuzzyEvaluator(classifier)
    evaluator.eval_fuzzy_model(
        X_train, y_train, X_test, y_test,
        plot_rules=True,
        export_path="reports/rule_graphs"
    )

Batch Metrics for Multiple Models
=================================

.. code-block:: python

    import pandas as pd
    from ex_fuzzy import eval_tools

    rows = []
    for name, model in models.items():
        evaluator = eval_tools.FuzzyEvaluator(model)
        rows.append(
            {
                "model": name,
                "accuracy": evaluator.get_metric("accuracy_score", X_test, y_test),
                "mcc": evaluator.get_metric("matthews_corrcoef", X_test, y_test),
            }
        )

    pd.DataFrame(rows).to_csv("reports/model_comparison.csv", index=False)

Next Steps
==========

- Save artifacts with :doc:`../user-guide/results-io`
- Adopt the :doc:`../user-guide/project-layout` for consistent outputs
- Use :doc:`../user-guide/troubleshooting` if any step fails
