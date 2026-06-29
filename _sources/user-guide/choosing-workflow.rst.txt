===================
Choosing a Workflow
===================

Ex-Fuzzy exposes several modeling paths. Choose the smallest workflow that
matches the uncertainty, interpretability, and performance needs of the task.

Classifier Choices
==================

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Goal
     - Recommended entry point
     - Notes
   * - Standard fuzzy rule classifier
     - :class:`ex_fuzzy.BaseFuzzyRulesClassifier`
     - Good default for most supervised classification tasks.
   * - Mine rules before optimization
     - :class:`ex_fuzzy.RuleMineClassifier`
     - Useful when frequent, high-confidence patterns should seed the model.
   * - Coverage-aware predictions
     - :class:`ex_fuzzy.ConformalFuzzyClassifier`
     - Produces prediction sets after calibration on held-out data.
   * - Lower-level rule mining
     - :mod:`ex_fuzzy.rule_mining`
     - Use when you need direct control over support, confidence, lift, or rule depth.

Fuzzy Set Types
===============

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Type
     - Use when
     - Tradeoff
   * - Type-1
     - You need a simple, fast, interpretable baseline.
     - Least expressive, easiest to inspect.
   * - Interval Type-2
     - Membership boundaries are uncertain or noisy.
     - More expressive, higher computational cost.
   * - General Type-2
     - You are doing advanced uncertainty modeling.
     - Most complex and usually best reserved for research workflows.

Backend Choices
===============

``backend="pymoo"``
  Default CPU backend. Use it for reproducibility, checkpoint support, and
  smaller or medium-sized datasets.

``backend="evox"``
  GPU-oriented backend. Use it for larger runs where JAX/EvoX is installed and
  the extra setup cost is justified.

Minimal Example
===============

.. code-block:: python

    from ex_fuzzy import BaseFuzzyRulesClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0, stratify=y
    )

    clf = BaseFuzzyRulesClassifier(nRules=10, nAnts=4, backend="pymoo")
    clf.fit(X_train, y_train, n_gen=30, pop_size=30)

    print(clf.score(X_test, y_test))

Reproducibility
===============

Evolutionary fitting is stochastic. For experiments, report the data split,
random seeds, population size, number of generations, backend, and fuzzy set
type. For benchmark tables, run multiple seeds and summarize the distribution
instead of relying on one run.
