===================
Choosing a Workflow
===================

Ex-Fuzzy exposes several modeling paths. Choose the smallest workflow that
matches the uncertainty, interpretability, and performance needs of the task.

Estimator Choices
=================

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Goal
     - Recommended entry point
     - Notes
   * - Standard fuzzy rule classifier
     - :class:`ex_fuzzy.BaseFuzzyRulesClassifier`
     - Good default for most supervised classification tasks.
   * - Interpretable fuzzy regressor
     - :class:`ex_fuzzy.BaseFuzzyRulesRegressor`
     - Type-1 regression with crisp Takagi-Sugeno or fuzzy Mamdani consequents.
   * - Mine rules before optimization
     - :class:`ex_fuzzy.RuleMineClassifier`
     - Useful when frequent, high-confidence patterns should seed the model.
   * - Greedy fuzzy rule tree with native evidential uncertainty
     - :class:`ex_fuzzy.FERL`
     - Returns belief, plausibility, ignorance, and prediction sets without a
       calibration split.
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
  GPU-oriented backend powered by EvoX and PyTorch. It runs on CUDA when
  available and otherwise falls back to CPU. Classification and Type-1
  regression objectives are evaluated in memory-aware batches.

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

Regression Example
==================

.. code-block:: python

    from ex_fuzzy import BaseFuzzyRulesRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=500, n_features=5, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    reg = BaseFuzzyRulesRegressor(
        nRules=20,
        nAnts=3,
        consequent_type="crisp",
        backend="evox",
    )
    reg.fit(X_train, y_train, n_gen=30, pop_size=50)

    print(reg.score(X_test, y_test))
    print(reg.optimization_device_)  # "cuda" or "cpu"

Reproducibility
===============

Evolutionary fitting is stochastic. For experiments, report the data split,
random seeds, population size, number of generations, backend, and fuzzy set
type. For benchmark tables, run multiple seeds and summarize the distribution
instead of relying on one run.

FERL's fixed-partition mode is deterministic. Learned-split FERL uses bootstrap
cut estimates; set ``random_state`` and report ``learned_n_boot`` for
reproducible experiments.
