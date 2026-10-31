Examples
========

Welcome to the Ex-Fuzzy examples gallery! Here you'll find practical examples demonstrating how to use Ex-Fuzzy for various machine learning tasks.

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Classification Examples
        :link: classification
        :link-type: doc

        Learn fuzzy classification with practical examples using the Iris dataset and other scenarios.

    .. grid-item-card:: Regression Examples
        :link: regression
        :link-type: doc

        Train interpretable regressors with crisp or fuzzy consequents on CPU or GPU.

    .. grid-item-card:: FERL
        :link: ferl
        :link-type: doc

        Learn a fuzzy rule tree with native belief, plausibility, ignorance, and prediction sets.

.. toctree::
   :maxdepth: 2
   :hidden:

   classification
   regression
   ferl

Working Examples
================

The repository ships seven executed notebooks in ``Demos/``. They render with
their outputs on GitHub and each runs in well under a minute; the Titanic and
California housing notebooks download their data through scikit-learn on the
first run.

.. list-table:: Demo notebooks
   :header-rows: 1
   :widths: 30 70

   * - Notebook
     - What it shows
   * - `01 Getting started <https://github.com/Fuminides/ex-fuzzy/blob/main/Demos/01_getting_started.ipynb>`_
     - Fit, score, read the rules, probabilities, per-sample explanations, partition plots.
   * - `02 Scikit-learn integration <https://github.com/Fuminides/ex-fuzzy/blob/main/Demos/02_scikit_learn_integration.ipynb>`_
     - Titanic data with categorical columns and missing values: a pipeline with imputation, cross-validation, grid search.
   * - `03 Rules and partitions <https://github.com/Fuminides/ex-fuzzy/blob/main/Demos/03_rules_and_partitions.ipynb>`_
     - Fuzzy sets and rules by hand, fixed versus optimised partitions, Type-2 sets, validation, inference modes, LaTeX export, saving and loading.
   * - `04 Controlling the search <https://github.com/Fuminides/ex-fuzzy/blob/main/Demos/04_controlling_the_search.ipynb>`_
     - Budget and early stopping, custom objectives, checkpoints, mined candidate rules, all classifiers compared.
   * - `05 Regression <https://github.com/Fuminides/ex-fuzzy/blob/main/Demos/05_regression.ipynb>`_
     - Crisp and Mamdani consequents on California housing, then inference by hand.
   * - `06 Uncertainty <https://github.com/Fuminides/ex-fuzzy/blob/main/Demos/06_uncertainty.ipynb>`_
     - Conformal prediction sets with coverage evaluation, next to FERL and DeepFERL evidential outputs.
   * - `07 Robustness <https://github.com/Fuminides/ex-fuzzy/blob/main/Demos/07_robustness.ipynb>`_
     - Pattern stability over repeated fits, permutation and bootstrap validation.

The EvoX backend comparison is a script, ``Demos/evox_backend_demo.py``, since
it needs the optional backend. To run the notebooks yourself, open them with
Jupyter after installing the package, or launch them
`in Binder <https://mybinder.org/v2/gh/Fuminides/ex-fuzzy/HEAD?urlpath=%2Fdoc%2Ftree%2FDemos>`_.

Example Categories
==================

**Beginner Examples**
  - Basic iris classification
  - Trainable Type-1 fuzzy regression
  - Simple pattern analysis
  - Visualization basics

**Intermediate Examples**  
  - Custom fuzzy sets
  - Multi-objective optimization
  - Performance comparison

**Advanced Examples**
  - Large-scale datasets
  - Real-time inference
  - Integration with other ML libraries

Contributing Examples
=====================

We welcome contributions of new examples! If you have an interesting use case or application of Ex-Fuzzy:

1. Create a clear, well-documented notebook
2. Include explanations and visualizations  
3. Test with sample data
4. Submit a pull request

See our :doc:`../contributing` guide for more details.
