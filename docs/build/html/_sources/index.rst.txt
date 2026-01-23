========================
Ex-Fuzzy Documentation
========================

.. image:: https://img.shields.io/pypi/v/ex-fuzzy.svg
   :target: https://pypi.org/project/ex-fuzzy/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/ex-fuzzy.svg
   :target: https://pypi.org/project/ex-fuzzy/
   :alt: Python versions

.. image:: https://img.shields.io/github/license/fuminides/ex-fuzzy.svg
   :target: https://github.com/fuminides/ex-fuzzy/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/github/stars/fuminides/ex-fuzzy.svg?style=social
   :target: https://github.com/fuminides/ex-fuzzy
   :alt: GitHub stars

**Ex-Fuzzy** is a powerful Python library for explainable fuzzy logic inference and approximate reasoning. 
It provides comprehensive tools for building, training, and analyzing fuzzy rule-based classifiers with 
a special focus on interpretability and explainability.

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: 🚀 Getting Started
        :link: getting-started
        :link-type: doc

        Get up and running with Ex-Fuzzy in minutes. Learn the basics of fuzzy classification
        and regression with practical examples.

    .. grid-item-card:: 📦 Installation
        :link: installation
        :link-type: doc

        Install Ex-Fuzzy and set up optional dependencies for GPU acceleration and plotting.

    .. grid-item-card:: 📖 User Guide
        :link: user-guide/index
        :link-type: doc

        Tutorials for building fuzzy classifiers and regressors, training models, and
        understanding core concepts.

    .. grid-item-card:: 🧪 Examples
        :link: examples/index
        :link-type: doc

        Real-world examples and case studies demonstrating Ex-Fuzzy's capabilities across
        different domains.

Key Features
============

.. grid:: 3
    :gutter: 2

    .. grid-item-card:: 🧠 Explainable AI
        :class-header: border-0

        Generate interpretable fuzzy rules that provide transparent decision-making processes
        for your machine learning models.

    .. grid-item-card:: ⚡ High Performance
        :class-header: border-0

        GPU-accelerated evolutionary optimization with EvoX backend. Optimized implementations 
        with support for both Type-1 and Type-2 fuzzy systems, automatic memory management,
        and 2-10x speedups on large datasets.

    .. grid-item-card:: 📊 Rich Visualizations
        :class-header: border-0

        Built-in plotting capabilities for fuzzy sets, rules, pattern stability analysis,
        and model performance metrics.

    .. grid-item-card:: 🔧 Flexible Architecture
        :class-header: border-0

        Modular design allows easy customization of fuzzy sets, membership functions,
        and inference mechanisms.

    .. grid-item-card:: 📈 Pattern Analysis
        :class-header: border-0

        Advanced tools for analyzing pattern stability, variable importance, and
        rule discovery consistency across multiple runs.

    .. grid-item-card:: 🔬 Research Ready
        :class-header: border-0

        Designed for academic research with comprehensive statistical testing,
        bootstrapping, and experimental validation tools.

Quick Example
=============

Here's a simple example to get you started:

.. code-block:: python

    import ex_fuzzy.evolutionary_fit as evf
    import ex_fuzzy.eval_tools as eval_tools
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # Load and prepare data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0
    )

    # Create and train fuzzy classifier
    classifier = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=4)
    classifier.fit(X_train, y_train, n_gen=50, pop_size=30)

    # Evaluate and visualize
    evaluator = eval_tools.FuzzyEvaluator(classifier)
    evaluator.eval_fuzzy_model(
        X_train, y_train, X_test, y_test,
        plot_rules=True, print_rules=True, plot_partitions=True
    )

Installation
============

Install Ex-Fuzzy using pip:

.. code-block:: bash

    pip install ex-fuzzy

Or install from source:

.. code-block:: bash

    git clone https://github.com/fuminides/ex-fuzzy.git
    cd ex-fuzzy
    pip install -e .


.. toctree::
    :maxdepth: 2
    :caption: Getting Started
    :hidden:

    getting-started

.. toctree::
    :maxdepth: 2
    :caption: Installation
    :hidden:

    installation

.. toctree::
    :maxdepth: 2
    :caption: User Guide
    :hidden:

    user-guide/index

.. toctree::
    :maxdepth: 2
    :caption: Examples
    :hidden:

    examples/index

.. toctree::
    :maxdepth: 2
    :caption: Reference
    :hidden:

    api/index
    evox_backend
    changelog
    contributing
    roadmap

.. toctree::
    :maxdepth: 1
    :caption: Legacy Documentation
    :hidden:

    api
    usage
    step1
    step2
    step3
    step4
    precom
    optimize
    gt2
    tmpfs
    extending
    classifiers
    pattern_stats

Community and Support
=====================

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: 💬 Discussion
        :link: https://github.com/fuminides/ex-fuzzy/discussions
        :link-type: url

        Join our community discussions, ask questions, and share your projects.

    .. grid-item-card:: 🐛 Report Issues
        :link: https://github.com/fuminides/ex-fuzzy/issues
        :link-type: url

        Found a bug or have a feature request? Let us know on GitHub.

    .. grid-item-card:: 📖 Contributing
        :link: contributing
        :link-type: doc

        Help improve Ex-Fuzzy by contributing code, documentation, or examples.

    .. grid-item-card:: 📧 Contact
        :link: mailto:your-email@example.com
        :link-type: url

        Get in touch with the development team for collaboration or support.

Citation
========

If you use Ex-Fuzzy in your research, please cite:

.. code-block:: bibtex

    @article{ex_fuzzy_2023,
        title={Ex-Fuzzy: A Python Library for Explainable Fuzzy Logic Inference},
        author={Fumanal Idocin, Javier},
        journal={Software Impacts},
        year={2023},
        publisher={Elsevier}
    }

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
