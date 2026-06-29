===============
Troubleshooting
===============

This page collects common installation and runtime issues.

Import Errors
=============

If importing Ex-Fuzzy fails after installing from source, verify that the
package was installed from the repository root:

.. code-block:: bash

    pip install -e .

For documentation builds and examples, install the docs extra:

.. code-block:: bash

    pip install -e ".[docs]"

EvoX or JAX Installation
========================

The EvoX backend is optional. Install it only when you need GPU-accelerated
optimization:

.. code-block:: bash

    pip install "ex-fuzzy[evox]"

If JAX cannot find a compatible accelerator, first confirm the CPU backend works:

.. code-block:: python

    from ex_fuzzy import BaseFuzzyRulesClassifier

    clf = BaseFuzzyRulesClassifier(backend="pymoo")

Then check the JAX installation that matches your platform and CUDA version.

Slow Training
=============

Training time grows with the number of rules, antecedents, generations, and
population size. Start with a small run and scale gradually:

.. code-block:: python

    clf.fit(X_train, y_train, n_gen=10, pop_size=20)

For larger datasets, compare ``backend="pymoo"`` and ``backend="evox"`` with the
same split and seed before committing to a backend.

Unexpected Accuracy Differences
===============================

Fuzzy rule optimization is stochastic. Differences can come from the train/test
split, optimizer seed, backend, fuzzy partitions, or population settings. For
published results, run multiple seeds and report the mean and standard
deviation.

Documentation Build Issues
==========================

From the repository root, install the documentation dependencies and rebuild:

.. code-block:: bash

    pip install -e ".[docs]"
    cd docs
    make clean html

The generated HTML is written to ``docs/build/html``.
