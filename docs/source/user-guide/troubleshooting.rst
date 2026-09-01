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

EvoX, PyTorch, or CUDA Installation
===================================

The EvoX backend is optional. Install it only when you need GPU-accelerated
optimization:

.. code-block:: bash

    pip install "ex-fuzzy[evox]"

If PyTorch cannot find a compatible CUDA device, first confirm that the default
CPU backend works:

.. code-block:: python

    from ex_fuzzy import BaseFuzzyRulesClassifier, BaseFuzzyRulesRegressor

    clf = BaseFuzzyRulesClassifier(backend="pymoo")
    reg = BaseFuzzyRulesRegressor(backend="pymoo")

Then install the CUDA-specific PyTorch wheel recommended by the PyTorch project
for your platform and driver. After an EvoX fit, inspect the actual device:

.. code-block:: python

    reg = BaseFuzzyRulesRegressor(backend="evox")
    reg.fit(X_train, y_train, n_gen=10, pop_size=20)

    print(reg.optimization_device_)  # "cuda" or "cpu"
    print(reg.gpu_accelerated_)      # True only when CUDA was used

An EvoX run on ``"cpu"`` is valid; it means CUDA was not available to
PyTorch. Both crisp and fuzzy regression consequents use the same device.

Slow Training
=============

Training time grows with the number of rules, antecedents, generations, and
population size. Start with a small run and scale gradually:

.. code-block:: python

    estimator.fit(X_train, y_train, n_gen=10, pop_size=20)

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
