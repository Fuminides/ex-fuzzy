.. _evox_backend:

=====================
EvoX Backend Guide
=====================

Overview
========

Ex-Fuzzy supports evolutionary optimization through the EvoX backend, using
PyTorch for its population operations. The amount of GPU acceleration depends
on the optimization problem and its fitness evaluator.

Classification uses the same exact objective as PyMoo, and EvoX reaches it
through the same fast paths: the array evaluator, the fit-local fitness and
firing caches, and population batching. EvoX hands over a whole generation at
once, so chromosomes repeated within a generation are also scored only once.

On a CUDA device, classification can additionally score whole generations with
an exact PyTorch implementation of the built-in Type-1 objective, for fixed or
optimized partitions. It reproduces the CPU objective bit for bit, so a GPU
changes how fast a fit runs, not what it finds. Each fit first checks a few
candidates of its first generation against the CPU and stays on the CPU if any
score differs; afterwards, whether the GPU or the CPU scores a generation is
decided by measurement. Custom
losses, Type-2 sets, ``ds_mode=2`` and categorical variables with optimized
partitions keep scoring on the CPU. No extra option is required;
:doc:`user-guide/training-performance` describes every fast path.

**When to use it.** The GPU objective is designed for very expensive fits:
datasets with tens of thousands of samples or more, many features, optimized
partitions, and large populations or long searches. For smaller problems the
default PyMoo backend on a CPU is usually the faster choice. In a pilot with
10,000 samples and 10 features, the GPU made EvoX fits 2–3× faster than EvoX on
the same node's CPU, yet they still took longer than the recorded PyMoo fit of
that workload on a desktop CPU.

Regression retains its separate batched Torch evaluator.

Installation
============

Basic Installation (PyMoo only)
--------------------------------

.. code-block:: bash

   pip install ex-fuzzy

With EvoX Support
-----------------

.. code-block:: bash

   pip install "ex-fuzzy[evox]"

For GPU support, ensure you have CUDA-compatible hardware and drivers installed.

EvoX fits do not import pymoo, so they also run in environments where pymoo is
missing or too old for the PyMoo backend.

Backend Selection
=================

Using PyMoo Backend (Default)
------------------------------

.. code-block:: python

   from ex_fuzzy import BaseFuzzyRulesClassifier
   
   classifier = BaseFuzzyRulesClassifier(
       nRules=30,
       nAnts=4,
       backend='pymoo'  # Explicit, but this is the default
   )
   
   classifier.fit(X_train, y_train)

Using EvoX Backend
------------------

.. code-block:: python

   from ex_fuzzy import BaseFuzzyRulesClassifier
   
   classifier = BaseFuzzyRulesClassifier(
       nRules=30,
       nAnts=4,
       backend='evox'  # Use GPU-accelerated backend
   )
   
   classifier.fit(X_train, y_train, 
                 n_gen=50,
                 pop_size=100)

   # Early stopping defaults: patience=10, min_delta=1e-4

Regression uses the same backend selection. Its crisp and Mamdani objectives
are evaluated as memory-aware PyTorch batches:

.. code-block:: python

   from ex_fuzzy import BaseFuzzyRulesRegressor

   regressor = BaseFuzzyRulesRegressor(
       nRules=30,
       nAnts=4,
       backend='evox'
   )
   regressor.fit(X_train, y_train, n_gen=50, pop_size=100)

   print(regressor.optimization_device_)  # 'cuda' or 'cpu'
   print(regressor.gpu_accelerated_)

Checking Available Backends
----------------------------

.. code-block:: python

   from ex_fuzzy import evolutionary_backends
   
   available = evolutionary_backends.list_available_backends()
   print(f"Available backends: {available}")
   
   # Check GPU availability
   import torch
   if torch.cuda.is_available():
       print(f"GPU: {torch.cuda.get_device_name(0)}")
   else:
       print("No GPU available, EvoX will use CPU")

Performance and memory
======================

Benchmark the same dataset, population size, generation budget and stopping
settings when comparing backends. EvoX and PyMoo run different genetic
algorithms, so their searches differ even with the same seed.

On the CPU, EvoX classification shares PyMoo's fit-local caches and batching.
On synthetic data (10 features, 20 rules, 30 generations, population 40) this
made complete EvoX fits 2.1–4.2× faster than scoring one chromosome at a time,
with identical results. With a population of 200 and 1,000 samples the gain was
2.1× for fixed and 1.1× for optimized partitions, where most offspring are new
chromosomes. These were measured on one shared machine and are not a promise
for your workload; ``benchmarks/benchmark_evox_routes.py`` reproduces them.

The GPU objective is meant for very expensive fits: many samples, many features,
large populations and optimized partitions. In a pilot on 10,000 samples and 10
features (20 rules, population 40, 5 generations), the complete fit took 1.89 s
on a GTX 1080 Ti against 3.85 s on the same node's CPU with fixed partitions,
and 1.66 s on an RTX 2080 against 4.79 s with optimized partitions, with
identical results. In a three-seed campaign on 100,000 samples and 200 features,
the retained route completed five-generation fits in a median 22.8 s on the GPU
against 521.3 s on the same nodes' CPU routes with fixed partitions (22.87×),
and 37.6 s against 729.2 s with optimized partitions (19.38×). All six sampled
device verifications matched bit for bit, all paired searches were identical,
and every device population evaluation used CUDA. Five workloads ran on GTX
1080 Ti nodes and one fixed-partition workload on an RTX 2080. These are large,
synthetic workloads on specific hardware, not a promise for other fits.

The GPU objective splits each generation into chunks
that use at most about a third of the free GPU memory. Each candidate needs
roughly ``8 × samples × (16 × rules + 6 × fuzzy sets)`` bytes with optimized
partitions and less with fixed ones; a problem whose single candidate does not
fit keeps scoring on the CPU.

The regression evaluator batches its Torch fitness operations in the same
memory-aware way. Larger datasets and rule bases need correspondingly more
memory.

Reduce ``pop_size`` and ``nRules`` for smaller trial runs. Increasing the search
budget changes the optimization task and should not be presented as a pure
execution speed comparison. Early stopping remains available through ``patience``
and ``min_delta``.

Examples
========

Basic Comparison
----------------

.. code-block:: python

   import time
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split
   from ex_fuzzy import BaseFuzzyRulesClassifier
   
   # Create a larger dataset
   X, y = make_classification(
       n_samples=50000,
       n_features=10,
       n_informative=8,
       n_classes=3,
       random_state=42
   )
   
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42
   )
   
   # Test PyMoo
   clf_pymoo = BaseFuzzyRulesClassifier(
       nRules=30, nAnts=4, backend='pymoo', verbose=True
   )
   start = time.time()
   clf_pymoo.fit(X_train, y_train, n_gen=30, pop_size=60)
   pymoo_time = time.time() - start
   
   # Test EvoX
   clf_evox = BaseFuzzyRulesClassifier(
       nRules=30, nAnts=4, backend='evox', verbose=True
   )
   start = time.time()
   clf_evox.fit(X_train, y_train, n_gen=30, pop_size=60)
   evox_time = time.time() - start
   
   print(f"PyMoo time: {pymoo_time:.2f}s")
   print(f"EvoX time: {evox_time:.2f}s")
   print(f"Speedup: {pymoo_time/evox_time:.2f}x")

Advanced Configuration
----------------------

.. code-block:: python

   from ex_fuzzy import BaseFuzzyRulesClassifier
   import ex_fuzzy
   
   # Construct custom fuzzy partitions
   partitions = ex_fuzzy.utils.construct_partitions(
       X_train, n_partitions=3
   )
   
   # Create classifier with custom settings
   classifier = BaseFuzzyRulesClassifier(
       nRules=40,
       nAnts=3,
       n_linguistic_variables=3,
       backend='evox',
       linguistic_variables=partitions,
       verbose=True
   )
   
   # Train with custom genetic algorithm parameters
   classifier.fit(
       X_train, y_train,
       n_gen=50,
       pop_size=100,
       sbx_eta=20.0,        # Crossover distribution index
       mutation_eta=20.0,   # Mutation distribution index
       random_state=42
   )
   
   # Evaluate
   accuracy = classifier.score(X_test, y_test)
   print(f"Test accuracy: {accuracy:.4f}")

Complete Demo
-------------

See the complete interactive demo in the repository:

- **Python Script**: ``Demos/evox_backend_demo.py``
- **Jupyter Notebook**: ``Demos/evox_backend_demo.ipynb``

The demo includes:

- Hardware detection and backend availability checking
- Side-by-side comparison of PyMoo vs EvoX
- Performance visualization
- Large dataset testing
- Memory usage analysis

Troubleshooting
===============

EvoX Not Available
------------------

If EvoX backend is not available:

.. code-block:: python

   from ex_fuzzy import evolutionary_backends
   
   available = evolutionary_backends.list_available_backends()
   if 'evox' not in available:
       print('EvoX not installed. Install with: pip install "ex-fuzzy[evox]"')

**Solution**: Install EvoX and PyTorch:

.. code-block:: bash

   pip install "ex-fuzzy[evox]"

GPU Not Detected
----------------

If GPU is not being used:

1. Check CUDA availability:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")

2. Ensure CUDA drivers are installed
3. Verify PyTorch CUDA version matches your CUDA drivers:

.. code-block:: bash

   # For CUDA 11.8
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121

Out of Memory Errors
--------------------

If you encounter out-of-memory errors:

1. **Reduce population size**:

.. code-block:: python

   classifier.fit(X_train, y_train, pop_size=30)  # Instead of 100

2. **Reduce rule count**: Population and GPU scoring hold arrays of ``samples × rules`` values per candidate; fewer rules shrink them.

3. **Use CPU mode for debugging**:

.. code-block:: python

   import torch
   torch.cuda.is_available = lambda: False  # Force CPU mode

4. **Monitor memory usage**: Use ``nvidia-smi`` (GPU) or system monitor (CPU)

Performance Not Improving
--------------------------

If EvoX is not faster than PyMoo:

1. **Dataset too small**: GPU overhead dominates on small datasets (<1000 samples)
2. **CPU bottleneck**: Ensure data transfer to GPU is not the bottleneck
3. **Try larger population**: GPU benefits scale with population size

API Reference
=============

Backend Selection Parameter
---------------------------

The same parameter is available on both estimators:

.. code-block:: python

   BaseFuzzyRulesClassifier(
       ...,
       backend='pymoo'  # or 'evox'
   )

   BaseFuzzyRulesRegressor(
       ...,
       backend='pymoo'  # or 'evox'
   )

**Parameters:**

- ``backend`` : str, default='pymoo'
    Backend for evolutionary optimization. Options:
    
    - ``'pymoo'``: Traditional CPU-based optimization
    - ``'evox'``: GPU-accelerated optimization with PyTorch

Backend Functions
-----------------

.. code-block:: python

   from ex_fuzzy import evolutionary_backends
   
   # List available backends
   available = evolutionary_backends.list_available_backends()
   
   # Returns: List[str], e.g., ['pymoo', 'evox']

See Also
========

- :ref:`ga` - Genetic Algorithm Details
- :doc:`optimize` - Optimization Guide
- :ref:`extending` - Extending Ex-Fuzzy
- `EvoX Documentation <https://evox.readthedocs.io/>`_
- `PyTorch Documentation <https://pytorch.org/docs/>`_
