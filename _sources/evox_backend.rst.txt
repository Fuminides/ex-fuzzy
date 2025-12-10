.. _evox_backend:

=====================
EvoX Backend Guide
=====================

Overview
========

Ex-Fuzzy now supports GPU-accelerated evolutionary optimization through the EvoX backend. 
This provides significant performance improvements for large datasets and complex rule bases
while maintaining full compatibility with the existing PyMoo backend.

Why EvoX?
=========

The EvoX backend offers several advantages:

- **GPU Acceleration**: Leverages PyTorch for GPU computation, providing 2-10x speedups
- **Large Dataset Support**: Efficient memory management for datasets with millions of samples
- **Automatic Batching**: Intelligent memory management prevents out-of-memory errors
- **Seamless Fallback**: Automatically uses CPU if GPU is unavailable
- **Modern Architecture**: Built on PyTorch ecosystem for easy integration

Installation
============

Basic Installation (PyMoo only)
--------------------------------

.. code-block:: bash

   pip install ex-fuzzy

With EvoX Support
-----------------

.. code-block:: bash

   pip install ex-fuzzy evox torch

For GPU support, ensure you have CUDA-compatible hardware and drivers installed.

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

Performance Comparison
======================

Backend Characteristics
-----------------------

+-----------+----------+------------------------+------------------+
| Backend   | Hardware | Best Use Case          | Speedup          |
+===========+==========+========================+==================+
| PyMoo     | CPU      | Small datasets         | Baseline         |
|           |          | (<10K samples)         |                  |
|           |          |                        |                  |
|           |          | Checkpoint support     |                  |
+-----------+----------+------------------------+------------------+
| EvoX      | GPU/CPU  | Large datasets         | 2-10x faster*    |
|           |          | (>10K samples)         |                  |
|           |          |                        |                  |
|           |          | Complex rule bases     |                  |
+-----------+----------+------------------------+------------------+

\*Speedup varies based on dataset size, rule complexity, and hardware configuration.

When to Use Each Backend
------------------------

**Use PyMoo if:**

- Working with small to medium datasets (<10,000 samples)
- Running on CPU-only systems
- Need checkpoint/resume functionality
- Memory is very limited
- Require maximum stability and compatibility

**Use EvoX if:**

- Have CUDA-compatible GPU available
- Working with large datasets (>10,000 samples)
- Training complex models (many rules, high generations)
- Speed is critical
- GPU memory is sufficient (>4GB recommended)

Performance Tips
================

Memory Management
-----------------

Both backends implement automatic memory management:

**PyMoo Backend:**
  - Automatically batches samples when dataset exceeds available memory
  - Conservative memory budget (30% of available RAM)
  - Processes samples in chunks while maintaining full accuracy

**EvoX Backend:**
  - Automatically batches population evaluation
  - Dynamic batch size based on available GPU/CPU memory
  - Uses 60% of GPU memory or 40% of CPU RAM
  - Rounds batch sizes for optimal performance

Optimization Tips
-----------------

1. **Start with EvoX**: Try EvoX first if you have a GPU - the performance gains are often significant

2. **Monitor Memory**: Watch GPU memory usage with ``nvidia-smi`` for GPU or system monitor for CPU

3. **Adjust Population Size**: Larger populations benefit more from GPU acceleration

.. code-block:: python

   # Smaller population for quick testing
   classifier.fit(X_train, y_train, pop_size=30, n_gen=20)
   
   # Larger population for better results (benefits from GPU)
   classifier.fit(X_train, y_train, pop_size=100, n_gen=50)

4. **Batch Size**: Both backends compute optimal batch sizes automatically, but you can monitor:

.. code-block:: python

   # EvoX will log batch size information in verbose mode
   classifier = BaseFuzzyRulesClassifier(
       nRules=30,
       nAnts=4,
       backend='evox',
       verbose=True  # Shows batch size decisions
   )

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
       print("EvoX not installed. Install with: pip install evox torch")

**Solution**: Install EvoX and PyTorch:

.. code-block:: bash

   pip install evox torch

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

2. **Let automatic batching handle it**: Both backends batch automatically, but if issues persist:

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

.. code-block:: python

   BaseFuzzyRulesClassifier(
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
- :ref:`optimize` - Optimization Guide  
- :ref:`extending` - Extending Ex-Fuzzy
- `EvoX Documentation <https://evox.readthedocs.io/>`_
- `PyTorch Documentation <https://pytorch.org/docs/>`_
