============
Installation
============

This guide provides detailed instructions for installing Ex-Fuzzy on different platforms and environments.

Quick Install
=============

The easiest way to install Ex-Fuzzy is using pip:

.. code-block:: bash

    pip install ex-fuzzy

This will install Ex-Fuzzy and all required dependencies.

Requirements
============

System Requirements
-------------------

- **Python**: 3.8 or later

Dependencies
------------

Ex-Fuzzy depends on several well-established scientific Python packages:

**Core Dependencies**:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Package
     - Version
     - Purpose
   * - numpy
     - ≥1.19.0
     - Numerical computations and array operations
   * - pandas
     - ≥1.3.0
     - Data manipulation and analysis
   * - scikit-learn
     - ≥1.0.0
     - Machine learning utilities and metrics
   * - matplotlib
     - ≥3.3.0
     - Plotting and visualization
   * - pymoo
     - ≥0.6.0
     - Multi-objective evolutionary optimization

**Optional Dependencies**:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Package
     - Version
     - Purpose
   * - evox
     - ≥0.8.0
     - GPU-accelerated evolutionary optimization
   * - torch
     - ≥1.9.0
     - PyTorch for GPU acceleration (required by EvoX)
   * - jupyter
     - ≥1.0.0
     - For running notebook examples
   * - seaborn
     - ≥0.11.0
     - Enhanced statistical visualizations
   * - plotly
     - ≥5.0.0
     - Interactive plotting (experimental)

Installation Methods
====================

Method 1: Using pip (Recommended)
----------------------------------

Install the latest stable release from PyPI:

.. code-block:: bash

    pip install ex-fuzzy

To install with GPU support (EvoX backend):

.. code-block:: bash

    pip install ex-fuzzy evox torch

For CUDA-specific PyTorch installation:

.. code-block:: bash

    # CUDA 11.8
    pip install ex-fuzzy evox
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    
    # CUDA 12.1
    pip install ex-fuzzy evox
    pip install torch --index-url https://download.pytorch.org/whl/cu121

To install with optional dependencies:

.. code-block:: bash

    pip install "ex-fuzzy[viz]"      # Enhanced visualization
    pip install "ex-fuzzy[jupyter]"  # Jupyter notebook support
    pip install "ex-fuzzy[gpu]"      # GPU support (EvoX + PyTorch)
    pip install "ex-fuzzy[all]"      # All optional dependencies


Method 2: Development Installation
----------------------------------

For contributors or users who want the latest features:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/fuminides/ex-fuzzy.git



Verification
============

Test your installation by running this simple Python script:

.. code-block:: python

    # test_installation.py
    try:
        import ex_fuzzy
        print(f"✅ Ex-Fuzzy successfully imported!")
        print(f"📦 Version: {ex_fuzzy.__version__}")
        
        # Test basic functionality
        import ex_fuzzy.fuzzy_sets as fs
        import ex_fuzzy.evolutionary_fit as evf
        import numpy as np
        
        # Create a simple dataset
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 100)
        
        # Create a classifier
        classifier = evf.BaseFuzzyRulesClassifier(nRules=3, verbose=False)
        print("Classifier created successfully!")
        
        print("Installation test passed!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please check your installation.")
    except Exception as e:
        print(f"Warning: {e}")
        print("Basic import works, but there might be issues with dependencies.")

Save this as ``test_installation.py`` and run:

.. code-block:: bash

    python test_installation.py

GPU Support
===========

Ex-Fuzzy supports GPU acceleration. GPU-based learning is performed through evox backend instead of Pymoo.

.. note::
    Even when not using Evox, some operations may also be compatible with pytorch. So you can still use solutions learned on CPU in GPU inference.

Prerequisites
-------------

To use GPU acceleration, you need:

1. **CUDA-compatible GPU** (NVIDIA) with compute capability ≥3.5 (maybe ROCm is compatible as well, but this has not been tested).
2. **CUDA Toolkit**


Verifying GPU Installation
---------------------------

Check if GPU is available:

.. code-block:: python

    import torch
    from ex_fuzzy import evolutionary_backends
    
    # Check available backends
    backends = evolutionary_backends.list_available_backends()
    print(f"Available backends: {backends}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU detected.")

Expected Output:

.. code-block:: text

    Available backends: ['pymoo', 'evox']
    ✓ GPU available: NVIDIA Whatever
      CUDA version: 11.8 or similar
      GPU memory: X GB

Troubleshooting GPU Setup
--------------------------

For troubleshooting be sure to check that Evox is working on your computer. exFuzzy should not generate additional problems once Evox is available.

.. note::
   GPU acceleration is most beneficial for large datasets (>10,000 samples) and
   complex rule bases. For small datasets, CPU (PyMoo) may be faster due to
   GPU transfer overhead (and that Pymoo is also well optimized).

Getting Help
============

If you encounter issues during installation:

1. **Check the logs**: Look for specific error messages in the installation output
2. **Check dependencies**: Ensure all required packages are compatible
3. **Ask for help**: Open an issue on `GitHub <https://github.com/fuminides/ex-fuzzy/issues>`_

Next Steps
==========

Once Ex-Fuzzy is installed, check out:

- :doc:`getting-started`: Learn the basics with a quick tutorial
- :doc:`examples/index`: See practical examples and use cases
- :doc:`user-guide/index`: Dive deeper into Ex-Fuzzy's capabilities
