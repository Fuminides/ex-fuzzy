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
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 2GB RAM (4GB+ recommended for large datasets)

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

To install with optional dependencies:

.. code-block:: bash

    pip install "ex-fuzzy[viz]"      # Enhanced visualization
    pip install "ex-fuzzy[jupyter]"  # Jupyter notebook support
    pip install "ex-fuzzy[all]"      # All optional dependencies

Method 2: Using conda
---------------------

.. note::
    Conda package is coming soon! For now, use pip even in conda environments.

If you're using conda, you can still install Ex-Fuzzy with pip:

.. code-block:: bash

    conda create -n exfuzzy python=3.9
    conda activate exfuzzy
    pip install ex-fuzzy

Method 3: Development Installation
----------------------------------

For contributors or users who want the latest features:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/your-username/ex-fuzzy.git
    cd ex-fuzzy

    # Install in development mode
    pip install -e .

    # Or with optional dependencies
    pip install -e ".[all]"

This installs Ex-Fuzzy in "editable" mode, so changes to the source code are immediately available.

Method 4: From Source Archive
-----------------------------

Download and install from a source archive:

.. code-block:: bash

    # Download the latest release
    wget https://github.com/your-username/ex-fuzzy/archive/v1.0.0.tar.gz
    tar -xzf v1.0.0.tar.gz
    cd ex-fuzzy-1.0.0

    # Install
    pip install .

Platform-Specific Instructions
===============================

Windows
-------

**Option 1: Using Python from python.org**

1. Download Python 3.8+ from `python.org <https://www.python.org/downloads/>`_
2. During installation, check "Add Python to PATH"
3. Open Command Prompt or PowerShell and run:

.. code-block:: cmd

    pip install ex-fuzzy

**Option 2: Using Anaconda**

1. Download Anaconda from `anaconda.com <https://www.anaconda.com/products/distribution>`_
2. Open Anaconda Prompt and run:

.. code-block:: cmd

    pip install ex-fuzzy

macOS
-----

**Option 1: Using Homebrew**

.. code-block:: bash

    # Install Python if not already installed
    brew install python

    # Install Ex-Fuzzy
    pip3 install ex-fuzzy

**Option 2: Using Anaconda**

1. Download Anaconda for macOS
2. Open Terminal and run:

.. code-block:: bash

    pip install ex-fuzzy

Linux (Ubuntu/Debian)
----------------------

.. code-block:: bash

    # Install Python and pip if not already installed
    sudo apt update
    sudo apt install python3 python3-pip

    # Install Ex-Fuzzy
    pip3 install ex-fuzzy

Linux (CentOS/RHEL/Fedora)
---------------------------

.. code-block:: bash

    # Install Python and pip if not already installed
    sudo yum install python3 python3-pip  # CentOS/RHEL
    # OR
    sudo dnf install python3 python3-pip  # Fedora

    # Install Ex-Fuzzy
    pip3 install ex-fuzzy

Virtual Environments
====================

It's highly recommended to use virtual environments to avoid dependency conflicts:

Using venv (Python built-in)
-----------------------------

.. code-block:: bash

    # Create virtual environment
    python -m venv exfuzzy_env

    # Activate it
    # On Windows:
    exfuzzy_env\\Scripts\\activate
    # On macOS/Linux:
    source exfuzzy_env/bin/activate

    # Install Ex-Fuzzy
    pip install ex-fuzzy

Using conda
------------

.. code-block:: bash

    # Create conda environment
    conda create -n exfuzzy python=3.9
    
    # Activate it
    conda activate exfuzzy
    
    # Install Ex-Fuzzy
    pip install ex-fuzzy

Using pipenv
------------

.. code-block:: bash

    # Install pipenv if not already installed
    pip install pipenv

    # Create environment and install Ex-Fuzzy
    pipenv install ex-fuzzy

    # Activate the environment
    pipenv shell

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
        print("✅ Classifier created successfully!")
        
        print("🎉 Installation test passed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please check your installation.")
    except Exception as e:
        print(f"⚠️  Warning: {e}")
        print("Basic import works, but there might be issues with dependencies.")

Save this as ``test_installation.py`` and run:

.. code-block:: bash

    python test_installation.py

GPU Support
===========

Ex-Fuzzy currently runs on CPU only. GPU acceleration is planned for future releases.

.. note::
    While Ex-Fuzzy doesn't directly use GPUs, some operations may benefit from optimized BLAS libraries like Intel MKL or OpenBLAS, which are automatically used by NumPy when available.

Troubleshooting
===============

Common Issues
-------------

**Issue**: ``pip install ex-fuzzy`` fails with permission errors

**Solution**: Use the ``--user`` flag or a virtual environment:

.. code-block:: bash

    pip install --user ex-fuzzy

**Issue**: Import errors related to missing dependencies

**Solution**: Ensure all dependencies are installed:

.. code-block:: bash

    pip install numpy pandas scikit-learn matplotlib pymoo

**Issue**: Older Python version

**Solution**: Ex-Fuzzy requires Python 3.8+. Upgrade your Python installation:

.. code-block:: bash

    # Check your Python version
    python --version

**Issue**: Installation fails on Apple Silicon Macs

**Solution**: Use Rosetta or install dependencies through conda:

.. code-block:: bash

    # Create conda environment with compatible packages
    conda create -n exfuzzy python=3.9
    conda activate exfuzzy
    conda install numpy pandas scikit-learn matplotlib
    pip install pymoo ex-fuzzy

Performance Optimization
========================

For better performance, consider installing optimized versions of NumPy and SciPy:

Intel-Optimized Packages
-------------------------

.. code-block:: bash

    # Uninstall existing numpy/scipy
    pip uninstall numpy scipy

    # Install Intel-optimized versions
    pip install intel-numpy intel-scipy

Or use conda with Intel MKL:

.. code-block:: bash

    conda install numpy scipy scikit-learn -c intel

Docker Installation
===================

Use our official Docker image for a consistent environment:

.. code-block:: bash

    # Pull the image
    docker pull exfuzzy/ex-fuzzy:latest

    # Run with Jupyter
    docker run -p 8888:8888 exfuzzy/ex-fuzzy:latest

Or build your own:

.. code-block:: dockerfile

    FROM python:3.9-slim

    RUN pip install ex-fuzzy[all]

    WORKDIR /workspace
    CMD ["python"]

Getting Help
============

If you encounter issues during installation:

1. **Check the logs**: Look for specific error messages in the installation output
2. **Update pip**: ``pip install --upgrade pip``
3. **Clear cache**: ``pip cache purge``
4. **Check dependencies**: Ensure all required packages are compatible
5. **Ask for help**: Open an issue on `GitHub <https://github.com/your-username/ex-fuzzy/issues>`_

Next Steps
==========

Once Ex-Fuzzy is installed, check out:

- :doc:`getting-started`: Learn the basics with a quick tutorial
- :doc:`examples/index`: See practical examples and use cases
- :doc:`user-guide/index`: Dive deeper into Ex-Fuzzy's capabilities
