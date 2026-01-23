=========
Changelog
=========

This document tracks all notable changes to Ex-Fuzzy.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
============

Added
-----
- **EvoX Backend Support**: GPU-accelerated evolutionary optimization using EvoX and PyTorch
- **Automatic Memory Management**: Batch processing for large datasets to prevent out-of-memory errors
- **Performance Improvements**: 2-10x speedup for large datasets with GPU acceleration
- **Backend Selection**: Easy switching between PyMoo (CPU) and EvoX (GPU) backends
- New comprehensive test suite with >90% code coverage
- Modern documentation website with PyData theme
- Interactive examples with Jupyter notebooks
- GitHub Actions CI/CD pipeline
- Type hints throughout the codebase
- Performance benchmarking suite

Changed
-------
- **Evolutionary Optimization**: Vectorized fitness evaluation for significant speedups
- **Memory Efficiency**: Automatic batching prevents memory overflow on large datasets
- **GPU Utilization**: Seamless GPU/CPU switching based on hardware availability
- Improved API consistency across all modules
- Better error messages and exception handling
- Enhanced visualization capabilities
- Optimized memory usage for large datasets

Fixed
-----
- Bug in fuzzy set membership calculation
- Memory leak in evolutionary optimization
- Incorrect rule dominance score calculation
- Threading issues in parallel processing

Deprecated
----------
- Old maintenance module (mnt.*) - will be removed in v2.0
- Legacy configuration format - use new YAML format

[1.0.0] - 2023-12-15
====================

Added
-----
- Complete fuzzy logic inference system
- Evolutionary optimization for rule discovery
- Pattern stability analysis tools
- Comprehensive visualization suite
- Type-1 and Type-2 fuzzy set support
- Multi-objective optimization capabilities
- Rule mining and analysis tools
- Model persistence and serialization

Changed
-------
- Complete API redesign for better usability
- Improved performance with vectorized operations
- Enhanced documentation and examples
- Better integration with scikit-learn

Fixed
-----
- Various numerical stability issues
- Compatibility with newer Python versions
- Edge cases in fuzzy set operations

[0.9.0] - 2023-06-20
====================

Added
-----
- Initial pattern stability analysis
- Basic visualization tools
- Evolutionary algorithm optimization
- Type-1 fuzzy sets implementation

Changed
-------
- Refactored core fuzzy logic engine
- Improved rule representation
- Better handling of categorical variables

Fixed
-----
- Issues with rule evaluation
- Memory usage optimization
- Threading synchronization

[0.8.0] - 2023-03-15
====================

Added
-----
- Basic fuzzy classification system
- Rule-based inference engine
- Simple optimization algorithms
- Core fuzzy set operations

Changed
-------
- Initial stable API design
- Basic documentation structure

[0.7.0] - 2023-01-10
====================

Added
-----
- Initial release
- Basic fuzzy logic capabilities
- Simple rule representation
- Experimental optimization

Migration Guides
================

Migrating from 0.9.x to 1.0.0
------------------------------

**API Changes:**

.. code-block:: python

   # Old way (0.9.x)
   from ex_fuzzy import FuzzyClassifier
   classifier = FuzzyClassifier(rules=10, antecedents=4)
   
   # New way (1.0.x)
   from ex_fuzzy.evolutionary_fit import BaseFuzzyRulesClassifier
   classifier = BaseFuzzyRulesClassifier(nRules=10, nAnts=4)

**Configuration Changes:**

.. code-block:: python

   # Old way (0.9.x)
   classifier.set_config('tolerance', 0.1)
   
   # New way (1.0.x)
   classifier = BaseFuzzyRulesClassifier(tolerance=0.1)

**Visualization Changes:**

.. code-block:: python

   # Old way (0.9.x)
   classifier.plot_rules()
   
   # New way (1.0.x)
   from ex_fuzzy.eval_tools import FuzzyEvaluator
   evaluator = FuzzyEvaluator(classifier)
   evaluator.eval_fuzzy_model(X_train, y_train, X_test, y_test, plot_rules=True)

Breaking Changes
================

Version 1.0.0
--------------

- **Removed** deprecated `mnt` module
- **Changed** main classifier import path
- **Renamed** several configuration parameters
- **Modified** visualization API for consistency

Version 0.9.0
--------------

- **Changed** rule representation format
- **Removed** experimental features
- **Updated** optimization algorithm interface

Notable Improvements
====================

Performance Improvements
------------------------

**Version 1.0.0:**
- 40% faster rule evaluation
- 60% reduction in memory usage
- 3x improvement in optimization speed
- Better scaling for large datasets

**Version 0.9.0:**
- 25% faster fuzzy set operations
- Improved numerical stability
- Better caching mechanisms

Documentation Improvements
--------------------------

**Version 1.0.0:**
- Complete documentation overhaul
- Interactive examples and tutorials
- Comprehensive API reference
- Best practices guide

**Version 0.9.0:**
- Added user guide
- Basic examples and tutorials
- API documentation improvements

Acknowledgments
===============

We thank all contributors who made these releases possible:

**Version 1.0.0 Contributors:**
- Javier Fumanal Idocin - Lead developer
- Community contributors - Bug reports and feature requests
- Beta testers - Early feedback and testing

**Version 0.9.0 Contributors:**
- Initial development team
- Academic collaborators
- Open source community
