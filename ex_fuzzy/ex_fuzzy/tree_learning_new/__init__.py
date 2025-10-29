"""
FuzzyCART Tree Modules

This package contains the modular components of the FuzzyCART (Fuzzy Classification and Regression Trees) algorithm.
The original monolithic tree_learning.py has been refactored into logical, maintainable modules.

Main Components:
- fuzzy_cart.py: Main FuzzyCART classifier class
- tree_builder.py: Tree construction and splitting algorithms  
- prediction_engine.py: All prediction and inference methods
- tree_structure.py: Tree traversal and node management
- pruning.py: Cost complexity pruning algorithms
- base.py: Shared base classes and utilities
- metrics/: Fuzzy metric functions (CCI, purity, coverage calculations)

For detailed information about each module, see README.md in this directory.
"""

from .fuzzy_cart import FuzzyCART

__all__ = ['FuzzyCART']