"""
Fuzzy classification metrics and evaluation functions.

This module provides reusable metrics for evaluating fuzzy classification
performance, including Complete Classification Index (CCI) and fuzzy purity measures.
"""

from .fuzzy_metrics import (
    compute_fuzzy_purity,
    compute_fuzzy_cci,
    _calculate_coverage,
    compute_purity,
    majority_class,
    class_probabilities,
    evaluate_fuzzy_feature_partition
)

__all__ = [
    'compute_fuzzy_purity',
    'compute_fuzzy_cci', 
    '_calculate_coverage',
    'compute_purity',
    'majority_class',
    'class_probabilities',
    'evaluate_fuzzy_feature_partition'
]