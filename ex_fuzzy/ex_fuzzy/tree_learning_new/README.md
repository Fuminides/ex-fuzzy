# FuzzyCART Tree Modules

This directory contains the modular architecture for the FuzzyCART (Fuzzy Classification and Regression Trees) algorithm. The original monolithic `tree_learning.py` (~2,079 lines) has been refactored into logical, maintainable modules to improve code organization, testing, and future development.

## 📁 Module Structure

```
tree_modules/
├── __init__.py              # Package initialization and exports
├── README.md               # This documentation
├── base.py                 # Base classes and shared utilities  
├── fuzzy_cart.py          # Main FuzzyCART classifier class
├── tree_builder.py        # Tree construction algorithms
├── prediction_engine.py   # Prediction and inference methods
├── tree_structure.py      # Tree navigation and node management
├── pruning.py             # Tree pruning algorithms
└── metrics/               # Fuzzy metrics and mathematical functions
    ├── __init__.py
    └── fuzzy_metrics.py
```

## 🎯 Module Responsibilities

### **fuzzy_cart.py** (~300-400 lines)
**Main FuzzyCART classifier class with public API**

**Key Functions:**
- `FuzzyCART.__init__()` - Constructor with parameters validation
- `FuzzyCART.fit(X, y)` - Train the fuzzy decision tree
- `FuzzyCART.predict(X)` - Predict class labels for samples
- `FuzzyCART.predict_with_path(X)` - Predict with membership paths
- `FuzzyCART.predict_proba(X)` - Predict class probabilities
- `FuzzyCART.predict_all_leaves(X)` - Get all leaf predictions
- `FuzzyCART.predict_all_leaves_matrix(X)` - Matrix format leaf predictions
- `FuzzyCART.print_tree()` - Display tree structure
- `FuzzyCART.get_tree_stats()` - Get tree statistics

**Purpose:** Provides the main scikit-learn compatible interface using composition pattern to integrate all other modules.

### **tree_builder.py** (~500-600 lines)
**Tree construction and node splitting algorithms**

**Key Functions:**
- `TreeBuilder.build_tree(X, y)` - Main tree construction algorithm
- `TreeBuilder.build_root(X, y)` - Initialize root node
- `TreeBuilder.split_node(node, X, y)` - Create child nodes
- `TreeBuilder.get_best_node_split(node, X, y)` - Find optimal split (purity-based)
- `TreeBuilder.get_best_node_split_cci(node, X, y)` - Find optimal split (CCI-based)
- `TreeBuilder.node_purity_checks(node, X, y)` - Evaluate purity improvement
- `TreeBuilder.node_cci_checks(node, X, y)` - Evaluate CCI improvement
- `TreeBuilder.get_best_possible_coverage(X, y)` - Calculate coverage metrics

**Purpose:** Handles all aspects of growing the fuzzy decision tree, including split evaluation, node creation, and stopping criteria.

### **prediction_engine.py** (~400-500 lines)
**All prediction algorithms and fuzzy membership computation**

**Key Functions:**
- `PredictionEngine.predict_all_nodes(X)` - Fuzzy prediction using all nodes
- `PredictionEngine.predict_proba_all_nodes(X)` - Probability prediction using all nodes
- `PredictionEngine.predict_recursive(x, node)` - Recursive tree traversal prediction
- `PredictionEngine.predict_proba_recursive(x, node)` - Recursive probability prediction
- `PredictionEngine.calculate_membership_to_leaf(X, leaf)` - Compute leaf membership
- `PredictionEngine.get_cached_memberships(X)` - Cached fuzzy set evaluations
- `PredictionEngine.clear_all_split_caches()` - Cache management

**Purpose:** Implements the core fuzzy inference engine that evaluates samples against the trained tree using fuzzy membership functions.

### **tree_structure.py** (~300-400 lines)
**Tree traversal, extraction, and node management**

**Key Functions:**
- `TreeStructure.extract_leaves()` - Get all leaf nodes with paths
- `TreeStructure.extract_all_nodes()` - Get all nodes with paths  
- `TreeStructure.get_all_leaf_nodes()` - Dictionary of leaf nodes
- `TreeStructure.find_node_by_name(name)` - Node lookup by name
- `TreeStructure.get_node_path(node)` - Get path from root to node
- `TreeStructure.get_path_to_leaf(leaf)` - Get fuzzy path to leaf
- `TreeStructure.invalidate_leaf_cache()` - Cache invalidation
- `TreeStructure.node_has_children(name)` - Check if node has children
- `TreeStructure.get_node_children_names(name)` - Get children names

**Purpose:** Manages tree structure, navigation, and provides utilities for accessing and manipulating tree nodes and paths.

### **pruning.py** (~300-400 lines)
**Cost complexity pruning and tree optimization**

**Key Functions:**
- `PruningEngine.cost_complexity_pruning(X, y, alpha)` - Main pruning algorithm
- `PruningEngine.fit_with_pruning(X, y, X_val, y_val)` - Training with validation pruning
- `PruningEngine.find_weakest_link(X, y)` - Identify nodes to prune
- `PruningEngine.prune_subtree(node, X, y)` - Remove subtree
- `PruningEngine.calculate_complexity_measure(node, X, y)` - Complexity calculation
- `PruningEngine.calculate_node_impurity(node, X, y)` - Node impurity
- `PruningEngine.calculate_subtree_impurity(node, X, y)` - Subtree impurity
- `PruningEngine.deep_copy_tree()` - Tree backup
- `PruningEngine.restore_tree(backup)` - Tree restoration

**Purpose:** Implements post-pruning techniques to reduce overfitting and improve generalization by removing less useful subtrees.

### **metrics/** (~200-300 lines)
**Fuzzy metrics and mathematical functions**

**Key Functions:**
- `majority_class(y, membership, classes_)` - Find majority class with fuzzy membership
- `class_probabilities(y, membership, classes_)` - Calculate class probability distributions
- `compute_fuzzy_purity(membership, y, threshold)` - Calculate fuzzy purity measure
- `compute_fuzzy_cci(membership, y, threshold)` - Calculate Complete Classification Index
- `_complete_classification_index(y_true, y_pred, membership)` - Core CCI computation
- `_calculate_coverage(membership, threshold)` - Coverage calculation
- `compute_purity(y)` - Standard purity calculation for crisp splits

**Purpose:** Mathematical foundation for fuzzy tree algorithms, providing metrics for split evaluation, node quality assessment, and classification performance measurement.

### **base.py** (~100-200 lines)
**Base classes, interfaces, and shared utilities**

**Key Functions:**
- `TreeComponent` - Abstract base class for tree components
- `TreeComponentMixin` - Mixin for shared tree operations
- `NodeValidator` - Node structure validation utilities
- `CacheManager` - Shared caching mechanisms
- `TreeMetrics` - Common tree evaluation utilities

**Purpose:** Provides common functionality and interfaces shared across all tree modules, ensuring consistency and reducing code duplication.

## 🔄 Module Interactions

```
FuzzyCART (main class)
├── TreeBuilder ──────────┐
├── PredictionEngine ─────┤
├── TreeStructure ────────┼── Uses base.py utilities
├── PruningEngine ────────┤
└── metrics.fuzzy_metrics ┘

TreeBuilder ←→ TreeStructure  (node creation/management)
PredictionEngine ←→ TreeStructure  (tree traversal)
PruningEngine ←→ TreeStructure  (tree modification)
All modules → metrics.fuzzy_metrics  (mathematical functions)
```

## 🚀 Usage Examples

### **Basic Usage (Unchanged Public API)**
```python
from ex_fuzzy.tree_modules import FuzzyCART

# Same API as before - backward compatible
classifier = FuzzyCART(fuzzy_partitions, max_depth=5)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```

### **Advanced Usage with Module Access**
```python
from ex_fuzzy.tree_modules.fuzzy_cart import FuzzyCART

classifier = FuzzyCART(fuzzy_partitions, max_depth=5)
classifier.fit(X_train, y_train)

# Access specific components for advanced operations
tree_stats = classifier.tree_structure.extract_all_nodes()
custom_pruning = classifier.pruning_engine.cost_complexity_pruning(X_val, y_val, alpha=0.01)
cached_memberships = classifier.prediction_engine.get_cached_memberships(X_test)
```

## 🧪 Testing Strategy

Each module should be testable independently:

```python
# Test tree building in isolation
tree_builder = TreeBuilder(fuzzy_partitions)
tree_builder.build_tree(X_train, y_train)

# Test prediction engine with mock tree
prediction_engine = PredictionEngine(mock_tree)
predictions = prediction_engine.predict_all_nodes(X_test)

# Test pruning algorithms
pruning_engine = PruningEngine(trained_tree)
pruned_tree = pruning_engine.cost_complexity_pruning(X_val, y_val)
```

## 🎯 Benefits of This Architecture

### **Maintainability**
- **Single Responsibility**: Each module has one clear purpose
- **Separation of Concerns**: Changes in one module don't affect others
- **Easier Debugging**: Issues are confined to specific modules

### **Extensibility**
- **Pluggable Components**: Easy to swap prediction algorithms
- **New Features**: Add pruning strategies without touching core logic
- **Algorithm Variants**: Implement different tree building approaches

### **Testing**
- **Unit Testing**: Test each component in isolation
- **Mock Dependencies**: Use mock objects for faster testing
- **Integration Testing**: Test component interactions

### **Performance**
- **Optimized Imports**: Only import what you need
- **Caching**: Shared caching mechanisms in base module
- **Memory Efficiency**: Better memory management per component

## 🔧 Development Guidelines

### **Adding New Features**
1. Identify which module the feature belongs to
2. Check if base classes need extension
3. Ensure proper dependency injection
4. Add tests for the new functionality
5. Update this README if needed

### **Modifying Existing Features**
1. Identify affected modules
2. Check for breaking changes in interfaces
3. Update dependent modules if needed
4. Ensure backward compatibility in public API
5. Update tests and documentation

## 📝 Migration Notes

This refactoring maintains **100% backward compatibility**. The original `tree_learning.py` will import from `tree_modules` and re-export `FuzzyCART`, so existing code continues to work without changes.

For new development, prefer importing from `tree_modules` directly:
```python
# New preferred import
from ex_fuzzy.tree_modules import FuzzyCART

# Still works for compatibility
from ex_fuzzy.tree_learning import FuzzyCART
```