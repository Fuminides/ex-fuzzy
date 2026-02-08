<p align="center">
  <img src="https://github.com/user-attachments/assets/74380868-0bee-4251-b09c-57e8ad65f2e5" width="200" height="200">
</p>

<h1 align="center">Ex-Fuzzy</h1>

<p align="center">
  <i>🚀 A modern, explainable fuzzy logic library for Python</i>
</p>

<p align="center">
  <a href="https://pypi.org/project/ex-fuzzy/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/ex-fuzzy?color=blue&style=flat-square">
  </a>
  <a href="https://pypi.org/project/ex-fuzzy/">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/ex-fuzzy?style=flat-square">
  </a>
  <a href="https://github.com/Fuminides/ex-fuzzy/actions/workflows/tests.yml">
    <img alt="Tests" src="https://github.com/Fuminides/ex-fuzzy/actions/workflows/tests.yml/badge.svg">
  </a>
  <a href="https://codecov.io/gh/Fuminides/ex-fuzzy">
    <img alt="codecov" src="https://codecov.io/gh/Fuminides/ex-fuzzy/branch/main/graph/badge.svg">
  </a>
  <a href="https://github.com/Fuminides/ex-fuzzy/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/Fuminides/ex-fuzzy?style=flat-square">
  </a>
  <a href="https://github.com/Fuminides/ex-fuzzy/stargazers">
    <img alt="GitHub Stars" src="https://img.shields.io/github/stars/Fuminides/ex-fuzzy?style=flat-square">
  </a>
  <a href="https://www.sciencedirect.com/science/article/pii/S0925231224008191">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-Neurocomputing-green?style=flat-square">
  </a>
</p>



---

## 🎯 Overview

**Ex-Fuzzy** is a comprehensive Python library for **explainable artificial intelligence** through fuzzy logic programming. Built with a focus on accessibility and visualization, it enables researchers and practitioners to create interpretable machine learning models using fuzzy association rules.

### Why Ex-Fuzzy?

- 🔍 **Explainable AI**: Create interpretable models that humans can understand. Support for classification and regression problems.
- 📊 **Rich Visualizations**: Beautiful plots and graphs for fuzzy sets and rules.
- 🛠️ **Scikit-learn Compatible**: Familiar API for machine learning practitioners.
- 🚀 **High Performance**: Optimized algorithms with optional GPU support using Evox (https://github.com/EMI-Group/evox).

## ✨ Features


### **Explainable Rule-Based Learning**
- **Fuzzy Association Rules**: For both classification and regression problems with genetic fine-tuning.
- **Out-of-the-box Results**: Complete compatibility with scikit-learn, minimal to none fuzzy knowledge required to obtain good results.
- **Complete Complexity Control**: Number of rules, rule length, linguistic variables, etc. can be specified by the user with strong and soft constrains.
- **Statistical Analysis of Results**: Confidence intervals for all rule quality metrics, repeated experiments for rule robustness.

###  **Complete Rule Base Visualization and Validation**
- **Comprehensive Plots**: Visualize fuzzy sets and rules.
- **Robustness Metrics**: Compute validation of rules, ensure linguistic meaning of fuzzy partitions, robustness metrics for rules and space partitions, reproducible experiments, etc.

###  **Advanced Learning Routines**
- **Multiple Backend Support**: Choose between PyMoo (CPU) and EvoX (GPU-accelerated) backends for evolutionary optimization.
- **Genetic Algorithms**: Rule base optimization supports fine-tuning of different hyperparameters, like tournament size, crossover rate, etc.
- **GPU Genetic Acceleration**: EvoX backend with PyTorch provides significant speedups for large datasets and complex rule bases.
- **Extensible Architecture**: Easy to extend with custom components.

### **Complete Fuzzy Logic Systems Support**
- **Multiple Fuzzy Set Types**: Classic, Interval-Valued Type-2, and General Type-2 fuzzy sets
- **Linguistic Variables**: Automatic generation with quantile-based optimization.

## 🚀 Quick Start

### Installation

Install Ex-Fuzzy using pip:

```bash
# Basic installation (CPU only, PyMoo backend)
pip install ex-fuzzy

# With GPU support (EvoX backend with PyTorch)
pip install ex-fuzzy evox torch
```

### Basic Usage

```python
import numpy as np
from ex_fuzzy import BaseFuzzyRulesClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train fuzzy classifier
classifier = BaseFuzzyRulesClassifier(
    n_rules=15,
    n_antecedents=4,
    fuzzy_type="t1",  # Type-1 fuzzy sets
    backend="pymoo"  # or "evox" for GPU acceleration
)

# Train the model
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)

# Evaluate and visualize
from ex_fuzzy.eval_tools import eval_fuzzy_model
eval_fuzzy_model(classifier, X_train, y_train, X_test, y_test, 
                plot_rules=True, plot_partitions=True)
```

## 📊 Visualizations

Ex-Fuzzy provides beautiful visualizations to understand your fuzzy models:

<p align="center">
  <img src="https://github.com/user-attachments/assets/858ae72b-6504-4173-b81b-b11a3caf802f" height="280" title="Type-1 Fuzzy Sets">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/6ffff71c-49e5-4437-94e3-3b821f799643" height="280" title="Type-2 Fuzzy Sets">
  <img src="https://github.com/Fuminides/ex-fuzzy/assets/12574757/b356a09f-4c66-45c9-8362-ebdbda684669" height="280" title="General Type-2 Fuzzy Sets">
</p>

### 📈 Statistical Analysis

Monitor pattern stability and variable usage across multiple runs:

<p align="center">
  <img src="https://github.com/user-attachments/assets/4e57469d-6cc6-4a9c-a256-dba052a91045" height="300" title="Usage per Class">
  <img src="https://github.com/user-attachments/assets/819f0988-deeb-4c8d-8cca-d8dd75e437f7" height="300" title="Usage per Variable">
</p>

### 🎯 Bootstrap Confidence Intervals

Obtain statistical confidence intervals for your metrics:

<p align="center">
  <img src="https://github.com/user-attachments/assets/4d5d9d77-4ac4-474e-8ac2-6a146085ae53" alt="Bootstrap Analysis" style="border: 2px solid #ddd; border-radius: 8px; padding: 10px;" />
</p>

## ⚡ Performance

### Backend Comparison

Ex-Fuzzy supports two evolutionary optimization backends:

| Backend | Hardware | Best For |
|---------|----------|----------|
| **PyMoo** | CPU | Small datasets (<10K samples), checkpoint support |
| **EvoX** | GPU | Large datasets with high generation counts |

### When to Use Each Backend

**Use PyMoo** when:
- Working with small to medium datasets
- Running on CPU-only environments
- Need checkpoint/resume functionality
- Memory is limited

**Use EvoX** when:
- Have GPU available (CUDA recommended)
- Working with large datasets (>10,000 samples)
- No checkpointing (Evox does not support checkpointing yet)

Both backends automatically batch operations to fit available memory and large datasets are processed in chunks to prevent out-of-memory errors.


## 🛠️ Examples

### 🔬 Interactive Jupyter Notebooks

Try our hands-on examples in Google Colab:

| Topic | Description | Colab Link |
|-------|-------------|------------|
| **Basic Classification** | Introduction to fuzzy classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1nEIcHEH-FqhJWK-ngPew_gqe82n1Dr2v/view?usp=sharing) |
| **Custom Loss Functions** | Advanced optimization techniques | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1ciajhHTK0PACgT2bGdfpcisCL8MRgiHa/view?usp=sharing) |
| **Rule File Loading** | Working with text-based rule files | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1vNAXfQDnLOdTktQ1gyrtEKwjSmNIlSUc/view?usp=sharing) |
| **Advanced Rules** | Using pre-computed rule populations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1jsjCcBDR9ZE-qEOJcCYCHmtNmwdrYvPh/view?usp=sharing) |
| **Temporal Fuzzy Sets** | Time-aware fuzzy reasoning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1J6T44KBIOdY06BbsO8AvE-X3gRohohIR/view?usp=sharing) |
| **Rule Mining** | Automatic rule discovery | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1qWlL-A_B21FpdtplMDHzg1M7r5tjbN6g/view?usp=sharing) |
| **EvoX Backend** | GPU-accelerated training with EvoX | [📓 Notebook](Demos/evox_backend_demo.ipynb) |

### 💻 Code Examples

<details>
<summary><b>🔍 Advanced Rule Mining</b></summary>

```python
from ex_fuzzy.rule_mining import mine_rulebase
from ex_fuzzy.utils import create_fuzzy_variables

# Create fuzzy variables
variables = create_fuzzy_variables(X_train, ['low', 'medium', 'high'])

# Mine rules from data
rules = mine_rulebase(X_train, variables, 
                     support_threshold=0.1, 
                     max_depth=3)

print(f"Discovered {len(rules)} rules")
```
</details>

<details>
<summary><b>📊 Custom Visualization</b></summary>

```python
from ex_fuzzy.vis_rules import visualize_rulebase

# Create custom rule visualization
visualize_rulebase(classifier.rule_base, 
                  export_path="my_rules.png",
                  layout="spring")

# Plot fuzzy variable partitions
classifier.plot_fuzzy_variables()
```
</details>

<details>
<summary><b>🚀 GPU-Accelerated Training (EvoX Backend)</b></summary>

```python
from ex_fuzzy import BaseFuzzyRulesClassifier

# Create classifier with EvoX backend for GPU acceleration
classifier = BaseFuzzyRulesClassifier(
    n_rules=30,
    n_antecedents=4,
    backend='evox',  # Use GPU-accelerated EvoX backend
    verbose=True
)

# Train with GPU acceleration
classifier.fit(X_train, y_train, 
              n_gen=50,
              pop_size=100)

# EvoX provides significant speedups for:
# - Large datasets (>10,000 samples)
# - Complex rule bases (many rules/antecedents)
# - High generation counts
print("Training completed with GPU acceleration!")
```
</details>

<details>
<summary><b>🧪 Bootstrap Analysis</b></summary>

```python
from ex_fuzzy.bootstrapping_test import generate_bootstrap_samples

# Generate bootstrap samples
bootstrap_samples = generate_bootstrap_samples(X_train, y_train, n_samples=100)

# Evaluate model stability
bootstrap_results = []
for X_boot, y_boot in bootstrap_samples:
    classifier_boot = BaseFuzzyRulesClassifier(n_rules=10)
    classifier_boot.fit(X_boot, y_boot)
    accuracy = classifier_boot.score(X_test, y_test)
    bootstrap_results.append(accuracy)

print(f"Bootstrap confidence interval: {np.percentile(bootstrap_results, [2.5, 97.5])}")
```
</details>

## 📚 Documentation

- **📖 [User Guide](https://github.com/Fuminides/ex-fuzzy/wiki)**: Comprehensive tutorials and examples
- **🔧 [API Reference](https://github.com/Fuminides/ex-fuzzy/wiki/API)**: Detailed function and class documentation
- **🚀 [Quick Start Guide](https://github.com/Fuminides/ex-fuzzy/wiki/Quick-Start)**: Get up and running fast
- **📊 [Examples Gallery](https://github.com/Fuminides/ex-fuzzy/tree/main/Demos)**: Real-world use cases

## 🛡️ Requirements

### Core Dependencies
- **Python** >= 3.7
- **NumPy** >= 1.19.0
- **Pandas** >= 1.2.0
- **Matplotlib** >= 3.3.0
- **PyMOO** >= 0.6.0

### Optional Dependencies
- **NetworkX** >= 2.6 (for rule visualization)
- **EvoX** >= 0.8.0 (for GPU-accelerated evolutionary optimization)
- **PyTorch** >= 1.9.0 (required by EvoX for GPU acceleration)
- **Scikit-learn** >= 0.24.0 (for compatibility examples)

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Bug Reports
Found a bug? Please [open an issue](https://github.com/Fuminides/ex-fuzzy/issues) with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information

### Feature Requests
Have an idea? [Submit a feature request](https://github.com/Fuminides/ex-fuzzy/issues) with:
- Clear use case description
- Proposed API design
- Implementation considerations

### 💻 Code Contributions
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with tests
4. Run the test suite: `pytest tests/ -v`
5. Submit a pull request

### 🧪 Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run tests with coverage report
pytest tests/ --cov=ex_fuzzy --cov-report=html

# Run specific test file
pytest tests/test_fuzzy_sets_comprehensive.py -v
```

### 📖 Documentation
Help improve documentation by:
- Adding examples
- Fixing typos
- Improving clarity
- Adding translations

## 📄 License

This project is licensed under the **AGPL v3 License** - see the [LICENSE](LICENSE) file for details.

## 📑 Citation

If you use Ex-Fuzzy in your research, please cite our paper:

```bibtex
@article{fumanalex2024,
  title = {Ex-Fuzzy: A library for symbolic explainable AI through fuzzy logic programming},
  journal = {Neurocomputing},
  pages = {128048},
  year = {2024},
  issn = {0925-2312},
  doi = {10.1016/j.neucom.2024.128048},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231224008191},
  author = {Javier Fumanal-Idocin and Javier Andreu-Perez}
}
```

## 👥 Main Authors

- **[Javier Fumanal-Idocin](https://github.com/Fuminides)** - *Lead Developer*
- **[Javier Andreu-Perez](https://github.com/jandreu)** - *Licensing officer*

## 🌟 Acknowledgments

- Special thanks to all [contributors](https://github.com/Fuminides/ex-fuzzy/graphs/contributors)
- This research has been supported by EU Horizon Europe under the Marie Skłodowska-Curie COFUND grant No 101081327 YUFE4Postdocs.
---

<p align="center">
  <b>⭐ Star us on GitHub if you find Ex-Fuzzy useful!</b><br>
  <a href="https://github.com/Fuminides/ex-fuzzy/stargazers">
    <img src="https://img.shields.io/github/stars/Fuminides/ex-fuzzy?style=social" alt="GitHub Stars">
  </a>
</p>
