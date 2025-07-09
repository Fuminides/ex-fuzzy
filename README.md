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

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-examples">Examples</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="#-contributing">Contributing</a> •
  <a href="#-citation">Citation</a>
</p>

---

## 🎯 Overview

**Ex-Fuzzy** is a comprehensive Python library for **explainable artificial intelligence** through fuzzy logic programming. Built with a focus on accessibility and visualization, it enables researchers and practitioners to create interpretable machine learning models using fuzzy association rules.

### Why Ex-Fuzzy?

- 🔍 **Explainable AI**: Create interpretable models that humans can understand
- 📊 **Rich Visualizations**: Beautiful plots and graphs for fuzzy sets and rules
- 🛠️ **Scikit-learn Compatible**: Familiar API for machine learning practitioners
- 🚀 **High Performance**: Optimized algorithms with optional GPU support
- 📚 **Comprehensive**: Support for classification, regression, and rule mining

## ✨ Features


### **Explainable Rule-Based Learning**
- **Fuzzy Association Rules**: For both classification and regression problems with genetic fine-tuning.
- **Out-of-the-box Results**: Complete compatibility with scikit-learn, minimal to none fuzzy knowledge required to obtain good results.
- **Complete Complexity Control**: Number of rules, rule length, linguistic variables, etc. can be specified by the user with strong and soft constrains.
- **Statistical Analysis of Results**: Confidence intervals for all rule quality metrics, .

###  **Complete Rule Base Visualization and Validation**
- **Comprehensive Plots**: Visualize fuzzy sets and rules.
- **Network Graphs**: Rule visualizations using NetworkX.
- **Robustness Metrics**: Compute validation of rules, ensure linguistic meaning of fuzzy partitions, robustness metrics for rules and space partitions, reproducible experiments, etc.

###  **Advanced Learning Routines**
- **Genetic Algorithms**: Rule base optimization using PyMOO supports fine-tuning of different hyperparameters, like tournament size, crossover rate, etc.
- **Pre-mining and Rule Search**: start with good initial or prior populations, and then refine those results to obtain a good classifier using genetic optimization.
- **Extensible Architecture**: Easy to extend with custom components.

### **Complete Fuzzy Logic Systems Support**
- **Multiple Fuzzy Set Types**: Classic, Interval-Valued Type-2, and General Type-2 fuzzy sets
- **Linguistic Variables**: Automatic generation with quantile-based optimization.
- **Linguistic Hedges**: Natural language modifiers for enhanced expressiveness.
- **Temporal Fuzzy Sets**: Time-aware fuzzy reasoning

## 🚀 Quick Start

### Installation

Install Ex-Fuzzy using pip:

```bash
pip install ex-fuzzy
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
    fuzzy_type="t1"  # Type-1 fuzzy sets
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
  <img src="https://user-images.githubusercontent.com/12574757/210235257-17b22ede-762b-406c-880a-497e06964f17.png" height="280" title="Fuzzy Rule Graph">
  <img src="https://github.com/user-attachments/assets/858ae72b-6504-4173-b81b-b11a3caf802f" height="280" title="Type-1 Fuzzy Sets">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/0daf546a-6f8b-46dd-9d7e-f97242ea5324" height="280" title="Type-2 Fuzzy Sets">
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
- **PyTorch** >= 1.9.0 (for GPU acceleration)
- **Scikit-learn** >= 0.24.0 (for compatibility examples)

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### 🐛 Bug Reports
Found a bug? Please [open an issue](https://github.com/Fuminides/ex-fuzzy/issues) with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information

### 🚀 Feature Requests
Have an idea? [Submit a feature request](https://github.com/Fuminides/ex-fuzzy/issues) with:
- Clear use case description
- Proposed API design
- Implementation considerations

### 💻 Code Contributions
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with tests
4. Submit a pull request

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
- **[Javier Andreu-Perez](https://github.com/javierandreuperez)** - *Co-author*

## 🌟 Acknowledgments

- Special thanks to all [contributors](https://github.com/Fuminides/ex-fuzzy/graphs/contributors)
- This research has been supported by the European Union and the University of Essex under a Marie Sklodowska-Curie YUFE4 postdoc action.

---

<p align="center">
  <b>⭐ Star us on GitHub if you find Ex-Fuzzy useful!</b><br>
  <a href="https://github.com/Fuminides/ex-fuzzy/stargazers">
    <img src="https://img.shields.io/github/stars/Fuminides/ex-fuzzy?style=social" alt="GitHub Stars">
  </a>
</p>
