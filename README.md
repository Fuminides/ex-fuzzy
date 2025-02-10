# Ex-Fuzzy
ex-Fuzzy is a fuzzy toolbox library for Python with special focus in its accesibility to use and visualization of results. In this way, we focus on the ex(-Fuzzy)plainable capacities of approximate reasoning.

Some of the tools available in this library include:

- Support for approximate reasoning using fuzzy association rules, for both classification and regression problems. This includes rule base optimization using genetic algorithms and rule visualization.
- Define rule bases as you want: number of rules, antecedents, etc. Everything can be explicitly set by the user.
- Quantile-based and problem-optimized fuzzy variables and their correspondent linguistic variables (i.e low, medium, high). We also support genetic fine tuning of the partitions that keep them interpretable by the user.
- Rule mining using support, confidence and lift measures. Customizable genetic optimization of the rule bases parameters.
- Support for various kinds of fuzzy sets, including classic fuzzy sets, IV-fuzzy sets and General Type 2 fuzzy sets. We also support linguistic hedges.

## Main Characteristics


### Easy to use

ex-Fuzzy is designed to be easy to use. Linguistic variables can be precomputed and optimized without any understading of its implementation. No need to know anything about fuzzy. Choosing one kind of fuzzy set only requires to set one flag. You can also see the demos to see how the basic uses are supported.

### Sci-py like interface

ex-Fuzzy is built taking into account the actual machine-learing frameworks used in Python. Training amd sing a rule base classifier works exactly as sci-kit learn classifier. Parameters such as the number of rules or antecedents are also built 

### Visualization

Use plots to visualize any kind of fuzzy sets, and use graphs to visualize rules or print them on screen.


<p align="center">
  <img src="https://user-images.githubusercontent.com/12574757/210235257-17b22ede-762b-406c-880a-497e06964f17.png" height="320" title="Fuzzy graph">
  <img src="https://github.com/user-attachments/assets/858ae72b-6504-4173-b81b-b11a3caf802f" height="320" title="Type 1 example">
  <img src="https://github.com/user-attachments/assets/0daf546a-6f8b-46dd-9d7e-f97242ea5324" height="350" title="Type 2 example">
  <img src="https://github.com/Fuminides/ex-fuzzy/assets/12574757/b356a09f-4c66-45c9-8362-ebdbda684669" height="350" title="General Type 2 example">
  
</p>

### Testing your patterns and building robust rules

ex-Fuzzy lets you study how reliable are your rules and their variable usage. You can easily repeat the experiments for statistical quantification of the results and then study the patterns obtained.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4e57469d-6cc6-4a9c-a256-dba052a91045" height="360" title="Usage per class">
  <img src="https://github.com/user-attachments/assets/819f0988-deeb-4c8d-8cca-d8dd75e437f7" height="360" title="Usage per variable">
</p>

You can also do bootstrapping to obtain the confidence intervals of all the metrics you are interested in. You can easily discard those that are not good enough for your standards:


<img src="https://github.com/user-attachments/assets/4d5d9d77-4ac4-474e-8ac2-6a146085ae53" alt="Exfuzzy iris example" style="border: 4px solid #ddd; border-radius: 15px; padding: 10px;" />

### Reusable code

Code is designed so that some parts can be easily extendable so that some use cases, like research, can be also supported. The rule base optimization is done using a Genetic Algorithm, but almost any other pymoo search algorithm will do. Fuzzy sets can be extended with ease, just as the kind of partitions, membership functions, etc.

## Try some demos! <img src="https://colab.research.google.com/img/colab_favicon_256px.png" height="40">
You can find them on Google colab:

- [Basic classification demo](https://drive.google.com/file/d/1nEIcHEH-FqhJWK-ngPew_gqe82n1Dr2v/view?usp=sharing)
- [Using a custom loss function](https://drive.google.com/file/d/1ciajhHTK0PACgT2bGdfpcisCL8MRgiHa/view?usp=sharing)
- [Loading a text rule file](https://drive.google.com/file/d/1vNAXfQDnLOdTktQ1gyrtEKwjSmNIlSUc/view?usp=sharing)
- [Using a good set of rules for initial population](https://drive.google.com/file/d/1jsjCcBDR9ZE-qEOJcCYCHmtNmwdrYvPh/view?usp=sharing)
- [Temporal fuzzy sets demo](https://drive.google.com/file/d/1J6T44KBIOdY06BbsO8AvE-X3gRohohIR/view?usp=sharing)
- [Rule mining classifiers](https://drive.google.com/file/d/1qWlL-A_B21FpdtplMDHzg1M7r5tjbN6g/view?usp=sharing)

## Dependencies

- Numpy
- Pandas
- Matplotlib
- Pymoo
- Networkx (optional: only for rule visualization)
- Pytorch (optional: only if you want to run fuzzy inference in a GPU)

## Installation

You can install ex-Fuzzy using pip, from the PyPi repository, with the following command:

`pip install ex-fuzzy`

## Citation

You can check our paper in [Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231224008191).

In case you find exFuzzy useful, please cite it in your papers:
```
@article{fumanalex2024,
title = {Ex-Fuzzy: A library for symbolic explainable AI through fuzzy logic programming},
journal = {Neurocomputing},
pages = {128048},
year = {2024},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.128048},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224008191},
author = {Javier Fumanal-Idocin and Javier Andreu-Perez},
}
```

If you find this work interesting or want to see papers that apply ex-Fuzzy, check the rest of our research as well in Google scholar. If you are interested in the code, a Github star is also greatly appreciated.

## Contributors
Javier Fumanal Idocin, Javier Andreu-Perez

This project is licensed under the terms of the AGLP v3 license, 2021-2024
