# Ex-Fuzzy
ex-Fuzzy is a fuzzy toolbox library for Python with special focus in its accesibility to use and visualization of results. In this way, we focus on the ex(-Fuzzy)plainable capacities of approximate reasoning.

Some of the tools available in this library include:

- Support for approximate reasoning using fuzzy association rules, for both classification and regression problems. This includes rule base optimization using genetic algorithms and rule visualization.
- Precomputed and optimized fuzzy variables and their correspondent linguistic variables (i.e low, medium, high).
- Support for various kinds of fuzzy sets, including classic fuzzy sets, IV-fuzzy sets and General Type 2 fuzzy sets.
- Rule mining using support, confidence and lift measures. Customizable genetic optimization of the rule bases parameters.

## Main Characteristics

### Easy to use

ex-Fuzzy is designed to be easy to use. Linguistic variables can be precomputed and optimized without any understading of its implementation. Choosing one kind of fuzzy set only requires to set one flag. 

### Reusable code

Code is designed so that some parts can be easily extendable so that some use cases, like research, can be also supported. The rule base optimization is done using a Genetic Algorithm, but almost any other pymoo search algorithm will do. Fuzzy sets can be extended with ease, just as the kind of partitions, membership functions, etc.

### Sci-py like interface

ex-Fuzzy is built taking into account the actual machine-learing frameworks used in Python. Training amd sing a rule base classifier works exactly as sci-kit learn classifier. Parameters such as the number of rules or antecedents are also built 

### Visualization

Use plots to visualize any kind of fuzzy sets, and use graphs to visualize rules or print them on screen.


<p align="center">
  <img src="https://user-images.githubusercontent.com/12574757/210235257-17b22ede-762b-406c-880a-497e06964f17.png" height="320" title="Fuzzy graph">
  <img src="https://user-images.githubusercontent.com/12574757/210235264-be98fff9-d1b6-4f3b-8b93-b11e0466a48c.png" height="320" title="Type 1 example">
  <img src="https://github.com/Fuminides/ex-fuzzy/assets/12574757/0a3f4508-6ab8-40b5-938b-d89b619c53a3" height="350" title="Type 2 example">
  <img src="https://github.com/Fuminides/ex-fuzzy/assets/12574757/b356a09f-4c66-45c9-8362-ebdbda684669" height="350" title="General Type 2 example">
  
</p>

## Dependencies

- Numpy
- Pandas
- Matplotlib
- Networkx
- Pymoo

## Installation

You can install ex-Fuzzy using pip, from the PyPi repository, with the following command:

`pip install ex-fuzzy`

## Preprint and Citation

You can check our preprint in SSRN in the following link:

[https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4766235
](https://www.sciencedirect.com/science/article/pii/S0925231224008191)

To cite the library please use the preprint until the final paper is accepted:
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

## Contributors
Javier Fumanal Idocin, Javier Andreu-Perez

This project is licensed under the terms of the AGLP v3 license, 2021-2024
