# Ex-Fuzzy
ex-Fuzzy is a fuzzy toolbox library for Python with special focus in its accesibility to use and visualization of results. In this way, we focus on the ex(-Fuzzy)plainable capacities of approximate reasoning.

Some of the tools available in this library include:

- Support for approximate reasoning using fuzzy association rules, for both classification and regression problems. This includes rule base optimization using genetic algorithms and rule visualization.
- Precomputed and optimized fuzzy variables and their correspondent linguistic variables (i.e low, medium, high).
- Support for various kinds of fuzzy sets, including classic fuzzy sets, IV-fuzzy sets and General Type 2 fuzzy sets.

## Main Characteristics

### Easy to use

ex-Fuzzy is designed to be easy to use. Linguistic variables can be precomputed and optimized without any understading of its implementation. Choosing one kind of fuzzy set only requires to set one flag. 

### Reusable code

Code is designed so that some parts can be easily extendable, so that some use cases, like research, can be also supported. The rule base optimization is done using a Genetic Algorithm, but almost any other pymoo search algorithm will do. Fuzzy sets can be extended with ease, just as the kind of partitions, membership functions, etc.

### Sci-py like interface

ex-Fuzzy is built taking into account the actual machine-learing frameworks used in Python. Training amd sing a rule base classifier works exactly as sci-kit learn classifier. Parameters such as the number of rules or antecedents are also built 

### Visualize the results

Use plots to visualize any kind of fuzzy sets, and use graphs to visualize rules or print them on screen.


<p align="center">
  <img src="https://user-images.githubusercontent.com/12574757/210235257-17b22ede-762b-406c-880a-497e06964f17.png" width="350" title="Fuzzy graph">
  <img src="https://user-images.githubusercontent.com/12574757/210235264-be98fff9-d1b6-4f3b-8b93-b11e0466a48c.png" width="350" title="Type 1 example">
  <img src="https://private-user-images.githubusercontent.com/12574757/310877934-89b7184e-5dcc-445f-8b5f-7d4e9388c56f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDk4MTUxMzAsIm5iZiI6MTcwOTgxNDgzMCwicGF0aCI6Ii8xMjU3NDc1Ny8zMTA4Nzc5MzQtODliNzE4NGUtNWRjYy00NDVmLThiNWYtN2Q0ZTkzODhjNTZmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzA3VDEyMzM1MFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTA0N2JhNDVlM2EwODFkMDNhNzkwN2VjZDI2Y2EyNDk4OTcwOWM3NTgwYTYyZGRhOGIwZGNkYzgwM2JlNWMwM2EmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.7SKQ0PRxT91Q-t6BN4KDk-l6WNjDtTjfhOgq8ih_cvo" width="350" title="Type 2 example">
  <img src="https://private-user-images.githubusercontent.com/12574757/310877940-cf4453fe-6f82-4f49-b418-c774729022f7.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDk4MTUxMzAsIm5iZiI6MTcwOTgxNDgzMCwicGF0aCI6Ii8xMjU3NDc1Ny8zMTA4Nzc5NDAtY2Y0NDUzZmUtNmY4Mi00ZjQ5LWI0MTgtYzc3NDcyOTAyMmY3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzA3VDEyMzM1MFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTdjY2NlYmNkYTBlMWJmOWUyZTViMjBlOTI2Y2Q5MTMxOTZlNzgwMjM4MjM5MmZkZjMyNTM2YTA2MzZlZDYzYWImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.D7dvwn_gkW5SVWBhOkNywlGgUiSzl-HABPcBte1j3gE" width="350" title="General Type 2 example">
  
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

## Home Page

Find all the information about ex-fuzzy in the github repository page:

[https://github.com/Fuminides/ex-fuzzy](https://github.com/Fuminides/ex-fuzzy)

## Documentation

The documentation for ex-Fuzzy is available in: 

[https://fuminides.github.io/ex-fuzzy/](https://fuminides.github.io/ex-fuzzy/)

## Try the demos!

The Demos folder contains a series of demos to try different features of the ex-fuzzy library. These are presented in two different formats: jupyter notebooks and python modules, which are stored under the demos_module folder. You dont need to install the library to execute them.

The list of demos is the following:

1. iris_demo: shows a simple classification example. It shows how to train a classifier, how to save checkpoints, how to show the rules in latex tabular format and to save them into a text file.
2. iris_demo_custom_loss: a classification example where the predefined loss is changed by other function.
3. iris_demo_persistence: a classification example where the rules are saved into a file and then imported for another classifier.
4. precandidate_rules_demo: a classification example where we first fit a fuzzy classifier as usual, and then, we look for the optimal subset of those rules.
5. regression_demo: an example of a regression problem using inerval-type 2 fuzzy sets.
6. occupancy_demo_temporal: an example of the use of temporal fuzzy sets.


## Contributors
Javier Fumanal Idocin, Javier Andreu-Perez

All rights reserved, 2021-2024


