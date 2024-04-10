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
  <img src="https://user-images.githubusercontent.com/12574757/210235257-17b22ede-762b-406c-880a-497e06964f17.png" width="350" title="Fuzzy graph">
  <img src="https://user-images.githubusercontent.com/12574757/210235264-be98fff9-d1b6-4f3b-8b93-b11e0466a48c.png" width="350" title="Type 1 example">
  <img src="https://private-user-images.githubusercontent.com/12574757/310877934-89b7184e-5dcc-445f-8b5f-7d4e9388c56f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTI3NTk5NTAsIm5iZiI6MTcxMjc1OTY1MCwicGF0aCI6Ii8xMjU3NDc1Ny8zMTA4Nzc5MzQtODliNzE4NGUtNWRjYy00NDVmLThiNWYtN2Q0ZTkzODhjNTZmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDEwVDE0MzQxMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWU0MzQ2YzhjMmQyM2M2MDlhMzc2MGUwMzUxYzFlNDgyNjk1OTU4NTY3ZGQ1Y2RhZDM2N2MzZDY2MTU2ZGVmMTImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.bnJll8XpKV6o7R6MXmjNB7wJQY8eYyBMANpwkPQjRo0" width="350" title="Type 2 example">
  <img src="https://private-user-images.githubusercontent.com/12574757/310877940-cf4453fe-6f82-4f49-b418-c774729022f7.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTI3NTk5NTAsIm5iZiI6MTcxMjc1OTY1MCwicGF0aCI6Ii8xMjU3NDc1Ny8zMTA4Nzc5NDAtY2Y0NDUzZmUtNmY4Mi00ZjQ5LWI0MTgtYzc3NDcyOTAyMmY3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDEwVDE0MzQxMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTcyNmQwNTFiNzNmYTljN2ZjZjcyYjY2ZTg3NWRjZDMxNmMyNGFmZTBlMGNkOTg4YTdlN2RkODNhYzc0OGE1NmUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.Q_LcDf9RsPHi_QKuqMxkaIQ0dKvx8-dSv0u-KcyRNIA" width="350" title="General Type 2 example">
  
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

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4766235

To cite the library please use the preprint until the final paper is accepted:
```
@article{fumanalex,
  title={Ex-Fuzzy: A Library for Symbolic Explainable AI Through Fuzzy Logic Programming},
  author={Fumanal Idocin, Javier and Andreu-Perez, Javier},
  journal={SSNR}
}
```

## Contributors
Javier Fumanal Idocin, Javier Andreu-Perez

This project is licensed under the terms of the AGLP v3 license, 2021-2024
