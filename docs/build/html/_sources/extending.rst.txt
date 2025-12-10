.. _extending:

Extending Ex-Fuzzy
=======================================

Some of the default behaviour/components can be easily extended to support more fuzzy/explainability tools. 

- Using other fuzzy sets: the whole library is programmed using object orientation, so that to extend some methods is to create classes that inherit from the correspondent classes. This is an easy way to implement additional to additional kinds of fuzzy sets. Nevertheless, as long as their membership values are numerical or 2-sized tuples, new fuzzy sets can be easily supported using the existing ``fs.FUZZY_SET`` enum. For example, ``fs.FUZZY_SET.t1`` expects a numerical membership value and  ``fs.FUZZY_SET.t2`` expects a tuple. You can take a look at the ``ex_fuzzy.temporal`` for an example on how to do this.
- In order to use other partitions than those given by the ``ex_fuzzzy.utils`` module, you can directly create the necessary objects through ``fs.FS()`` objects and ``fs.fuzzyVariable()``. 
- You can change the loss function used in the genetic optimization using the ``new_loss`` method of ``ex_fuzzy.evolutionary_fit.BaseFuzzyRulesClassifier``.


