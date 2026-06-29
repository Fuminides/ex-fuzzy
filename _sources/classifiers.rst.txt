.. _classifiers:

Advanced classifiers
=======================================

Besides ``ex_fuzzy.evolutionary_fit.BaseFuzzyRulesClassifier``, it is possible to use the classifiers in the ``ex_fuzzy.classifiers`` module,
which contains classifiers that take the base classifier and combine it with other techniques. There are two main additions to the base classification
class: rule mining using support, confidence and lift measures; and using a double genetic tuning, so that first a large number of rules can be
considered as potential good rules, and then the second optimization step choose the best combination of them.

The three kind of classifiers are:

1. ``ex_fuzzy.classifiers.RuleMineClassifier``: first mines the rules by checking all the possible combinations of antecedents. It looks for rules that present a minumum of the quality measures, (support, confidence and lift) and then uses them as candidate rules to find an optimal subset of them.
2. ``ex_fuzzy.classifiers.FuzzyRulesClassifier``: performs a double genetic optimization. First, it finds a good rule base and then it uses it as the initial population for another round of genetic optimization.
3. ``ex_fuzzy.classifiers.RuleFineTuneClassifier``: combines both previous approaches. First, searchs for all rules that hold the quality metrics. Then, uses them as candidate rules and finds a good subset of them. Finally, uses that rulebase as initial population for another round of genetic optimization, which gives the final result.

----------------------------
Support, Confidence and lift
----------------------------
1. Support: The definition of support is the percentage of appearance of the antecedent of a rule in the whole dataset. We compute it as the average of the membership values of that antecedent in each sample for the dataset. The membership for each sample to that antecedent is computed  using the minimum t-norm in this case.
2. Confidence: is the ratio between the support of an antecedent for a particular class and for the whole dataset.
3. Lift: is the ratio between the confidence and the expected confidence. It is computed as the ratio between the confidence of the rule and the percentage of samples of the rule class in the dataset.

