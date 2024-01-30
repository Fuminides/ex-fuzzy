.. _persistence:

Persistence
====================================
Rules can be saved and loaded using plain text. The specification for this format is the same the print format of the rules.
We can extract the rules from a model using the ``ex_fuzzy.eval_tools.eval_fuzzy_model`` method, which can can return the rules in string format if the ``return_rules`` parameter is set to ``True``::



    import pandas as pd

    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    import sys

    import ex_fuzzy.fuzzy_sets as fs
    import ex_fuzzy.evolutionary_fit as GA
    import ex_fuzzy.utils as  utils
    import ex_fuzzy.eval_tools as eval_tools
    import ex_fuzzy.persistence as persistence
    import ex_fuzzy.vis_rules as vis_rules

    n_gen = 5
    n_pop = 30
    nRules = 15
    nAnts = 4
    vl = 3
    fz_type_studied = fs.FUZZY_SETS.t1

    # Import some data to play with
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    # We create a FRBC with the precomputed partitions and the specified fuzzy set type, 
    fl_classifier = GA.BaseFuzzyRulesClassifier(nRules=nRules, linguistic_variables=precomputed_partitions, nAnts=nAnts, 
                                                n_linguist_variables=vl, fuzzy_type=fz_type_studied)
    fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=n_pop, checkpoints=1)

    str_rules = eval_tools.eval_fuzzy_model(fl_classifier, X_train, y_train, X_test, y_test, 
                            plot_rules=True, print_rules=True, plot_partitions=True, return_rules=True)

    # Save the rules as a plain text file
    with open('rules_iris_t1.txt', 'w') as f:
        f.write(str_rules)



The rules can be loaded from a file using the ``load_rules`` method of the ``FuzzyModel`` class::

    # Load the rules from a file
    mrule_base = persistence.load_fuzzy_rules(str_rules, precomputed_partitions)

    fl_classifier = GA.FuzzyRulesClassifier(precomputed_rules=mrule_base)

If we already created the ``FuzzyRulesClassifier`` object, we can load the rules using the ``load_master_rule_base`` method::

    fl_classifier.load_master_rule_base(mrule_base)

You can also save the best rulebase found each x steps of the genetic tuning if you set the ``checkpoint`` parameter to that x number of steps.

    