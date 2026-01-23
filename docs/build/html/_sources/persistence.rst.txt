.. _persistence:

Persistence
====================================
Rules and fuzzy partitions can be saved and loaded using plain text. The specification for the rule file format is the same the print format of the rules.
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

    # Build fuzzy partitions from training data
    precomputed_partitions = utils.construct_partitions(X_train, fz_type_studied)

    # We create a FRBC with the precomputed partitions and the specified fuzzy set type, 
    fl_classifier = GA.BaseFuzzyRulesClassifier(nRules=nRules, linguistic_variables=precomputed_partitions, nAnts=nAnts, 
                                                n_linguist_variables=vl, fuzzy_type=fz_type_studied)
    fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=n_pop, checkpoints=1)

    str_rules = eval_tools.eval_fuzzy_model(fl_classifier, X_train, y_train, X_test, y_test, 
                            plot_rules=True, print_rules=True, plot_partitions=True, return_rules=True)

    # Save the rules as a plain text file
    with open('rules_iris_t1.txt', 'w') as f:
        f.write(str_rules)



The rules can be loaded from a file using ``persistence.load_fuzzy_rules`` and assigned to a classifier::

    # Load the rules from a file
    mrule_base = persistence.load_fuzzy_rules(str_rules, precomputed_partitions)

    fl_classifier = GA.BaseFuzzyRulesClassifier(precomputed_rules=mrule_base)

If we already created the ``BaseFuzzyRulesClassifier`` object, we can load the rules using ``load_master_rule_base``::

    fl_classifier.load_master_rule_base(mrule_base)

You can also save the best rulebase found each x steps of the genetic tuning if you set the ``checkpoint`` parameter to that x number of steps.

For the fuzzy partitions, a separate text file is needed. Each file is comprised of a section per variable, introduced as: "$$$ Linguistic variable:", after the :, we introduce the name of the variable. Each of the subsequent lpines contains the info per each of the fuzzy sets used to partitionate that variable. Those lines follow the scheme: Name, Domain, trapezoidal or gaussian membership (trap|gaus), and the parameters of the fuzzy membership. The separator between different fields is always the ,. When using a t2 partition, the parameters of the other membership function appear after the previous one. This is an example for the Iris dataset::

    $$$ Linguistic variable: sepal length (cm)
    Very Low;4.3,7.9;trap;4.3,4.3,5.0,5.36
    Low;4.3,7.9;trap;5.04,5.2,5.6,5.779999999999999
    Medium;4.3,7.9;trap;5.44,5.6,6.05,6.2749999999999995
    High;4.3,7.9;trap;5.85,6.05,6.5,6.68
    Very High;4.3,7.9;trap;6.34,6.7,7.9,7.9

    $$$ Linguistic variable: sepal width (cm)
    Very Low;2.0,4.4;trap;2.0,2.0,2.7,2.88
    Low;2.0,4.4;trap;2.72,2.8,2.95,2.995
    Medium;2.0,4.4;trap;2.91,2.95,3.1,3.1900000000000004
    High;2.0,4.4;trap;3.02,3.1,3.3083333333333345,3.405833333333335
    Very High;2.0,4.4;trap;3.221666666666667,3.4166666666666683,4.4,4.4

You can load this file using the ``load_fuzzy_variables`` function from the persistence module::

    # Load the saved fuzzy partitions from a file
    with open('iris_partitions.txt', 'r') as f:
        loaded_partitions = persistence.load_fuzzy_variables(f.read())
