.. _usage:

Getting Started
====================================


The most straightforward way to use Ex-Fuzzy is to fit a fuzzy rule based classifier to a dataset, and then explore the results and the rules obtained.
A couple of examples of this can be found in the "demos" folder.

A brief piece of code that does this case of use is the following::

    import ex_fuzzy.fuzzy_sets as fs
    import ex_fuzzy.evolutionary_fit as GA
    import ex_fuzzy.utils as  utils
    import ex_fuzzy.eval_tools as eval_tools

    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    fl_classifier = GA.BaseFuzzyRulesClassifier(nRules=10, nAnts=4, n_linguist_variables=3,
                                                 fuzzy_type=fs.FUZZY_SETS.t2, tolerance=0.001)
    fl_classifier.fit(X_train, y_train, n_gen=50, pop_size=30)

    eval_tools.eval_fuzzy_model(fl_classifier, X_train, y_train, X_test, y_test, 
                            plot_rules=True, print_rules=True, plot_partitions=True)

This code trains the classifier and also plots the rules, prints them on screen and show the linguistic variables optimized in the process.

In the following, we will explain how the different processes to perform fuzzy inference are automated in this code, and how they can be perfomed manually.

The next step is :ref:`step1`.
