.. _step3:

Optimizing a Fuzzy rule base for a classification problem
=========================================================

--------------------------------------
Fuzzy rule based classifier
--------------------------------------
Usually, in classification inference we compute the matching degree of a sample for each rule in the rule base 
(we refer as "rule base" to both ``ex_fuzzy.rules.RuleBase`` and ``ex_fuzzy.rules.MasterRuleBase`` objects as they are conceptually equivalent).
Then, the predicted class is the consequent class of that rule. In this library, besides the matching degree, we also use a prior, the Dominance Scores,
that are multiplied by the matching degree. 

The Dominance Score is the product of the support and confidence of a rule, so that we rely more on those rules that are more general, and that
cover different patterns than those covered by other rules.

For more info about the dominance scores, you can see [Fach23].

--------------------------------------
Training a fuzzy rule based classifier
--------------------------------------
In order to train a fuzzy rule based classifier, Ex-Fuzzy uses a Genetic algorithm to tune the rules to the 
desired classification task. The interface to use this kind of classifiers is analogous to the standard used
in scikit-learn, so it requires no previous knowledge about fuzzy logic in order to work.

For example, we load the iris dataset and split it in train and test::

    
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

Once the data has been loaded, we just need to create a classifier with the proper parameters, number of rules,
maximum number of antecedents per rule, number of linguist variables per fuzzy variable and tolerance, which will explained
in the evaluation part of this section::


    import ex_fuzzy.evolutionary_fit as GA

    fl_classifier = GA.BaseFuzzyRulesClassifier(nRules=10, nAnts=4, n_linguist_variables=3,
                                                 fuzzy_type=fs.FUZZY_SETS.t2, tolerance=0.001)

These instructions will optimize the linguistic variables in each fuzzy variable, using IV fuzzy sets, using three linguistic variables and ten rules with up to four antecedents.
It is also possible to give a precomputed set of linguistic variables as a list of fuzzy variables. A convenient way to compute
these with easy can be found on the ``utils`` module, by means of the ``ex_fuzzy.utils.construct_partitions`` function.

Once the classifier has been created, the next thing is tranining it. Since we are using a Genetic algorithm, we can specify the number
of generations and the population size::

    fl_classifier.fit(X_train, y_train, n_gen=50, pop_size=30)

And then we can use forward or predict just as with a scikit-learn classifier.

-----------------
Evaluation
-----------------
The genetic algorithm needs a fitness measure to evaluate the quality of each solution. In order to obtain the best possible set of rules,
Ex-Fuzzy uses three different criteria.

1. Matthew Correlation Coefficient: it is a metric that ranges from [-1, 1] that measures the quality of a classification performance. It less sensible to imbalance classification than the standard accuracy.
2. Less antecedents: the less antecedents per rule, the better.
3. Less rules: rule bases with less rules are prefered.
    

[Fach23] Fumanal-Idocin, J., Andreu-Perez, J., Cord, O., Hagras, H., & Bustince, H. (2023). Artxai: Explainable artificial intelligence curates deep representation learning for artistic images using fuzzy techniques. IEEE Transactions on Fuzzy Systems.