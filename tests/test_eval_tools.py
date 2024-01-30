import numpy as np
import ex_fuzzy
import sys
sys.path.append('../ex_fuzzy/')

def test_eval_fuzzy_model():
    '''
    Test the eval_fuzzy_model function.

    If it works, it should print the following:
    Accuracy: 0.5 aprox
    Matthews correlation coefficient: 0.0 aprox
    '''
    sample_size = 1000
    sample = np.random.random_sample((sample_size, 5))
    targets = np.random.randint(0, 2, sample_size)

    model = ex_fuzzy.evolutionary_fit.BaseFuzzyRulesClassifier(10, 3, ex_fuzzy.fuzzy_sets.FUZZY_SETS.t1, verbose=False, tolerance=0.0, n_linguist_variables=3)
    model.fit(sample, targets)

    ex_fuzzy.eval_tools.eval_fuzzy_model(model, sample, targets, sample, targets, 
                        plot_rules=False, print_rules=False, plot_partitions=False)

