'''
Module that contains classifiers that uses two step genetic optimization and rule mining based on the support of the candidate rules. 

'''

import numpy as np
from sklearn.base import ClassifierMixin

try:
    from . import fuzzy_sets as fs
    from . import evolutionary_fit as evf
    from . import rule_mining as rm
    from . import utils
    from . import maintenance as mnt
except:
    import fuzzy_sets as fs
    import evolutionary_fit as evf
    import rule_mining as rm
    import utils
    import maintenance as mnt


class RuleMineClassifier(ClassifierMixin):
    """A classifier that works by mining a set of candidate rules with a minimum support, confidence and lift, and then using a genetic algorithm that chooses
    the optimal combination of those rules."""

    def __init__(self, nRules: int = 30, nAnts: int = 4, fuzzy_type: fs.FUZZY_SETS = None, tolerance: float = 0.0,
                verbose=False, n_class: int=None, runner: int=1, linguistic_variables: list[fs.fuzzyVariable]=None) -> None:
        '''
        Inits the optimizer with the corresponding parameters.

        :param nRules: number of rules to optimize.
        :param nAnts: max number of antecedents to use.
        :param fuzzy type: FUZZY_SET enum type in fuzzy_sets module. The kind of fuzzy set used.
        :param tolerance: tolerance for the support/dominance score of the rules.
        :param verbose: if True, prints the progress of the optimization.
        :param n_class: number of classes in the problem. If None (default) the classifier will compute it empirically.
        :param runner: number of threads to use.
        :param linguistic_variables: linguistic variables per antecedent.
        '''
        if mnt.save_usage_flag:
            mnt.usage_data[mnt.usage_categories.Classification]['data_mining'] += 1
        
        self.nAnts = nAnts
        self.fl_classifier = evf.BaseFuzzyRulesClassifier(nRules=nRules, linguistic_variables=linguistic_variables, 
                                            fuzzy_type=fuzzy_type, verbose=verbose, tolerance=tolerance, runner=runner, n_class=n_class)
        self.fuzzy_type = fuzzy_type
        self.tolerance = tolerance


    def fit(self, X: np.array, y: np.array, n_gen:int=30, pop_size:int=50):
        '''
        Trains the model with the given data.

        :param X: samples to train.
        :param y: labels for each sample.
        '''
        fuzzy_vars = utils.construct_partitions(X, self.fuzzy_type)
        candidate_rules = rm.multiclass_mine_rulebase(X, y, fuzzy_vars, self.tolerance, max_depth=self.nAnts)
        self.fl_classifier.fit(X, y, checkpoints=0, candidate_rules=candidate_rules, n_gen=n_gen, pop_size=pop_size)


    def predict(self, X: np.array) -> np.array:
        '''
        Predict for each sample the corresponding class.

        :param X: samples to predict.
        :return: a class for each sample.
        '''
        # Make predictions using the fitted model
        y_pred = self.fl_classifier.predict(X)

        return y_pred
    

    def internal_classifier(self) -> evf.BaseFuzzyRulesClassifier:
        # Returns the classifier that performs the final predictions
        return self.fl_classifier
    


class FuzzyRulesClassifier(ClassifierMixin):
    """A classifier that works by performing a double optimization process. First, it creates a candidate rule base using genetic optimization
    and then uses it as a basis to obtain a better one that satisfies the constrain of antecedents and number of rules."""

    def __init__(self, nRules: int = 30, nAnts: int = 4, fuzzy_type: fs.FUZZY_SETS = None, tolerance: float = 0.0, 
                 verbose=False, n_class: int=None, runner: int=1, expansion_factor:int=1, linguistic_variables: list[fs.fuzzyVariable]=None) -> None:
        '''
        Inits the optimizer with the corresponding parameters.

        :param nRules: number of rules to optimize.
        :param nAnts: max number of antecedents to use.
        :param fuzzy type: FUZZY_SET enum type in fuzzy_sets module. The kind of fuzzy set used.
        :param tolerance: tolerance for the dominance score of the rules.
        :param verbose: if True, prints the progress of the optimization.
        :param n_class: number of classes in the problem. If None (default) the classifier will compute it empirically.
        :param runner: number of threads to use.
        :param expansion_factor: if > 1, it will compute inthe first optimization process n times the nRules parameters. (So that the search space for the second step is bigger)
        :param linguistic_variables: linguistic variables per antecedent.
        '''
        if mnt.save_usage_flag:
            mnt.usage_data[mnt.usage_categories.Classification]['double_go'] += 1


        self.fl_classifier1 = evf.BaseFuzzyRulesClassifier(nRules=nRules* 1, linguistic_variables=linguistic_variables, nAnts=nAnts, # We give this one more number rules so that then the second optimization has a bigger search space
                                            fuzzy_type=fuzzy_type, verbose=verbose, tolerance=tolerance, runner=runner, n_class=n_class) 
        self.fl_classifier2 = evf.BaseFuzzyRulesClassifier(nRules=nRules, linguistic_variables=linguistic_variables, nAnts=nAnts, 
                                            fuzzy_type=fuzzy_type, verbose=verbose, tolerance=tolerance, runner=runner, n_class=n_class)
        self.fuzzy_type = fuzzy_type
        self.tolerance = tolerance


    def fit(self, X: np.array, y: np.array, n_gen:int=30, pop_size:int=50, checkpoints:int=0, n_runner:int=1):
        '''
        Trains the model with the given data.

        :param X: samples to train.
        :param y: labels for each sample.
        :param n_gen: number of generations to compute in the genetic optimization.
        :param pop_size: number of subjects per generation.
        :param checkpoints: if bigger than 0, will save the best subject per x generations in a text file.
        :param n_runner: number of threds to use.
        '''
        self.fl_classifier1.fit(X, y, n_gen, pop_size, checkpoints)
        self.phase1_rules = self.fl_classifier1.rule_base
        self.fl_classifier2.fit(X, y, n_gen, pop_size, checkpoints, initial_rules=self.phase1_rules, n_runner=n_runner)
        

    def predict(self, X: np.array) -> np.array:
        '''
        Predcit for each sample the correspondent class.

        :param X: samples to predict.
        :return: a class for each sample.
        '''
        # Make predictions using the fitted model
        y_pred = self.fl_classifier2.predict(X)

        return y_pred
    

    def internal_classifier(self) -> evf.BaseFuzzyRulesClassifier:
        # Returns the classifier that performs the final predictions
        return self.fl_classifier2
    

class RuleFineTuneClassifier(ClassifierMixin):
    """A classifier that works by mining a set of candidate rules with a minimum support and then uses a two step genetic optimization that chooses
    the optimal combination of those rules and fine tunes them."""

    def __init__(self, nRules: int = 30, nAnts: int = 4, fuzzy_type: fs.FUZZY_SETS = None, tolerance: float = 0.0, 
                 verbose=False, n_class: int=None, runner: int=1, expansion_factor:int=1, linguistic_variables: list[fs.fuzzyVariable]=None) -> None:
        '''
        Inits the optimizer with the corresponding parameters.

        :param nRules: number of rules to optimize.
        :param nAnts: max number of antecedents to use.
        :param fuzzy type: FUZZY_SET enum type in fuzzy_sets module. The kind of fuzzy set used.
        :param tolerance: tolerance for the dominance score of the rules.
        :param verbose: if True, prints the progress of the optimization.
        :param n_class: number of classes in the problem. If None (default) the classifier will compute it empirically.
        :param linguistic_variables: linguistic variables per antecedent.
        '''
        if mnt.save_usage_flag:
            mnt.usage_data[mnt.usage_categories.Classification]['double_go'] += 1
            mnt.usage_data[mnt.usage_categories.Classification]['data_mining'] += 1

        self.fl_classifier1 = evf.BaseFuzzyRulesClassifier(nRules=nRules* expansion_factor, linguistic_variables=linguistic_variables, nAnts=nAnts, # We give this one more number rules so that then the second optimization has a bigger search space
                                            fuzzy_type=fuzzy_type, verbose=verbose, tolerance=tolerance, runner=runner, n_class=n_class) 
        self.fl_classifier2 = evf.BaseFuzzyRulesClassifier(nRules=nRules, linguistic_variables=linguistic_variables, nAnts=nAnts, 
                                            fuzzy_type=fuzzy_type, verbose=verbose, tolerance=tolerance, runner=runner, n_class=n_class)
        self.fuzzy_type = fuzzy_type
        self.tolerance = tolerance


    def fit(self, X: np.array, y: np.array, n_gen:int=30, pop_size:int=50, checkpoints:int=0, n_runner:int=1):
        '''
        Trains the model with the given data.

        :param X: samples to train.
        :param y: labels for each sample.
        :param n_gen: number of generations to compute in the genetic optimization.
        :param pop_size: number of subjects per generation.
        :param checkpoints: if bigger than 0, will save the best subject per x generations in a text file.
        :param n_runner: number of threds to use.
        '''
        candidate_rules = rm.multiclass_mine_rulebase(X, y, self.fl_classifier1.lvs, self.tolerance)

        self.fl_classifier1.fit(X, y, n_gen, pop_size, checkpoints, candidate_rules=candidate_rules)
        self.phase1_rules = self.fl_classifier1.rule_base
        self.fl_classifier2.fit(X, y, n_gen, pop_size, checkpoints, initial_rules=self.phase1_rules, n_runner=n_runner)


    def predict(self, X: np.array) -> np.array:
        '''
        Predcit for each sample the correspondent class.

        :param X: samples to predict.
        :return: a class for each sample.
        '''
        # Make predictions using the fitted model
        y_pred = self.fl_classifier.predict(X)

        return y_pred
    

    def internal_classifier(self) -> evf.BaseFuzzyRulesClassifier:
        # Returns the classifier that performs the final predictions
        return self.fl_classifier2