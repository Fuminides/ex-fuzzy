'''
Expansion of the base fuzzy sets, adding temporal fuzzy sets.

Contains functions to model the temporal fuzzy sets, temporal fuzzy variables and temporal rule bases.
It also contains functions to evaluate the fuzzy rulebases obtained.

'''
import enum

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.variable import Integer
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization

try:
    from . import fuzzy_sets as fs
    from . import maintenance as mnt
    from . import rules as rl
    from . import evolutionary_fit as evf
    from . import vis_rules
    from . import eval_rules as evr
except:
    import fuzzy_sets as fs
    import maintenance as mnt
    import rules as rl
    import evolutionary_fit as evf
    import vis_rules
    import eval_rules as evr


TMP_FUZZY_SETS = enum.Enum(
    "NEW_FUZZY_SETS",
    ['temporal', 'temporal_t2', 'temporal_gt2']
)
NEW_FUZZY_SETS = enum.Enum(
    "FUZZY_SETS",
    [(es.name, es.value) for es in fs.FUZZY_SETS] + [(es.name, es.name) for es in TMP_FUZZY_SETS]
)

fs.FUZZY_SETS = NEW_FUZZY_SETS
### DEFINE THE FUZZY SET ####
class temporalFS(fs.FS):
    '''
    Class to implement temporal fuzzy sets.
    '''

    def __init__(self, std_fuzzy_set: fs.FS, conditional_variable: np.array) -> None:
        '''
        Creates a temporal fuzzy set.

        :param std_fuzzy_set: FS. Standard fuzzy set that contains the non-temporal aware memberhsip function.
        :param conditional_variable: np.array. The variable that expresses the different temporal moments. Shape (time discrete moments, ).
        '''
        if mnt.save_usage_flag:
            mnt.usage_data[mnt.usage_categories.FuzzySets][self.type().name] += 1
            
        self.std_set = std_fuzzy_set
        self.tmp_function = conditional_variable
        self.time = None
        self.name = std_fuzzy_set.name

        if std_fuzzy_set.type() == fs.FUZZY_SETS.t1:
            self.membership_parameters = std_fuzzy_set.membership_parameters
        elif std_fuzzy_set.type() == fs.FUZZY_SETS.t2:
            self.secondMF_upper = std_fuzzy_set.secondMF_upper
            self.secondMF_lower = std_fuzzy_set.secondMF_lower
        elif std_fuzzy_set.type() == fs.FUZZY_SETS.gt2:
            self.secondary_memberships = std_fuzzy_set.secondary_memberships
            self.alpha_cuts = std_fuzzy_set.alpha_cuts


    def membership(self, input: np.array, time: int=None) -> np.array:
        '''
        Computes the membership of each sample and in each time for the fs.

        :param input: array temporal_time x samples.
        :time: int. Time moment to compute the membership. If none, looks for a fixed time
        :return: array temporal_time x samples.
        '''
        if time is None:
            assert self.time is not None, 'Temporal fuzzy set has no fixed time. Please, fix a time or provide a time to compute the membership.'
            time = self.time

        return self.std_set.membership(input) * self.tmp_function[time]


    def type(self) -> fs.FUZZY_SETS:
        '''
        Returns the type of the fuzzy set. (temporal)
        '''
        return fs.FUZZY_SETS.temporal
    

    def inside_type(self) -> fs.FUZZY_SETS:
        '''
        Returns the type of the og fuzzy set computed before the time dependency.
        '''
        return self.std_set.type()
    

    def fix_time(self, time: int) -> None:
        '''
        Fixes the time of the temporal fuzzy set.

        :param time: int. Time moment to fix.
        :return: FS. Fuzzy set with the fixed time.
        '''
        self.time = time



#### DEFINE THE FUZZY VARIABLE ####
class temporalFuzzyVariable(fs.fuzzyVariable):
    '''
    Class to implement a temporal fuzzy variable.
    '''

    def __init__(self, name: str, fuzzy_sets: list[temporalFS]) -> None:
        '''
        Creates a temporal fuzzy variable.

        :param str: name of the variable.
        :param fuzzy_sets: list of the fuzzy sets pre-time dependencies.
        '''
        self.linguistic_variables = []
        self.name = name
        self.time = None
        for ix, fs in enumerate(fuzzy_sets):
            self.linguistic_variables.append(fs)

        self.fs_type = self.linguistic_variables[0].type()


    def fix_time(self, time: int) -> None:
        '''
        Fixes the time of the temporal fuzzy variable.

        :param time: int. Time moment to fix.
        '''
        self.time = time


    def compute_memberships(self, x: np.array, time: int=None) -> dict:
        '''
        Computes the membership to each of the FS in the fuzzy variables.

        :param x: numeric value or array. Computes the membership to each of the IVFS in the fuzzy variables.
        :param time: int. Time moment to compute the membership.
        :return: list of floats. Membership to each of the FS in the fuzzy variables.
        '''
        if time is None:
            assert self.time is not None, 'Temporal fuzzy variable has no fixed time. Please, fix a time or provide a time to compute the membership.'
            time = self.time

        res = []

        for fuzzy_set in self.linguistic_variables:
            res.append(fuzzy_set.membership(x, time))

        return res
    

    def n_time_moments(self) -> int:
        '''
        Returns the number of time moments of the temporal fuzzy variable.

        :return: int. Number of time moments of the temporal fuzzy variable.
        '''
        return self.linguistic_variables[0].tmp_function.shape[0]


#### DEFINE THE RULE BASE WITH TEMPORAL DEPENDENCIES ####
class temporalMasterRuleBase(rl.MasterRuleBase):
    '''
    This class is a temporal extension of the MasterRuleBase class. It includes a list of
    rule bases for each time step.
    '''
    def __init__(self, rule_base: list[rl.MasterRuleBase], time_step_names: list[str]=None):
        '''
        Constructor of the temporalMasterRuleBase class.

        :param rule_base: list of rule bases.
        :param time_steps: number of time steps.
        '''
        super().__init__(rule_base)
        self.time_steps = np.arange(len(rule_base))
        self.time_mrule_bases = rule_base

        if time_step_names is None:
            self.time_step_names = [str(x) for x in self.time_steps]
        else:
            self.time_step_names = time_step_names
        
        for ix, mrb in enumerate(rule_base):
            for jx, rb in enumerate(mrb):
                for antecedent in rb.antecedents:
                    antecedent.fix_time(ix)
        
        self.antecedents = rule_base[0].antecedents
                     

    def add_rule(self, rule: rl.RuleSimple, consequent: int, time_step: int) -> None:
        '''
        Adds a rule to the rule base of the given consequent.

        :param rule: rule to add.
        :param consequent: index of the rule base to add the rule.
        :param time_step: time step of the rule base to add the rule.
        '''
        self.time_mrule_bases[time_step][consequent].add_rule(rule)


    def get_rulebase_matrix(self) -> list[np.array]:
        '''
        Returns a list with the rulebases for each antecedent in matrix format.

        :return: list with the rulebases for each antecedent in matrix format.
        '''
        return [a.get_rulebase_matrix() for x in self.time_mrule_bases for a in x]


    def get_scores(self) -> np.array:
        '''
        Returns the dominance score for each rule in all the rulebases.

        :return: array with the dominance score for each rule in all the rulebases.
        '''
        res = []
        for rule_bases in self.time_mrule_bases:    
            for rb in rule_bases:
                res.append(rb.scores())

        res = [x for x in res if len(x) > 0]

        return np.concatenate(res, axis=0)


    def compute_firing_strenghts(self, X: np.array, time_moments: list[int]) -> np.array:
        '''
        Computes the firing strength of each rule for each sample.

        :param X: array with the values of the inputs.
        :return: array with the firing strength of each rule for each sample.
        '''
        aux = []
        time_moments = np.array(time_moments)

        for time_moment, rule_bases in enumerate(self.time_mrule_bases):
            actual_moment = np.equal(time_moments, time_moment)
            for ix in range(len(rule_bases)):
                if rule_bases.fuzzy_type() == fs.FUZZY_SETS.t2:
                    aux.append(np.mean(rule_bases[ix].compute_rule_antecedent_memberships(X), axis=2) * np.expand_dims(actual_moment, axis=1))
                elif rule_bases.fuzzy_type() == fs.FUZZY_SETS.gt2:
                    aux.append(np.mean(rule_bases[ix].compute_rule_antecedent_memberships(X), axis=2) * np.expand_dims(actual_moment, axis=1))
                else:
                    aux.append(rule_bases[ix].compute_rule_antecedent_memberships(X) * np.expand_dims(actual_moment, axis=1))

        # Firing strengths shape: samples x rules
        return np.concatenate(aux, axis=1)


    def winning_rule_predict(self, X: np.array, time_moments: int) -> np.array:
        '''
        Returns the winning rule for each sample. Takes into account dominance scores if already computed.

        :param X: array with the values of the inputs.
        :return: array with the winning rule for each sample.
        '''
        consequents = []
        for ix, mrb in enumerate(self.time_mrule_bases):
            for jx, rb in enumerate(mrb):
                for rule in rb:
                    consequents.append(jx)

        # consequents = sum([[ix]*len(self[ix].get_rules())
        #                  for ix in range(len(self.rule_bases))], [])  # The sum is for flatenning
        firing_strengths = self.compute_firing_strenghts(X, time_moments)

        if self.time_mrule_bases[0].fuzzy_type() == fs.FUZZY_SETS.t2 or self.time_mrule_bases[0].fuzzy_type() == fs.FUZZY_SETS.gt2:
            association_degrees = np.mean(self.get_scores(), axis=1) * firing_strengths
        else:
            association_degrees = self.get_scores() * firing_strengths


        winning_rules = np.argmax(association_degrees, axis=1)

        return np.array([consequents[ix] for ix in winning_rules])


    def add_rule_base(self, rule_base: rl.RuleBase, time: int) -> None:
        '''
        Adds a rule base to the list of rule bases.

        :param rule_base: rule base to add.
        '''
        self.time_mrule_bases[time].add_rule_base(rule_base)


    def print_rules(self, return_rules=False) -> None:
        '''
        Print all the rules for all the consequents.
        '''
        res = ''
        for zx, time in enumerate(self.time_mrule_bases):
            res += 'Rules for time step: ' + self.time_step_names[zx] + '\n'
            res += '----------------\n' 
            for ix, ruleBase in enumerate(time):
                res += 'Consequent: ' + time.consequent_names[ix] + '\n'
                res += ruleBase.print_rules(True)
                res += '\n'
            res += '----------------\n'

        if return_rules:
            return res
        else:
            print(res)


    def get_rules(self) -> list[rl.RuleSimple]:
        '''
        Returns a list with all the rules.

        :return: list with all the rules.
        '''
        return [rule for mruleBase in self.rule_bases for rule in mruleBase.get_rules()]


    def fuzzy_type(self) -> fs.FUZZY_SETS:
        '''
        Returns the correspoing type of the RuleBase using the enum type in the fuzzy_sets module.

        :return: the corresponding fuzzy set type of the RuleBase.
        '''
        return self.time_mrule_bases[0].rule_bases[0].fuzzy_type()


    def purge_rules(self, tolerance=0.001) -> None:
        '''
        Delete the roles with a dominance score lower than the tolerance.

        :param tolerance: tolerance to delete the rules.
        '''
        for mruleBase in self.time_mrule_bases:
            mruleBase.purge_rules(tolerance)


    def __getitem__(self, item: int) -> rl.RuleBase:
        '''
        Returns the corresponding time rulebase.

        :param item: index of the rulebase.
        :return: the corresponding rulebase.
        '''
        return self.time_mrule_bases[item]


    def __len__(self) -> int:
        '''
        Returns the number of rule bases.
        '''
        return len(self.time_mrule_bases)


    def get_consequents(self) -> list[int]:
        '''
        Returns a list with the consequents of each rule base.

        :return: list with the consequents of each rule base.
        '''
        return sum([x.get_consequents() for ix, x in enumerate(self.time_mrule_bases)], [])
    

    def get_rulebases(self) -> list[rl.RuleBase]:
        '''
        Returns a list with all the rules.

        :return: list
        '''
        rule_bases = []

        for ruleBase in self.time_mrule_bases:
            for rule in ruleBase:
                rule_bases.append(rule)
        
        return rule_bases
    

    def _winning_rules(self, X: np.array, temporal_moments: list[int]) -> np.array:
        '''
        Returns the winning rule for each sample. Takes into account dominance scores if already computed.

        :param X: array with the values of the inputs.
        :return: array with the winning rule for each sample.
        '''
        
        firing_strengths = self.compute_firing_strenghts(X, temporal_moments)

        association_degrees = self.get_scores() * firing_strengths

        if (self[0].fuzzy_type() == fs.FUZZY_SETS.t2) or (self[0].fuzzy_type() == fs.FUZZY_SETS.gt2):
            association_degrees = np.mean(association_degrees, axis=2)
        elif self[0].fuzzy_type() == fs.FUZZY_SETS.gt2:
            association_degrees = np.mean(association_degrees, axis=3)

        winning_rules = np.argmax(association_degrees, axis=1)

        return winning_rules
        

#### DEFINE THE FUZZY CLASSIFIER USING TEMPORAL FUZZY SETS ####
class TemporalFuzzyRulesClassifier(evf.BaseFuzzyRulesClassifier):
    '''
    Class that is used as a classifier for a fuzzy rule based system with time dependencies. Supports precomputed and optimization of the linguistic variables.
    '''

    def __init__(self,  nRules: int = 30, nAnts: int = 4, fuzzy_type: fs.FUZZY_SETS = None, tolerance: float = 0.0,
                 n_linguist_variables: int = 0, verbose=False, linguistic_variables: list[fs.fuzzyVariable] = None,
                 domain: list[float] = None, n_class: int=None, precomputed_rules: rl.MasterRuleBase =None, runner: int=1) -> None:
        '''
        Inits the optimizer with the corresponding parameters.

        :param nRules: number of rules to optimize.
        :param nAnts: max number of antecedents to use.
        :param fuzzy type: FUZZY_SET enum type in fuzzy_sets module. The kind of fuzzy set used.
        :param tolerance: tolerance for the dominance score of the rules.
        :param n_linguist_variables: number of linguistic variables per antecedent.
        :param verbose: if True, prints the progress of the optimization.
        :param linguistic_variables: list of fuzzyVariables type. If None (default) the optimization process will init+optimize them.
        :param domain: list of the limits for each variable. If None (default) the classifier will compute them empirically.

        kwargs:
            - n_classes: number of classes to predict. Default deduces from data.
        '''
        super().__init__(nRules=nRules, nAnts=nAnts, fuzzy_type=fuzzy_type, tolerance=tolerance, 
                         n_linguist_variables=n_linguist_variables, verbose=verbose, linguistic_variables=linguistic_variables, 
                         domain=domain, n_class=n_class, precomputed_rules=precomputed_rules, runner=runner)
    

    def _contruct_tempRuleBase(self, problems, best_individuals):
        ruleBase_temp = []

        for problem, subject in zip(problems, best_individuals):
            ruleBase_time_moment = problem._construct_ruleBase(subject, 
                                                                self.fuzzy_type)
            ruleBase_temp.append(ruleBase_time_moment)

        return ruleBase_temp


    def _fix_time(self, lvs: list[temporalFuzzyVariable], time: int) -> None:
        '''
        Fixes the time of the temporal fuzzy variables.

        :param lvs: list of temporal fuzzy variables.
        :param time: integer. Time moment to fix.
        :return: None. The time is fixed.
        '''
        for lv in lvs:
            lv.fix_time(time)


    def fit(self, X: np.array, y: np.array, n_gen:int=50, pop_size:int=10, time_moments: np.array=None, checkpoints:int=0):
        '''
        Fits a fuzzy rule based classifier using a genetic algorithm to the given data.

        :param X: numpy array samples x features
        :param y: labels. integer array samples (x 1)
        :param n_gen: integer. Number of generations to run the genetic algorithm.
        :param pop_size: integer. Population size for each gneration.
        :param time_moments: array of integers. Time moments associated to each sample (when temporal dependencies are present)
        :return: None. The classifier is fitted to the data.
        '''
        problems = []
        for ix in range(len(np.unique(time_moments))):
            X_problem = X[time_moments == ix]
            y_problem = y[time_moments == ix]
            
            if self.lvs is None:
                # If Fuzzy variables need to be optimized.
                problem = evf.FitRuleBase(X_problem, y_problem, nRules=self.nRules, nAnts=self.nAnts, tolerance=self.tolerance,
                                    n_linguist_variables=self.n_linguist_variables, fuzzy_type=self.fuzzy_type, domain=self.domain, 
                                    n_classes=self.n_class, thread_runner=self.thread_runner)
            else:
                import copy
                time_lvs = [copy.deepcopy(aux) for aux in self.lvs]
                self._fix_time(time_lvs, ix)
                # If Fuzzy variables are already precomputed.       
                problem = evf.FitRuleBase(X_problem, y_problem, nRules=self.nRules, nAnts=self.nAnts,
                                    linguistic_variables=time_lvs, domain=self.domain, tolerance=self.tolerance, 
                                    n_classes=self.classes_, thread_runner=self.thread_runner)
            
            problems.append(problem)

        

        best_individuals = []
        self.performance = {}
        for time, problem in enumerate(problems):
            algorithm = GA(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=.3, eta=3.0),
            mutation=PolynomialMutation(eta=7.0),
            eliminate_duplicates=True)

            if checkpoints > 0:
                if self.verbose:
                    print('=================================================')
                    print('n_gen  |  n_eval  |     f_avg     |     f_min    ')
                    print('=================================================')
                algorithm.setup(problem, seed=33, termination=('n_gen', n_gen)) # 33? Soon...
                for k in range(n_gen):
                    algorithm.next()
                    res = algorithm
                    if self.verbose:
                        print('%-6s | %-8s | %-8s | %-8s' % (res.n_gen, res.evaluator.n_eval, round(res.pop.get('F').mean(), 8), round(res.pop.get('F').min(), 8)))
                    if k % checkpoints == 0:
                        with open("checkpoint_" + str(time) + '_' + str(algorithm.n_gen), "w") as f:
                            pop = algorithm.pop
                            fitness_last_gen = pop.get('F')
                            best_solution_arg = np.argmin(fitness_last_gen)
                            best_individual = pop.get('X')[best_solution_arg, :]

                            rule_base = problem._construct_ruleBase(
                                best_individual, self.fuzzy_type)
                            eval_performance = evr.evalRuleBase(
                                rule_base, np.array(X), y)
                            
                            eval_performance.add_full_evaluation()  
                            # self.rename_fuzzy_variables() This wont work on checkpoints!
                            rule_base.purge_rules(self.tolerance)
                            checkpoint_rules = rule_base.print_rules(True)
                            f.write(checkpoint_rules)     

            else:
                res = minimize(problem,
                            algorithm,
                            # termination,
                            ("n_gen", n_gen),
                            copy_algorithm=False,
                            save_history=False,
                            verbose=self.verbose)

            pop = res.pop
            fitness_last_gen = pop.get('F')
            best_solution = np.argmin(fitness_last_gen)
            best_individual = pop.get('X')[best_solution, :]
            best_individuals.append(best_individual)

            if self.verbose:
                print('Rule based fit for time ' + str(time) + ' completed.')

        
            self.performance[time] = 1 - fitness_last_gen[best_solution]

        try:
            self.var_names = list(X.columns)
            self.X = X.values
        except AttributeError:
            self.X = X
            self.var_names = [str(ix) for ix in range(X.shape[1])]

        
        ruleBase_temp = self._contruct_tempRuleBase(problems, best_individuals)

        self.rule_base = temporalMasterRuleBase(ruleBase_temp)

        self.eval_performance = evr.evalRuleBase(self.rule_base, np.array(X), y, time_moments)

        self.eval_performance.add_full_evaluation()  
        self.rename_fuzzy_variables()
        self.rule_base.purge_rules(self.tolerance)


    def forward(self, X: np.array, time_moments: list[int] = None) -> np.array:
        '''
        Returns the predicted class for each sample.

        :param X: np array samples x features.
        :param time_moments: list of integers. Time moments associated to each sample (when temporal dependencies are present)
        :return: np array samples (x 1) with the predicted class.
        '''
        try:
            X = X.values  # If X was a numpy array
        except AttributeError:
            pass
        
        return self.rule_base.winning_rule_predict(X, time_moments)



    def plot_fuzzy_variables(self) -> None:
        '''
        Plot the fuzzy partitions in each fuzzy variable.
        '''
        fuzzy_variables = self.rule_base.rule_bases[0].antecedents

        for ix, fv in enumerate(fuzzy_variables):
            vis_rules.plot_fuzzy_variable(fv)


    def get_rulebase(self) -> list[np.array]:
        '''
        Get the rulebase obtained after fitting the classifier to the data.

        :return: a matrix format for the rulebase.
        '''
        return self.rule_base.get_rulebase_matrix()


def eval_temporal_fuzzy_model(fl_classifier: evf.BaseFuzzyRulesClassifier, X_train: np.array, y_train: np.array,
                     X_test: np.array, y_test: np.array, time_moments: list[int] = None, test_time_moments: list[int] = None,
                     plot_rules=True, print_rules=True, plot_partitions=True, return_rules=False, print_accuracy=True, print_matthew=True) -> None:
    '''
    Function that evaluates a fuzzy rule based model. It also plots the rules and the fuzzy partitions.

    :param fl_classifier: Fuzzy rule based model.
    :param X_train: Training data.
    :param y_train: Training labels.
    :param X_test: Test data.
    :param y_test: Test labels.
    :param plot_rules: If True, it plots the rules.
    :param print_rules: If True, it prints the rules.
    :param plot_partitions: If True, it plots the fuzzy partitions.
    :return: None
    '''

    if print_accuracy:
      print('ACCURACY')
      print('Train performance: ' +
            str(np.mean(np.equal(y_train, fl_classifier.forward(X_train, time_moments)))))
      print('Test performance: ' +
            str(np.mean(np.equal(y_test, fl_classifier.forward(X_test, test_time_moments)))))
      print('------------')
    if print_matthew:
      print('MATTHEW CORRCOEF')
      print('Train performance: ' +
            str(matthews_corrcoef(y_train, fl_classifier.forward(X_train, time_moments))))
      print('Test performance: ' +
            str(matthews_corrcoef(y_test, fl_classifier.forward(X_test, test_time_moments))))
      print('------------')

    for ix in np.unique(time_moments):
      try:
            X_aux = X_train[time_moments == ix, :]
            X_aux_test = X_test[time_moments == ix, :]
      except pd.core.indexing.InvalidIndexError:
            X_aux = X_train.iloc[time_moments == ix, :]
            X_aux_test = X_test.iloc[test_time_moments == ix, :]

      y_aux = y_train[time_moments == ix]
      y_aux_test = y_test[test_time_moments == ix]

      if print_matthew:
            print('MOMENT ' + str(ix))
            print('------------')
            print('MATTHEW CORRCOEF')
            print('Train performance: ' +
                  str(matthews_corrcoef(y_aux, fl_classifier.forward(X_aux, np.array([ix] * X_aux.shape[0])))))
            print('Test performance: ' +
                  str(matthews_corrcoef(y_aux_test, fl_classifier.forward(X_aux_test, np.array([ix] * X_aux_test.shape[0])))))
            print('------------')

    if plot_rules:
        vis_rules.visualize_rulebase(fl_classifier)
    if print_rules or return_rules:
        res = fl_classifier.print_rules(return_rules)
    if plot_partitions:
        fl_classifier.plot_fuzzy_variables()
      
    if return_rules:
        return res
    else:
        print(res)

