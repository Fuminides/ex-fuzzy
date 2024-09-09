"""
This is a the source file that contains the class to train/fit the rulebase using a genetic algorithm.

"""
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.base import ClassifierMixin
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
    from . import rules
    from . import eval_rules as evr
    from . import vis_rules
    from . import maintenance as mnt
except ImportError:
    import fuzzy_sets as fs
    import rules
    import eval_rules as evr
    import vis_rules
    import maintenance as mnt



class BaseFuzzyRulesClassifier(ClassifierMixin):
    '''
    Class that is used as a classifier for a fuzzy rule based system. Supports precomputed and optimization of the linguistic variables.
    '''

    def __init__(self,  nRules: int = 30, nAnts: int = 4, fuzzy_type: fs.FUZZY_SETS = fs.FUZZY_SETS.t1, tolerance: float = 0.0, class_names: list[str] = None,
                 n_linguistic_variables: list[int]|int = 3, verbose=False, linguistic_variables: list[fs.fuzzyVariable] = None,
                 domain: list[float] = None, n_class: int=None, precomputed_rules: rules.MasterRuleBase=None, runner: int=1, ds_mode: int = 0, fuzzy_modifiers:bool=False, allow_unknown:bool=False) -> None:
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
        :param n_class: names of the classes in the problem. If None (default) the classifier will compute it empirically.
        :param precomputed_rules: MasterRuleBase object. If not None, the classifier will use the rules in the object and ignore the conflicting parameters.
        :param runner: number of threads to use. If None (default) the classifier will use 1 thread.
        :param ds_mode: mode for the dominance score. 0: normal dominance score, 1: rules without weights, 2: weights optimized for each rule based on the data.
        :param fuzzy_modifiers: if True, the classifier will use the modifiers in the optimization process.
        :param allow_unknown: if True, the classifier will allow the unknown class in the classification process. (Which would be a -1 value)
        '''
        if mnt.save_usage_flag:
            mnt.usage_data[mnt.usage_categories.Funcs]['fit'] += 1

        if precomputed_rules is not None:
            self.nRules = len(precomputed_rules.get_rules())
            self.nAnts = len(precomputed_rules.get_rules()[0].antecedents)
            self.n_class = len(precomputed_rules)
            self.nclasses_ = len(precomputed_rules.consequent_names)
            self.classes_names = precomputed_rules.consequent_names
            self.rule_base = precomputed_rules
        else:
            self.nRules = nRules
            self.nAnts = nAnts
            self.nclasses_ = n_class
            if not (class_names is None):
                if isinstance(class_names, np.ndarray):
                    self.classes_names = list(class_names)
                else:
                    self.classes_names = class_names
            else:
                self.classes_names = class_names

        self.custom_loss = None
        self.verbose = verbose
        self.tolerance = tolerance
        self.ds_mode = ds_mode
        self.fuzzy_modifiers = fuzzy_modifiers
        self.allow_unknown = allow_unknown

        if runner > 1:
            pool = ThreadPool(runner)
            self.thread_runner = StarmapParallelization(pool.starmap)
        else:
            self.thread_runner = None
        
        if linguistic_variables is not None:
            if mnt.save_usage_flag:
                mnt.usage_data[mnt.usage_categories.Funcs]['precompute_labels'] += 1
            # If the linguistic variables are precomputed then we act accordingly
            self.lvs = linguistic_variables
            self.n_linguist_variables = [len(lv.linguistic_variable_names()) for lv in self.lvs]
            self.domain = None
            self.fuzzy_type = self.lvs[0].fuzzy_type()

            if self.nAnts > len(linguistic_variables):
                self.nAnts = len(linguistic_variables)
                if verbose:
                    print('Warning: The number of antecedents is higher than the number of variables. Setting nAnts to the number of linguistic variables. (' + str(len(linguistic_variables)) + ')')

        else:
            if mnt.save_usage_flag:
                mnt.usage_data[mnt.usage_categories.Funcs]['opt_labels'] += 1

            # If not, then we need the parameters sumistered by the user.
            self.lvs = None
            self.fuzzy_type = fuzzy_type
            self.n_linguist_variables = n_linguistic_variables
            self.domain = domain

        self.alpha_ = 0.0
        self.beta_ = 0.0


    def customized_loss(self, loss_function):
        '''
        Function to customize the loss function used for the optimization.

        :param loss_function: function that takes as input the true labels and the predicted labels and returns a float.
        :return: None
        '''
        self.custom_loss = loss_function


    def fit(self, X: np.array, y: np.array, n_gen:int=70, pop_size:int=30,
            checkpoints:int=0, candidate_rules:rules.MasterRuleBase=None, initial_rules:rules.MasterRuleBase=None, random_state:int=33) -> None:
        '''
        Fits a fuzzy rule based classifier using a genetic algorithm to the given data.

        :param X: numpy array samples x features
        :param y: labels. integer array samples (x 1)
        :param n_gen: integer. Number of generations to run the genetic algorithm.
        :param pop_size: integer. Population size for each gneration.
        :param checkpoints: integer. Number of checkpoints to save the best rulebase found so far.
        :param candidate_rules: if these rules exist, the optimization process will choose the best rules from this set. If None (default) the rules will be generated from scratch.
        :return: None. The classifier is fitted to the data.
        '''
        if mnt.save_usage_flag:
            mnt.usage_data[mnt.usage_categories.Funcs]['fit'] += 1
            
        if self.classes_names is None:
            self.classes_names = [aux for aux in np.unique(y)]
        
        if self.nclasses_ is None:
            self.nclasses_ = len(self.classes_names)

        if isinstance(y[0], str):
            y = np.array([self.classes_names.index(str(aux)) for aux in y])
            
        if candidate_rules is None:
            if initial_rules is not None:
                self.fuzzy_type = initial_rules.fuzzy_type()
                self.n_linguist_variables = initial_rules.n_linguistic_variables()
                self.domain = [fv.domain for fv in initial_rules[0].antecedents]
                self.nRules = len(initial_rules.get_rules())
                self.nAnts = len(initial_rules.get_rules()[0].antecedents)

            if self.lvs is None:
                # Check if self.n_linguist_variables is a list or a single value.
                if isinstance(self.n_linguist_variables, int):
                    self.n_linguist_variables = [self.n_linguist_variables for _ in range(X.shape[1])]
                
                if self.nAnts > X.shape[1]:
                    self.nAnts = X.shape[1]
                    if self.verbose:
                        print('Warning: The number of antecedents is higher than the number of variables. Setting nAnts to the number of variables. (' + str(X.shape[1]) + ')') 

                # If Fuzzy variables need to be optimized.
                problem = FitRuleBase(X, y, nRules=self.nRules, nAnts=self.nAnts, tolerance=self.tolerance, n_classes=len(np.unique(y)),
                                    n_linguistic_variables=self.n_linguist_variables, fuzzy_type=self.fuzzy_type, domain=self.domain, thread_runner=self.thread_runner,
                                    alpha=self.alpha_, beta=self.beta_, ds_mode=self.ds_mode, encode_mods=self.fuzzy_modifiers,
                                    allow_unknown=self.allow_unknown)
            else:
                # If Fuzzy variables are already precomputed.
                problem = FitRuleBase(X, y, nRules=self.nRules, nAnts=self.nAnts, n_classes=len(np.unique(y)),
                                    linguistic_variables=self.lvs, domain=self.domain, tolerance=self.tolerance, thread_runner=self.thread_runner,
                                    alpha=self.alpha_, beta=self.beta_, ds_mode=self.ds_mode, encode_mods=self.fuzzy_modifiers,
                                    allow_unknown=self.allow_unknown)
        else:
            self.fuzzy_type = candidate_rules.fuzzy_type()
            self.n_linguist_variables = candidate_rules.n_linguistic_variables()
            problem = ExploreRuleBases(X, y, n_classes=len(np.unique(y)), candidate_rules=candidate_rules, thread_runner=self.thread_runner, nRules=self.nRules)

        if self.custom_loss is not None:
            problem.fitness_func = self.custom_loss

        if initial_rules is not None:
            rules_gene = problem.encode_rulebase(initial_rules, self.lvs is None)
            rules_gene = (np.ones((pop_size, len(rules_gene))) * rules_gene).astype(int)
        else:
            rules_gene = IntegerRandomSampling()

        algorithm = GA(
            pop_size=pop_size,
            crossover=SBX(prob=.3, eta=3.0),
            mutation=PolynomialMutation(eta=7.0),
            sampling=rules_gene,
            eliminate_duplicates=False)
        

        if checkpoints > 0:
            if self.verbose:
                print('=================================================')
                print('n_gen  |  n_eval  |     f_avg     |     f_min    ')
                print('=================================================')
            algorithm.setup(problem, seed=random_state, termination=('n_gen', n_gen)) 
            for k in range(n_gen):
                algorithm.next()
                res = algorithm
                if self.verbose:
                    print('%-6s | %-8s | %-8s | %-8s' % (res.n_gen, res.evaluator.n_eval, res.pop.get('F').mean(), res.pop.get('F').min()))
                if k % checkpoints == 0:
                    with open("checkpoint_" + str(algorithm.n_gen), "w") as f:
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
                        rule_base.rename_cons(self.classes_names)
                        checkpoint_rules = rule_base.print_rules(True)
                        f.write(checkpoint_rules)     

        else:
            res = minimize(problem,
                        algorithm,
                        # termination,
                        ("n_gen", n_gen),
                        seed=random_state,
                        copy_algorithm=False,
                        save_history=False,
                        verbose=self.verbose)
        
        pop = res.pop
        fitness_last_gen = pop.get('F')
        best_solution = np.argmin(fitness_last_gen)
        best_individual = pop.get('X')[best_solution, :]

        
        self.performance = 1 - fitness_last_gen[best_solution]

        try:
            self.var_names = list(X.columns)
            self.X = X.values
        except AttributeError:
            self.X = X
            self.var_names = [str(ix) for ix in range(X.shape[1])]

        self.rule_base = problem._construct_ruleBase(
        best_individual, fz_type=self.fuzzy_type)
        self.rule_base.rename_cons(self.classes_names)

        self.eval_performance = evr.evalRuleBase(
        self.rule_base, np.array(X), y)

        self.eval_performance.add_full_evaluation()  
        if self.lvs is None:
            self.rename_fuzzy_variables()
        self.eval_performance.add_classification_metrics()
        self.rule_base.purge_rules(self.tolerance)
        

    def load_master_rule_base(self, rule_base: rules.MasterRuleBase) -> None:
        '''
        Loads a master rule base to be used in the prediction process.

        :param rule_base: ruleBase object.
        :return: None
        '''
        self.rule_base = rule_base
        self.nRules = len(rule_base.get_rules())
        self.nAnts = len(rule_base.get_rules()[0].antecedents)
        self.nclasses_ = len(rule_base)
        

    def forward(self, X: np.array) -> np.array:
        '''
        Returns the predicted class for each sample.

        :param X: np array samples x features.
        :return: np array samples (x 1) with the predicted class.
        '''
        try:
            X = X.values  # If X was a pandas dataframe
        except AttributeError:
            pass
        
        return self.rule_base.winning_rule_predict(X)
        

    def predict(self, X: np.array) -> np.array:
        '''
        Returns the predicted class for each sample.

        :param X: np array samples x features.
        :return: np array samples (x 1) with the predicted class.
        '''
        return self.forward(X)
    

    def predict_proba(self, X: np.array) -> np.array:
        '''
        Returns the predicted class probabilities for each sample.

        :param X: np array samples x features.
        :return: np array samples x classes with the predicted class probabilities.
        '''
        try:
            X = X.values  # If X was a pandas dataframe
        except AttributeError:
            pass

        return self.rule_base.compute_association_degrees(X)


    def print_rules(self, return_rules:bool=False) -> None:
        '''
        Print the rules contained in the fitted rulebase.
        '''
        return self.rule_base.print_rules(return_rules)


    def plot_fuzzy_variables(self) -> None:
        '''
        Plot the fuzzy partitions in each fuzzy variable.
        '''
        fuzzy_variables = self.rule_base.rule_bases[0].antecedents

        for ix, fv in enumerate(fuzzy_variables):
            vis_rules.plot_fuzzy_variable(fv)


    def rename_fuzzy_variables(self) -> None:
        '''
        Renames the linguist labels so that high, low and so on are consistent. It does so usually after an optimization process.

        :return: None. Names are sorted accorded to the central point of the fuzzy memberships.
        '''

        for ix in range(len(self.rule_base)):
            fuzzy_variables = self.rule_base.rule_bases[ix].antecedents

            for jx, fv in enumerate(fuzzy_variables):
                new_order_values = []
                possible_names = FitRuleBase.vl_names[self.n_linguist_variables[jx]]

                for zx, fuzzy_set in enumerate(fv.linguistic_variables):
                    studied_fz = fuzzy_set.type()
                    
                    if studied_fz == fs.FUZZY_SETS.temporal:
                        studied_fz = fuzzy_set.inside_type()

                    if studied_fz == fs.FUZZY_SETS.t1:
                        f1 = np.mean(
                            fuzzy_set.membership_parameters[0] + fuzzy_set.membership_parameters[1])
                    elif (studied_fz == fs.FUZZY_SETS.t2):
                        f1 = np.mean(
                            fuzzy_set.secondMF_upper[0] + fuzzy_set.secondMF_upper[1])
                    elif studied_fz == fs.FUZZY_SETS.gt2:
                        sec_memberships = fuzzy_set.secondary_memberships.values()
                        f1 = float(list(fuzzy_set.secondary_memberships.keys())[np.argmax(
                            [fzm.membership_parameters[2] for ix, fzm in enumerate(sec_memberships)])])

                    new_order_values.append(f1)

                new_order = np.argsort(np.array(new_order_values))
                fuzzy_sets_vl = fv.linguistic_variables

                for jx, x in enumerate(new_order):
                    fuzzy_sets_vl[x].name = possible_names[jx]


    def get_rulebase(self) -> list[np.array]:
        '''
        Get the rulebase obtained after fitting the classifier to the data.

        :return: a matrix format for the rulebase.
        '''
        return self.rule_base.get_rulebase_matrix()
    

    def reparametrice_loss(self, alpha:float, beta:float) -> None:
        '''
        Changes the parameters in the loss function. 

        :note: Does not check for convexity preservation. The user can play with these parameters as it wills.
        :param alpha: controls the MCC term.
        :param beta: controls the average rule size loss.
        '''
        self.alpha_ = alpha
        self.beta_ = beta



class ExploreRuleBases(Problem):
    '''
    Class to model as pymoo problem the fitting of a rulebase to a set of data given a series of candidate rules for a classification problem using Evolutionary strategies
    Supports type 1 and t2.
    '''

    def __init__(self, X: np.array, y: np.array, nRules: int, n_classes: int, candidate_rules: rules.MasterRuleBase, thread_runner: StarmapParallelization=None, tolerance:float = 0.01) -> None:
        '''
        Cosntructor method. Initializes the classifier with the number of antecedents, linguist variables and the kind of fuzzy set desired.

        :param X: np array or pandas dataframe samples x features.
        :param y: np vector containing the target classes. vector sample
        :param n_class: number of classes in the problem. If None (as default) it will be computed from the data.
        :param cancidate_rules: MasterRuleBase object. If not None, the classifier will use the rules in the object and ignore the conflicting parameters.
        '''
        try:
            self.var_names = list(X.columns)
            self.X = X.values
        except AttributeError:
            self.X = X
            self.var_names = [str(ix) for ix in range(X.shape[1])]

        self.tolerance = tolerance
        self.fuzzy_type = candidate_rules.fuzzy_type()
        self.y = y
        self.nCons = 1  # This is fixed to MISO rules.
        self.n_classes = n_classes
        self.candidate_rules = candidate_rules
        self.nRules = nRules
        self._precomputed_truth = rules.compute_antecedents_memberships(candidate_rules.get_antecedents(), X)

        self.fuzzy_type = self.candidate_rules[0].antecedents[0].fuzzy_type()

        self.min_bounds = np.min(self.X, axis=0)
        self.max_bounds = np.max(self.X, axis=0)

        nTotalRules = len(self.candidate_rules.get_rules())
        # Each var is using or not a rule. 
        vars = {ix: Integer(bounds=[0, nTotalRules - 1]) for ix in range(self.nRules)}
        varbound = np.array([[0, nTotalRules- 1]] * self.nRules)

        nVar = len(vars.keys())
        if thread_runner is not None:
            super().__init__(
                vars=vars,
                n_var=nVar,
                n_obj=1,
                elementwise=True,
                vtype=int,
                xl=varbound[:, 0],
                xu=varbound[:, 1],
                elementwise_runner=thread_runner)
        else:
            super().__init__(
                vars=vars,
                n_var=nVar,
                n_obj=1,
                elementwise=True,
                vtype=int,
                xl=varbound[:, 0],
                xu=varbound[:, 1])


    def _construct_ruleBase(self, x: np.array, fuzzy_type: fs.FUZZY_SETS, ds_mode:int=0, allow_unknown:bool=True) -> rules.MasterRuleBase:
        '''
        Creates a valid rulebase from the given subject and the candidate rules.

        :param x: gen of a rulebase. type: dict.
        :param fuzzy_type: FUZZY_SET enum type in fuzzy_sets module. The kind of fuzzy set used.
        :param ds_mode: int. Mode for the dominance score. 0: normal dominance score, 1: rules without weights, 2: weights optimized for each rule based on the data.
        :param allow_unknown: if True, the classifier will allow the unknown class in the classification process. (Which would be a -1 value)
        
        :return: a Master rulebase object.
        '''
        x = x.astype(int)
        # Get all rules and their consequents
        diff_consequents = np.arange(len(self.candidate_rules))
        
        # Choose the selected ones in the gen
        total_rules = self.candidate_rules.get_rules()
        chosen_rules = [total_rules[ix] for ix, val in enumerate(x)]
        rule_consequents = sum([[ix] * len(rule) for ix, rule in enumerate(self.candidate_rules)], [])
        chosen_rules_consequents = [rule_consequents[val] for ix, val in enumerate(x)]
        # Create a rule base for each consequent with the selected rules
        rule_list = [[] for _ in range(self.n_classes)]
        rule_bases = []
        for ix, consequent in enumerate(diff_consequents):
            for rx, rule in enumerate(chosen_rules):
                if chosen_rules_consequents[rx] == consequent:
                    rule_list[ix].append(rule)

            if len(rule_list[ix]) > 0:
                if fuzzy_type == fs.FUZZY_SETS.t1:
                    rule_base_cons = rules.RuleBaseT1(
                        self.candidate_rules[0].antecedents, rule_list[ix])
                elif fuzzy_type == fs.FUZZY_SETS.t2:
                    rule_base_cons = rules.RuleBaseT2(
                        self.candidate_rules[0].antecedents, rule_list[ix])
                elif fuzzy_type == fs.FUZZY_SETS.gt2:
                    rule_base_cons = rules.RuleBaseGT2(
                        self.candidate_rules[0].antecedents, rule_list[ix])
                    
                rule_bases.append(rule_base_cons)
            
        # Create the Master Rule Base object with the individual rule bases
        newMasterRuleBase = rules.MasterRuleBase(rule_bases, diff_consequents, ds_mode=ds_mode, allow_unknown=allow_unknown)    

        return newMasterRuleBase


    def _evaluate(self, x: np.array, out: dict, *args, **kwargs):
        '''
        :param x: array of train samples. x shape = features
            those features are the parameters to optimize.

        :param out: dict where the F field is the fitness. It is used from the outside.
        '''
        try:
            ruleBase = self._construct_ruleBase(x, self.fuzzy_type)

            score = self.fitness_func(ruleBase, self.X, self.y, self.tolerance, precomputed_truth=self._precomputed_truth)
            

            out["F"] = 1 - score
        except rules.RuleError:
            out["F"] = 1

    
    def fitness_func(self, ruleBase: rules.RuleBase, X:np.array, y:np.array, tolerance:float, alpha:float=0.0, beta:float=0.0, precomputed_truth=None) -> float:
        '''
        Fitness function for the optimization problem.
        :param ruleBase: RuleBase object
        :param X: array of train samples. X shape = (n_samples, n_features)
        :param y: array of train labels. y shape = (n_samples,)
        :param tolerance: float. Tolerance for the size evaluation.
        :return: float. Fitness value.
        '''
        ev_object = evr.evalRuleBase(ruleBase, X, y, precomputed_truth=precomputed_truth)
        ev_object.add_rule_weights()

        score_acc = ev_object.classification_eval()
        score_rules_size = ev_object.size_antecedents_eval(tolerance)
        score_nrules = ev_object.effective_rulesize_eval(tolerance)

        score = score_acc + score_rules_size * alpha + score_nrules * beta

        return score
    


class FitRuleBase(Problem):
    '''
    Class to model as pymoo problem the fitting of a rulebase for a classification problem using Evolutionary strategies. 
    Supports type 1 and iv fs (iv-type 2)
    '''

    def _init_optimize_vl(self, fuzzy_type: fs.FUZZY_SETS, n_linguist_variables: int, domain: list[(float, float)] = None):
        '''
        Inits the corresponding fields if no linguistic partitions were given.

        :param fuzzy type: FUZZY_SET enum type in fuzzy_sets module. The kind of fuzzy set used.
        :param n_linguistic_variables: number of linguistic variables per antecedent.
        :param domain: list of the limits for each variable. If None (default) the classifier will compute them empirically.
        '''
        self.lvs = None
        self.vl_names = [FitRuleBase.vl_names[n_linguist_variables[nn]] if n_linguist_variables[nn] < 6 else list(map(str, np.arange(nn))) for nn in range(len(n_linguist_variables))]
        

        self.fuzzy_type = fuzzy_type
        self.n_lv_possible = n_linguist_variables
        self.domain = domain
        self._precomputed_truth = None

    def _init_precomputed_vl(self, linguist_variables: list[fs.fuzzyVariable], X: np.array):
        '''
        Inits the corresponding fields if linguistic partitions for each variable are given.

        :param linguistic_variables: list of fuzzyVariables type.
        :param X: np array samples x features.
        '''
        self.lvs = linguist_variables
        self.vl_names = [lv.linguistic_variable_names() for lv in self.lvs]
        self.n_lv_possible = [len(lv.linguistic_variable_names()) for lv in self.lvs]
        self.fuzzy_type = self.lvs[0].fs_type
        self.domain = None
        self._precomputed_truth = rules.compute_antecedents_memberships(linguist_variables, X)

    vl_names = [  # Linguistic variable names prenamed for some specific cases.
        [],
        [],
        ['Low', 'High'],
        ['Low', 'Medium', 'High'],
        ['Low', 'Medium', 'High', 'Very High'],
        ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    ]

    def __init__(self, X: np.array, y: np.array, nRules: int, nAnts: int, n_classes: int, thread_runner: StarmapParallelization=None, 
                 linguistic_variables:list[fs.fuzzyVariable]=None, n_linguistic_variables:int=3, fuzzy_type=fs.FUZZY_SETS.t1, domain:list=None,
                 tolerance:float=0.01, alpha:float=0.0, beta:float=0.0, ds_mode: int =0, encode_mods: bool=False, allow_unknown:bool=True) -> None:
        '''
        Cosntructor method. Initializes the classifier with the number of antecedents, linguist variables and the kind of fuzzy set desired.

        :param X: np array or pandas dataframe samples x features.
        :param y: np vector containing the target classes. vector sample
        :param nRules: number of rules to optimize.
        :param nAnts: max number of antecedents to use.
        :param n_class: number of classes in the problem. If None (as default) it will be computed from the data.
        :param linguistic_variables: list of linguistic variables precomputed. If given, the rest of conflicting arguments are ignored.
        :param n_linguistic_variables: number of linguistic variables per antecedent.
        :param fuzzy_type: Define the fuzzy set or fuzzy set extension used as linguistic variable.
        :param domain: list with the upper and lower domains of each input variable. If None (as default) it will stablish the empirical min/max as the limits.
        :param tolerance: float. Tolerance for the size evaluation.
        :param alpha: float. Weight for the rulebase size term in the fitness function. (Penalizes number of rules)
        :param beta: float. Weight for the average rule size term in the fitness function.
        :param ds_mode: int. Mode for the dominance score. 0: normal dominance score, 1: rules without weights, 2: weights optimized for each rule based on the data.
        :param encode_mods: bool. If True, the optimization process will include the modifiers for the membership functions.
        :param allow_unknown: if True, the classifier will allow the unknown class in the classification process. (Which would be a -1 value)
        '''
        try:
            self.var_names = list(X.columns)
            self.X = X.values
        except AttributeError:
            self.X = X
            self.var_names = [str(ix) for ix in range(X.shape[1])]

        try:
            self.tolerance = tolerance
        except KeyError:
            self.tolerance = 0.001

        self.y = y
        self.classes_names = np.unique(y)
        self.nRules = nRules
        self.nAnts = nAnts
        self.nCons = 1  # This is fixed to MISO rules.
        self.ds_mode = ds_mode
        self.encode_mods = encode_mods
        self.allow_unknown = allow_unknown

        if n_classes is not None:
            self.n_classes = n_classes
        else:
            self.n_classes = len(np.unique(y))

        if linguistic_variables is not None:
            self._init_precomputed_vl(linguistic_variables, X)
        else:
            if isinstance(n_linguistic_variables, int):
                n_linguistic_variables = [n_linguistic_variables] * self.X.shape[1]
            self._init_optimize_vl(
                fuzzy_type, n_linguistic_variables)

        if self.domain is None:
            # If all the variables are numerical, then we can compute the min/max of the domain.
            if np.all([np.issubdtype(self.X[:, ix].dtype, np.number) for ix in range(self.X.shape[1])]):
                self.min_bounds = np.min(self.X, axis=0)
                self.max_bounds = np.max(self.X, axis=0)
            else:
                self.min_bounds = np.zeros(self.X.shape[1])
                self.max_bounds = np.zeros(self.X.shape[1])

                for ix in range(self.X.shape[1]):
                    if np.issubdtype(self.X[:, ix].dtype, np.number):
                        self.min_bounds[ix] = np.min(self.X[:, ix])
                        self.max_bounds[ix] = np.max(self.X[:, ix])
                    else:
                        self.min_bounds[ix] = 0
                        self.max_bounds[ix] = len(np.unique(self.X[:, ix]))
        else:
            self.min_bounds, self.max_bounds = self.domain

        self.antecedents_referencial = [np.linspace(
            self.min_bounds[ix], self.max_bounds[ix], 100) for ix in range(self.X.shape[1])]

        possible_antecedent_bounds = np.array(
            [[0, self.X.shape[1] - 1]] * self.nAnts * self.nRules)  
        vl_antecedent_bounds = np.array(
            [[-1, self.n_lv_possible[ax] - 1] for ax in range(self.nAnts)] * self.nRules) # -1 means not caring
        antecedent_bounds = np.concatenate(
            (possible_antecedent_bounds, vl_antecedent_bounds))
        vars_antecedent = {ix: Integer(
            bounds=antecedent_bounds[ix]) for ix in range(len(antecedent_bounds))}
        aux_counter = len(vars_antecedent)

        if self.lvs is None:

            self.feature_domain_bounds = np.array(
                [[0, 99] for ix in range(self.X.shape[1])])
            size_multiplier = 4 if self.fuzzy_type == fs.FUZZY_SETS.t1 else 8
            membership_bounds = np.concatenate(
                [[self.feature_domain_bounds[ixx]] * size_multiplier * self.n_lv_possible[ixx] for ixx in range(len(self.n_lv_possible))])
            
            vars_memberships = {
                aux_counter + ix: Integer(bounds=membership_bounds[ix]) for ix in range(len(membership_bounds))}
            aux_counter += len(vars_memberships)

        final_consequent_bounds = np.array(
            [[-1, self.n_classes - 1]] * self.nRules)
        vars_consequent = {aux_counter + ix: Integer(
            bounds=final_consequent_bounds[ix]) for ix in range(len(final_consequent_bounds))}

        if self.lvs is None:
            vars = {key: val for d in [
                vars_antecedent, vars_memberships, vars_consequent] for key, val in d.items()}
            varbound = np.concatenate(
                (antecedent_bounds, membership_bounds, final_consequent_bounds), axis=0)
        else:
            vars = {key: val for d in [vars_antecedent,
                                       vars_consequent] for key, val in d.items()}
            varbound = np.concatenate(
                (antecedent_bounds, final_consequent_bounds), axis=0)
        
        if self.ds_mode == 2:
            weights_bounds = np.array([[0, 99] for ix in range(self.nRules)])

            vars_weights = {max(vars.keys()) + 1 + ix: Integer(
                bounds=weights_bounds[ix]) for ix in range(len(weights_bounds))}
            vars = {key: val for d in [vars, vars_weights] for key, val in d.items()}
            varbound = np.concatenate((varbound, weights_bounds), axis=0)

        if encode_mods:
            # Now we add modifiers exponents for the membership functions.
            rule_mods = np.array([[0, len(rules.modifiers_names.keys()) - 1]] * self.nAnts * self.nRules)
            vars_modifiers = {max(vars.keys()) + 1 + ix: Integer(
                bounds=rule_mods[ix]) for ix in range(len(rule_mods))}
            vars = {key: val for d in [vars, vars_modifiers] for key, val in d.items()}
            varbound = np.concatenate((varbound, rule_mods), axis=0)

        nVar = len(varbound)
        self.single_gen_size = nVar
        self.alpha_ = alpha
        self.beta_ = beta

        if thread_runner is not None:
            super().__init__(
                vars=vars,
                n_var=nVar,
                n_obj=1,
                elementwise=True,
                vtype=int,
                xl=varbound[:, 0],
                xu=varbound[:, 1],
                elementwise_runner=thread_runner)
        else:
            super().__init__(
                vars=vars,
                n_var=nVar,
                n_obj=1,
                elementwise=True,
                vtype=int,
                xl=varbound[:, 0],
                xu=varbound[:, 1])


    def encode_rulebase(self, rule_base: rules.MasterRuleBase, optimize_lv: bool, encode_mods:bool=False) -> np.array:
        '''
        Given a rule base, constructs the corresponding gene associated with that rule base.

        GENE STRUCTURE

        First: antecedents chosen by each rule. Size: nAnts * nRules (index of the antecedent)
        Second: Variable linguistics used. Size: nAnts * nRules
        Third: Parameters for the fuzzy partitions of the chosen variables. Size: nAnts * self.n_linguistic_variables * 8|4 (2 trapezoidal memberships if t2)
        Four: Consequent classes. Size: nRules

        :param rule_base: rule base object.
        :param optimize_lv: if True, the gene is prepared to optimize the membership functions.
        :param encode_mods: if True, the gene is prepared to encode the modifiers for the membership functions.
        :return: np array of size self.single_gen_size.
        '''
        gene = np.zeros((self.single_gen_size,))

        n_lv_possible = len(rule_base.rule_bases[0].antecedents[0].linguistic_variables)
        fz_type = rule_base.fuzzy_type()
        rule_consequents = rule_base.get_consequents()
        nreal_rules = len(rule_consequents)
        mf_size = 4 if fz_type == fs.FUZZY_SETS.t1 else 8

        # Pointer to the fourth section of the gene: consequents
        if optimize_lv:
            # If lv memberships are optimized.
            fourth_pointer = 2 * self.nAnts * self.nRules + \
                len(rule_base.antecedents) * n_lv_possible * mf_size
        else:
            # If no memberships are optimized.
            fourth_pointer = 2 * self.nAnts * self.nRules

        # Pointer to the fifth section of the gene: weights (if they exist)
        fifth_pointer = fourth_pointer + self.nRules
        if rule_base.ds_mode == 2:
            for ix, rule in enumerate(rule_base.get_rules()):
                gene[fifth_pointer + ix] = rule.weight

        # Last pointer to the gene: modifiers for the membership functions
        if encode_mods:
            if rule_base.ds_mode == 2:
                sixth_pointer = fifth_pointer + rule_base.get_rules()
            else:
                sixth_pointer = fifth_pointer
            
            for ix, rule in enumerate(rule_base.get_rules()):
                for jx, modifier in enumerate(rule.modifiers):
                    mod_idx = list(rules.modifiers_names.keys()).index(modifier)
                    gene[sixth_pointer + ix * self.nAnts + jx] = mod_idx

        # First and second sections of the gene: antecedents and linguistic variables
        for i0, rule in enumerate(rule_base.get_rules()):  # Reconstruct the rules
            first_pointer = i0 * self.nAnts
            second_pointer = (self.nRules * self.nAnts) + i0 * self.nAnts

            for ax, linguistic_variable in enumerate(rule.antecedents):
                gene[first_pointer + ax] = ax
                gene[second_pointer + ax] = linguistic_variable
            
            # Update the fourth section of the gene: consequents using the fourth pointer
            gene[fourth_pointer + i0] = rule_consequents[i0]

        # Fill the rest of the rules with don't care values
        nvoid_rules = self.nRules - nreal_rules
        for vx in range(nvoid_rules):
            first_pointer = nreal_rules * self.nAnts + vx * self.nAnts
            second_pointer = (self.nRules * self.nAnts) + nreal_rules * self.nAnts + vx * self.nAnts

            for ax, linguistic_variable in enumerate(rule.antecedents):
                gene[first_pointer + ax] = ax
                gene[second_pointer + ax] = -1
            
            # Update the fourth section of the gene: consequents using the fourth pointer
            gene[fourth_pointer + nreal_rules + vx] = -1

        if optimize_lv:
            # If lv memberships are optimized.
            third_pointer = 2 * self.nAnts * self.nRules
            aux_pointer = 0
            for ix, fuzzy_variable in enumerate(rule_base.get_antecedents()):
                for linguistic_variable in range(n_lv_possible):
                    fz_parameters = fuzzy_variable[linguistic_variable].membership_parameters
                    for jx, fz_parameter in enumerate(fz_parameters):
                        closest_idx = (np.abs(np.asarray(self.antecedents_referencial[ix]) - fz_parameter)).argmin()
                        gene[third_pointer + aux_pointer] = closest_idx
                        aux_pointer += 1
                    
        return np.array(list(map(int, gene)))
        


    def _construct_ruleBase(self, x: np.array, fz_type: fs.FUZZY_SETS, **kwargs) -> rules.MasterRuleBase:
        '''
        Given a subject, it creates a rulebase according to its specification.

        :param x: gen of a rulebase. type: dict.
        :param fz_type: a enum type. Check fuzzy_sets for complete specification (two fields, t1 and t2, to mark which fs you want to use)
        :param kwargs: additional parameters to pass to the rule
        :return: a rulebase object.

        kwargs:
            - time_moment: if temporal fuzzy sets are used with different partitions for each time interval, 
                            then this parameter is used to specify which time moment is being used.
        '''

        rule_list = [[] for _ in range(self.n_classes)]

        mf_size = 4 if fz_type == fs.FUZZY_SETS.t1 else 8
        '''
        GEN STRUCTURE

        First: antecedents chosen by each rule. Size: nAnts * nRules
        Second: Variable linguistics used. Size: nAnts * nRules
        Third: Parameters for the fuzzy partitions of the chosen variables. Size: X.shape[1] * self.n_linguistic_variables * 8|4 (2 trapezoidal memberships if t2)
        Four: Consequent classes. Size: nRules
        Five: Weights for each rule. Size: nRules (only if ds_mode == 2)
        '''
        if self.lvs is None:
            # If memberships are optimized.
            fourth_pointer = 2 * self.nAnts * self.nRules + \
                sum(self.n_lv_possible) * mf_size
        else:
            # If no memberships are optimized.
            fourth_pointer = 2 * self.nAnts * self.nRules

        if self.ds_mode == 2:
            fifth_pointer = fourth_pointer + self.nRules
        else:
            fifth_pointer = fourth_pointer

        if self.ds_mode == 2:
            sixth_pointer = fifth_pointer + self.nRules
        else:
            sixth_pointer = fifth_pointer
            
        
        aux_pointer = 0
        min_domain = np.min(self.X, axis=0)
        max_domain = np.max(self.X, axis=0)

        # Integer sampling doesnt work fine in pymoo, so we do this (which is btw what pymoo is really doing if you just set integer optimization)
        try:
            # subject might come as a dict.
            x = np.array(list(x.values())).astype(int)
        except AttributeError:
            x = x.astype(int)

        for i0 in range(self.nRules):  # Reconstruct the rules
            first_pointer = i0 * self.nAnts
            chosen_ants = x[first_pointer:first_pointer + self.nAnts]

            second_pointer = (i0 * self.nAnts) + (self.nAnts * self.nRules)
            # Shape: self.nAnts + self.n_lv_possible  + 1
            antecedent_parameters = x[second_pointer:second_pointer+self.nAnts]

            init_rule_antecedents = np.zeros(
                (self.X.shape[1],)) - 1  # -1 is dont care
            for jx, ant in enumerate(chosen_ants):
                if self.lvs is not None:
                    antecedent_parameters[jx] = min(antecedent_parameters[jx], len(self.lvs[ant]) - 1)
                else:
                    antecedent_parameters[jx] = min(antecedent_parameters[jx], self.n_lv_possible[ant] - 1)

                init_rule_antecedents[ant] = antecedent_parameters[jx]

            consequent_idx = x[fourth_pointer + aux_pointer]
            assert consequent_idx < self.n_classes, "Consequent class is not valid. Something in the gene is wrong."
            aux_pointer += 1
 
            if self.ds_mode == 2:
                rule_weight = x[fifth_pointer + i0] / 100
            else:
                rule_weight = 1.0
            
            # Last pointer to the gene: modifiers for the membership functions
            if self.encode_mods:
                #for jx, modifier in enumerate(rule.modifiers):
                rule_modifiers = np.ones((len(self.lvs),)) * -1
                idx_mods = x[sixth_pointer + i0 * self.nAnts: sixth_pointer + (i0+1)*self.nAnts]
                for jx, ant in enumerate(chosen_ants):
                    rule_modifiers[ant] = list(rules.modifiers_names.keys())[idx_mods[jx]]
                

            else:
                rule_modifiers = None

            if consequent_idx != -1 and np.any(init_rule_antecedents != -1):
                rs_instance = rules.RuleSimple(init_rule_antecedents, 0, rule_modifiers)
                if self.ds_mode == 1 or self.ds_mode == 2:
                    rs_instance.weight = rule_weight

                rule_list[consequent_idx].append(
                    rs_instance)

            
        # If we optimize the membership functions
        if self.lvs is None:
            third_pointer = 2 * self.nAnts * self.nRules
            aux_pointer = 0
            antecedents = []

            for fuzzy_variable in range(self.X.shape[1]):
                linguistic_variables = []

                for linguistic_variable in range(self.n_lv_possible[fuzzy_variable]):
                    parameter_pointer = third_pointer + aux_pointer
                    fz_parameters_idx = x[parameter_pointer:parameter_pointer + mf_size]
                    fz_parameters = self.antecedents_referencial[fuzzy_variable][fz_parameters_idx]
                    aux_pointer += mf_size

                    if fz_type == fs.FUZZY_SETS.t2:
                        fz_parameters[0:6] = np.sort(fz_parameters[0:6])
                        mu = [np.min(fz_parameters[0:2]), fz_parameters[2],
                              fz_parameters[3], np.max(fz_parameters[4:6])]
                        ml = [np.max(fz_parameters[0:2]), fz_parameters[2],
                              fz_parameters[3], np.min(fz_parameters[4:6])]
                        height = fz_parameters[6] / np.max(fz_parameters)

                        ivfs = fs.IVFS(self.vl_names[fuzzy_variable][linguistic_variable], ml, mu,
                                       (min_domain[fuzzy_variable], max_domain[fuzzy_variable]), lower_height=height)
                    else:
                        ivfs = fs.FS(self.vl_names[fuzzy_variable][linguistic_variable], np.sort(fz_parameters[0:4]),
                                     (min_domain[fuzzy_variable], max_domain[fuzzy_variable]))
                    linguistic_variables.append(ivfs)

                antecedents.append(fs.fuzzyVariable(
                    self.var_names[fuzzy_variable], linguistic_variables))
        else:
            try:
                antecedents = self.lvs[kwargs['time_moment']]
            except:
                antecedents = self.lvs


        for i in range(self.n_classes):
            if fz_type == fs.FUZZY_SETS.temporal:
                fz_type = self.lvs[0][0].inside_type()

            if fz_type == fs.FUZZY_SETS.t1:
                rule_base = rules.RuleBaseT1(antecedents, rule_list[i])
            elif fz_type == fs.FUZZY_SETS.t2:
                rule_base = rules.RuleBaseT2(antecedents, rule_list[i])
            elif fz_type == fs.FUZZY_SETS.gt2:
                rule_base = rules.RuleBaseGT2(antecedents, rule_list[i])
            

            if i == 0:
                res = rules.MasterRuleBase([rule_base], self.classes_names, ds_mode=self.ds_mode, allow_unknown=self.allow_unknown)
            else:
                res.add_rule_base(rule_base)

        res.rename_cons(self.classes_names)

        return res


    def _evaluate(self, x: np.array, out: dict, *args, **kwargs):
        '''
        :param x: array of train samples. x shape = features
            those features are the parameters to optimize.

        :param out: dict where the F field is the fitness. It is used from the outside.
        '''
        ruleBase = self._construct_ruleBase(x, self.fuzzy_type)

        if len(ruleBase.get_rules()) > 0:
            score = self.fitness_func(ruleBase, self.X, self.y, self.tolerance, self.alpha_, self.beta_, self._precomputed_truth)
        else:
            score = 0.0
        
        out["F"] = 1 - score
    

    def fitness_func(self, ruleBase: rules.RuleBase, X:np.array, y:np.array, tolerance:float, alpha:float=0.0, beta:float=0.0, precomputed_truth:np.array=None) -> float:
        '''
        Fitness function for the optimization problem.
        :param ruleBase: RuleBase object
        :param X: array of train samples. X shape = (n_samples, n_features)
        :param y: array of train labels. y shape = (n_samples,)
        :param tolerance: float. Tolerance for the size evaluation.
        :param alpha: float. Weight for the accuracy term.
        :param beta: float. Weight for the average rule size term.
        :param precomputed_truth: np array. If given, it will be used as the truth values for the evaluation.
        :return: float. Fitness value.
        '''
        if precomputed_truth is None:
            precomputed_truth = rules.compute_antecedents_memberships(ruleBase.antecedents, X)

        ev_object = evr.evalRuleBase(ruleBase, X, y, precomputed_truth=precomputed_truth)
        ev_object.add_full_evaluation()
        ruleBase.purge_rules(tolerance)

        if len(ruleBase.get_rules()) > 0: 
            score_acc = ev_object.classification_eval()
            score_rules_size = ev_object.size_antecedents_eval(tolerance)
            score_nrules = ev_object.effective_rulesize_eval(tolerance)

            score = score_acc + score_rules_size * alpha + score_nrules * beta
        else:
            score = 0.0
            
        return score
    

