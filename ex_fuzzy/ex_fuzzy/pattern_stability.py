import numpy as np
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from sklearn.model_selection import train_test_split
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


def add_dicts(dict1, dict2):
    for key in dict2:
        try:
            dict1[key] += dict2[key]
        except KeyError:
            dict1[key] = dict2[key]
    
    return dict1

def concatenate_dicts(dict1, dict2):
    for key in dict2:
        try:
            dict1[key]
        except KeyError:
            dict1[key] = dict2[key]
    
    return dict1

class pattern_stabilizer():

    def __init__(self,  X, y, nRules: int = 30, nAnts: int = 4, fuzzy_type: fs.FUZZY_SETS = fs.FUZZY_SETS.t1, tolerance: float = 0.0, class_names: list[str] = None,
                 n_linguistic_variables: int = 3, verbose=False, linguistic_variables: list[fs.fuzzyVariable] = None,
                 domain: list[float] = None, n_class: int=None, runner: int=1) -> None:
        
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
        '''
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
        

        if runner > 1:
            pool = ThreadPool(runner)
            self.thread_runner = StarmapParallelization(pool.starmap)
        else:
            self.thread_runner = None
        
        if linguistic_variables is not None:
            # If the linguistic variables are precomputed then we act accordingly
            self.lvs = linguistic_variables
            self.n_linguist_variables = [len(lv.linguistic_variable_names()) for lv in self.lvs]
            self.domain = None
            self.fuzzy_type = self.lvs[0].fuzzy_type()
        else:
            # If not, then we need the parameters sumistered by the user.
            self.lvs = None
            self.fuzzy_type = fuzzy_type
            self.n_linguist_variables = n_linguistic_variables
            self.domain = domain

        self.alpha_ = 0.0
        self.beta_ = 0.0

        self.X = X
        self.y = y


    def generate_solutions(self, n=30):
        rule_bases = []
        accs = []

        for ix in range(n):
            fl_classifier = evf.BaseFuzzyRulesClassifier(nRules=10, linguistic_variables=self.lvs, nAnts=3, n_linguistic_variables=5, fuzzy_type=self.fuzzy_type, verbose=False, tolerance=0.01, runner=1)
            # Generate train test partition
            
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=ix)
            fl_classifier.fit(X_train, np.array(y_train), n_gen=10, pop_size=10, checkpoints=0)

            rule_bases.append(fl_classifier.rule_base)
            accuracy = np.mean(np.equal(fl_classifier.forward(X_test), np.array(y_test)))
            accs.append(accuracy)
        
        return rule_bases, accs
    

    def count_unique_patterns(self, rule_base: rl.RuleBase):
        # We will count the number of unique patterns in the rule base
        unique_patterns = {}
        patterns_ds = {}
        var_used = {}

        for ix, rule in enumerate(rule_base.get_rulebase_matrix()):
            pattern = str(rule)
            patterns_ds[pattern] = rule_base[ix].score
            if pattern in unique_patterns:
                unique_patterns[pattern] += 1
            else:
                unique_patterns[pattern] = 1

            for jx, var in enumerate(rule):
                try:
                    var_used[jx][var] += 1
                except:
                    try:
                        var_used[jx][var] = 1
                    except:
                        var_used[jx] = {}
                        var_used[jx][var] = 1
        

        
        return unique_patterns, patterns_ds, var_used
    

    def count_unique_patterns_all_classes(self, mrule_base: rl.MasterRuleBase, class_patterns: dict[list] = None, patterns_dss: dict[list] = None, class_vars: dict[list] = None):
        if class_patterns is None:
            class_patterns = {ix: {} for ix in range(len(mrule_base))}
            class_vars = {}
            for key in range(len(mrule_base)):
                class_vars[key] = {}
                for jx in range(len(mrule_base.n_linguistic_variables())):
                    class_vars[key][jx] = {zx: 0 for zx in np.arange(-1, mrule_base.n_linguistic_variables()[key])}

            patterns_dss = {ix: {} for ix in range(len(mrule_base))}

        for ix, rule_base in enumerate(mrule_base):
            unique_patterns, patterns_ds, var_used = self.count_unique_patterns(rule_base)
            class_patterns[ix] = add_dicts(class_patterns[ix], unique_patterns)
            for key, value in class_vars.items():
                class_vars[ix][key] = add_dicts(class_vars[ix][key], var_used[key])

            patterns_dss[ix] = concatenate_dicts(patterns_dss[ix], patterns_ds)
            

        return class_patterns, patterns_dss, class_vars
    

    def get_patterns_scores(self, n=30):
        rule_bases, accuracies = self.generate_solutions(n)

        for ix, mrule_base in enumerate(rule_bases):
            if ix == 0:
                class_patterns, patterns_dss, class_vars = self.count_unique_patterns_all_classes(mrule_base)
            else:
                class_patterns, patterns_dss, class_vars = self.count_unique_patterns_all_classes(mrule_base, class_patterns, patterns_dss, class_vars)
            

        return class_patterns, patterns_dss, class_vars, accuracies


    @staticmethod
    def transform_unique_patterns2matrix(unique_patterns: dict):
        # We will transform the unique patterns into a matrix
        matrix = np.zeros((len(unique_patterns), len(unique_patterns[0])))
        for ix, pattern in enumerate(unique_patterns):
            matrix[ix, :] = pattern
        
        return matrix
    


            
