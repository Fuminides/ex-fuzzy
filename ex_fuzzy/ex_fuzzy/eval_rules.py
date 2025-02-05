"""
This file contains the classes to perform rule classification evaluation.

"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

try:
    from . import rules
    from . import fuzzy_sets as fs
    from . import permutation_test as pt
    from . import bootstrapping_test as bt
except ImportError:
    import rules
    import fuzzy_sets as fs
    import permutation_test as pt
    import bootstrapping_test as bt


class evalRuleBase():
    '''
    Class to evaluate a set of rules given a evaluation dataset.
    '''

    def __init__(self, mrule_base: rules.MasterRuleBase, X: np.array, y: np.array, time_moments: np.array=None, precomputed_truth=None) -> None:
        '''
        Creates the object with the rulebase to evaluate and the data to use in the evaluation.

        :param mrule_base: The rule base to evaluate.
        :param X: array shape samples x features. The data to evaluate the rule base.
        :param y: array shape samples x 1. The labels of the data.
        :param time_moments: array shape samples x 1. The time moments of the samples. (Only for temporal rule bases)
        :return: None
        '''
        self.mrule_base = mrule_base
        self.X = X
        self.y = y
        self.time_moments = time_moments

        self.precomputed_truth = precomputed_truth

        if isinstance(y[0], str):
            consequents_names = self.mrule_base.get_consequents_names()
            self.y = np.array([list(consequents_names).index(str(y)) for y in y])




    def compute_antecedent_pattern_support(self, X: np.array=None) -> np.array:
        '''
        Computes the pattern support for each of the rules for the given X.
        Each pattern support firing strength is the result of the tnorm for all the antecedent memeberships,
        dvided by their number.

        :return: array of shape rules x 2
        '''
        data_X = X if X is not None else self.X
        precomputed_truth = self.precomputed_truth if self.precomputed_truth is not None else None

        if self.time_moments is None:
            antecedent_memberships = self.mrule_base.compute_firing_strenghts(
                data_X, precomputed_truth=precomputed_truth)
        else:
            antecedent_memberships = self.mrule_base.compute_firing_strenghts(
                data_X, self.time_moments)


        patterns = self._get_all_rules()

        if self.mrule_base.fuzzy_type() == fs.FUZZY_SETS.t1:
            res = np.zeros((len(patterns), ))
        elif self.mrule_base.fuzzy_type() == fs.FUZZY_SETS.t2:
            res = np.zeros((len(patterns), 2))
        elif self.mrule_base.fuzzy_type() == fs.FUZZY_SETS.gt2:
            res = np.zeros((len(patterns), 2))

        for ix, pattern in enumerate(patterns):
            pattern_firing_strength = antecedent_memberships[:, ix]
            res[ix] = np.mean(pattern_firing_strength)
            
        return res

    def compute_pattern_support(self, X: np.array=None, y: np.array=None) -> np.array:
        '''
        Computes the pattern support for each of the rules for the given X.
        Each pattern support firing strength is the result of the tnorm for all the antecedent memeberships,
        dvided by their number.

        :return: array of shape rules x 2
        '''
        data_X = X if X is not None else self.X
        data_y = y if y is not None else self.y
        precomputed_truth = self.precomputed_truth if self.precomputed_truth is not None else None

        if self.time_moments is None:
            antecedent_memberships = self.mrule_base.compute_firing_strenghts(
                data_X, precomputed_truth=precomputed_truth)
        else:
            antecedent_memberships = self.mrule_base.compute_firing_strenghts(
                data_X, self.time_moments)


        patterns = self._get_all_rules()

        if self.mrule_base.fuzzy_type() == fs.FUZZY_SETS.t1:
            res = np.zeros((len(patterns), ))
        elif self.mrule_base.fuzzy_type() == fs.FUZZY_SETS.t2:
            res = np.zeros((len(patterns), 2))
        elif self.mrule_base.fuzzy_type() == fs.FUZZY_SETS.gt2:
            res = np.zeros((len(patterns), 2))
        consequents = self.mrule_base.get_consequents()
        for ix, pattern in enumerate(patterns):
            consequent_match = np.equal(data_y, consequents[ix])
            pattern_firing_strength = antecedent_memberships[:, ix]



            if pattern_firing_strength[consequent_match].shape[0] > 0:
                    res[ix] = np.mean(pattern_firing_strength[consequent_match])
            

        return res


    def compute_aux_pattern_support(self) -> np.array:
        '''
        Computes the pattern support for each of the rules for each of the classes for the given X.
        Each pattern support firing strength is the result of the tnorm for all the antecedent memeberships,
        dvided by their number.

        :return: array of shape rules x 2
        '''
        if self.time_moments is None:
            antecedent_memberships = self.mrule_base.compute_firing_strenghts(
                self.X)
        else:
            antecedent_memberships = self.mrule_base.compute_firing_strenghts(
                self.X, self.time_moments)

        patterns = self._get_all_rules()
        n_classes = len(np.unique(self.y))

        if self.mrule_base.fuzzy_type() == fs.FUZZY_SETS.t1:
            res = np.zeros((len(patterns), n_classes, ))
        elif self.mrule_base.fuzzy_type() == fs.FUZZY_SETS.t2:
            res = np.zeros((len(patterns), n_classes, 2))
        elif self.mrule_base.fuzzy_type() == fs.FUZZY_SETS.gt2:
            res = np.zeros((len(patterns), n_classes, 2))

        for con_ix in range(n_classes):
            for ix, pattern in enumerate(patterns):
                consequent_match = self.y == con_ix
                pattern_firing_strength = antecedent_memberships[:, ix]

                # / pattern_firing_strength.shape[0]
                if pattern_firing_strength[consequent_match].shape[0] > 0:
                    res[ix, con_ix] = np.mean(pattern_firing_strength[consequent_match])
                else:
                    res[ix, con_ix] = pattern_firing_strength[consequent_match]

        return res


    def _get_all_rules(self) -> list[rules.RuleSimple]:
        '''
        Returns a list of all the rules in the master rule base.

        :return: list of rules.
        '''
        res = []
        for jx in self.mrule_base.get_rules():
                res.append(jx)

        return res


    def compute_pattern_confidence(self, X: np.array=None, y: np.array=None, precomputed_truth=None) -> np.array:
        '''
        Computes the pattern confidence for each of the rules for the given X.
        Each pattern confidence is the normalized firing strength.

        :returns: array of shape 1 x rules 
        '''
        data_X = X if X is not None else self.X
        data_y = y if y is not None else self.y
        precomputed_truth = self.precomputed_truth if self.precomputed_truth is not None else None

        if self.time_moments is None:
            antecedent_memberships = self.mrule_base.compute_firing_strenghts(
            data_X, precomputed_truth=precomputed_truth)
        else:
            antecedent_memberships = self.mrule_base.compute_firing_strenghts(
            data_X, self.time_moments, precomputed_truth=precomputed_truth)


        patterns = self._get_all_rules()

        if self.mrule_base.fuzzy_type() == fs.FUZZY_SETS.t1:
            res = np.zeros((len(patterns), ))
        elif self.mrule_base.fuzzy_type() == fs.FUZZY_SETS.t2:
            res = np.zeros((len(patterns), 2))
        elif self.mrule_base.fuzzy_type() == fs.FUZZY_SETS.gt2:
            res = np.zeros((len(patterns), 2))
        consequents = self.mrule_base.get_consequents()
        for ix, pattern in enumerate(patterns):
            antecedent_consequent_match = np.equal(data_y, consequents[ix])
            pattern_firing_strength = antecedent_memberships[:, ix]
            dem = np.sum(pattern_firing_strength)
            if dem == 0:


                res[ix] = 0
            else:
                res[ix] = np.sum(
                    pattern_firing_strength[antecedent_consequent_match]) / dem

        return res


    def compute_aux_pattern_confidence(self) -> np.array:
        '''
        Computes the pattern confidence for each of the rules for the given X.
        Each pattern confidence is the normalized firing strength.

        :returns: array of shape rules x classes
        '''
        if self.time_moments is None:
            antecedent_memberships = self.mrule_base.compute_firing_strenghts(
            self.X)
        else:
            antecedent_memberships = self.mrule_base.compute_firing_strenghts(
            self.X, self.time_moments)

        patterns = self._get_all_rules()
        n_classes = len(np.unique(self.y))
        if self.mrule_base.fuzzy_type() == fs.FUZZY_SETS.t1:
            res = np.zeros((len(patterns), n_classes, ))
        elif self.mrule_base.fuzzy_type() == fs.FUZZY_SETS.t2:
            res = np.zeros((len(patterns), n_classes, 2))
        elif self.mrule_base.fuzzy_type() == fs.FUZZY_SETS.gt2:
            res = np.zeros((len(patterns), n_classes, 2))

        for consequent in range(n_classes):
            for ix, pattern in enumerate(patterns):
                antecedent_consequent_match = self.y == consequent
                pattern_firing_strength = antecedent_memberships[:, ix]
                dem = np.sum(pattern_firing_strength)
                if dem == 0:
                    res[ix, consequent] = 0
                else:
                    res[ix, consequent] = np.sum(
                        pattern_firing_strength[antecedent_consequent_match]) / dem

        return res


    def dominance_scores(self) -> np.array:
        '''
        Returns the dominance score of each pattern for each rule.

        :return: array of shape rules x 2
        '''
        return self.compute_pattern_confidence() * self.compute_pattern_support()


    def association_degree(self) -> np.array:
        '''
        Returns the association degree of each rule for each sample.

        :return: vector of shape rules
        '''
        firing_strengths = self.mrule_base.compute_firing_strenghts(self.X)
        res = self.dominance_scores() * firing_strengths

        if (self.mrule_base[0].fuzzy_type() == fs.FUZZY_SETS.t2) or (self.mrule_base[0].fuzzy_type() == fs.FUZZY_SETS.gt2):
            res = np.mean(res, axis=2)

        return res


    def aux_dominance_scores(self) -> np.array:
        '''
        Returns the dominance score of each pattern for each rule.

        :return: array of shape rules x 2
        '''
        return self.compute_aux_pattern_confidence() * self.compute_aux_pattern_support()


    def add_rule_weights(self) -> None:
        '''
        Add dominance score field to each of the rules present in the master Rule Base.
        '''
        supports = self.compute_pattern_support()
        confidences = self.compute_pattern_confidence()
        scores = self.dominance_scores()

        aux_counter = 0
        rules = self.mrule_base.get_rules()
        for jx in range(len(rules)):
                rules[jx].score = scores[aux_counter]
                rules[jx].support = supports[aux_counter]
                rules[jx].confidence = confidences[aux_counter]
                
                aux_counter += 1
    

    def add_auxiliary_rule_weights(self) -> None:
        '''
        Add dominance score field to each of the rules present in the master Rule Base for each consequent.
        They are labeled as aux_score, aux_support and aux_confidence. (Because they are not the main rule weights)
        '''
        supports = self.compute_aux_pattern_support()
        confidences = self.compute_aux_pattern_confidence()
        scores = self.aux_dominance_scores()

        aux_counter = 0
        rules = self.mrule_base.get_rules()
        for jx in range(len(rules)):
                rules[jx].aux_score = scores[aux_counter]
                rules[jx].aux_support = supports[aux_counter]
                rules[jx].aux_confidence = confidences[aux_counter]
                
                aux_counter += 1


    def add_classification_metrics(self, X: np.array=None, y: np.array=None) -> None:
        '''
        Adds the accuracy of each rule in the master rule base. It also adds the f1, precision and recall scores.
        If X and y are None uses the train set.

        :param X: array of shape samples x features
        :param y: array of shape samples

        '''
        if X is not None:
            actual_X = X
            actual_y = y
        else:
            actual_X = self.X
            actual_y = self.y

        if isinstance(actual_y, list):
            actual_y = np.array(actual_y)
        
        if not hasattr(self.mrule_base.get_rules()[0], 'score'):
            self.add_rule_weights()

        if self.time_moments is None:
            winning_rules = self.mrule_base._winning_rules(actual_X, precomputed_truth=self.precomputed_truth, allow_unkown=self.mrule_base.allow_unknown)
            preds = self.mrule_base.winning_rule_predict(actual_X, precomputed_truth=self.precomputed_truth)
        else:
            winning_rules = self.mrule_base._winning_rules(actual_X, self.time_moments)
            preds = self.mrule_base.winning_rule_predict(actual_X, self.time_moments)

        # If preds and labels are not instances of the same type, we convert them to the same type
        consequents_names = self.mrule_base.get_consequents_names()
        if type(preds[0]) != type(actual_y[0]):
            if isinstance(actual_y[0], str):
                preds = np.array([consequents_names[p].index(str(p)) for p in preds])
            elif isinstance(preds[0], str):
                preds = np.array([consequents_names.index(str(p)) for p in preds])
                

        rules = self.mrule_base.get_rules()
        for jx in range(len(rules)):
                relevant_samples = winning_rules == jx
                if np.sum(relevant_samples) > 0:
                    relevant_labels = actual_y[relevant_samples]
                    relevant_preds = preds[relevant_samples]

                    rules[jx].accuracy = accuracy_score(relevant_labels, relevant_preds)
                else:
                    rules[jx].accuracy = 0.0
                
    
    def classification_eval(self) -> float:
        '''
        Returns the matthews correlation coefficient for a classification task using
        the rules evaluated.

        :return: mattews correlation coefficient. (float in [-1, 1])
        '''
        from sklearn.metrics import matthews_corrcoef
        self.add_rule_weights()
        preds = self.mrule_base.winning_rule_predict(self.X, precomputed_truth=self.precomputed_truth)

        self.mcc = matthews_corrcoef(self.y, preds)
        self.acc = accuracy_score(self.y, preds)

        return self.mcc


    def size_antecedents_eval(self, tolerance=0.1) -> float:
        '''
        Returns a score between 0 and 1, where 1 means that the rule base only contains almost no antecedents.

        0 means that the rule base contains all rules with more than {tolerance} DS, there are many of them and they have all possible antecedents.
        The more rules and antecedent per rules the lower this score is.

        :param tolerance: float in [0, 1]. The tolerance for the dominance score. Default 0.1
        :return: float in [0, 1] with the score.
        '''
        possible_rule_size = 0
        effective_rule_size = 0

        for rule_base in self.mrule_base.get_rulebases():
            if len(rule_base) > 0:
                for rule in rule_base:
                    rscore = np.mean(rule.score)
                    if rscore > tolerance:
                        possible_rule_size += len(rule.antecedents)
                        # No antecedents for this rule
                        if sum(np.array(rule.antecedents) != -1) == 0:
                            effective_rule_size += len(rule.antecedents)
                        else:
                            effective_rule_size += sum(
                                np.array(rule.antecedents) != -1)

            else:
                return 0.0  # If one consequent does not have rules, then we return 0.0

        try:
            rule_density = 1 - effective_rule_size / possible_rule_size  # Antecedents used
        except ZeroDivisionError:
            rule_density = 0.0

        return rule_density
    

    def effective_rulesize_eval(self, tolerance=0.1) -> float:
        '''
        Returns a score between 0 and 1, where 1 means that the rule base only contains almost no antecedents.

        0 means that the rule base contains all rules with more than {tolerance} DS, there are many of them and they have all possible antecedents.
        The more rules and antecedent per rules the lower this score is.

        :param tolerance: float in [0, 1]. The tolerance for the dominance score. Default 0.1
        :return: float in [0, 1] with the score.
        '''
        possible_rules = len(self.mrule_base.get_rules())
        effective_rules = 0

        for rule_base in self.mrule_base.get_rulebases():
            if len(rule_base) > 0:
                for rule in rule_base:
                    rscore = np.mean(rule.score)
                    if rscore > tolerance:
                        # No antecedents for this rule
                        if not np.all(np.equal(np.array(rule.antecedents), -1)):
                            effective_rules += 1
                        
            else:
                return 0.0  # If one consequent does not have rules, then we return 0.0

        try:
            rule_density =  effective_rules / possible_rules
        except ZeroDivisionError:
            rule_density = 0.0

        return rule_density


    def p_permutation_classifier_validation(self, n=100, r=10) -> float:
        '''
        Performs a boostrap test to evaluate the performance of the rule base.
        Returns the p-valuefor the label permutation test and the feature coalition test.

        :param n: int. Number of boostrap samples.
        :param r: int. Number of repetitions to estimate the original error rate.
        :return: p-value of the permutation test.
        '''
        test1 = pt.permutation_labels_test(self.mrule_base, self.X, self.y, k=n, r=r)
        test2 = pt.permute_columns_class_test(self.mrule_base, self.X, self.y, k=n, r=r)

        pt.rulewise_label_permutation_test(self.mrule_base, self.X, self.y, k=n, r=r)
        pt.rulewise_column_permutation_test(self.mrule_base, self.X, self.y, k=n, r=r)

        self.mrule_base.p_value_class_structure = test1
        self.mrule_base.p_value_feature_coalition = test2

        return test1, test2
    

    def p_bootstrapping_rules_validation(self, n=100) -> float:
        rules_p_values =  bt.compute_rule_p_value(self.mrule_base, self.X, self.y, n).flatten()

        for jx, rule in enumerate(self.mrule_base.get_rules()):
            rule.boot_p_value = rules_p_values[jx]
        
        confidence_interval = self.bootstrap_confidence_confinterval(self.X, self.y, n)
        support_interval = self.bootstrap_support_confinterval(self.X, self.y, n)
        joint_interval = self.jointprob_confinterval(self.X, self.y, n)

        for jx, rule in enumerate(self.mrule_base.get_rules()):
            rule.boot_confidence_interval = confidence_interval[:, jx]
            rule.boot_support_interval = support_interval[:, jx]
            rule.boot_jointprob_interval = joint_interval[:, jx]



    def add_full_evaluation(self):

        '''
        Adds classification scores, both Dominance Scores and accuracy metrics, for each individual rule.
        '''
        self.add_rule_weights()
        self.add_classification_metrics()
        self.classification_eval()
    

    def bootstrap_support_rules(self, X: np.array, y: np.array, n_samples: int):
        '''
        Bootstraps the support of the rules in the classifier.
        '''
        samples = bt.generate_bootstrap_samples(X, y, n_samples)
        supports = np.zeros((n_samples, len(self.mrule_base.get_rules())))
    

        for i, sample in enumerate(samples):
            supports[i] = self.compute_pattern_support(sample[0], sample[1])
    
        return supports


    def bootstrap_confidence_rules(self, X: np.array, y: np.array, n_samples: int):
        '''
        Bootstraps the confidence of the rules in the classifier.
        '''
        samples = bt.generate_bootstrap_samples(X, np.array(y), n_samples)
        confidences = np.zeros((n_samples, len(self.mrule_base.get_rules())))

        for i, sample in enumerate(samples):
            confidences[i] = self.compute_pattern_confidence(sample[0], sample[1])
    
        return confidences


    def bootstrap_jointprob_rules(self, X: np.array, y: np.array, n_samples: int):
        '''
        Bootstraps the joint probability of the rules in the classifier.
        '''
        samples = bt.generate_bootstrap_samples(X, np.array(y), n_samples)
        ant_support = np.zeros((n_samples, len(self.mrule_base.get_rules())))
        confidences = np.zeros((n_samples, len(self.mrule_base.get_rules())))


        for i, sample in enumerate(samples):
            confidences[i] = self.compute_pattern_confidence(sample[0], sample[1])
            ant_support[i] = self.compute_antecedent_pattern_support(sample[0])

        return ant_support * confidences


    def bootstrap_support_confinterval(self, X: np.array, y: np.array, n_samples: int):
        '''
        Bootstraps the support of the rules in the classifier.
        '''
        supports = self.bootstrap_support_rules(X, np.array(y), n_samples)
        conf_interval = np.percentile(supports, [2.5, 97.5], axis=0)
        return conf_interval



    def bootstrap_confidence_confinterval(self, X: np.array, y: np.array, n_samples: int):
        '''
        Bootstraps the confidence of the rules in the classifier.
        '''
        confidences = self.bootstrap_confidence_rules(X, np.array(y), n_samples)
        conf_interval = np.percentile(confidences, [2.5, 97.5], axis=0)
        return conf_interval
    

    def jointprob_confinterval(self, X: np.array, y: np.array, n_samples: int):
        '''
        Bootstraps the joint probability of the rules in the classifier.
        '''
        joint_probabilities = self.bootstrap_jointprob_rules(X, np.array(y), n_samples)
        joint_interval = np.percentile(joint_probabilities, [2.5, 97.5], axis=0)
        return joint_interval
