# -*- coding: utf-8 -*-
"""
This is a the source file that contains the Rule, RuleBase and MasterRuleBase classes to perform fuzzy inference.

"""
import abc

import numpy as np
try:
    from . import fuzzy_sets as fs
    from . import centroid
except ImportError:
    import fuzzy_sets as fs
    import centroid


class RuleError(Exception):
    '''
    Exception raised when a rule is not well defined.
    '''

    def __init__(self, message: str) -> None:
        '''
        Constructor of the RuleError class.

        :param message: message of the error.
        '''
        super().__init__(message)

def _myprod(x: np.array, y: np.array) -> np.array:
    '''
    Proxy function to change the product operation interface
    '''
    return x*y


class Rule():
    '''
    Class of Rule designed to work with one single rule. It contains the whole inference functionally in itself.
    '''

    def __init__(self, antecedents: list[fs.FS], consequent: fs.FS) -> None:
        '''
        Creates a rule with the given antecedents and consequent.

        :param antecedents: list of fuzzy sets.
        :param consequent: fuzzy set.
        '''
        self.antecedents = antecedents
        self.consequent = consequent

    def membership(self, x: np.array, tnorm=_myprod) -> np.array:
        '''
        Computes the membership of one input to the antecedents of the rule.

        :param x: input to compute the membership.
        :param tnorm: t-norm to use in the inference process.
        '''
        for ix, antecedent in enumerate(self.antecedents):
            if ix == 0:
                res = antecedent.membership(x)
            else:
                res = tnorm(res, antecedent.membership(x))

        return res

    def consequent_centroid(self) -> np.array:
        '''
        Returns the centroid of the consequent using a Karnik and Mendel algorithm.
        '''
        try:
            return self.centroid_consequent
        except AttributeError:
            consequent_domain = self.consequent.domain
            domain_linspace = np.arange(
                consequent_domain[0], consequent_domain[1], 0.05)
            consequent_memberships = self.consequent.membership(
                domain_linspace)

            self.centroid_consequent = centroid.compute_centroid_fs(
                domain_linspace, consequent_memberships)

            return self.centroid_consequent


class RuleSimple():
    '''
    Class designed to represent rules in its simplest form to optimize the computation in a rulebase.

    It indicates the antecedent values using a vector coded in this way:
        ith entry: -1 means this variable is not used. 0-N indicates that the variable linguistic used for the ith input is that one.
    '''

    def __init__(self, antecedents: list[int], consequent: int=0) -> None:
        '''
        Creates a rule with the given antecedents and consequent.

        :param antecedents: list of integers indicating the linguistic variable used for each input.
        :param consequent: integer indicating the linguistic variable used for the consequent.
        '''
        self.antecedents = list(map(int, antecedents))
        self.consequent = int(consequent)


    def __getitem__(self, ix):
        '''
        Returns the antecedent value for the given index.

        :param ix: index of the antecedent to return.
        '''
        return self.antecedents[ix]


    def __setitem__(self, ix, value):
        '''
        Sets the antecedent value for the given index.

        :param ix: index of the antecedent to set.
        :param value: value to set.
        '''
        self.antecedents[ix] = value
    

    def __str__(self):
        '''
        Returns a string representation of the rule.
        '''
        return 'Rule: antecedents: ' + str(self.antecedents) + ' consequent: ' + str(self.consequent)
    

    def __len__(self):
        '''
        Returns the number of antecedents in the rule.
        '''
        return len(self.antecedents)
    

    def __eq__(self, other):
        '''
        Returns True if the two rules are equal.
        '''
        return self.antecedents == other.antecedents and self.consequent == other.consequent
    

    def __hash__(self):
        '''
        Returns the hash of the rule.
        '''
        return hash(str(self))


class RuleBase():
    '''
    Class optimized to work with multiple rules at the same time. Right now supports only one consequent. (Solution: use one rulebase per consequent to study)
    '''

    def __init__(self, antecedents: list[fs.fuzzyVariable], rules: list[RuleSimple], consequent: fs.fuzzyVariable = None, tnorm=np.prod) -> None:
        '''
        Creates a rulebase with the given antecedents, rules and consequent.

        :param antecedents: list of fuzzy sets.
        :param rules: list of rules.
        :param consequent: fuzzy set.
        :param tnorm: t-norm to use in the inference process.
        '''
        rules = self.delete_rule_duplicates(rules)
        self.rules = rules
        self.antecedents = antecedents
        self.consequent = consequent
        self.tnorm = tnorm
        self.consequent_centroids = np.zeros(
            (len(consequent.linguistic_variable_names()), 2))
        for ix, (name, vl_consequent) in enumerate(consequent.linguistic_variables.items()):
            consequent_domain = vl_consequent.domain
            domain_linspace = np.arange(
                consequent_domain[0], consequent_domain[1], 0.05)
            consequent_memberships = vl_consequent.membership(domain_linspace)

            self.consequent_centroids[ix, :] = centroid.compute_centroid_iv(
                domain_linspace, consequent_memberships)

        self.consequent_centroids_rules = np.zeros((len(self.rules), 2))
        for ix, rule in enumerate(self.rules):
            consequent_ix = rule.consequent
            self.consequent_centroids_rules[ix] = self.consequent_centroids[consequent_ix]

        self.delete_duplicates()


    def get_rules(self) -> list[RuleSimple]:
        '''
        Returns the list of rules in the rulebase.
        '''
        return self.rules


    def add_rule(self, new_rule: RuleSimple):
        '''
        Adds a new rule to the rulebase.
        :param new_rule: rule to add.
        '''
        self.rules.append(new_rule)


    def add_rules(self, new_rules: list[RuleSimple]):
        '''
        Adds a list of new rules to the rulebase.

        :param new_rules: list of rules to add.
        '''
        self.rules += new_rules


    def remove_rule(self, ix: int) -> None:
        '''
        Removes the rule in the given index.
        :param ix: index of the rule to remove.
        '''
        del self.rules[ix]


    def remove_rules(self, delete_list: list[int]) -> None:
        '''
        Removes the rules in the given list of indexes.

        :param delete_list: list of indexes of the rules to remove.
        '''
        self.rules = [rule for ix, rule in enumerate(
            self.rules) if ix not in delete_list]


    def get_rulebase_matrix(self):
        '''
        Returns a matrix with the antecedents values for each rule.
        '''
        res = np.zeros((len(self.rules), len(self.antecedents)))

        for ix, rule in enumerate(self.rules):
            res[ix] = rule

        return res


    def get_scores(self):
        '''
        Returns an array with the dominance score for each rule.
        (Must been already computed by an evalRule object)

        '''
        res = np.zeros((len(self.rules, )))
        for ix, rule in enumerate(self.rules):
            res[ix] = rule.score

        return res


    def delete_rule_duplicates(self, list_rules:list[RuleSimple]):
        # Delete the rules that are duplicated in the rule list
        unique = {}
        for ix, rule in enumerate(list_rules):
            try:
                unique[rule]
            except KeyError:
                unique[rule] = ix

        new_list = [list_rules[x] for x in unique.values()]
        
        return new_list
        

    def compute_antecedents_memberships(self, x: np.array) -> list[dict]:
        '''
        Returns a list of of dictionaries that contains the memberships for each x value to the ith antecedents, nth linguistic variable.
        x must be a vector (only one sample)

        :param x: vector with the values of the inputs.
        :return: a list with the antecedent truth values for each one. Each list is comprised of a list with n elements, where n is the number of linguistic variables in each variable.
        '''
        if len(self.rules) > 0:
            cache_antecedent_memberships = []

            for ix, antecedent in enumerate(self.antecedents):
                cache_antecedent_memberships.append(
                    antecedent.compute_memberships(x[:, ix]))

            return cache_antecedent_memberships

        else:
            if self.fuzzy_type() == fs.FUZZY_SETS.t1:
                return [np.zeros((x.shape[0], 1))]
            elif self.fuzzy_type() == fs.FUZZY_SETS.t2:
                return [np.zeros((x.shape[0], 1, 2))]
            elif self.fuzzy_type() == fs.FUZZY_SETS.gt2:
                return [np.zeros((x.shape[0], len(self.alpha_cuts), 2))]


    def compute_rule_antecedent_memberships(self, x: np.array, scaled=False) -> np.array:
        '''
        Computes the antecedent truth value of an input array.

        Return an array in shape samples x rules (x 2) (last is iv dimension)

        :param x: array with the values of the inputs.
        :param scaled: if True, the memberships are scaled according to their sums for each sample.
        :return: array with the memberships of the antecedents for each rule.
        '''
        if self.fuzzy_type() == fs.FUZZY_SETS.t2:
            res = np.zeros((x.shape[0], len(self.rules), 2))
        elif self.fuzzy_type() == fs.FUZZY_SETS.t1:
            res = np.zeros((x.shape[0], len(self.rules), ))
        elif self.fuzzy_type() == fs.FUZZY_SETS.gt2:
            res = np.zeros(
                (x.shape[0], len(self.rules), len(self.alpha_cuts), 2))

        antecedents_memberships = self.compute_antecedents_memberships(x)

        for jx, rule in enumerate(self.rules):
            rule_antecedents = rule.antecedents
            initiated = False
            for ix, vl in enumerate(rule_antecedents):
                if vl >= 0:
                    if not initiated:
                        membership = list(antecedents_memberships[ix])[vl]
                        initiated = True
                    else:
                        membership = self.tnorm(membership, list(
                            antecedents_memberships[ix])[vl])

            try:
                res[:, jx] = membership
            except UnboundLocalError:
                pass  # All the antecedents are dont care.

        if scaled:
            if self.fuzzy_type() == fs.FUZZY_SETS.t1:
                non_zero_rows = np.sum(res, axis=1) > 0
                res[non_zero_rows] = res[non_zero_rows] / \
                    np.sum(res[non_zero_rows], axis=1, keepdims=True)

            elif self.fuzzy_type() == fs.FUZZY_SETS.t2:
                non_zero_rows = np.sum(res[:, :, 0], axis=1) > 0
                res[non_zero_rows, :, 0] = res[non_zero_rows, :, 0] / \
                    np.sum(res[non_zero_rows, :, 0], axis=1, keepdims=True)
                non_zero_rows = np.sum(res[:, :, 1], axis=1) > 0
                res[non_zero_rows, :, 1] = res[non_zero_rows, :, 1] / \
                    np.sum(res[non_zero_rows, :, 1], axis=1, keepdims=True)
            
            elif self.fuzzy_type() == fs.FUZZY_SETS.gt2:

                for ix, alpha in enumerate(self.alpha_cuts):
                    relevant_res = res[:, :, ix, :]

                    non_zero_rows = np.sum(relevant_res[:, :, 0], axis=1) > 0
                    relevant_res[non_zero_rows, :, 0] = relevant_res[non_zero_rows, :, 0] / \
                        np.sum(relevant_res[non_zero_rows, :, 0], axis=1, keepdims=True)
                    non_zero_rows = np.sum(relevant_res[:, :, 1], axis=1) > 0
                    relevant_res[non_zero_rows, :, 1] = relevant_res[non_zero_rows, :, 1] / \
                        np.sum(relevant_res[non_zero_rows, :, 1], axis=1, keepdims=True)

                    res[:, :, ix, :] = relevant_res
                    
        return res


    def print_rules(self, return_rules:bool=False) -> None:
        '''
        Print the rules from the rule base.
        '''
        all_rules = ''
        for ix, rule in enumerate(self.rules):
            initiated = False
            str_rule = 'IF '
            for jx, vl_ix in enumerate(rule.antecedents):
                antecedent = self.antecedents[jx]
                keys = antecedent.linguistic_variable_names()

                if rule[jx] != -1:
                    if not initiated:
                        str_rule += str(antecedent.name) + ' IS ' + str(keys[vl_ix])
                        initiated = True
                    else:
                        str_rule += ' AND ' + str(antecedent.name) + \
                            ' IS ' + str(keys[vl_ix])

            try:
                score = rule.score if self.fuzzy_type() == fs.FUZZY_SETS.t1 else np.mean(rule.score)
                str_rule += ' WITH DS ' + str(score)

                # If the classification scores have been computed, print them.
                try:
                    str_rule += ', ACC ' + str(rule.accuracy)

                except AttributeError:
                    pass
            except AttributeError:
                str_rule += ' THEN consequent vl is ' + str(rule.consequent)
            
            all_rules += str_rule + '\n'

        if not return_rules:
            print(all_rules)
        else:
            return all_rules


    @abc.abstractmethod
    def inference(self, x: np.array) -> np.array:
        '''
        Computes the fuzzy output of the fl inference system.

        Return an array in shape samples x 2 (last is iv dimension)

        :param x: array with the values of the inputs.
        :return: array with the memberships of the consequents for each rule.
        '''
        raise NotImplementedError


    @abc.abstractmethod
    def forward(self, x: np.array) -> np.array:
        '''
        Computes the deffuzified output of the fl inference system.

        Return a vector of size (samples, )

        :param x: array with the values of the inputs.
        :return: array with the deffuzified output.
        '''
        raise NotImplementedError


    @abc.abstractmethod
    def fuzzy_type(self) -> fs.FUZZY_SETS:
        '''
        Returns the corresponding type of the RuleBase using the enum type in the fuzzy_sets module.

        :return: the type of fuzzy set used in the RuleBase.
        '''
        raise NotImplementedError


    def __len__(self):
        '''
        Returns the number of rules in the rule base.
        '''
        return len(self.rules)


    def prune_bad_rules(self, tolerance=0.01) -> None:
        '''
        Delete the rules from the rule base that do not have a dominance score superior to the threshold or have 0 accuracy in the training set.

        :param tolerance: threshold for the dominance score.
        '''
        delete_list = []
        try:
            for ix, rule in enumerate(self.rules):
                score = rule.score
                if (self.fuzzy_type() == fs.FUZZY_SETS.t2) or (self.fuzzy_type() == fs.FUZZY_SETS.gt2):
                    score = np.mean(score)

                if score < tolerance or rule.accuracy == 0.0:
                    delete_list.append(ix)
        except AttributeError:
            assert False, 'Dominance scores not computed for this rulebase'

        self.remove_rules(delete_list)


    def scores(self) -> np.array:
        '''
        Returns the dominance score for each rule.

        :return: array with the dominance score for each rule.
        '''
        scores = []
        for rule in self.rules:
            scores.append(rule.score)

        return np.array(scores)


    def __getitem__(self, item: int) -> RuleSimple:
        '''
        Returns the corresponding rulebase.

        :param item: index of the rule.
        :return: the corresponding rule.
        '''
        return self.rules[item]


    def __setitem__(self, key: int, value: RuleSimple) -> None:
        '''
        Set the corresponding rule.

        :param key: index of the rule.
        :param value: new rule.
        '''
        self.rules[key] = value
    

    def __iter__(self):
        '''
        Returns an iterator for the rule base.
        '''
        return iter(self.rules)


    def __str__(self):
        '''
        Returns a string with the rules in the rule base.
        '''
        str_res = ''
        for rule in self.rules:
            str_res += str(rule) + '\n'
        return str_res
    

    def __eq__(self, other):
        '''
        Returns True if the two rule bases are equal.
        '''
        return self.rules == other.rules
    

    def __hash__(self):
        '''
        Returns the hash of the rule base.
        '''
        return hash(str(self))
    

    def __add__(self, other):
        '''
        Adds two rule bases.
        '''
        return RuleBase(self.antecedents, self.rules + other.rules, self.consequent, self.tnorm)
    

    def n_linguist_variables(self) -> int:
        '''
        Returns the number of linguistic variables in the rule base.
        '''
        return len(self.antecedents)

    
class RuleBaseT2(RuleBase):
    '''
    Class optimized to work with multiple rules at the same time. Supports only one consequent. 
    (Use one rulebase per consequent to study classification problems. Check MasterRuleBase class for more documentation)

    This class supports iv approximation for t2 fs.
    '''

    def __init__(self, antecedents: list[fs.fuzzyVariable], rules: list[RuleSimple], consequent: fs.fuzzyVariable = None, tnorm=_myprod) -> None:
        '''
        Constructor of the RuleBaseT2 class.

        :param antecedents: list of fuzzy variables that are the antecedents of the rules.
        :param rules: list of rules.
        :param consequent: fuzzy variable that is the consequent of the rules.
        :param tnorm: t-norm used to compute the fuzzy output.
        '''
        rules = self.delete_rule_duplicates(rules)
        self.rules = rules
        self.antecedents = antecedents
        self.consequent = consequent
        self.tnorm = tnorm

        if consequent is not None:
            self.consequent_centroids = np.zeros(
                (len(consequent.linguistic_variable_names()), 2))

            for ix, vl_consequent in enumerate(consequent.linguistic_variables):
                consequent_domain = vl_consequent.domain
                domain_linspace = np.arange(
                    consequent_domain[0], consequent_domain[1], 0.05)
                consequent_memberships = vl_consequent.membership(
                    domain_linspace)

                self.consequent_centroids[ix, :] = centroid.compute_centroid_iv(
                    domain_linspace, consequent_memberships)

            self.consequent_centroids_rules = np.zeros((len(self.rules), 2))
            # If 0, we are classifying and we do not need the consequent centroids.
            if len(self.consequent_centroids) > 0:
                for ix, rule in enumerate(self.rules):
                    consequent_ix = rule.consequent
                    self.consequent_centroids_rules[ix] = self.consequent_centroids[consequent_ix]


    def inference(self, x: np.array) -> np.array:
        '''
        Computes the iv output of the t2 inference system.

        Return an array in shape samples x 2 (last is iv dimension)

        :param x: array with the values of the inputs.
        :return: array with the memberships of the consequents for each sample.
        '''
        res = np.zeros((x.shape[0], 2))

        antecedent_memberships = self.compute_rule_antecedent_memberships(x)
        for sample in range(antecedent_memberships.shape[0]):
            res[sample, :] = centroid.consequent_centroid(
                antecedent_memberships[sample], self.consequent_centroids_rules)

        return res


    def forward(self, x: np.array) -> np.array:
        '''
        Computes the deffuzified output of the t2 inference system.

        Return a vector of size (samples, )

        :param x: array with the values of the inputs.
        :return: array with the deffuzified output for each sample.
        '''
        return np.mean(self.inference(x))


    def fuzzy_type(self) -> fs.FUZZY_SETS:
        '''
        Returns the correspoing type of the RuleBase using the enum type in the fuzzy_sets module.

        :return: the corresponding fuzzy set type of the RuleBase.
        '''
        return fs.FUZZY_SETS.t2


class RuleBaseGT2(RuleBase):
    '''
    Class optimized to work with multiple rules at the same time. Supports only one consequent. 
    (Use one rulebase per consequent to study classification problems. Check MasterRuleBase class for more documentation)

    This class supports gt2 fs. (ONLY FOR CLASSIFICATION PROBLEMS)
    '''

    def __init__(self, antecedents: list[fs.fuzzyVariable], rules: list[RuleSimple], consequent: fs.fuzzyVariable = None, tnorm=_myprod) -> None:
        '''
        Constructor of the RuleBaseGT2 class.

        :param antecedents: list of fuzzy variables that are the antecedents of the rules.
        :param rules: list of rules.
        :param consequent: fuzzy variable that is the consequent of the rules.
        :param tnorm: t-norm used to compute the fuzzy output.
        '''
        rules = self.delete_rule_duplicates(rules)
        self.rules = rules
        self.antecedents = antecedents
        self.consequent = consequent
        self.tnorm = tnorm
        self.alpha_cuts = antecedents[0][0].alpha_cuts


    def inference(self, x: np.array) -> np.array:
        '''
        Computes the output of the gt2 inference system.

        Return an array in shape samples x alpha_cuts

        :param x: array with the values of the inputs.
        :return: array with the memberships of the consequents for each sample.
        '''
        res = np.zeros((x.shape[0], 2))

        antecedent_memberships = self.compute_rule_antecedent_memberships(x)
        for sample in range(antecedent_memberships.shape[0]):
            res[sample, :] = centroid.consequent_centroid(
                antecedent_memberships[sample], self.consequent_centroids_rules)

        return res


    def _alpha_reduction(self, x) -> np.array:
        '''
        Computes the type reduction to reduce the alpha cuts to one value.

        :param x: array with the values of the inputs.
        :return: array with the memberships of the consequents for each sample.
        '''
        formtatted = np.expand_dims(np.expand_dims(np.expand_dims(
            np.array(self.alpha_cuts), axis=1), axis=0), axis=0)
        return np.sum(formtatted * x, axis=2) / np.sum(self.alpha_cuts)


    def forward(self, x: np.array) -> np.array:
        '''
        Computes the deffuzified output of the t2 inference system.

        Return a vector of size (samples, )

        :param x: array with the values of the inputs.
        :return: array with the deffuzified output for each sample.
        '''
        return np.sum(np.array(self.alpha_cuts) * (self.inference(x)), axis=1) / np.sum(self.alpha_cuts)


    def fuzzy_type(self) -> fs.FUZZY_SETS:
        '''
        Returns the correspoing type of the RuleBase using the enum type in the fuzzy_sets module.

        :return: the corresponding fuzzy set type of the RuleBase.
        '''
        return fs.FUZZY_SETS.gt2


    def compute_rule_antecedent_memberships(self, x: np.array, scaled=True) -> np.array:
        '''
        Computes the membership for the antecedents performing the alpha_cut reduction.

        :param x: array with the values of the inputs.
        :param scaled: if True, the memberships are scaled to sum 1 in each sample.
        '''
        antecedent_membership = super().compute_rule_antecedent_memberships(x, scaled)
        return self._alpha_reduction(antecedent_membership)


    def alpha_compute_rule_antecedent_memberships(self, x: np.array, scaled=True) -> np.array:
        '''
        Computes the membership for the antecedents for all the alpha cuts.

        :param x: array with the values of the inputs.
        :param scaled: if True, the memberships are scaled to sum 1 in each sample.
        :return: array with the memberships of the antecedents for each sample.
        '''
        return super().compute_rule_antecedent_memberships(x, scaled)


class RuleBaseT1(RuleBase):
    '''
    Class optimized to work with multiple rules at the same time. Supports only one consequent.
    (Use one rulebase per consequent to study classification problems. Check MasterRuleBase class for more documentation)

    This class supports t1 fs.
    '''

    def __init__(self, antecedents: list[fs.fuzzyVariable], rules: list[RuleSimple], consequent: fs.fuzzyVariable = None, tnorm=_myprod) -> None:
        '''
        Constructor of the RuleBaseT1 class.

        :param antecedents: list of fuzzy variables that are the antecedents of the rules.
        :param rules: list of rules.
        :param consequent: fuzzy variable that is the consequent of the rules. ONLY on regression problems.
        :param tnorm: t-norm used to compute the fuzzy output.
        '''
        rules = self.delete_rule_duplicates(rules)
        self.rules = rules
        self.antecedents = antecedents
        self.consequent = consequent
        self.tnorm = tnorm

        if consequent is not None:
            self.consequent_centroids = np.zeros(
                (len(consequent.linguistic_variable_names()), ))

            for ix, (name, vl_consequent) in enumerate(consequent.linguistic_variables.items()):
                consequent_domain = vl_consequent.domain
                domain_linspace = np.arange(
                    consequent_domain[0], consequent_domain[1], 0.05)
                consequent_memberships = vl_consequent.membership(
                    domain_linspace)

                self.consequent_centroids[ix] = centroid.center_of_masses(
                    domain_linspace, consequent_memberships)

            self.consequent_centroids_rules = np.zeros((len(self.rules), ))
            for ix, rule in enumerate(self.rules):
                consequent_ix = rule.consequent
                self.consequent_centroids_rules[ix] = self.consequent_centroids[consequent_ix]


    def inference(self, x: np.array) -> np.array:
        '''
        Computes the output of the t1 inference system.

        Return an array in shape samples.

        :param x: array with the values of the inputs.
        :return: array with the output of the inference system for each sample.
        '''
        res = np.zeros((x.shape[0], 2))

        antecedent_memberships = self.compute_rule_antecedent_memberships(x)

        for sample in range(antecedent_memberships.shape[0]):
            res[sample, :] = centroid.center_of_masses(
                self.consequent_centroids_rules, antecedent_memberships[sample])

        return res

    def forward(self, x: np.array) -> np.array:
        '''
        Same as inference() in the t1 case.

        Return a vector of size (samples, )

        :param x: array with the values of the inputs.
        :return: array with the deffuzified output for each sample.
        '''
        return self.inference(x)


    def fuzzy_type(self) -> fs.FUZZY_SETS:
        '''
        Returns the correspoing type of the RuleBase using the enum type in the fuzzy_sets module.

        :return: the corresponding fuzzy set type of the RuleBase.
        '''
        return fs.FUZZY_SETS.t1


class MasterRuleBase():
    '''
    This Class encompasses a list of rule bases where each one corresponds to a different class.
    '''

    def __init__(self, rule_base: list[RuleBase], consequent_names: list[str]=None) -> None:
        '''
        Constructor of the MasterRuleBase class.

        :param rule_base: list of rule bases.
        '''
        if len(rule_base) == 0:
            raise RuleError('No rule bases given!')
        
        self.rule_bases = rule_base
        self.antecedents = rule_base[0].antecedents

        if consequent_names is None:
            self.consequent_names = [str(ix) for ix in range(len(self.rule_bases))]
        else:
            self.consequent_names = consequent_names


    def rename_cons(self, consequent_names: list[str]) -> None:
        '''
        Renames the consequents of the rule base.
        '''
        self.consequent_names = consequent_names


    def add_rule(self, rule: RuleSimple, consequent: int) -> None:
        '''
        Adds a rule to the rule base of the given consequent.

        :param rule: rule to add.
        :param consequent: index of the rule base to add the rule.
        '''
        self.rule_bases[consequent].add_rule(rule)
    

    def get_consequents(self) -> list[int]:
        '''
        Returns a list with the consequents of each rule base.

        :return: list with the consequents of each rule base.
        '''
        return sum([[ix]*len(x) for ix, x in enumerate(self.rule_bases)], [])


    def get_rulebase_matrix(self) -> list[np.array]:
        '''
        Returns a list with the rulebases for each antecedent in matrix format.

        :return: list with the rulebases for each antecedent in matrix format.
        '''
        return [x.get_rulebase_matrix() for x in self.rule_bases]


    def get_scores(self) -> np.array:
        '''
        Returns the dominance score for each rule in all the rulebases.

        :return: array with the dominance score for each rule in all the rulebases.
        '''
        res = []
        for rb in self.rule_bases:
            res.append(rb.scores())

        res = [x for x in res if len(x) > 0]
        
        return np.concatenate(res, axis=0)


    def compute_firing_strenghts(self, X) -> np.array:
        '''
        Computes the firing strength of each rule for each sample.

        :param X: array with the values of the inputs.
        :return: array with the firing strength of each rule for each sample.
        '''
        aux = []
        for ix in range(len(self.rule_bases)):
            aux.append(self[ix].compute_rule_antecedent_memberships(X))

        # Firing strengths shape: samples x rules
        return np.concatenate(aux, axis=1)


    def _winning_rules(self, X: np.array) -> np.array:
        '''
        Returns the winning rule for each sample. Takes into account dominance scores if already computed.

        :param X: array with the values of the inputs.
        :return: array with the winning rule for each sample.
        '''
        
        firing_strengths = self.compute_firing_strenghts(X)

        association_degrees = self.get_scores() * firing_strengths

        if (self[0].fuzzy_type() == fs.FUZZY_SETS.t2) or (self[0].fuzzy_type() == fs.FUZZY_SETS.gt2):
            association_degrees = np.mean(association_degrees, axis=2)
        elif self[0].fuzzy_type() == fs.FUZZY_SETS.gt2:
            association_degrees = np.mean(association_degrees, axis=3)

        winning_rules = np.argmax(association_degrees, axis=1)

        return winning_rules
    
    
    def winning_rule_predict(self, X: np.array) -> np.array:
        '''
        Returns the winning rule for each sample. Takes into account dominance scores if already computed.

        :param X: array with the values of the inputs.
        :return: array with the winning rule for each sample.
        '''
        consequents = sum([[ix]*len(self[ix].rules)
                          for ix in range(len(self.rule_bases))], [])  # The sum is for flatenning the list
        winning_rules = self._winning_rules(X)

        return np.array([consequents[ix] for ix in winning_rules])


    def add_rule_base(self, rule_base: RuleBase) -> None:
        '''
        Adds a rule base to the list of rule bases.

        :param rule_base: rule base to add.
        '''
        self.rule_bases.append(rule_base)

        if len(self.rule_bases) != len(self.consequent_names):
            # We did not give proper names to the consequents
            self.consequent_names = [str(ix) for ix in range(len(self.rule_bases))]


    def print_rules(self, return_rules=False) -> None:
        '''
        Print all the rules for all the consequents.

        :param autoprint: if True, the rules are printed. If False, the rules are returned as a string.
        '''
        res = ''
        for ix, ruleBase in enumerate(self.rule_bases):    
            res += 'Rules for consequent: ' + str(self.consequent_names[ix]) + '\n'
            
            res += '----------------\n'
            res += ruleBase.print_rules(return_rules=True) + '\n'
        
        
        if return_rules:
            return res
        else:
            print(res)


    def get_rules(self) -> list[RuleSimple]:
        '''
        Returns a list with all the rules.

        :return: list with all the rules.
        '''
        return [rule for ruleBase in self.rule_bases for rule in ruleBase.rules]


    def fuzzy_type(self) -> fs.FUZZY_SETS:
        '''
        Returns the correspoing type of the RuleBase using the enum type in the fuzzy_sets module.

        :return: the corresponding fuzzy set type of the RuleBase.
        '''
        return self.rule_bases[0].fuzzy_type()


    def purge_rules(self, tolerance=0.001) -> None:
        '''
        Delete the roles with a dominance score lower than the tolerance.

        :param tolerance: tolerance to delete the rules.
        '''
        for ruleBase in self.rule_bases:
            ruleBase.prune_bad_rules(tolerance)


    def __getitem__(self, item) -> RuleBase:
        '''
        Returns the corresponding rulebase.

        :param item: index of the rulebase.
        :return: the corresponding rulebase.
        '''
        return self.rule_bases[item]


    def __len__(self) -> int:
        '''
        Returns the number of rule bases.
        '''
        return len(self.rule_bases)


    def __str__(self) -> str:
        '''
        Returns a string with the rules for each consequent.
        '''
        return self.print_rules(return_rules=True)
    

    def __eq__(self, __value: object) -> bool:
        '''
        Returns True if the two rule bases are equal.

        :param __value: object to compare.
        :return: True if the two rule bases are equal.
        '''
        if not isinstance(__value, MasterRuleBase):
            return False
        else:
            return self.rule_bases == __value.rule_bases
        
    
    def __call__(self, X: np.array) -> np.array:
        '''
        Gives the prediction for each sample (same as winning_rule_predict)

        :param X: array of dims: samples x features.
        :return: vector of predictions, size: samples,
        '''
        return self.winning_rule_predict(X)
        
        
    def get_rulebases(self) -> list[RuleBase]:
        '''
        Returns a list with all the rules.

        :return: list
        '''
        return self.rule_bases


    def n_linguist_variables(self) -> int:
        '''
        Returns the number of linguistic variables in the rule base.
        '''
        return len(self.antecedents[0])
    

    def get_antecedents(self) -> list[fs.fuzzyVariable]:
        '''
        Returns the antecedents of the rule base.
        '''
        return self.antecedents


def list_rules_to_matrix(rule_list: list[RuleSimple]) -> np.array:
    '''
    Returns a matrix out of the rule list.

    :param rule_list: list of rules.
    :return: matrix with the antecedents of the rules.
    '''
    assert len(rule_list) > 0, 'No rules to list!'

    res = np.zeros((len(rule_list), len(rule_list[0].antecedents)))
    for ix, rule in enumerate(rule_list):
        res[ix, :] = rule.antecedents

    return res

