"""
Fuzzy Rules and Inference Engine for Ex-Fuzzy Library

This module contains the core classes and functions for fuzzy rule definition, management,
and inference. It implements a complete fuzzy inference system supporting Type-1, Type-2,
and General Type-2 fuzzy sets with various t-norm operations and defuzzification methods.

Main Components:
    - Rule classes: RuleSimple for individual rule representation  
    - RuleBase classes: RuleBaseT1, RuleBaseT2, RuleBaseGT2 for rule collections
    - MasterRuleBase: Container for multiple rule bases (multi-class problems)
    - Inference engines: Support for Mamdani and Takagi-Sugeno inference
    - Defuzzification: Centroid, height, and other defuzzification methods
    - T-norm operations: Product, minimum, and other aggregation functions

The module supports complex fuzzy reasoning workflows including rule firing strength
computation, aggregation across multiple rules, and final defuzzification to crisp outputs.
It also includes support for rule modifiers (e.g., "very", "somewhat") and dominance scores
for rule quality assessment.
"""
import abc
import numbers
import copy

import numpy as np
try:
    from . import fuzzy_sets as fs
    from . import centroid
except ImportError:
    import fuzzy_sets as fs
    import centroid

modifiers_names = {0.5: 'Somewhat', 1.0: '', 1.3: 'A little', 1.7: 'Slightly', 2.0: 'Very', 3.0: 'Extremely', 4.0: 'Very very'}

def compute_antecedents_memberships(antecedents: list[fs.fuzzyVariable], x: np.array) -> list[dict]:
    """
    Compute membership degrees for input values across all fuzzy variables.
    
    This function calculates the membership degrees of input values for each linguistic
    variable in the antecedents. It returns a structured representation that can be
    used for efficient rule evaluation and inference.
    
    Args:
        antecedents (list[fs.fuzzyVariable]): List of fuzzy variables representing
            the antecedents (input variables) of the fuzzy system
        x (np.array): Input vector with values for each antecedent variable.
            Shape should be (n_samples, n_variables) or (n_variables,) for single sample
            
    Returns:
        list[dict]: List containing membership dictionaries for each antecedent variable.
            Each dictionary maps linguistic term indices to their membership degrees.
            
    Example:
        >>> # For 2 variables with 3 linguistic terms each
        >>> antecedents = [temp_var, pressure_var]  
        >>> x = np.array([25.0, 101.3])  # temperature=25°C, pressure=101.3kPa
        >>> memberships = compute_antecedents_memberships(antecedents, x)
        >>> # memberships[0] contains temperature memberships: {0: 0.2, 1: 0.8, 2: 0.0}
        >>> # memberships[1] contains pressure memberships: {0: 0.0, 1: 0.6, 2: 0.4}
        
    Note:
        This function is typically used internally during rule evaluation but can be
        useful for debugging membership degree calculations or analyzing input fuzzification.
    """
    x = np.array(x)
    cache_antecedent_memberships = []

    for ix, antecedent in enumerate(antecedents):
        cache_antecedent_memberships.append(
            antecedent.compute_memberships(x[:, ix]))

    return cache_antecedent_memberships

        
            
class RuleError(Exception):
    """
    Exception raised when a fuzzy rule is incorrectly defined or invalid.
    
    This exception is used throughout the rules module to indicate various
    rule-related errors such as invalid antecedent specifications, inconsistent
    membership functions, or malformed rule structures.
    
    Attributes:
        message (str): Human-readable description of the error
        
    Example:
        >>> try:
        ...     rule = RuleSimple([-1, -1, -1], 0)  # Invalid: all don't-care antecedents
        ... except RuleError as e:
        ...     print(f"Rule error: {e}")
    """

    def __init__(self, message: str) -> None:
        """
        Initialize the RuleError exception.

        Args:
            message (str): Descriptive error message explaining what went wrong
        """
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
        if x.ndim == 1:
            x = x.reshape(1, -1)
        for ix, antecedent in enumerate(self.antecedents):
            if ix == 0:
                res = antecedent.membership(x[:,ix])
            else:
                res = tnorm(res, antecedent.membership(x[:,ix]))

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
    """
    Simplified Rule Representation for Optimized Computation.
    
    This class represents fuzzy rules in a simplified format optimized for
    computational efficiency in rule base operations. It uses integer encoding
    for antecedents and consequents to minimize memory usage and speed up
    rule evaluation processes.
    
    Attributes:
        antecedents (list[int]): Integer-encoded antecedents where:
            - -1: Variable not used in the rule
            - 0-N: Index of the linguistic variable used for the ith input
        consequent (int): Integer index of the consequent linguistic variable
        modifiers (np.array): Optional modifiers for rule adaptation
        
    Example:
        >>> # Rule: IF x1 is Low AND x2 is High THEN y is Medium
        >>> # Assuming Low=0, High=1, Medium=1
        >>> rule = RuleSimple([0, 1], consequent=1)
        >>> print(rule.antecedents)  # [0, 1]
        >>> print(rule.consequent)  # 1
        
    Note:
        This simplified representation is designed for high-performance
        rule evaluation in large rule bases where memory and speed are critical.
    """

    def __init__(self, antecedents: list[int], consequent: int = 0, modifiers: np.array = None) -> None:
        """
        Creates a rule with the given antecedents and consequent.
        
        Args:
            antecedents (list[int]): List of integers indicating the linguistic 
                variable used for each input (-1 for unused variables)
            consequent (int, optional): Integer indicating the linguistic variable 
                used for the consequent. Defaults to 0.
            modifiers (np.array, optional): Array of modifier values for rule adaptation.
                Defaults to None.
                
        Example:
            >>> # Create a rule with two antecedents and one consequent
            >>> rule = RuleSimple([0, 2, -1], consequent=1)  # x1=0, x2=2, x3=unused, y=1
        """
        self.antecedents = list(map(int, antecedents))
        self.consequent = int(consequent)
        self.modifiers = modifiers

    def __getitem__(self, ix):
        """
        Returns the antecedent value for the given index.
        
        Args:
            ix (int): Index of the antecedent to return
            
        Returns:
            int: The antecedent value at the specified index
            
        Example:
            >>> rule = RuleSimple([0, 1, 2])
            >>> print(rule[1])  # 1
        """
        return self.antecedents[ix]

    def __setitem__(self, ix, value):
        """
        Sets the antecedent value for the given index.
        
        Args:
            ix (int): Index of the antecedent to set
            value (int): Value to set at the specified index
            
        Example:
            >>> rule = RuleSimple([0, 1, 2])
            >>> rule[1] = 3  # Change second antecedent to 3
        """
        self.antecedents[ix] = value

    def __str__(self):
        """
        Returns a string representation of the rule.
        
        Returns:
            str: Human-readable string representation of the rule
            
        Example:
            >>> rule = RuleSimple([0, 1], consequent=2)
            >>> print(rule)  # Rule: antecedents: [0, 1] consequent: 2
        """
        aux = 'Rule: antecedents: ' + str(self.antecedents) + ' consequent: ' + str(self.consequent)

        try:
            if self.modifiers is not None:
                aux += ' modifiers: ' + str(self.modifiers)
        except AttributeError:
            pass
        
        try:
            aux += ' score: ' + str(self.score)
        except AttributeError:
            pass

        try:
            aux += ' weight: ' + str(self.weight)
        except AttributeError:
            pass
    
        try:
            aux += ' accuracy: ' + str(self.accuracy)
        except AttributeError:
            pass

        try:
            aux += ' p-value class structure: ' + str(self.p_value_class_structure)
        except AttributeError:
            pass

        try:
            aux += ' p-value feature coalitions: ' + str(self.p_value_feature_coalitions)
        except AttributeError:
            pass

        try:
            aux += ' p-value bootstrapping membership validation: ' + str(self.boot_p_value)
        except AttributeError:
            pass

        try:
            aux += ' bootstrapping confidence conf. interval: ' + str(self.boot_confidence_interval)
        except AttributeError:
            pass

        try:
            aux += ' bootstrapping support conf. interval: ' + str(self.boot_support_interval)
        except AttributeError:
            pass
            

        return aux

    def __len__(self):
        """
        Returns the number of antecedents in the rule.
        
        Returns:
            int: Number of antecedents in the rule
            
        Example:
            >>> rule = RuleSimple([0, 1, 2])
            >>> print(len(rule))  # 3
        """
        return len(self.antecedents)

    def __eq__(self, other):
        """
        Returns True if the two rules are equal.
        
        Args:
            other (RuleSimple): Another rule to compare with
            
        Returns:
            bool: True if rules have identical antecedents and consequent
            
        Example:
            >>> rule1 = RuleSimple([0, 1], consequent=2)
            >>> rule2 = RuleSimple([0, 1], consequent=2)
            >>> print(rule1 == rule2)  # True
        """
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

    def __init__(self, antecedents: list[fs.fuzzyVariable], rules: list[RuleSimple], consequent: fs.fuzzyVariable=None, tnorm=np.prod) -> None:
        '''
        Creates a rulebase with the given antecedents, rules and consequent.

        :param antecedents: list of fuzzy sets.
        :param rules: list of rules.
        :param consequent: fuzzy set.
        :param tnorm: t-norm to use in the inference process.
        :param fuzzy_modifiers: array with the fuzzy modifiers for each rule. If None, no modifiers are used. (Fuzzy modifiers are modifiers to the antecedent memberships, also known as linguistic hedges). Only exponentiation is supported. (x**a, being a the modifier in the matrix). Defaults to None. They can also be specified in the RuleSimple list of rules.
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
    

    def get_weights(self):
        '''
        Returns an array with the weights for each rule.
        (Different from dominance scores: must been already computed by an optimization algorithm)

        '''
        res = np.zeros((len(self.rules, )))
        for ix, rule in enumerate(self.rules):
            res[ix] = rule.weight

        return res


    def delete_rule_duplicates(self, list_rules:list[RuleSimple]):
        """Return a list with duplicate rules removed, preserving first occurrences."""
        # Delete the rules that are duplicated in the rule list.
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
                # Check if x is pandas 
                if hasattr(x, 'values'):
                    x = x.values

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


    def compute_rule_antecedent_memberships(self, x: np.array, scaled=False, antecedents_memberships:list[np.array]=None) -> np.array:
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

        if antecedents_memberships is None:
            antecedents_memberships = self.compute_antecedents_memberships(x)
        
        for jx, rule in enumerate(self.rules):
            rule_antecedents = rule.antecedents
            try:
                fuzzy_modifier = rule.modifiers
            except AttributeError:
                fuzzy_modifier = None

            if self.fuzzy_type() == fs.FUZZY_SETS.t1:
                membership = np.zeros((x.shape[0], len(rule_antecedents)))
            elif self.fuzzy_type() == fs.FUZZY_SETS.t2:
                membership = np.zeros((x.shape[0], len(rule_antecedents), 2))
            elif self.fuzzy_type() == fs.FUZZY_SETS.gt2:
                membership = np.zeros(
                    (x.shape[0], len(rule_antecedents), len(self.alpha_cuts), 2))

            n_nonvl = 0
            for ix, vl in enumerate(rule_antecedents):
                if vl >= 0:
                    membership_antecedent = list(antecedents_memberships[ix])[vl]

                    if fuzzy_modifier is not None:
                        if fuzzy_modifier[ix] != -1:
                            membership[:, ix] = membership_antecedent**fuzzy_modifier[ix]
                        else:
                            membership[:, ix] = membership_antecedent
                    else:
                        membership[:, ix] = membership_antecedent

                    n_nonvl += 1
                else:
                    membership[:, ix] = 1.0

            if n_nonvl == 0:
                membership[:, ix] = 0.0

            membership = self.tnorm(membership, axis=1)
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


    def print_rules(self, return_rules:bool=False, bootstrap_results:bool=True) -> None:
        '''
        Print the rules from the rule base.

        :param return_rules: if True, the rules are returned as a string.
        '''
        all_rules = ''
        for ix, rule in enumerate(self.rules):
            str_rule = generate_rule_string(rule, self.antecedents, bootstrap_results)
            
            all_rules += str_rule + '\n'

        if not return_rules:
            print(all_rules)
        else:
            return all_rules

    def print_rules_regression(self, return_rules:bool=False, output_name:str='output', 
                              show_memberships:bool=False) -> None:
        '''
        Print the rules from the rule base for regression problems.
        Formats rules with numeric consequents in a human-readable way.

        :param return_rules: if True, the rules are returned as a string.
        :param output_name: name of the output variable (default: 'output').
        :param show_memberships: if True, shows membership function parameters.
        '''
        all_rules = ''
        for ix, rule in enumerate(self.rules):
            str_rule = generate_rule_string_regression(
                rule, self.antecedents, ix + 1, output_name, show_memberships
            )
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

                try:
                    if score < tolerance or rule.accuracy == 0.0:
                        delete_list.append(ix)
                except AttributeError:
                    if score < tolerance:
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
    

    def n_linguistic_variables(self) -> int:
        '''
        Returns the number of linguistic variables in the rule base.
        '''
        return [len(amt) for amt in self.antecedents]


    def print_rule_bootstrap_results(self) -> None:
        '''
        Prints the bootstrap results for each rule.
        '''
        for ix, rule in enumerate(self.rules):
            print('Rule ' + str(ix) + ': ' + str(rule) + ' - Confidence: ' + str(rule.boot_confidence_interval) + ' - Support: ' + str(rule.boot_support_interval))


    def copy(self):
        '''
        Creates a copy of the RuleBase.
        
        :param deep: if True, creates a deep copy. If False, creates a shallow copy.
        :return: a copy of the RuleBase.
        '''
        # Deep copy all components
        copied_rules = copy.deepcopy(self.rules)
        copied_antecedents = copy.deepcopy(self.antecedents)
        copied_consequent = copy.deepcopy(self.consequent) if self.consequent is not None else None
        
        # Create new instance based on the type
        if isinstance(self, RuleBaseT1):
            return RuleBaseT1(copied_antecedents, copied_rules, copied_consequent, self.tnorm)
        elif isinstance(self, RuleBaseT2):
            return RuleBaseT2(copied_antecedents, copied_rules, copied_consequent, self.tnorm)
        elif isinstance(self, RuleBaseGT2):
            return RuleBaseGT2(copied_antecedents, copied_rules, copied_consequent, self.tnorm)
        else:
            # Base RuleBase class
            new_rb = RuleBase.__new__(RuleBase)
            new_rb.rules = copied_rules
            new_rb.antecedents = copied_antecedents
            new_rb.consequent = copied_consequent
            new_rb.tnorm = self.tnorm
            return new_rb
   

class RuleBaseT2(RuleBase):
    '''
    Class optimized to work with multiple rules at the same time. Supports only one consequent. 
    (Use one rulebase per consequent to study classification problems. Check MasterRuleBase class for more documentation)

    This class supports iv approximation for t2 fs.
    '''

    def __init__(self, antecedents: list[fs.fuzzyVariable], rules: list[RuleSimple], consequent: fs.fuzzyVariable = None, tnorm=np.prod) -> None:
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

    def __init__(self, antecedents: list[fs.fuzzyVariable], rules: list[RuleSimple], consequent: fs.fuzzyVariable = None, tnorm=np.prod) -> None:
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

        try:
            # We try to get the modifiers from the rules, else, we will use the ones given in the constructor.
            self.fuzzy_modifiers = np.array([rule.modifiers for rule in rules])
        except AttributeError:
            self.fuzzy_modifier = fuzzy_modifiers


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


    def compute_rule_antecedent_memberships(self, x: np.array, scaled=True, antecedents_memberships=None) -> np.array:
        '''
        Computes the membership for the antecedents performing the alpha_cut reduction.

        :param x: array with the values of the inputs.
        :param scaled: if True, the memberships are scaled to sum 1 in each sample.
        :param antecedents_memberships: precomputed antecedent memberships. Not supported for GT2.
        :return: array with the memberships of the antecedents for each sample.
        '''
        
        rules_truth_values = super().compute_rule_antecedent_memberships(x, scaled, antecedents_memberships=antecedents_memberships)
            
        return self._alpha_reduction(rules_truth_values)


    def alpha_compute_rule_antecedent_memberships(self, x: np.array, scaled=True, antecedents_memberships=None) -> np.array:
        '''
        Computes the membership for the antecedents for all the alpha cuts.

        :param x: array with the values of the inputs.
        :param scaled: if True, the memberships are scaled to sum 1 in each sample.
        :param antecedents_memberships: precomputed antecedent memberships. Not supported for GT2.
        :return: array with the memberships of the antecedents for each sample.
        '''
        return super().compute_rule_antecedent_memberships(x, scaled)



class RuleBaseT1(RuleBase):
    '''
    Class optimized to work with multiple rules at the same time. Supports only one consequent.
    (Use one rulebase per consequent to study classification problems. Check MasterRuleBase class for more documentation)

    This class supports t1 fs.
    '''

    def __init__(self, antecedents: list[fs.fuzzyVariable], rules: list[RuleSimple], consequent: fs.fuzzyVariable = None, tnorm=np.prod) -> None:
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

            for ix, vl_consequent in enumerate(consequent.linguistic_variables):
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
        res = np.zeros((x.shape[0]))

        antecedent_memberships = self.compute_rule_antecedent_memberships(x)

        for sample in range(antecedent_memberships.shape[0]):
            res[sample] = centroid.center_of_masses(
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

    def __init__(self, rule_base: list[RuleBase], consequent_names: list[str]=None, ds_mode: int = 0, allow_unknown:bool=False) -> None:
        '''
        Constructor of the MasterRuleBase class.

        :param rule_base: list of rule bases.
        '''
        if len(rule_base) == 0:
            raise RuleError('No rule bases given!')
        
        self.rule_bases = rule_base
        self.antecedents = rule_base[0].antecedents

        if consequent_names is None:
            self.consequent_names = [ix for ix in range(len(self.rule_bases))]
        else:
            self.consequent_names = consequent_names
        self.ds_mode = ds_mode
        self.allow_unknown = allow_unknown


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

    
    def remove_rule(self, consequent: int, ix: int) -> None:
        '''
        Remove a rule from the rule base of the given consequent.

        :param consequent: index of the rule base to remove the rule from.
        :param idx: index of the rule to remove.
        '''
        self.rule_bases[consequent].remove_rule(ix)
    

    def get_consequents(self) -> list[int]:
        '''
        Returns a list with the consequents of each rule base.

        :return: list with the consequents of each rule base.
        '''
        return sum([[ix]*len(x) for ix, x in enumerate(self.rule_bases)], [])


    def get_consequents_names(self) -> list[str]:
        '''
        Returns a list with the names of the consequents.

        :return: list with the names of the consequents.
        '''
        return self.consequent_names
    

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
    

    def get_weights(self) -> np.array:
        '''
        Returns the weights for each rule in all the rulebases.

        :return: array with the weights for each rule in all the rulebases.
        '''
        res = []
        for rb in self.rule_bases:
            res.append(rb.get_weights())

        res = [x for x in res if len(x) > 0]
        
        try:
            return np.concatenate(res, axis=0)
        except ValueError:
            return np.array([])


    def compute_firing_strenghts(self, X, precomputed_truth=None) -> np.array:
        '''
        Computes the firing strength of each rule for each sample.

        :param X: array with the values of the inputs.
        :param precomputed_truth: if not None, the antecedent memberships are already computed. (Used for sped up in genetic algorithms)
        :return: array with the firing strength of each rule for each sample.
        '''
        aux = []
        for ix in range(len(self.rule_bases)):
            aux.append(self[ix].compute_rule_antecedent_memberships(X, antecedents_memberships=precomputed_truth))

        # Firing strengths shape: samples x rules (x 2) (last is iv dimension) or (x alpha_cuts x 2) for gt2
        return np.concatenate(aux, axis=1)


    def _winning_rules(self, X: np.array, precomputed_truth=None, allow_unkown=True) -> np.array:
        """Compute the winning rule index and firing strength for each sample."""
        association_degrees = self.compute_association_degrees(X, precomputed_truth)
        firing_strengths = self.compute_firing_strenghts(X, precomputed_truth=precomputed_truth)

        winning_rules = np.argmax(association_degrees, axis=1)
        winning_association_degrees = np.max(firing_strengths, axis=1)

        if allow_unkown:
            # If there is no rule that fires, we set the consequent to -1
            winning_rules[np.max(association_degrees, axis=1) == 0.0] = -1

        return winning_rules, winning_association_degrees


    def compute_association_degrees(self, X, precomputed_truth=None) -> np.array:
        '''
        Returns the winning rule for each sample. Takes into account dominance scores if already computed.
        :param X: array with the values of the inputs.
        :return: array with the winning rule for each sample.
        '''
        firing_strengths = self.compute_firing_strenghts(X, precomputed_truth=precomputed_truth)
        
        if self.ds_mode == 0:
            rulesw = self.get_scores()
            if self.fuzzy_type() == fs.FUZZY_SETS.t2 and len(rulesw.shape) == 1:
                rulesw = rulesw[None, :, None]
                
            association_degrees = rulesw * firing_strengths
        elif self.ds_mode == 1:
            association_degrees = firing_strengths
        elif self.ds_mode == 2:
            rulesw = self.get_weights()
            if self.fuzzy_type() == fs.FUZZY_SETS.t2:
                rulesw = rulesw[None, :, None] 
            
            association_degrees = rulesw * firing_strengths

        if (self[0].fuzzy_type() == fs.FUZZY_SETS.t2) or (self[0].fuzzy_type() == fs.FUZZY_SETS.gt2):
            association_degrees = np.mean(association_degrees, axis=2)
        elif self[0].fuzzy_type() == fs.FUZZY_SETS.gt2:
            association_degrees = np.mean(association_degrees, axis=3)
            
        return association_degrees
    
    
    def winning_rule_predict(self, X: np.array, precomputed_truth=None, out_class_names=False) -> np.array:
        '''
        Returns the winning rule for each sample. Takes into account dominance scores if already computed.

        :param X: array with the values of the inputs.
        :param precomputed_truth: if not None, the antecedent memberships are already computed. (Used for sped up in genetic algorithms)
        :return: array with the winning rule for each sample.
        '''
        # Raise an error if there no rules
        if len(self.get_rules()) == 0:
            raise RuleError('No rules to predict!')
        
        if out_class_names:
            consequents = sum([[self.consequent_names[ix]]*len(self[ix].rules)
                          for ix in range(len(self.rule_bases))], [])  # The sum is for flatenning the list
        else:
            consequents = sum([[ix]*len(self[ix].rules)
                          for ix in range(len(self.rule_bases))], [])  # The sum is for flatenning the list
            
        winning_rules, _winning_association_degrees = self._winning_rules(X, precomputed_truth=precomputed_truth, allow_unkown=self.allow_unknown)

        if out_class_names:
            res = []
        else:
            res = np.zeros((X.shape[0], ))

        for ix, winning_rule in enumerate(winning_rules):
            if winning_rule != -1:
                if out_class_names:
                    res.append(consequents[winning_rule])
                else:
                    res[ix] = consequents[winning_rule]
            else:
                if out_class_names:
                    res.append('Unknown')
                else:
                    res[ix] = -1

        return np.array(res)


    def explainable_predict(self, X: np.array, out_class_names=False, precomputed_truth=None) -> np.array:
        '''
        Returns the predicted class for each sample.

        :param X: array with the values of the inputs.
        :param out_class_names: if True, the output will be the class names instead of the class index.
        :return: np array samples (x 1) with the predicted class.
        '''
        # Raise an error if there no rules
        if len(self.get_rules()) == 0:
            raise RuleError('No rules to predict!')
        
        if out_class_names:
            consequents = sum([[self.consequent_names[ix]]*len(self[ix].rules)
                          for ix in range(len(self.rule_bases))], [])  # The sum is for flatenning the list
        else:
            consequents = sum([[ix]*len(self[ix].rules)
                          for ix in range(len(self.rule_bases))], [])  # The sum is for flatenning the list
            
        winning_rules, winning_association_degrees = self._winning_rules(X, precomputed_truth=precomputed_truth, allow_unkown=self.allow_unknown)
        try:
            confidence_intervals = [ rule.boot_confidence_interval for rule in self.get_rules()]
            winning_rule_confidence_intervals = [confidence_intervals[ix] for ix in winning_rules]
        except AttributeError:
            winning_rule_confidence_intervals = np.ones((X.shape[0], 1))

        if out_class_names:
            res = []
        else:
            res = np.zeros((X.shape[0], ))

        for ix, winning_rule in enumerate(winning_rules):
            if winning_rule != -1:
                if out_class_names:
                    res.append(consequents[winning_rule])
                else:
                    res[ix] = consequents[winning_rule]
            else:
                if out_class_names:
                    res.append('Unknown')
                else:
                    res[ix] = -1

        if len(winning_association_degrees.shape) == 1:
            winning_association_degrees = winning_association_degrees[:, None]

        return np.array(res), winning_rules, winning_association_degrees, np.array(winning_rule_confidence_intervals) * winning_association_degrees



    def add_rule_base(self, rule_base: RuleBase) -> None:
        '''
        Adds a rule base to the list of rule bases.


        :param rule_base: rule base to add.
        '''
        self.rule_bases.append(rule_base)

        if len(self.rule_bases) != len(self.consequent_names):
            # We did not give proper names to the consequents
            self.consequent_names = [ix for ix in range(len(self.rule_bases))]


    def print_rules(self, return_rules=False, bootstrap_results=True) -> None:
        '''
        Print all the rules for all the consequents.

        :param autoprint: if True, the rules are printed. If False, the rules are returned as a string.
        '''
        res = ''
         
        for ix, ruleBase in enumerate(self.rule_bases):    
            res += 'Rules for consequent: ' + str(self.consequent_names[ix]) + '\n'
            res += '----------------\n'
            res += ruleBase.print_rules(return_rules=True, bootstrap_results=bootstrap_results) + '\n'
        
        
        if return_rules:
            return res
        else:
            print(res)
    

    def print_rule_bootstrap_results(self) -> None:
        '''
        Prints the bootstrap results for each rule.
        '''
        for ix, ruleBase in enumerate(self.rule_bases):
            print('Rules for consequent: ' + str(self.consequent_names[ix]))
            ruleBase.print_rule_bootstrap_results()

    def print_rules_regression(self, return_rules=False, output_name=None, show_memberships=False) -> None:
        '''
        Print all the rules for regression problems with numeric consequents.

        :param return_rules: if True, the rules are returned as a string.
        :param output_name: name of the output variable. If None, uses consequent_names.
        :param show_memberships: if True, shows membership function parameters.
        '''
        res = ''
        
        for ix, ruleBase in enumerate(self.rule_bases):
            if output_name is None:
                out_name = str(self.consequent_names[ix])
            else:
                out_name = output_name
                
            res += ruleBase.print_rules_regression(
                return_rules=True, 
                output_name=out_name, 
                show_memberships=show_memberships
            )
        
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
        return self.print_rules(return_rules=True, bootstrap_results=False)
    

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
        aux = self.winning_rule_predict(X)
         
        # Convert the predictions to the names of the consequents
        #return np.array(self.consequent_names)[aux]
        return aux
    

    def predict(self, X: np.array) -> np.array:
        '''
        Gives the prediction for each sample (same as winning_rule_predict)

        :param X: array of dims: samples x features.
        :return: vector of predictions, size: samples,
        '''
        return self(X)
        

    def get_rulebases(self) -> list[RuleBase]:
        '''
        Returns a list with all the rules.

        :return: list
        '''
        return self.rule_bases


    def n_linguistic_variables(self) -> list[int]:
        '''
        Returns the number of linguistic variables in the rule base.
        '''
        return [len(amt) for amt in self.antecedents]
    

    def get_antecedents(self) -> list[fs.fuzzyVariable]:
        '''
        Returns the antecedents of the rule base.
        '''
        return self.antecedents
    
    # Add to MasterRuleBase class
    def copy(self, deep=True):
        '''
        Creates a copy of the MasterRuleBase.
        
        :param deep: if True, creates a deep copy. If False, creates a shallow copy.
        :return: a copy of the MasterRuleBase.
        '''
        if deep:
            # Deep copy all rule bases and other attributes
            copied_rule_bases = [rb.copy(deep=True) for rb in self.rule_bases]
            copied_consequent_names = copy.deepcopy(self.consequent_names)
        else:
            # Shallow copy - copy the list but not the rule bases themselves
            copied_rule_bases = self.rule_bases.copy()
            copied_consequent_names = self.consequent_names.copy()
        
        # Create new MasterRuleBase instance
        new_master_rb = MasterRuleBase(
            copied_rule_bases, 
            copied_consequent_names, 
            self.ds_mode, 
            self.allow_unknown
        )
        
        return new_master_rb


def construct_rule_base(rule_matrix: np.array, nclasses:int, consequents: np.array, antecedents: list[fs.fuzzyVariable], rule_weights: np.array, class_names: list=None) -> MasterRuleBase:
    '''
    Constructs a rule base from a matrix of rules.

    :param rule_matrix: matrix with the rules.
    :param consequents: array with the consequents per rule.
    :param antecedents: list of fuzzy variables.
    :param class_names: list with the names of the classes.
    '''
    rule_lists = {ix:[] for ix in range(nclasses)}
    fs_studied = antecedents[0].fuzzy_type()
    for ix, consequent in enumerate(consequents):
        if not np.equal(rule_matrix[ix], -1).all():
            rule_object = RuleSimple(rule_matrix[ix])
            rule_object.score = rule_weights[ix]
            rule_lists[consequent].append(rule_object)

    for ix, consequent in enumerate(np.unique(consequents)):
        if fs_studied == fs.FUZZY_SETS.t1:
            rule_base = RuleBaseT1(antecedents, rule_lists[ix])
        elif fs_studied == fs.FUZZY_SETS.t2:
            rule_base = RuleBaseT2(antecedents, rule_lists[ix])
        elif fs_studied == fs.FUZZY_SETS.gt2:
            rule_base = RuleBaseGT2(antecedents, rule_lists[ix])
        
        if ix == 0:
            res = MasterRuleBase([rule_base], np.unique(consequents))
        else:
            res.add_rule_base(rule_base)

    if class_names is not None:
        res.rename_cons(class_names)

    return res


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


def generate_rule_string_regression(rule: RuleSimple, antecedents: list, rule_num: int, 
                                    output_name: str = 'output', show_memberships: bool = False) -> str:
    '''
    Generates a string representation for a regression rule with numeric consequent.

    :param rule: rule to generate the string.
    :param antecedents: list of fuzzy variables.
    :param rule_num: rule number for display.
    :param output_name: name of the output variable.
    :param show_memberships: if True, shows membership function parameters.
    :return: formatted string representation of the rule.
    '''
    str_rule = f'Rule {rule_num}: '
    
    # Build antecedent part
    antecedent_parts = []
    for jx, antecedent in enumerate(antecedents):
        if rule[jx] != -1:
            keys = antecedent.linguistic_variable_names()
            lv_name = keys[rule[jx]]
            
            if show_memberships:
                # Show membership function parameters
                lv = antecedent.linguistic_variables[rule[jx]]
                params = lv.membership_parameters
                lv_name += f' {params}'
            
            antecedent_parts.append(f'{antecedent.name} is {lv_name}')
            
            # Check for modifiers
            try:
                relevant_modifier = rule.modifiers[jx]
                if relevant_modifier != 1:
                    if relevant_modifier in modifiers_names.keys():
                        str_mod = modifiers_names[relevant_modifier]
                    else:
                        str_mod = str(relevant_modifier)
                    antecedent_parts[-1] += f' ({str_mod})'
            except (AttributeError, TypeError):
                pass
    
    if antecedent_parts:
        str_rule += 'IF ' + ' AND '.join(antecedent_parts) + ' '
    
    # Build consequent part (numeric value)
    consequent_value = rule.consequent
    if isinstance(consequent_value, (int, float, np.number)):
        str_rule += f'THEN {output_name} = {consequent_value:.4f}'
    else:
        str_rule += f'THEN {output_name} = {consequent_value}'
    
    return str_rule


def generate_rule_string(rule: RuleSimple, antecedents: list, bootstrap_results: bool=True) -> str:
    '''
    Generates a string with the rule.

    :param rule: rule to generate the string.
    :param antecedents: list of fuzzy variables.
    :param modifiers: array with the modifiers for the antecedents.
    '''
    def format_p_value(p_value):
        """Convert a numeric p-value into significance stars."""
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**' 
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'
        
    initiated = False
    str_rule = 'IF '
    for jx, antecedent in enumerate(antecedents):
        keys = antecedent.linguistic_variable_names()

        if rule[jx] != -1:
            if not initiated:
                initiated = True
            else:
                str_rule += ' AND '

            str_rule += str(antecedent.name) + ' IS ' + str(keys[rule[jx]])
            
            try:
                relevant_modifier = rule.modifiers[jx]
                if relevant_modifier != 1:
                    if relevant_modifier in modifiers_names.keys():
                        str_mod = modifiers_names[relevant_modifier]
                    else:
                        str_mod = str(relevant_modifier)

                    str_rule += ' (MOD ' + str_mod + ')'
            except AttributeError:
                pass
            except TypeError:
                pass


    try:
        score = rule.score if antecedents[0].fuzzy_type() == fs.FUZZY_SETS.t1 else np.mean(rule.score)
        str_rule += ' WITH DS ' + str(score)

        # If the classification scores have been computed, print them.
        try:
            str_rule += ', ACC ' + str(rule.accuracy)

        except AttributeError:
            pass
        # Check if they have weights
        try:
            str_rule += ', WGHT ' + str(rule.weight)
        except AttributeError:
            pass
    except AttributeError:
        try:
            str_rule += ' THEN consequent vl is ' + str(rule.consequent)
        except AttributeError:
            pass
    
    if bootstrap_results:
        try:
            p_value_class_structure = rule.p_value_class_structure
            p_value_feature_coalition = rule.p_value_feature_coalitions


            pvc = format_p_value(p_value_class_structure)
            pvf = format_p_value(p_value_feature_coalition)

            str_rule += ' (p-value Permutation Class Structure: ' + pvc + ', Feature Coalition: ' + pvf 
            p_value_bootstrap = rule.boot_p_value

            pbs = format_p_value(p_value_bootstrap)
            str_rule += ' Membership Validation: ' + pbs
            str_rule += ')'

            str_rule += ' Confidence Interval: ' + str(rule.boot_confidence_interval)
            str_rule += ' Support Interval: ' + str(rule.boot_support_interval)
        except AttributeError:
            pass
    return str_rule
