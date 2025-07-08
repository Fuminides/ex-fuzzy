"""
Persistence Module for Ex-Fuzzy Library

This module provides functionality for loading and saving fuzzy rule systems and fuzzy variables
from/to plain text format. It enables serialization and deserialization of fuzzy systems 
for persistence and portability.

Main Components:
    - Fuzzy rule loading and saving
    - Fuzzy variable loading and saving  
    - Text-based serialization format support
    - Support for Type-1 and Type-2 fuzzy systems

The text format follows a specific structure that allows complete reconstruction
of fuzzy systems including membership functions, linguistic variables, and rule bases.
"""
import numpy as np
import re

modifier_string_pattern = r'\(MOD\s+(\w+)\)'

try:
    from . import fuzzy_sets as fs
    from . import rules
except ImportError:
    import fuzzy_sets as fs
    import rules


def _extract_mod_word(text):
    """
    Extract modifier word from a string containing MOD pattern.
    
    Args:
        text (str): Input text containing modifier pattern like "(MOD word)"
        
    Returns:
        str or None: The extracted modifier word, or None if no pattern found
        
    Example:
        >>> _extract_mod_word("variable IS value (MOD very)")
        'very'
    """
    pattern = r'\(MOD\s+([^)]+)\)'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def _remove_mod_completely(text):
    """
    Remove modifier pattern completely from text.
    
    Args:
        text (str): Input text containing modifier pattern like "(MOD word)"
        
    Returns:
        str: Text with modifier pattern removed
        
    Example:
        >>> _remove_mod_completely("variable IS value (MOD very)")
        'variable IS value '
    """
    pattern = r'\(MOD\s+([^)]+)\)'
    replacement = ''
    return re.sub(pattern, replacement, text)


import re

def remove_parentheses(text):
    """
    Remove all content within parentheses from text.
    
    Args:
        text (str): Input text that may contain parentheses
        
    Returns:
        str: Text with all parenthetical content removed
        
    Example:
        >>> remove_parentheses("DS 0.85 (ACC 0.92), (WGHT 1.0)")
        'DS 0.85 , '
    """
    return re.sub(r'\(.*?\)', '', text)


def load_fuzzy_rules(rules_printed: str, fuzzy_variables: list) -> rules.MasterRuleBase:
    """
    Load fuzzy rules from a text string representation.
    
    This function parses a text-based representation of fuzzy rules and constructs
    a MasterRuleBase object containing all the rules organized by consequent classes.
    
    Args:
        rules_printed (str): Text representation of fuzzy rules following the 
            ex-fuzzy format. Each rule should start with "IF" and contain 
            antecedents connected by "AND", followed by "WITH" and consequent 
            information including dominance score (DS), accuracy (ACC), and 
            optionally weight (WGHT).
        fuzzy_variables (list): List of fuzzyVariable objects that define the 
            linguistic variables and their membership functions used in the rules.
            
    Returns:
        rules.MasterRuleBase: A MasterRuleBase object containing all parsed rules
            organized by consequent classes, with proper rule bases for each class.
            
    Raises:
        ValueError: If the text format is invalid or contains unrecognized elements
        IndexError: If referenced linguistic variables or values are not found
        
    Example:
        >>> rules_text = '''Rules for class_1:
        ... IF var1 IS low AND var2 IS high WITH DS 0.85 (ACC 0.92)
        ... Rules for class_2:
        ... IF var1 IS high WITH DS 0.78 (ACC 0.88)'''
        >>> master_rb = load_fuzzy_rules(rules_text, fuzzy_variables)
        
    Note:
        The text format supports:
        - Multiple rule bases for different consequent classes
        - Rule modifiers using (MOD modifier_name) syntax
        - Accuracy and weight information in parentheses
        - Don't care conditions (variables not mentioned in antecedents)
    """
    consequent = 0
    linguistic_variables_names = [linguistic_variable.name for linguistic_variable in fuzzy_variables]
    value_names = {fuzzy_variables[ix].name : [x.name for x in fuzzy_variables[ix]] for ix in range(len(fuzzy_variables))}
    fz_type = fuzzy_variables[0].fuzzy_type()
    consequent_names = []
    detected_modifiers = False
    for line in rules_printed.splitlines():
        if line.startswith('IF'):
            #Is a rule
            antecedents , consequent_ds = line.split('WITH')
            # Try to look for weight and accuracy in the rule
            rule_acc = None
            rule_weight = None
            for jx, stat in enumerate(consequent_ds.split(',')):
                if 'ACC' in stat:
                    rule_acc = stat.strip()
                elif 'WGHT' in stat:
                    rule_weight = stat.strip()
            
            consequent_ds = remove_parentheses(consequent_ds)
            consequent_ds = consequent_ds.split(',')[0].strip()
            modifiers = np.ones((len(fuzzy_variables),))
            init_rule_antecedents = np.zeros(
                (len(fuzzy_variables),)) - 1  # -1 is dont care
            
            for lx, antecedent in enumerate(antecedents.split('AND')):
                antecedent = antecedent.replace('IF', '').strip()
                if 'MOD' in antecedent:
                    detected_modifiers = True
                    modifier_value = _extract_mod_word(antecedent)
                    antecedent = _remove_mod_completely(antecedent)
                    if modifier_value in rules.modifiers_names.keys():
                        modifiers[lx] = rules.modifiers_names[modifier_value]

                antecedent_name, antecedent_value = antecedent.split('IS')
                antecedent_name = antecedent_name.strip()
                antecedent_value = antecedent_value.strip()
                antecedent_index = linguistic_variables_names.index(antecedent_name)
                antecedent_value_index = value_names[antecedent_name].index(antecedent_value)

                init_rule_antecedents[antecedent_index] = antecedent_value_index
                
            rule_simple = rules.RuleSimple(init_rule_antecedents, 0)
            rule_simple.score = float(consequent_ds[3:].strip()) # We remove the 'DS ' and the last space
            rule_simple.accuracy = float(rule_acc[4:].strip()) # We remove the 'ACC ' and the last space
            try:
                rule_simple.weight = float(rule_weight[4:].strip())
                ds_mode = 2
            except:
                ds_mode = 0
            
            rule_simple.modifiers = modifiers
            
            reconstructed_rules.append(rule_simple)

        elif line.startswith('Rules'):
            #New consequent
            consequent_name = line.split(':')[-1].strip()
            consequent_names.append(consequent_name)
            if consequent > 0:
                if fz_type == fs.FUZZY_SETS.t1:
                    rule_base = rules.RuleBaseT1(fuzzy_variables, reconstructed_rules)
                elif fz_type == fs.FUZZY_SETS.t2:
                    rule_base = rules.RuleBaseT2(fuzzy_variables, reconstructed_rules)
                elif fz_type == fs.FUZZY_SETS.gt2:
                    rule_base = rules.RuleBaseGT2(fuzzy_variables, reconstructed_rules)
                                    
            if consequent == 1:
                mrule_base = rules.MasterRuleBase([rule_base], ds_mode=ds_mode)
            elif consequent > 1:
                mrule_base.add_rule_base(rule_base)

            reconstructed_rules = []
            consequent += 1
    
    # We add the last rule base
    if fz_type == fs.FUZZY_SETS.t1:
        rule_base = rules.RuleBaseT1(fuzzy_variables, reconstructed_rules)
    elif fz_type == fs.FUZZY_SETS.t2:
        rule_base = rules.RuleBaseT2(fuzzy_variables, reconstructed_rules)
    elif fz_type == fs.FUZZY_SETS.gt2:
        rule_base = rules.RuleBaseGT2(fuzzy_variables, reconstructed_rules)
        
    mrule_base.add_rule_base(rule_base) 
    mrule_base.rename_cons(consequent_names)

    return mrule_base


def load_fuzzy_variables(fuzzy_variables_printed: str) -> list:
    """
    Load fuzzy variables from a text string representation.
    
    This function parses a text-based representation of fuzzy variables and constructs
    a list of fuzzyVariable objects with their associated fuzzy sets and membership functions.
    
    Args:
        fuzzy_variables_printed (str): Text representation of fuzzy variables following
            the ex-fuzzy format. Should contain variable definitions starting with 
            '$$$' (for fuzzy variables) or '$Categorical' (for categorical variables),
            followed by fuzzy set definitions with membership function parameters.
            
    Returns:
        list: List of fuzzyVariable objects containing all parsed linguistic variables
            with their fuzzy sets and membership functions properly configured.
            
    Raises:
        ValueError: If the text format is invalid or contains unrecognized elements
        TypeError: If membership function parameters cannot be converted to proper types
        
    Example:
        >>> fvars_text = '''$$$ Linguistic variable: temperature
        ... low;0.0,50.0;trap;0.0,0.0,20.0,30.0
        ... high;0.0,50.0;trap;25.0,35.0,50.0,50.0'''
        >>> fuzzy_vars = load_fuzzy_variables(fvars_text)
        
    Note:
        The text format supports:
        - Type-1 and Type-2 fuzzy sets (trapezoidal and gaussian)
        - Categorical variables with different data types
        - Variable units specification
        - Multiple membership function types per variable
    """
    fuzzy_set_type = fs.FUZZY_SETS.t1 

    fuzzy_variables = []
    active_linguistic_variables = False
    lines = fuzzy_variables_printed.splitlines()
    for line in lines:
        if line.startswith('$$$') or line.startswith('$Fuzzy'):
            #New linguistic variable
            if active_linguistic_variables:
                object_fvar = fs.fuzzyVariable(linguistic_variable_name, linguistic_var_fuzzy_sets, fvar_units)

                fuzzy_variables.append(object_fvar)

            linguistic_var_fuzzy_sets = []
            linguistic_variable_name = line.split(':')[1].strip()
            try:
                fvar_units = line.split(':')[2].strip()
            except:
                fvar_units = None

            active_linguistic_variables = True
        elif line.startswith('$Categorical'):
            #New categorical variable
            if active_linguistic_variables:
                object_fvar = fs.fuzzyVariable(linguistic_variable_name, linguistic_var_fuzzy_sets, fvar_units)

                fuzzy_variables.append(object_fvar)
            linguistic_var_fuzzy_sets = []
            linguistic_variable_name = line.split(':')[1].strip()

            active_linguistic_variables = True
            
        elif line == '':
            pass
        else:
            processes_line = line.split(';')

            if processes_line[0].startswith('Categorical'):
                data_type = processes_line[0].split(' ')[1].strip()
                if data_type == 'float':
                    data_cast_func = float
                elif data_type == 'str':
                    data_cast_func = str
                else:
                    data_cast_func = str

                categories = processes_line[1].split(',')[:-1]

                if fuzzy_set_type == fs.FUZZY_SETS.t1:
                    fscat_categories = [fs.categoricalFS(category, data_cast_func(category)) for category in categories]
                elif fuzzy_set_type == fs.FUZZY_SETS.t2:
                    fscat_categories = [fs.categoricalIVFS(category, data_cast_func(category)) for category in categories]

                #We know there is one categorical variable active
                for fscat in fscat_categories:
                    linguistic_var_fuzzy_sets.append(fscat)

            else:
                #We know there is one fuzzy variable active
                fields = processes_line
                if len(fields) == 4:
                    name, domain, membership_type, mem1 = fields
                    domain = [float(x) for x in domain.split(',')]
                    mem = [float(x) for x in mem1.split(',')]
                elif len(fields) > 4:
                    name, domain, membership_type, mem1, mem2, height = fields
                    fuzzy_set_type = fs.FUZZY_SETS.t2
                    mem2 = [float(x) for x in mem2.split(',')]
                    height = float(height)
                    domain = [float(x) for x in domain.split(',')]
                    mem = [float(x) for x in mem1.split(',')]
                elif len(fields) == 2: # This is a categorical/custom variable
                    name, key_value = fields
                    try:
                        key_value = float(key_value)
                    except ValueError:
                        pass
                    
                    membership_type = 'Categorical'
                    

                if membership_type == 'gauss':
                    if fuzzy_set_type == fs.FUZZY_SETS.t1:
                        constructed_fs = fs.gaussianFS(name, mem, domain)
                    elif fuzzy_set_type == fs.FUZZY_SETS.t2:
                        constructed_fs = fs.gaussianIVFS(name, mem1, mem2, domain, height)

                elif membership_type == 'trap':
                    if fuzzy_set_type == fs.FUZZY_SETS.t1:
                        constructed_fs = fs.FS(name, mem, domain)
                    elif fuzzy_set_type == fs.FUZZY_SETS.t2:
                        constructed_fs = fs.IVFS(name, mem, mem2, domain, height)
                
                elif membership_type == 'Categorical':
                    if fuzzy_set_type == fs.FUZZY_SETS.t1:
                        constructed_fs = fs.categoricalFS(name, key_value)
                    elif fuzzy_set_type == fs.FUZZY_SETS.t2:
                        constructed_fs = fs.categoricalIVFS(name, key_value)
                
                linguistic_var_fuzzy_sets.append(constructed_fs)

    if active_linguistic_variables:
        object_fvar = fs.fuzzyVariable(linguistic_variable_name, linguistic_var_fuzzy_sets, fvar_units)
        fuzzy_variables.append(object_fvar)

    return fuzzy_variables


def print_fuzzy_variable(fuzzy_variable: fs.fuzzyVariable) -> str:
    """
    Convert a fuzzy variable to its text string representation.
    
    This function serializes a fuzzyVariable object into a text format that can be
    saved to files and later loaded using load_fuzzy_variables().
    
    Args:
        fuzzy_variable (fs.fuzzyVariable): The fuzzy variable object to be serialized.
            Should contain properly configured fuzzy sets with membership functions.
            
    Returns:
        str: Text representation of the fuzzy variable following the ex-fuzzy format.
            Includes variable name, units (if any), and all fuzzy set definitions
            with their membership function parameters.
            
    Example:
        >>> # For a fuzzy variable with trapezoidal sets
        >>> text = print_fuzzy_variable(temperature_var)
        >>> print(text)
        $$$ Linguistic variable: temperature
        low;0.0,50.0;trap;0.0,0.0,20.0,30.0
        high;0.0,50.0;trap;25.0,35.0,50.0,50.0
        
    Note:
        The output format varies based on fuzzy set types:
        - Categorical variables use '$Categorical variable:' prefix
        - Regular fuzzy variables use '$$$ Linguistic variable:' prefix
        - Type-2 fuzzy sets include additional parameters for upper membership functions
        - Supports trapezoidal, triangular, and gaussian membership functions
    """
    if isinstance(fuzzy_variable[0], fs.categoricalFS):
        fuzzy_variable_printed = '$Categorical variable: ' + fuzzy_variable.name
        cat_type = 'float' if isinstance(fuzzy_variable[0].category, float) or isinstance(fuzzy_variable[0].category, int) else 'str'
        if fuzzy_variable.units is not None:
            fuzzy_variable_printed += ' : ' + fuzzy_variable.units
        fuzzy_variable_printed += '\n'

        fuzzy_variable_printed += 'Categorical ' + cat_type + ';'
        for fuzzy_set in fuzzy_variable:
            fuzzy_variable_printed += fuzzy_set.name + ','
        fuzzy_variable_printed += '\n'
    else:
        fuzzy_variable_printed = '$$$ Linguistic variable: ' + fuzzy_variable.name
        if fuzzy_variable.units is not None:
            fuzzy_variable_printed += ' : ' + fuzzy_variable.units
        fuzzy_variable_printed += '\n'

        for fuzzy_set in fuzzy_variable.linguistic_variables:
            if isinstance(fuzzy_set, fs.gaussianIVFS):
                fuzzy_variable_printed += fuzzy_set.name + ';' + ','.join([str(x) for x in fuzzy_set.domain]) + ';' + 'gauss;' + ','.join([str(x) for x in fuzzy_set.secondMF_lower]) + ';' + ','.join([str(x) for x in fuzzy_set.secondMF_upper]) + ';' + str(fuzzy_set.lower_height) + '\n' 
            elif isinstance(fuzzy_set, fs.IVFS):
                fuzzy_variable_printed += fuzzy_set.name + ';' + ','.join([str(x) for x in fuzzy_set.domain]) + ';' + 'trap;' + ','.join([str(x) for x in fuzzy_set.secondMF_lower]) + ';' + ','.join([str(x) for x in fuzzy_set.secondMF_upper]) + ';' + str(fuzzy_set.lower_height) + '\n'  
            elif isinstance(fuzzy_set, fs.gaussianFS):
                fuzzy_variable_printed += fuzzy_set.name + ';' + ','.join([str(x) for x in fuzzy_set.domain]) + ';' + 'gauss;' + ','.join([str(x) for x in fuzzy_set.membership_parameters]) + '\n'
            elif isinstance(fuzzy_set, fs.FS):
                fuzzy_variable_printed += fuzzy_set.name + ';' + ','.join([str(x) for x in fuzzy_set.domain]) + ';' + 'trap;' + ','.join([str(x) for x in fuzzy_set.membership_parameters]) + '\n'
            
    return fuzzy_variable_printed


def save_fuzzy_variables(fuzzy_variables: list) -> str:
    """
    Save multiple fuzzy variables to a text string representation.
    
    This function serializes a list of fuzzyVariable objects into a single text string
    that can be saved to files and later loaded using load_fuzzy_variables().
    
    Args:
        fuzzy_variables (list): List of fuzzyVariable objects to be serialized.
            Each variable should contain properly configured fuzzy sets with 
            membership functions.
            
    Returns:
        str: Text representation of all fuzzy variables concatenated with newlines.
            Each variable is separated and formatted according to the ex-fuzzy 
            text specification.
            
    Example:
        >>> variables = [temperature_var, pressure_var, flow_var]
        >>> text = save_fuzzy_variables(variables)
        >>> # Save to file
        >>> with open('fuzzy_vars.txt', 'w') as f:
        ...     f.write(text)
        
    Note:
        This function calls print_fuzzy_variable() for each variable and concatenates
        the results with newline separators for proper file formatting.
    """
    fuzzy_variables_printed = ''
    for fvar in fuzzy_variables:
        fuzzy_variables_printed += print_fuzzy_variable(fvar) + '\n'

    return fuzzy_variables_printed