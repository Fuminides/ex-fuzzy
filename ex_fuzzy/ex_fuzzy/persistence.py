'''
Load the rules of a fuzzy rules system using plain text format.

'''
import numpy as np
import re

modifier_string_pattern = r'\(MOD\s+(\w+)\)'

try:
    from . import fuzzy_sets as fs
    from . import rules
    from . import maintenance as mnt

except ImportError:
    import fuzzy_sets as fs
    import rules
    import maintenance as mnt


def _extract_mod_word(text):
    pattern = r'\(MOD\s+([^)]+)\)'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def _remove_mod_completely(text):
    pattern = r'\(MOD\s+([^)]+)\)'
    replacement = ''
    return re.sub(pattern, replacement, text)


import re

def remove_parentheses(text):
    return re.sub(r'\(.*?\)', '', text)


def load_fuzzy_rules(rules_printed: str, fuzzy_variables: list) -> rules.MasterRuleBase:
    '''
    Load the rules from a string.
    
    :param rules_printed: string with the rules. Follows the specification given by the same printing method of rules.MasterRuleBase
    :param fuzzy_variables: list with the linguistic variables. Objects of FuzzyVariable class.
    :return mrule_base: object of MasterRuleBase class that contains the rules.
    '''
    if mnt.save_usage_flag:
        mnt.usage_data[mnt.usage_categories.Persistence]['persistence_read'] += 1
        
    consequent = 0
    linguistic_variables_names = [linguistic_variable.name for linguistic_variable in fuzzy_variables]
    value_names = [x.name for x in fuzzy_variables[0]]
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
                antecedent_value_index = value_names.index(antecedent_value)

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
    '''
    Load the linguistic variables from a string.
    
    :param fuzzy_variables_printed: string with the linguistic variables. Follows the specification given by the same printing method of FuzzyVariable class.
    :return fuzzy_variables: list with the linguistic variables. Objects of FuzzyVariable class.
    '''
    if mnt.save_usage_flag:
        mnt.usage_data[mnt.usage_categories.Persistence]['persistence_read'] += 1
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
        elif line == '':
            pass
        else:
            processes_line = line.split(';')

            if processes_line[1] == 'Categorical':
                categories = processes_line[0].split(',')

                if fuzzy_set_type == fs.FUZZY_SETS.t1:
                    fscat_categories = [fs.categoricalFS(category, category) for category in categories]
                elif fuzzy_set_type == fs.FUZZY_SETS.t2:
                    fscat_categories = [fs.categoricalFS(category, category) for category in categories]

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
    '''
    Save the linguistic variable to a string.
    
    :param fuzzy_variable: linguistic variable. Object of FuzzyVariable class.
    :return fuzzy_variable_printed: string with the linguistic variable. Follows the specification given by the same printing method of FuzzyVariable class.
    '''
    if mnt.save_usage_flag:
        mnt.usage_data[mnt.usage_categories.Persistence]['persistence_write'] += 1

    if isinstance(fuzzy_variable[0], fs.categoricalFS):
        fuzzy_variable_printed = '$Categorical variable: ' + fuzzy_variable.name
        if fuzzy_variable.units is not None:
            fuzzy_variable_printed += ' : ' + fuzzy_variable.units
        fuzzy_variable_printed += '\n'

        for fuzzy_set in fuzzy_variable.fuzzy_sets:
            fuzzy_variable_printed += fuzzy_set.name + ','
        fuzzy_variable_printed += 'Categorical\n'
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
    '''
    Save the linguistic variables to a string.
    
    :param fuzzy_variables: list with the linguistic variables. Objects of FuzzyVariable class.
    :return fuzzy_variables_printed: string with the linguistic variables. Follows the specification given by the same printing method of FuzzyVariable class.
    '''
    if mnt.save_usage_flag:
        mnt.usage_data[mnt.usage_categories.Persistence]['persistence_write'] += 1

    fuzzy_variables_printed = ''
    for fvar in fuzzy_variables:
        fuzzy_variables_printed += print_fuzzy_variable(fvar) + '\n'

    return fuzzy_variables_printed