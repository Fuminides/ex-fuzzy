'''
Load the rules of a fuzzy rules system using plain text format.

'''
import numpy as np

try:
    from . import fuzzy_sets as fs
    from . import rules
    from . import maintenance as mnt

except ImportError:
    import fuzzy_sets as fs
    import rules
    import maintenance as mnt


def load_fuzzy_rules(rules_printed: str, fuzzy_variables: list) -> rules.MasterRuleBase:
    '''
    Load the rules from a string.
    
    :param rules_printed: string with the rules. Follows the specification given by the same printing method of rules.MasterRuleBase
    :param fuzzy_variables: list with the linguistic variables. Objects of FuzzyVariable class.
    
    '''
    if mnt.save_usage_flag:
        mnt.usage_data[mnt.usage_categories.Persistence]['persistence_read'] += 1
        
    consequent = 0
    linguistic_variables_names = [linguistic_variable.name for linguistic_variable in fuzzy_variables]
    value_names = [x.name for x in fuzzy_variables[0]]
    fz_type = fuzzy_variables[0].fuzzy_type()
    consequent_names = []
    for line in rules_printed.splitlines():
        if line.startswith('IF'):
            #Is a rule
            antecedents , consequent_ds = line.split('WITH')
            consequent_ds = consequent_ds.split(',')[0].strip()
            init_rule_antecedents = np.zeros(
                (len(fuzzy_variables),)) - 1  # -1 is dont care
            
            for antecedent in antecedents.split('AND'):
                antecedent = antecedent.replace('IF', '').strip()
                antecedent_name, antecedent_value = antecedent.split('IS')
                antecedent_name = antecedent_name.strip()
                antecedent_value = antecedent_value.strip()
                antecedent_index = linguistic_variables_names.index(antecedent_name)
                antecedent_value_index = value_names.index(antecedent_value)

                init_rule_antecedents[antecedent_index] = antecedent_value_index
                
            rule_simple = rules.RuleSimple(init_rule_antecedents, 0)
            rule_simple.score = float(consequent_ds[3:].strip()) # We remove the 'DS ' and the last space
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
                mrule_base = rules.MasterRuleBase([rule_base])
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

