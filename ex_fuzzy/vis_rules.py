# -*- coding: utf-8 -*-
"""
Plot fuzzy partitions and format fuzzy rule bases for inspection.

The plotting helper supports Type-1, interval Type-2, general Type-2, Gaussian,
and categorical fuzzy variables. Rule bases can also be converted to tabular or
LaTeX representations.
"""
import numpy as np
import pandas as pd

from . import rules
from . import fuzzy_sets as fs


def plot_fuzzy_variable(fuzzy_variable: fs.fuzzyVariable) -> None:
    """
    Plots a fuzzy variable using trapezoidal membership functions.

    Args:
        fuzzy_variable: a fuzzy variable from the fuzzyVariable class in fuzzy_set module.

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    fz_studied =  fuzzy_variable.linguistic_variables[0].type()
    set_shape = fuzzy_variable.linguistic_variables[0].shape()
    
    if set_shape != 'categorical':
        if fuzzy_variable.linguistic_variables[0].type() != fs.FUZZY_SETS.gt2:
            fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

        unit_resolution = 0.01
        unit_range_sampled = np.arange(
                0, 1 + unit_resolution, unit_resolution)

        colors = ['b', 'r', 'g', 'orange', 'purple']

        for ix, fuzzy_set in enumerate(fuzzy_variable.linguistic_variables):
            name = fuzzy_set.name
            color = colors[ix % len(colors)]
            initiated = False
            if getattr(fuzzy_set, 'domain', None) is None:
                continue
            domain_sampled = unit_range_sampled * (fuzzy_set.domain[1] - fuzzy_set.domain[0]) + fuzzy_set.domain[0]
            # Trapezoids are drawn through their four vertices.
            memberships = [0, 1, 1, 0]

            if fz_studied == fs.FUZZY_SETS.t1:
                if fuzzy_set.shape() == 'gaussian':
                    memberships = fuzzy_set.membership(domain_sampled)
                    ax.plot(domain_sampled, memberships, color=color, label=name)
                    ax.fill_between(domain_sampled, memberships, alpha=0.3)
                else:
                    ax.plot(fuzzy_set.membership_parameters, memberships, color=color, label=name)
                    ax.fill_between(fuzzy_set.membership_parameters, memberships, alpha=0.3)

            elif fz_studied == fs.FUZZY_SETS.t2 and fuzzy_set.shape() == 'gaussian':
                memberships = fuzzy_set.membership(domain_sampled)
                ax.fill_between(domain_sampled, memberships[:, 0], memberships[:, 1], color=color, alpha=0.5, label=name)

            elif fz_studied == fs.FUZZY_SETS.t2:
                ax.plot(fuzzy_set.secondMF_lower, np.array(memberships) * fuzzy_set.lower_height, 'black')
                ax.plot(fuzzy_set.secondMF_upper, np.array(memberships), 'black')

                # Compute the memberships for the lower/upper membership points. We do it in this way because non-exact 0/1s give problems.
                x_lower = fuzzy_set.secondMF_lower
                x_lower_lmemberships = [0.0 ,fuzzy_set.lower_height ,fuzzy_set.lower_height, 0.0]
                x_lower_umemberships = [fuzzy_set(x_lower[0])[1] , 1.0, 1.0 , fuzzy_set(x_lower[3])[1]]

                x_upper = fuzzy_set.secondMF_upper
                x_upper_lmemberships  = [0.0 , fuzzy_set(x_upper[1])[0], fuzzy_set(x_upper[2])[0], 0.0]
                x_upper_umemberships  = [0.0 ,1.0 ,1.0, 0.0]

                x_values = list(x_lower) + list(x_upper)
                lmembership_values = list(x_lower_lmemberships) + list(x_upper_lmemberships)
                umembership_values = list(x_lower_umemberships) + list(x_upper_umemberships)
                aux_df = pd.DataFrame(zip(x_values, lmembership_values, umembership_values),  columns=['x', 'l', 'u'])

                if len(aux_df['x']) != len(set(aux_df['x'])): # There are repeated elements, so we use an order that should work in this case
                    # u0 l0 u1 l1 l2 u2 l3 u3
                    x = list((x_upper[0], x_lower[0], x_upper[1], x_lower[1], x_lower[2], x_upper[2], x_lower[3], x_upper[3]))
                    l_memberships = list((x_upper_lmemberships[0], x_lower_lmemberships[0], x_upper_lmemberships[1], x_lower_lmemberships[1], x_lower_lmemberships[2], x_upper_lmemberships[2], x_lower_lmemberships[3], x_upper_lmemberships[3]))
                    u_memberships = list((x_upper_umemberships[0], x_lower_umemberships[0], x_upper_umemberships[1], x_lower_umemberships[1], x_lower_umemberships[2], x_upper_umemberships[2], x_lower_umemberships[3], x_upper_umemberships[3]))

                    ax.fill_between(x, l_memberships, u_memberships, color=color, alpha=0.5, label=name)
                else:
                    aux_df.sort_values('x', inplace=True)
                    ax.fill_between(aux_df['x'], aux_df['l'], aux_df['u'], color=color, alpha=0.5, label=name)
            else:
                for key, value in fuzzy_set.secondary_memberships.items():

                    gt2_memberships = value(fuzzy_set.sample_unit_domain)
                    key_plot = [float(key)]*sum(gt2_memberships > 0)
                    if initiated:
                        ax.plot(key_plot, fuzzy_set.sample_unit_domain[gt2_memberships > 0], gt2_memberships[gt2_memberships > 0], color=color)
                    else:
                        ax.plot(key_plot,  fuzzy_set.sample_unit_domain[gt2_memberships > 0], gt2_memberships[gt2_memberships > 0], color=color, label=name)
                        initiated = True
        
        # Enhance the overall style
        plt.legend(bbox_to_anchor=(1.05, 1))
        ax.set_ylabel('Membership degree', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1.1)
        plt.style.use('seaborn-v0_8-whitegrid')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        
        if fuzzy_variable.units is not None:
            ax.set_xlabel(fuzzy_variable.units, fontsize=12)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Add space for the title
        plt.title(fuzzy_variable.name, fontsize=16, fontweight='bold')
        fig.show(warn=False)
    else:
        # Sample data
        categories = [fuzzy_set.name for fuzzy_set in fuzzy_variable.linguistic_variables]

        # Create a table
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis('off')  # Hide axes
        table = ax.table(cellText=[[cat] for cat in categories], loc='center', cellLoc='center', colLabels=['Categories'])

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 2)  # Scale cell sizes

        # Show the name of the fuzzy variable
        plt.title(fuzzy_variable.name, fontsize=16, fontweight='bold')
        
        # Show plot
        plt.tight_layout()
        plt.show()


def matrix_rule_base_form(rule_base: rules.Rule) -> pd.DataFrame:
    """
    Returns a matrix with the rule base in the form of a matrix to visualize.

    Args:
        mrule_base: Rule base to transform.

    Returns:
        Matrix with the rule base in the form of a matrix.
    """

    n_rules = len(rule_base.rules)
    antecedents = len(rule_base.antecedents)

    res = pd.DataFrame(np.zeros((n_rules, antecedents)), columns=[jx.name for jx in rule_base.antecedents])

    for ix, rule in enumerate(rule_base):
        for jx, antecedent in enumerate(rule_base.antecedents):
            res.loc[ix, antecedent.name] = rule.antecedents[jx]
    
    return res


def filter_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the columns where every rule has a don't care (-1) value.

    Args:
        df: Dataframe to filter.

    Returns:
        Filtered dataframe.
    """
    for column in df.columns:
        if (df[column] == -1).all():
            df.drop(column, axis=1, inplace=True)

    return df


def rules_to_latex(rule_base:rules.MasterRuleBase) -> str:
    r"""
    Prints the rule base in a latex format.

    Args:
        rule_base: the master rule base to print.

    Returns:
        the String as a latex tabular.

    Note:
        if the rule base has three different linguistic labels, it will use custom commands for the partitions. You can define these commands (\low, \mid, \hig, \dc) to show colors, figures, etc. Be sure to recheck the DS, ACC values in this case, because 1.0 values of them are also converted to these commands.
    """
    class proxy_dict():

        def __init__(self) -> None:
            self.cell_colors = {
                -1: '\\dc',
                0: '\\low',
                1: '\\med',
                2: '\\hig'
            }
        
        def __getitem__(self, value) -> str:
            if value in self.cell_colors.keys():
                return self.cell_colors[value]
            else:
                return "{:.2f}".format(value)


    # Define the mapping for cell colors
    rules_matrix = rule_base.get_rulebase_matrix()

    # Add the consequent to the rules
    cons_rules_matrix = []
    for ix_cons, ruls in enumerate(rules_matrix):
        for rule_list in ruls:
            cons_rules_matrix.append(np.append(np.ones((1, )) * ix_cons, rule_list))

    # Interval dominance scores are summarised by their mean.
    dominance_scores = np.array([[np.mean(x.score)] for x in rule_base.get_rules()])

    accs = np.array([x.accuracy for x in rule_base.get_rules()])
    accs = np.expand_dims(accs, axis=1)
    cons_rules_matrix = np.append(np.append(np.array(cons_rules_matrix), dominance_scores, axis=1), accs, axis=1)
    column_order = ['Consequent'] + [a.name for a in rule_base.antecedents] + ['DS', 'Acc']
    df = pd.DataFrame(cons_rules_matrix, columns=column_order)
    cell_colors = proxy_dict()

    # Create the LaTeX table
    latex_table = "\\begin{tabular}{" + "c" * (len(column_order)-2) + "|cc}\n"
    latex_table += "\t\\toprule\n"
    latex_table += "\t" + " & ".join(column_order) + " \\\\\n"
    latex_table += "\t\\midrule\n"

    i = 0
    for cluster, group in df.groupby('Consequent'):
            latex_table += f"\t\\multirow{{{len(group)}}}{{*}}{{{cluster}}}"
            for _, row in group.iterrows():
                if i % 2 == 0: # Add a shade of grey
                    latex_table += " & \\cellcolor{gray!25}" + " & \\cellcolor{gray!25}".join([cell_colors[val] for val in row[column_order[1:]]]) + " \\\\\n"
                else:
                    latex_table += " & " + " & ".join([cell_colors[val] for val in row[column_order[1:]]]) + " \\\\\n"
                i += 1
            if cluster != len(rules_matrix) - 1:
                latex_table += "\t\\midrule\n"            
                
    latex_table += "\t\\bottomrule\n"
    latex_table += "\\end{tabular}"

    print(latex_table)
    return latex_table
