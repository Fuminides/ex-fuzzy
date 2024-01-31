# -*- coding: utf-8 -*-
"""
This is a the source file that contains the functions necessary to visualize the set of rules.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

try:
    from . import rules
    from . import evolutionary_fit as evf
    from . import fuzzy_sets as fs
    from . import maintenance as mnt
except ImportError:
    import rules
    import evolutionary_fit as evf
    import fuzzy_sets as fs
    import maintenance as mnt


def _column_histogram(rule_matrix: np.array) -> dict:
    '''
    Computes the histogram for all the unique values in a column.

    :param rule_matrix: vector with the numerical values.
    :return: dictionary with the histogram.
    '''
    res = {}
    for x in np.unique(rule_matrix):
        if x != -1:
            res[x] = np.sum(rule_matrix == x)

    return res


def _histogram(rule_matrix: np.array) -> list[dict]:
    '''
    Returns a list with the histogram for each antecedent according to linguist variables.

    :param rule_matrix: matrix with the rules.
    :return: list with the histogram in dictionary format for each antecedent.
    '''
    res = []

    for column_ix in range(rule_matrix.shape[1]):
        res.append(_column_histogram(rule_matrix[:, column_ix]))

    return res


def _max_values(appearances: list) -> tuple[int, int]:
    '''
    Returns the antecedent and its linguistic value most repeated.

    :param appearances: list with the histogram for each antecedent.
    :return: tuple with the antecedent and its linguistic value most repeated.
    '''
    res = 0
    antecedent = None
    vl = None

    for ix, apperances_an in enumerate(appearances):
        for key, value in apperances_an.items():
            if value > res:
                res = value
                antecedent = ix
                vl = key

    return antecedent, vl


def create_graph_connection(rules, possible_vl):
    '''
    Returns a square matrix where each number indicates if two nodes are connected.
    Connectes by checking if both are in the same rule.

    :param rules: list with the rules.
    :param possible_vl: number of linguistic variables.
    :return: square matrix where each number indicates if two nodes are connected.
    '''
    def generate_index(ant, vl0): return int(possible_vl * ant + vl0)
    res = np.zeros(
        (possible_vl * rules.shape[1], possible_vl * rules.shape[1]))

    for rule in rules:
        for antecedent, vl in enumerate(rule):
            if vl > -1:
                for antecedent2, vl2 in enumerate(rule):
                    if vl2 > -1:
                        res_index1 = generate_index(antecedent, vl)
                        res_index2 = generate_index(antecedent2, vl2)

                        res[res_index1, res_index2] += 1

    return res / 2


def choose_popular_rules(rule_matrix: np.array) -> np.array:
    '''
    Returns the index of the rules that contain the most popular antecedent in the dataset.

    :param rule_matrix: matrix with the rules.
    :return: numpy array with the rules that contain the most popular antecedent in the dataset.
    '''
    appearances = _histogram(rule_matrix)
    antecedent, vl = _max_values(appearances)

    chosen_rules = rule_matrix[:, antecedent] == vl

    return chosen_rules


def connect_rulebase(rulebase: rules.RuleBase) -> list[np.array]:
    '''
    Connects antecedents connected by checking if both are in the same rule.

    :param rulebase: Rule base to connect.
    :return: List of pandas dataframes with the connections in adjacency matrix format.
    '''

    # We choose those rules to explain, as those who have the most common antecedent.
    rule_matrix = rules.list_rules_to_matrix(rulebase.rules)
    res = []
    antecedents_names = [
        x.name + ' ' + y for x in rulebase.antecedents for y in rulebase.antecedents[0].linguistic_variable_names()]

    while rule_matrix.shape[0] > 0:
        rules_to_viz = rule_matrix[choose_popular_rules(rule_matrix), :]
        rule_matrix = rule_matrix[(
            1 - choose_popular_rules(rule_matrix)).astype(bool), :]

        graph_rule = create_graph_connection(rules_to_viz, len(
            rulebase.antecedents[0].linguistic_variable_names()))
        res.append(pd.DataFrame(
            graph_rule, columns=antecedents_names, index=antecedents_names))

    return res


def connect_master_rulebase(mrule_base: rules.MasterRuleBase) -> list[list[np.array]]:
    '''
    Connects antecedents connected by checking if both are in the same rule.

    :param mrule_base: Master rule base to connect.
    :return: List of list of pandas dataframes with the connections in adjacency matrix format.
    '''
    res = []
    for rule_base in mrule_base.rule_bases:
        res.append(connect_rulebase(rule_base))

    return res


def visualize_rulebase(mrule_base: rules.MasterRuleBase, export_path: str=None) -> None:
    '''
    Visualize a rule base using low, medium and high partitions with 1 rule in common -> 1 edge connections.

    :param mrule_base: Master rule base to visualize.
    :param export_path: Path to export the graph.
    '''

    if mnt.save_usage_flag:
        mnt.usage_data[mnt.usage_categories.Visualization]['plot_graph'] += 1

    def color_func(a):
        '''
        Returns the color of the node according to the linguistic variable.

        :param a: Node name.
        :return: Color of the node.
        '''
        node_colors = ['blue', 'yellow', 'red']

        if ' L' in a:
            return node_colors[0]
        elif ' M' in a:
            return node_colors[1]
        else:
            return node_colors[2]

    def vl_prune(a): return a.replace('High', 'H').replace(
        'Medium', 'M').replace('Low', 'L').strip()

    if isinstance(mrule_base, evf.BaseFuzzyRulesClassifier):
        mrule_base = mrule_base.rule_base

    connected_mrule_base = connect_master_rulebase(mrule_base)

    for ix, connected_rule_base in enumerate(connected_mrule_base):
        for jx, rule in enumerate(connected_rule_base):
            G = nx.from_pandas_adjacency(rule)
            isolated_nodes = [
                node for node in G.nodes() if G.degree(node) == 0]
            G.remove_nodes_from(isolated_nodes)
            auto_edges = nx.selfloop_edges(G)
            G.remove_edges_from(auto_edges)

            new_node_names = [vl_prune(node) for node in G.nodes()]
            mapping = dict(zip(G, new_node_names))
            G = nx.relabel_nodes(G, mapping)

            if jx == 0:
                G_final = G
            else:
                G_final = nx.compose(G_final, G)

        fig, ax = plt.subplots()
        try:
            os = nx.nx_agraph.graphviz_layout(G_final, prog='sfdp')
        except ImportError:
            os = nx.kamada_kawai_layout(G_final)

        node_colors = [color_func(node) for node in G_final.nodes()]
        nx.draw(G_final, with_labels=True, ax=ax,
                pos=os, node_color=node_colors)
        plt.title('Consequent: ' + str(ix))
        fig.show()

        if export_path is not None:
            nx.write_gexf(G_final, os.path.join(export_path,
                          'consequent_' + str(ix) + '.gexf'))


def plot_fuzzy_variable(fuzzy_variable: fs.fuzzyVariable) -> None:
    '''
    Plots a fuzzy variable using trapezoidal membership functions.

    :param fuzzy_variable: a fuzzy variable from the fuzzyVariable class in fuzzy_set module.
    :return: None
    '''
    if mnt.save_usage_flag:
        mnt.usage_data[mnt.usage_categories.Visualization]['plot_fuzzy_variable'] += 1

    if fuzzy_variable.linguistic_variables[0].type() != fs.FUZZY_SETS.gt2:
        fig, ax = plt.subplots()
    else:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

    memberships = [0, 1, 1, 0]

    colors = ['b', 'r', 'g', 'orange', 'y']
    for ix, fuzzy_set in enumerate(fuzzy_variable.linguistic_variables):
        name = fuzzy_set.name
        initiated = False
        fz_studied =  fuzzy_set.type()

        if  fz_studied == fs.FUZZY_SETS.t1:
            ax.plot(fuzzy_set.membership_parameters,
                    memberships, colors[ix], label=name)
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

                ax.fill_between(x, l_memberships, u_memberships, color=colors[ix], alpha=0.5, label=name)
            else:
                aux_df.sort_values('x', inplace=True)
                ax.fill_between(aux_df['x'], aux_df['l'], aux_df['u'], color=colors[ix], alpha=0.5, label=name)

        elif fz_studied == fs.FUZZY_SETS.gt2:
            for key, value in fuzzy_set.secondary_memberships.items():
                
                gt2_memberships = value(fuzzy_set.sample_unit_domain)
                key_plot = [float(key)]*sum(gt2_memberships > 0)
                if initiated:
                    ax.plot(key_plot, fuzzy_set.sample_unit_domain[gt2_memberships > 0], gt2_memberships[gt2_memberships > 0], color=colors[ix])
                else:
                    ax.plot(key_plot,  fuzzy_set.sample_unit_domain[gt2_memberships > 0], gt2_memberships[gt2_memberships > 0], color=colors[ix], label=name)
                    initiated = True

    ax.legend(loc='upper right', shadow=True)
    plt.title(fuzzy_variable.name)
    fig.show()


def matrix_rule_base_form(rule_base: rules.Rule) -> pd.DataFrame:
    '''
    Returns a matrix with the rule base in the form of a matrix to visualize.

    :param mrule_base: Rule base to transform.
    :return: Matrix with the rule base in the form of a matrix.
    '''

    n_rules = len(rule_base.rules)
    antecedents = len(rule_base.antecedents)

    res = pd.DataFrame(np.zeros((n_rules, antecedents)), columns=[jx.name for jx in rule_base.antecedents])

    for ix, rule in enumerate(rule_base):
        for jx, antecedent in enumerate(rule_base.antecedents):
            res.loc[ix, antecedent.name] = rule.antecedents[jx]
    
    return res


def filter_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Filter columns with only one value.

    :param df: Dataframe to filter.
    :return: Filtered dataframe.
    '''
    for column in df.columns:
        if df[column].unique()[0] == -1:
            df.drop(column, axis=1, inplace=True)

    return df


def rules_to_latex(rule_base:rules.MasterRuleBase) -> str:
    '''
    Prints the rule base in a latex format. It prints

    :note: if the rule base has three different linguistic labels, it will use custom commands for the partitions. You can define these commands (\low, \mid, \hig, \dc) to show colors, figures, etc. Be sure to recheck the DS, ACC values in this case, because 1.0 values of them are also converted to these commands.
    
    :param rule_base: the master rule base to print.
    :returns: the String as a latex tabular.
    '''
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
            if len(rule_list.shape) == 2:
                cons_rules_matrix.append(np.append(np.ones((rule_list.shape[0])) * ix_cons, rule_list, axis=1))
            else:
                cons_rules_matrix.append(np.append(np.ones((1, )) * ix_cons, rule_list))
                     
    dominance_scores = np.array([x.score for x in rule_base.get_rules()])
    if len(dominance_scores.shape) == 2:
        dominance_scores = np.mean(dominance_scores, axis=1, keepdims=True)
    else:
        dominance_scores = np.expand_dims(dominance_scores, axis=1)

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
                    latex_table += " & \cellcolor{gray!25}" + " & \cellcolor{gray!25}".join([cell_colors[val] for val in row[column_order[1:]]]) + " \\\\\n"
                else:
                    latex_table += " & " + " & ".join([cell_colors[val] for val in row[column_order[1:]]]) + " \\\\\n"
                i += 1
            if cluster != len(rules_matrix) - 1:
                latex_table += "\t\\midrule\n"            
                
    latex_table += "\t\\bottomrule\n"
    latex_table += "\\end{tabular}"

    print(latex_table)