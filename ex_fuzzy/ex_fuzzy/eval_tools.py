"""
Functions that contain some general functions to eval already fitted fuzzy rule based models.
It can also be used to visualize rules and fuzzy partitions.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

try:
      from . import evolutionary_fit as evf
      from . import vis_rules
except ImportError:
      import evolutionary_fit as evf
      import vis_rules


def eval_fuzzy_model(fl_classifier: evf.BaseFuzzyRulesClassifier, X_train: np.array, y_train: np.array,
                     X_test: np.array, y_test: np.array, plot_rules=True, print_rules=True, plot_partitions=True, 
                     return_rules=False, print_accuracy=True, print_matthew=True, export_path:str=None) -> None:
    '''
    Function that evaluates a fuzzy rule based model. It also plots the rules and the fuzzy partitions.

    :param fl_classifier: Fuzzy rule based model.
    :param X_train: Training data.
    :param y_train: Training labels.
    :param X_test: Test data.
    :param y_test: Test labels.
    :param plot_rules: If True, it plots the rules.
    :param print_rules: If True, it prints the rules.
    :param plot_partitions: If True, it plots the fuzzy partitions.
    :return: None
    '''
    # Get the unique classes from the classifier
    unique_classes = fl_classifier.classes_
    # Convert the names from the labels to the corresponding class
    y_train = np.array([list(unique_classes).index(str(y)) for y in y_train])
    y_test = np.array([list(unique_classes).index(str(y)) for y in y_test])
    
    if print_accuracy:
      print('------------')
      print('ACCURACY')
      print('Train performance: ' +
            str(np.mean(np.equal(y_train, fl_classifier.predict(X_train)))))
      print('Test performance: ' +
            str(np.mean(np.equal(y_test, fl_classifier.predict(X_test)))))
      print('------------')
    if print_matthew:
      print('MATTHEW CORRCOEF')
      print('Train performance: ' +
            str(matthews_corrcoef(y_train, fl_classifier.predict(X_train))))
      print('Test performance: ' +
            str(matthews_corrcoef(y_test, fl_classifier.predict(X_test))))
      print('------------')

    if plot_rules:
        vis_rules.visualize_rulebase(fl_classifier.rule_base, export_path=export_path)
    if print_rules or return_rules:
        res = fl_classifier.print_rules(return_rules)

        if print_rules:
            print(res)

    if plot_partitions:
        fl_classifier.plot_fuzzy_variables()

    if return_rules:
        return res
    