"""
Evaluation Tools for Fuzzy Rule-Based Models

This module provides comprehensive evaluation and analysis tools for fuzzy classification
models. It includes performance metrics, statistical analysis, visualization capabilities,
and model interpretation tools specifically designed for fuzzy rule-based systems.

Main Components:
    - FuzzyEvaluator: Core evaluation class for fuzzy models
    - Performance metrics: Accuracy, F1-score, precision, recall, and fuzzy-specific metrics
    - Statistical analysis: Bootstrap confidence intervals and significance testing
    - Rule analysis: Rule importance, coverage, and interpretability metrics
    - Visualization integration: Hooks for rule and partition plotting
    - Model comparison: Tools for comparing different fuzzy models

Key Features:
    - Scikit-learn compatible metric evaluation
    - Fuzzy-specific evaluation measures (rule coverage, dominance scores)
    - Bootstrap statistical analysis for robust performance assessment
    - Integration with visualization tools for rule inspection
    - Support for multi-class and imbalanced dataset evaluation
    - Comprehensive reporting with statistical significance

The module is designed to provide both quick evaluation capabilities and in-depth
analysis tools for understanding fuzzy model behavior and performance characteristics.
"""
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

try:
      from . import evolutionary_fit as evf
      from . import vis_rules
except ImportError:
      import evolutionary_fit as evf
      import vis_rules
    
    
def eval_fuzzy_model(fl_classifier: evf.BaseFuzzyRulesClassifier, X_train:np.array, y_train:np.array, X_test:np.array, y_test:np.array, plot_rules=False,print_rules:bool=True, plot_partitions:bool=False, return_rules:bool=True, bootstrap_results_print:bool=True) -> str:
    """
    Comprehensive evaluation of a fitted fuzzy rule-based classifier.
    
    This function provides a complete evaluation workflow for fuzzy classifiers including
    performance metrics, rule analysis, visualization options, and statistical testing.
    It serves as a convenient wrapper around the FuzzyEvaluator class.
    
    Args:
        fl_classifier (evf.BaseFuzzyRulesClassifier): Fitted fuzzy rule-based classifier
        X_train (np.array): Training feature data used for model fitting
        y_train (np.array): Training target labels used for model fitting  
        X_test (np.array): Test feature data for evaluation
        y_test (np.array): Test target labels for evaluation
        plot_rules (bool, optional): Whether to generate rule visualization plots. Defaults to False.
        print_rules (bool, optional): Whether to print rule text representations. Defaults to True.
        plot_partitions (bool, optional): Whether to plot fuzzy variable partitions. Defaults to False.
        return_rules (bool, optional): Whether to include rule text in return string. Defaults to True.
        bootstrap_results_print (bool, optional): Whether to perform bootstrap statistical analysis. Defaults to True.
        
    Returns:
        str: Comprehensive evaluation report containing performance metrics, rule analysis,
            and statistical results formatted as a readable text report.
            
    Example:
        >>> classifier = BaseFuzzyRulesClassifier()
        >>> classifier.fit(X_train, y_train)
        >>> report = eval_fuzzy_model(classifier, X_train, y_train, X_test, y_test)
        >>> print(report)
        
    Note:
        This function creates a FuzzyEvaluator instance internally and calls its
        eval_fuzzy_model method. For more control over the evaluation process,
        consider using FuzzyEvaluator directly.
    """
    fuzzy_evaluator = FuzzyEvaluator(fl_classifier)
    res = fuzzy_evaluator.eval_fuzzy_model(X_train, y_train, X_test, y_test, 
                      plot_rules=plot_rules, print_rules=print_rules, plot_partitions=plot_partitions, return_rules=return_rules, bootstrap_results_print=bootstrap_results_print)
    

    return res

class FuzzyEvaluator():
    """
    Comprehensive evaluation and analysis tool for fuzzy rule-based classifiers.
    
    This class provides a complete suite of evaluation methods for fuzzy classification
    models, including performance metrics, rule analysis, statistical testing, and
    visualization capabilities. It is designed to work with fuzzy classifiers that
    follow the BaseFuzzyRulesClassifier interface.
    
    Attributes:
        fl_classifier (evf.BaseFuzzyRulesClassifier): The fuzzy classifier to evaluate
        
    Example:
        >>> evaluator = FuzzyEvaluator(trained_classifier)
        >>> predictions = evaluator.predict(X_test)
        >>> accuracy = evaluator.get_metric('accuracy_score', X_test, y_test)
        >>> report = evaluator.eval_fuzzy_model(X_train, y_train, X_test, y_test)
        
    Note:
        The FuzzyEvaluator assumes the classifier has been fitted before evaluation.
        It provides both individual metric computation and comprehensive evaluation reports.
    """
    def __init__(self,fl_classifier: evf.BaseFuzzyRulesClassifier):
        """
        Initialize the FuzzyEvaluator with a fitted fuzzy classifier.
        
        Args:
            fl_classifier (evf.BaseFuzzyRulesClassifier): A fitted fuzzy rule-based
                classifier that implements the standard fit/predict interface.
        """
        self.fl_classifier = fl_classifier


    
    def predict(self, X: np.array) -> np.array:
        """
        Generate predictions for input data using the wrapped fuzzy classifier.
        
        This method provides a unified interface for prediction that can be used
        with scikit-learn evaluation metrics and other analysis tools.
        
        Args:
            X (np.array): Feature data for prediction with shape (n_samples, n_features)
        
        Returns:
            np.array: Predicted class labels with shape (n_samples,)
        """
        return self.fl_classifier.predict(X)
    

    def get_metric(self, metric: str, X_true: np.array, y_true: np.array, **kwargs) -> float:
        """
        Compute a specific classification metric for the fuzzy model.
        
        This method provides a unified interface for computing various scikit-learn
        classification metrics on the fuzzy model predictions. It handles class
        label conversion and error handling for unsupported metrics.
        
        Args:
            metric (str): Name of the sklearn.metrics function to compute (e.g., 'accuracy_score', 'f1_score')
            X_true (np.array): Feature data for prediction
            y_true (np.array): True class labels
            **kwargs: Additional arguments for the specific metric function
            
        Returns:
            float: The computed metric value, or error string if metric is unavailable
            
        Example:
            >>> evaluator = FuzzyEvaluator(classifier)
            >>> accuracy = evaluator.get_metric('accuracy_score', X_test, y_test)
            >>> f1 = evaluator.get_metric('f1_score', X_test, y_test, average='weighted')
            
        Note:
            The method automatically handles string class labels by converting them
            to numeric indices based on the classifier's classes_names attribute.
        """
        # Get y predictions
        y_pred = self.predict(X_true)
        y_true = np.array(y_true)
        # Convert str classes to numbers in corresponding class if necessary
        unique_classes = self.fl_classifier.classes_names

        if isinstance(y_true[0], str):
            y_true = np.array([list(unique_classes).index(str(y)) for y in y_true])
      
        #Find metrics requested in sklearn library, if not found 
        try:
             # Get the metric function dynamically from sklearn.metrics
              metric_function = getattr(metrics, metric)
             # Call the metric function with y_true, y_pred, and any additional keyword arguments
              return metric_function(y_true, y_pred, **kwargs)
        except AttributeError:
              return f"Metric '{metric}' not found in sklearn.metrics."
        except TypeError:
              return f"Invalid arguments passed for the metric '{metric}'."
        
        
    def eval_fuzzy_model(self, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, 
                         plot_rules=True, print_rules=True, plot_partitions=True, 
                         return_rules=False, print_accuracy=True, print_matthew=True, export_path: str = None, bootstrap_results_print: bool = True) -> None:
        """
        Comprehensive evaluation of the fuzzy rule-based model.
        
        This method provides a complete evaluation workflow including performance metrics,
        rule visualization, partition plotting, and statistical analysis. It combines
        multiple evaluation aspects into a single convenient interface.
        
        Args:
            X_train (np.array): Training feature data
            y_train (np.array): Training target labels
            X_test (np.array): Test feature data
            y_test (np.array): Test target labels
            plot_rules (bool, optional): Whether to generate rule visualization plots. Defaults to True.
            print_rules (bool, optional): Whether to print rule text representations. Defaults to True.
            plot_partitions (bool, optional): Whether to plot fuzzy variable partitions. Defaults to True.
            return_rules (bool, optional): Whether to return rule text in output. Defaults to False.
            print_accuracy (bool, optional): Whether to print accuracy metrics. Defaults to True.
            print_matthew (bool, optional): Whether to print Matthews correlation coefficient. Defaults to True.
            export_path (str, optional): Path to export rule visualization plots. Defaults to None.
            bootstrap_results_print (bool, optional): Whether to perform bootstrap statistical analysis. Defaults to True.
            
        Returns:
            str or None: Rule text representation if return_rules=True, otherwise None
            
        Example:
            >>> evaluator = FuzzyEvaluator(classifier)
            >>> report = evaluator.eval_fuzzy_model(X_train, y_train, X_test, y_test,
            ...                                     plot_rules=True, print_rules=True)
            
        Note:
            This method handles string class labels automatically and provides
            comprehensive output including performance metrics and rule analysis.
        """
        # Get the unique classes from the classifier
        unique_classes = self.fl_classifier.classes_names

        # Convert the names from the labels to the corresponding class if necessary
        if isinstance(y_train[0], str):
            y_train = np.array([list(unique_classes).index(str(y)) for y in y_train])
            y_test = np.array([list(unique_classes).index(str(y)) for y in y_test])
      
        if print_accuracy:
            print('------------')
            print('ACCURACY')
            print('Train performance: ' +
                  str(self.get_metric('accuracy_score', X_train, y_train)))
            print('Test performance: ' +
                  str(self.get_metric('accuracy_score', X_test, y_test)))
            print('------------')
            
        if print_matthew:
            print('MATTHEW CORRCOEF')
            print('Train performance: ' +
                  str(self.get_metric('matthews_corrcoef', X_train, y_train))
                  )
            print('Test performance: ' +
                  str(self.get_metric('matthews_corrcoef', X_test, y_test))
                  )
            print('------------')

        if print_rules or return_rules:
            res = self.fl_classifier.print_rules(True, bootstrap_results=bootstrap_results_print)

            if print_rules:
                print(res)

        if plot_partitions:
            self.fl_classifier.plot_fuzzy_variables()

        if plot_rules:
            vis_rules.visualize_rulebase(self.fl_classifier.rule_base, export_path=export_path)
            
        if return_rules:
            return res