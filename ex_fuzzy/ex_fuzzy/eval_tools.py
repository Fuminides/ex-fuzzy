"""
Functions that contain some general functions to eval already fitted fuzzy rule based models.
It can also be used to visualize rules and fuzzy partitions.
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
    
    
class FuzzyEvaluator():
    '''
    Takes a model and associated data and permits rule evaluation
    '''
    def __init__(self,fl_classifier: evf.BaseFuzzyRulesClassifier):
            '''
            :param fl_classifier: Fuzzy rule based model
            '''
            self.fl_classifier = fl_classifier


    
    def predict(self,X: np.array) -> np.array:
        # Predict y for given X for use in metric evaluation
        return self.fl_classifier.predict(X)
    

    def get_metric(self,metric:str,X_true:np.array,y_true:np.array,**kwargs) -> float:
        '''
        :param metric: named metric in string format available in sklearn library for evaluation
        :param X_true: np.array of X values for prediction
        :param y_true: np.array of true class outcomes for X values
        :param **kwargs: additional arguments for different sklearn.metrics functions
        '''
        #Get y predictions
        y_pred = self.predict(X_true)
        y_true = np.array(y_true)
        #Convert str classes to numbers in corresponding class if necessary
        unique_classes = self.fl_classifier.classes_names

        if isinstance(y_true[0],str):
            y_true = np.array([list[unique_classes].index(str(y)) for y in y_true])
      
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
        
        
    def eval_fuzzy_model(self,X_train: np.array, y_train: np.array,X_test: np.array, y_test: np.array, 
                         plot_rules=True, print_rules=True, plot_partitions=True, 
                     return_rules=False, print_accuracy=True, print_matthew=True, export_path:str=None) -> None:
      '''
      Function that evaluates a fuzzy rule based model. It also plots the rules and the fuzzy partitions.

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
      unique_classes = self.fl_classifier.classes_names

      # Convert the names from the labels to the corresponding class if necessary
      if isinstance(y_train[0], str):
            y_train = np.array([list(unique_classes).index(str(y)) for y in y_train])
            y_test = np.array([list(unique_classes).index(str(y)) for y in y_test])
      
      if print_accuracy:
            print('------------')
            print('ACCURACY')
            print('Train performance: ' +
                  str(self.get_metric('accuracy_score',X_train,y_train)))
            print('Test performance: ' +
                  str(self.get_metric('accuracy_score',X_test,y_test)))
            print('------------')
      if print_matthew:
            print('MATTHEW CORRCOEF')
            print('Train performance: ' +
                  str(self.get_metric('matthews_corrcoef',X_train,y_train))
                  )
            print('Test performance: ' +
                  str(self.get_metric('matthews_corrcoef',X_test,y_test))
                  )
            print('------------')


      if print_rules or return_rules:
            res = self.fl_classifier.print_rules(True)

            if print_rules:
                  print(res)

      if plot_partitions:
            self.fl_classifier.plot_fuzzy_variables()

      if plot_rules:
            vis_rules.visualize_rulebase(self.fl_classifier.rule_base, export_path=export_path)
            
      if return_rules:
            return res