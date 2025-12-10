This folder contains a series of demos to try different features of the ex-fuzzy library. These are presented in two different formats: jupyter notebooks and python modules, which are stored under the demos_module folder. You dont need to install the library to execute them.

The list of demos is the following:

1. iris_demo: shows a simple classification example. It shows how to train a classifier, how to save checkpoints, how to show the rules in latex tabular format and to save them into a text file.
2. iris_demo_custom_loss: a classification example where the predefined loss is changed by other function.
3. iris_demo_persistence: a classification example where the rules are saved into a file and then imported for another classifier.
4. precandidate_rules_demo: a classification example where we first fit a fuzzy classifier as usual, and then, we look for the optimal subset of those rules.
5. regression_demo: an example of a regression problem using interval-type 2 fuzzy sets.
6. occupancy_demo_temporal: an example of the use of temporal fuzzy sets.
7. iris_demo_advanced_classifiers: in this example we show the different training procedures that can be found in classifiers.py file.

## Regression Demos (in demos_module/):
8. demo_california_housing.py: Complete regression example using California Housing dataset, showing both crisp and fuzzy consequents with Mamdani inference
9. demo_additive_regression.py: Demonstrates additive vs sufficient rule modes in regression
10. demo_print_regression_rules.py: Shows how to print and display regression rules in various formats
11. demo_simple_print.py: Simple example of using print() with regression models
12. demo_complete_workflow.py: End-to-end regression workflow from training to prediction


