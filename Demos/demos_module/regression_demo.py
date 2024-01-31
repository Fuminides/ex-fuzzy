# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:35:55 2021
All rights reserved

@author: Javier Fumanal Idocin - University of Essex
@author: Javier Andreu-Perez - University of Essex


This is a the source file that contains a demo for a tip computation example, where a diferent set of IVFS are used to compute
a t2 reasoning approach.

"""
import sys
# In case you run this without installing the package, you need to add the path to the package

# This is for launching from root folder path
sys.path.append('./ex_fuzzy/')
sys.path.append('./ex_fuzzy/ex_fuzzy/')

# This is for launching from Demos folder
sys.path.append('../ex_fuzzy/')
sys.path.append('../ex_fuzzy/ex_fuzzy/')

import numpy as np

import ex_fuzzy.fuzzy_sets as t2
import ex_fuzzy.rules as rules

# Define the fuzzy sets
food_rancid_lower = [0, 0, 0.5, 4.5]
food_rancid_upper = [0, 0, 1, 5]
food_delicious_lower = [4.5, 8.5, 9, 9]
food_delicious_upper = [4, 8, 9, 9]

food_rancid = t2.IVFS('Rancid', food_rancid_lower, food_rancid_upper, [0,9])
food_delicious = t2.IVFS('Delicious', food_delicious_lower, food_delicious_upper, [0,9])

#Use the fuzzy sets to define a fuzzy variable with its linguistic partitions.
food = t2.fuzzyVariable('Food', [food_rancid, food_delicious])

service_poor_lower = [0, 0, 0.5, 2.5]
service_poor_upper = [0, 0, 1, 3]
service_good_lower = [1.5, 3.5, 4.5, 6.5]
service_good_upper = [1, 3, 5, 7]
service_excellent_lower = [5.5, 7.5, 9, 9]
service_excellent_upper = [5, 7, 9, 9]

service_poor = t2.IVFS('Poor', service_poor_lower, service_poor_upper, [0,9])
service_good = t2.IVFS('Good', service_good_lower, service_good_upper, [0,9])
service_excellent = t2.IVFS('Excellent', service_excellent_lower, service_excellent_upper, [0,9])

service = t2.fuzzyVariable('Service', [service_poor, service_good, service_excellent])

tip_cheap_lower = [2, 6, 6, 10]
tip_cheap_upper = [0, 6, 6, 12]
tip_average_lower = [12, 15, 15, 18]
tip_average_upper = [10, 15, 15, 20]
tip_genereous_lower = [20, 24, 24, 28]
tip_generous_upper = [18, 24, 24, 30]

tip_cheap = t2.IVFS('Cheap', tip_cheap_lower, tip_cheap_upper, [0,30])
tip_average = t2.IVFS('Average', tip_average_lower, tip_average_upper, [0,30])
tip_genereous = t2.IVFS('Generous', tip_genereous_lower, tip_generous_upper, [0,30])

tip = t2.fuzzyVariable('Tip', [tip_cheap, tip_average, tip_genereous])

rule_list =[
    rules.RuleSimple([0, 0], 0),
    rules.RuleSimple([0, 1], 0),
    rules.RuleSimple([0, 2], 1),
    rules.RuleSimple([1, 0], 1),
    rules.RuleSimple([1, 1], 1),
    rules.RuleSimple([1, 2], 2)
]

inference_module = rules.RuleBaseT2([food, service], rule_list, tip)
input = np.array([4,2.5]).reshape((1,2))
print(inference_module.inference(input))