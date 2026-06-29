.. _step2:

Using Fuzzy Rules
=================

-----------------
Fuzzy Rules
-----------------
Fuzzy rules can be used to solve both regression and classification problems. 

The most straightforward way to construct a rule is to give a series of antecedents and a consequent. 
For the case of classification, the consequent will be a class, and for regression, a fuzzy set.
Following the temperature example. Suppose we have these fuzzy sets as consequents to module
the use of air conditioner::
 
    activate_small = fs.FS('Small', [0.0, 0.0, 0.1, 0.2],  [0,1])
    activate_medium = fs.FS('Small', [0.1, 0.4, 0.4, 0.5],  [0,1])
    activate_large = fs.FS('Small', [0.5, 0.8, 1.0, 1.0],  [0,1])

    activate = fs.fuzzyVariable('Activate', [activate_small, activate_medium, activate_large])

We can construct a rule for regression using the ``ex_fuzzy.rules.Rule`` class. 
For example, the rule IF temperature IS hot THEN conditioner IS large can be implemented as::

    import ex_fuzzy.rules as frule
    frule.Rule([hot], activate_large)

Then, we can use the ``membership`` method to obtain the degree of truth for a value in a rule, and the ``centroid`` method to
compute the centroid of the consequent.

This implementation, however, can be problematic when there is a considerable number of rules with repeated antecedents, 
because we do not want to compute the degree of truth for a value for the same antecedents over and over. So, instead
of using the ``Rule`` class, it is more practical to use ``RuleBase`` and ``RuleSimple`` classes.

-----------------
Rule Bases
-----------------

``RuleSimple`` is a class that simplifies the way in which rules are expressed. Its antecedents are expressed as a list, denoting the
linguistic variable relevant to the rule. The previous rule would be expressed as a ``RuleSimple`` as this::

    my_rule = frule.RuleSimple([2], 2)

The length of the first list is the number of antecedents, and the second argument denotes that the consequent fuzzy set is "activates_large".
``RuleSimple`` is used by ``RuleBase`` class to efficiently compute the degrees of truth for all the antecedents for all the data,
and then use them when necessary. In order to create one rule base, we need the list of all the fuzzy variables to use, the consequent
and the rules expressed as ``RuleSimple`` objects::

    my_rulebase = frule.RuleBaseT1([temperature], [my_rule], activate) 

This is quite a simple case because we are using only one fuzzy variable and one rule, but the process is the same for more rules and variables.
Then, we can use "my_rule" using the ``inference`` method::

    my_rulebase.inference(np.array([8.2]))

Which will return the defuzzified result of the fuzzy inference process. The process is the same for the rest of the fuzzy sets, but other
classes are required: ``RuleBaseT2``, ``RuleBaseGT2``.

---------------------------------------------
Classification problems and Master Rule Bases
---------------------------------------------
Up to now, we have discussed how to model a regression problem. Classification problems perform the inference in a different way, which require another kind of object: the ``ex_fuzzy.rules.MasterRuleBase``.
This is because the way in which Ex-Fuzzy handles classification problems is by using one Rule Base per consequent. 
So, the ``rules.MasterRuleBase`` class is used to handle the rule bases created for each class. An object of this class is created using
a list of rule bases, and its main method is ``rules.MasterRuleBase.winning_rule_predict()`` which returns the class obtained from the rule with highest association degree.
You can find more the specifics of the classification inference in the next steps.


The next step is :ref:`step3`.
