.. _precom:

Computing fuzzy partitions
=======================================

One of the most typical ways to compute fuzzy partitions is to use quantiles of the data. The module ``utils`` contains a series of functions
to generate fuzzy partitions for all the supported kinds of fuzzy sets.
The easiest way to compute these partitions is with the ``utils.construct_partitions`` function, specifying the fuzzy set desired::

    from ex_fuzzy import utils

    fz_type_studied = fs.FUZZY_SETS.t2
    precomputed_partitions = utils.construct_partitions(X, fz_type_studied)

---------------------------------
Categorical variables
---------------------------------
Quantiles are meaningless for a variable that holds names, or one that is just a 0/1 flag. ``construct_partitions``
detects those variables and gives them a crisp partition instead: one fuzzy set per category, which is 1 exactly on
that category and 0 everywhere else. A variable is detected as categorical when its values are not numeric, or when
it only takes whole numbers and at most five different ones::

    categorical_mask = utils.detect_categorical_mask(X)  # what construct_partitions detects on its own

Pass ``categorical_mask`` yourself to decide it instead of the detection, or ``detect_categorical=False`` to treat
every variable as numerical. ``BaseFuzzyRulesClassifier`` accepts the same two arguments and uses them when it
optimizes the partitions itself.

---------------------------------
Partitions from your own bands
---------------------------------
Quantile terms say little when the variable has an agreed vocabulary, such as the official bands of an air quality
index. ``construct_breakpoint_partition`` builds a trapezoidal variable from the boundaries you give: one term per band, shoulders
at both ends of the domain, and a linear crossover of the width you choose around each boundary, so the two neighbouring
terms reach 0.5 at the breakpoint::

    no2 = utils.construct_breakpoint_partition(
        'NO2', ['Good', 'Fair', 'Poor'], breakpoints=[40, 90], domain=[0, 400], transition=10, units='ug/m3')

Pass such variables as ``linguistic_variables`` to any classifier or regressor. The domain is not learned from data, so
inputs outside it are clipped to it, as with the other partitions.

--------------------------------
About the precomputed partitions
--------------------------------
Partitions computed using these method use three linguistic variables per fuzzy variable. We chose that number as it creates easily understandable
low, medium and high partitions. For the case of IV-fuzzy sets, the trapezoids constructed, both the lower and upper memberships 
present 1 values in the same points. For the case of General Type 2 Fuzzy sets check :ref:`gt2`.
