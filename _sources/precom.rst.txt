.. _precom:

Computing fuzzy partitions
=======================================

One of the most typical ways to compute fuzzy partitions is to use quantiles of the data. The module ``utils`` contains a series of functions
to generate fuzzy partitions for all the supported kinds of fuzzy sets.
The easiest way to compute these partitions is with the ``utils.construct_partitions`` function, specifying the fuzzy set desired::

    import utils

    fz_type_studied = fs.FUZZY_SETS.t2
    precomputed_partitions = utils.construct_partitions(X, fz_type_studied)

--------------------------------
About the precomputed partitions
--------------------------------
Partitions computed using these method use three linguistic variables per fuzzy variable. We chose that number as it creates easily understandable
low, medium and high partitions. For the case of IV-fuzzy sets, the trapezoids constructed, both the lower and upper memberships 
present 1 values in the same points. For the case of General Type 2 Fuzzy sets check :ref:`gt2`.
