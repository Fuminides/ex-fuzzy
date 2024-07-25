.. _tempfs:

Temporal Fuzzy Sets
=======================================

Temporal Fuzzy Sets (TFS) are a generalization of fuzzy sets to include a temporal variable that influences the membership values. 
A comprehensive explanation of such fuzzy sets can be found in [Kiah].

Temporal fuzzy sets thus require the additional temporal variable, which can be spceified in the dedicated functions that work with this kind of fuzzy sets.
The way in which is the temporal variable is used is by first discretizing the the temporal variable from a continuous into a discrete time space. For example,
our time variable is the seconds of the day, we can do the following to define the different stages of the day::


    cut_point_morning0 = '00:00:00'
    cut_point_morning1 = '10:00:00'
    cut_points_morning = [cut_point_morning0, cut_point_morning1]
    cut_point_daytime0 = '11:00:00'
    cut_point_daytime1 = '19:00:00'
    cut_points_daytime = [cut_point_daytime0, cut_point_daytime1]
    cut_point_evening0 = '20:00:00'
    cut_point_evening1 = '23:00:00'
    cutpoints_evening = [cut_point_evening0, cut_point_evening1]

Once we have defined this cut points, there are various functions in the ``ex_fuzzy.utils`` module to assign each of the observatio to one of the temporal moments::

    temporal_boolean_markers = utils.temporal_cuts(X_total, cutpoints=[cut_points_morning, cut_points_daytime, cutpoints_evening], time_resolution='hour')
    time_moments = np.array([utils.assign_time(a, temporal_boolean_markers) for a in range(X_total.shape[0])])

We can also partition the dataset equally in order to have balanced partitions in each of the temporal moments::

    partitions, partition_markers = utils.temporal_assemble(X_total, y_total, temporal_moments=temporal_boolean_markers)
    X_train, X_test, y_train, y_test = partitions
    train_markers, test_markers = partition_markers


Given the time moments and the original fuzzy partitions, we can convert them into temporal fuzzy partitions::

    temp_partitions = utils.create_tempVariables(X_total_array, time_moments, precomputed_partitions) 

The temporal fuzzy partitions are then used to train the temporal fuzzy classifier::

    X_train = X_train.drop(columns=['date'])
    X_test = X_test.drop(columns=['date'])
    fl_classifier = temporal.TemporalFuzzyRulesClassifier(nRules=nRules, nAnts=nAnts, 
        linguistic_variables=temp_partitions, n_linguist_variables=3, 
        fuzzy_type=fz_type_studied, verbose=True, tolerance=0.001, n_classes=2)
    fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=pop_size, time_moments=train_time_moments)

The temporal fuzzy classifier can be evaluated using the ``eval_temporal_fuzzy_model`` function in the ``ex_fuzzy.eval_tools`` module::

    eval_tools.eval_temporal_fuzzy_model(fl_classifier, X_train, y_train, X_test, y_test, 
                                time_moments=train_time_moments, test_time_moments=test_time_moments,
                                plot_rules=False, print_rules=True, plot_partitions=False)


.. _references::
    [Kiah] Kiani, M., Andreu-Perez, J., & Hagras, H. (2022). A Temporal Type-2 Fuzzy System for Time-dependent Explainable Artificial Intelligence. IEEE Transactions on Artificial Intelligence.