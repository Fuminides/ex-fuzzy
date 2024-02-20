# -*- coding: utf-8 -*-
"""
Functions that are not fuzzy-specific, but util for some computations. 
Dedicated mostly to compute quantiles for fuzzy partitions.

"""
import numpy as np
import pandas as pd

try:
    from . import fuzzy_sets as fs
    from . import maintenance as mnt
    from . import temporal
    from . import rules
    from . import eval_rules as evr
except ImportError:
    import fuzzy_sets as fs
    import maintenance as mnt
    import temporal
    import rules
    import eval_rules as evr


from sklearn.model_selection import train_test_split


def quartile_compute(x: np.array) -> list[float]:
    '''
    Computes the quartiles for each feature.

    :param x: array samples x features
    :return: list of quartiles for each feature
    '''
    return np.quantile(x, [0, 0.25, 0.5, 1], axis=0)


def fixed_quantile_compute(x: np.array) -> list[float]:
    '''
    Computes a series of quantiles for each feature in numpy array.
    Quantiles: [0, 0.20, 0.30, 0.45, 0.55, 0.7, 0.8, 1]

    :param x: array samples x features
    :return: list of quantiles for each feature
    '''
    return np.quantile(x, [0, 0.20, 0.30, 0.45, 0.55, 0.7, 0.8, 1], axis=0)


def partition3_quantile_compute(x: np.array) -> list[float]:
    '''
    Computes a series of quantiles partitioning the variable in 3 cases.

    Quantiles: [0.00, 0.20, 0.50, 0.80, 1.00]

    :param x: array samples x features
    :return: list of quantiles for each feature
    '''
    return np.quantile(x, [0, 0.20, 0.50, 0.80, 1.00], axis=0)


def t1_simple_partition(x: np.array) -> np.array:
    '''
    Partitions the fuzzy variable in four trapezoidal memberships.

    :param x: numpy array, vector of shape (samples, ).
    :return: numpy array, vector of shape (variables, 4, 4).
    '''
    

    n_partitions = 4
    trap_memberships_size = 4
    quantile_numbers = fixed_quantile_compute(x)

    partition_parameters = np.zeros(
        (x.shape[1], n_partitions, trap_memberships_size))
    for partition in range(n_partitions):
        if partition == 0:
            partition_parameters[:, partition, 0] = quantile_numbers[0]
            partition_parameters[:, partition, 1] = quantile_numbers[0]
            partition_parameters[:, partition, 2] = quantile_numbers[1]
            partition_parameters[:, partition, 3] = quantile_numbers[2]
        elif partition == n_partitions - 1:
            partition_parameters[:, partition, 0] = quantile_numbers[-3]
            partition_parameters[:, partition, 1] = quantile_numbers[-2]
            partition_parameters[:, partition, 2] = quantile_numbers[-1]
            partition_parameters[:, partition, 3] = quantile_numbers[-1]
        else:
            pointer = 1 if partition == 1 else 4
            partition_parameters[:, partition, 0] = quantile_numbers[pointer]
            partition_parameters[:, partition,
                                 1] = quantile_numbers[pointer + 1]
            partition_parameters[:, partition,
                                 2] = quantile_numbers[pointer + 2]
            partition_parameters[:, partition,
                                 3] = quantile_numbers[pointer + 3]

    return partition_parameters


def t1_three_partition(x: np.array) -> np.array:
    '''
    Partitions the fuzzy variable in three trapezoidal memberships.

    :param x: numpy array, vector of shape (samples, ).
    :return: numpy array, vector of shape (variables, 3, 4).
    '''
    n_partitions = 3
    trap_memberships_size = 4
    quantile_numbers = partition3_quantile_compute(x)

    partition_parameters = np.zeros(
        (x.shape[1], n_partitions, trap_memberships_size))
    for partition in range(n_partitions):
        if partition == 0:
            partition_parameters[:, partition, 0] = quantile_numbers[0]
            partition_parameters[:, partition, 1] = quantile_numbers[0]
            partition_parameters[:, partition, 2] = quantile_numbers[1]
            partition_parameters[:, partition, 3] = quantile_numbers[2]
        elif partition == 1:
            partition_parameters[:, partition, 0] = quantile_numbers[1]
            partition_parameters[:, partition, 1] = (
                quantile_numbers[1] + quantile_numbers[2]) / 2
            partition_parameters[:, partition, 2] = (
                quantile_numbers[3] + quantile_numbers[2]) / 2
            partition_parameters[:, partition, 3] = quantile_numbers[3]
        else:
            partition_parameters[:, partition, 0] = quantile_numbers[2]
            partition_parameters[:, partition, 1] = quantile_numbers[3]
            partition_parameters[:, partition, 2] = quantile_numbers[4]
            partition_parameters[:, partition, 3] = quantile_numbers[4]

    return partition_parameters


def t2_simple_partition(x: np.array) -> np.array:
    '''
    Partitions the fuzzy variable in three trapezoidal memberships.

    :param x: numpy array, vector of shape (samples, ).
    :return: numpy array, vector of shape (variables, 3, 4, 2).
    '''
    n_partitions = 3
    trap_memberships_size = 4
    quantile_numbers = partition3_quantile_compute(x)

    partition_parameters = np.zeros(
        (x.shape[1], n_partitions, trap_memberships_size, 2))
    for partition in range(n_partitions):
        if partition == 0:
            partition_parameters[:, partition, 0, 1] = quantile_numbers[0]
            partition_parameters[:, partition, 1, 1] = quantile_numbers[0]
            partition_parameters[:, partition, 2, 1] = quantile_numbers[1]
            partition_parameters[:, partition, 3, 1] = quantile_numbers[2]

            partition_parameters[:, partition, 0, 0] = quantile_numbers[0]
            partition_parameters[:, partition, 1, 0] = quantile_numbers[0]
            partition_parameters[:, partition, 2, 0] = quantile_numbers[1]
            partition_parameters[:, partition, 3, 0] = quantile_numbers[1] + \
                0.9 * (quantile_numbers[2] - quantile_numbers[1])

        elif partition == 1:
            partition_parameters[:, partition, 0, 1] = quantile_numbers[1]
            partition_parameters[:, partition, 1, 1] = (
                quantile_numbers[1] + quantile_numbers[2]) / 2
            partition_parameters[:, partition, 2, 1] = (
                quantile_numbers[2] + quantile_numbers[3]) / 2
            partition_parameters[:, partition, 3, 1] = quantile_numbers[3]

            partition_parameters[:, partition, 0, 0] = quantile_numbers[1] + \
                0.1 * (quantile_numbers[2] - quantile_numbers[1])
            partition_parameters[:, partition, 1, 0] = (
                quantile_numbers[1] + quantile_numbers[2]) / 2
            partition_parameters[:, partition, 2, 0] = (
                quantile_numbers[3] + quantile_numbers[2]) / 2
            partition_parameters[:, partition, 3, 0] = quantile_numbers[2] + \
                0.9 * (quantile_numbers[3] - quantile_numbers[2])

        else:
            partition_parameters[:, partition, 0, 1] = quantile_numbers[2]
            partition_parameters[:, partition, 1, 1] = quantile_numbers[3]
            partition_parameters[:, partition, 2, 1] = quantile_numbers[4]
            partition_parameters[:, partition, 3, 1] = quantile_numbers[4]

            partition_parameters[:, partition, 0, 0] = quantile_numbers[2] + \
                0.1 * (quantile_numbers[3] - quantile_numbers[2])
            partition_parameters[:, partition, 1, 0] = quantile_numbers[3]
            partition_parameters[:, partition, 2, 0] = quantile_numbers[4]
            partition_parameters[:, partition, 3, 0] = quantile_numbers[4]

    return partition_parameters


def t1_fuzzy_partitions_dataset(x0: np.array) -> list[fs.fuzzyVariable]:
    '''
    Partitions the dataset features into different fuzzy variables. Parameters are prefixed.
    Use it for simple testing and initial solution.

    :param x: numpy array|pandas dataframe, shape samples x features.
    :return: list of fuzzy variables.
    '''
    tripartition_names = ['Low', 'Medium', 'High']
    try:
        fv_names = x0.columns
        x = x0.values
    except AttributeError:
        fv_names = [str(ix) for ix in range(x0.shape[1])]
        x = x0

    mins = np.min(x, axis=0)
    maxs = np.max(x, axis=0)
    fz_memberships = t1_three_partition(x)
    res = []
    for fz_parameter in range(fz_memberships.shape[0]):
        fzs = [fs.FS(tripartition_names[ix], fz_memberships[fz_parameter, ix, :], [
                     mins[fz_parameter], maxs[fz_parameter]]) for ix in range(fz_memberships.shape[1])]
        res.append(fs.fuzzyVariable(fv_names[fz_parameter], fzs))

    return res


def t2_fuzzy_partitions_dataset(x0: np.array) -> list[fs.fuzzyVariable]:
    '''
    Partitions the dataset features into different fuzzy variables using iv fuzzy sets. Parameters are prefixed.
    Use it for simple testing and initial solution.

    :param x: numpy array|pandas dataframe, shape samples x features.
    :return: list of fuzzy variables.
    '''
    try:
        fv_names = x0.columns
        x = x0.values
    except AttributeError:
        fv_names = [str(ix) for ix in range(x0.shape[1])]
        x = x0

    mins = np.min(x, axis=0)
    maxs = np.max(x, axis=0)
    fz_memberships = t2_simple_partition(x)
    res = []
    for fz_parameter in range(fz_memberships.shape[0]):
        fzs = [fs.IVFS(str(ix), fz_memberships[fz_parameter, ix, :, 0], fz_memberships[fz_parameter, ix, :, 1], [
                       mins[fz_parameter], maxs[fz_parameter]], lower_height=0.8) for ix in range(fz_memberships.shape[1])]
        res.append(fs.fuzzyVariable(fv_names[fz_parameter], fzs))

    return res


def gt2_fuzzy_partitions_dataset(x0: np.array, resolution_exp:int=1) -> list[fs.fuzzyVariable]:
    '''
    Partitions the dataset features into different fuzzy variables using gt2 fuzzy sets. Parameters are prefixed.
    Use it for simple testing and initial solution.

    :param x: numpy array|pandas dataframe, shape samples x features.
    :param resolution_exp: exponent of the resolution of the partition. Default is -2, which means 0.01. (Number of significant decimals)
    :return: list of fuzzy variables.
    '''
    try:
        fv_names = x0.columns
        x = x0.values
    except AttributeError:
        fv_names = [str(ix) for ix in range(x0.shape[1])]
        x = x0

    mins = np.min(x, axis=0)
    maxs = np.max(x, axis=0)
    iv_simple_partition = t2_fuzzy_partitions_dataset(x)
    resolution = 10.0**-np.abs(resolution_exp)
    res = []
    # We iterate through all possible variables
    for ix_var, fz_var in enumerate(iv_simple_partition):
        domain_resolution = np.arange(
            mins[ix_var], maxs[ix_var] + resolution, resolution)
        fzs = []
        for ix_lv, fz_lv in enumerate(fz_var.get_linguistic_variables()):
            memberships = fz_lv.membership(domain_resolution)
            fs_domain = {}
            for ix_z, x in enumerate(domain_resolution):
                membership_z = memberships[ix_z]
                fs_domain[x] = fs.FS(str(x), [membership_z[0], np.mean(
                   membership_z), np.mean(membership_z), membership_z[1]], [0.0, 1.0])

            fzs.append(fs.GT2(fz_lv.name, fs_domain, [
                       mins[ix_var], maxs[ix_var]], significant_decimals=np.abs(resolution_exp), unit_resolution=0.01))

        res.append(fs.fuzzyVariable(fv_names[ix_var], fzs))

    return res


def construct_partitions(X : np.array, fz_type_studied:fs.FUZZY_SETS, categorical_mask: np.array=None) -> list[fs.fuzzyVariable]:
    '''
    Returns a list of linguistic variables according to the kind of fuzzy specified.

    :param X: numpy array|pandas dataframe, shape samples x features.
    :param fz_type_studied: fuzzy set type studied.
    :param categorial_mask: a boolean mask vector that indicates for each variables if its categorical or not.
    '''
    if mnt.save_usage_flag:
        mnt.usage_data[mnt.usage_categories.Funcs]['precompute_labels'] += 1
        mnt.usage_data[mnt.usage_categories.FuzzySets][fz_type_studied.name] += 1

    if fz_type_studied == fs.FUZZY_SETS.t1:
        precomputed_partitions = t1_fuzzy_partitions_dataset(X)
    elif fz_type_studied == fs.FUZZY_SETS.t2:
        precomputed_partitions = t2_fuzzy_partitions_dataset(X)
    elif fz_type_studied == fs.FUZZY_SETS.gt2:
        precomputed_partitions = gt2_fuzzy_partitions_dataset(X)

    if categorical_mask is not None:
        for ix, elem in enumerate(categorical_mask):
            if elem:
                if isinstance(X, pd.DataFrame):
                    name = X.columns[ix]
                else:
                    name = str(ix)
                cat_var = construct_crisp_categorical_partition(np.array(X)[:, ix], name, fz_type_studied)

                precomputed_partitions[ix] = cat_var

    return precomputed_partitions


def construct_crisp_categorical_partition(x: np.array, name: str, fz_type_studied: fs.FUZZY_SETS):
    '''
    Creates a fuzzy variable for a categorical feature. 

    :param x: array with values of the categorical variable.
    :param name of the fuzzy variable.
    :return: a fuzzy variable that works as a categorical crips variable (each fuzzy set is 1 exactly on each class value, and 0 on the rest)
    '''
    possible_values = np.unique(x)
    possible_fuzzy_values = np.arange(len(possible_values))

    epsilon = 1e-5
    fuzzy_sets = []

    # Create a fuzzy sets for each possible value
    for ix, value in enumerate(possible_values):
        mem_function = [possible_fuzzy_values[ix] - epsilon, possible_fuzzy_values[ix], possible_fuzzy_values[ix], possible_fuzzy_values[ix] + epsilon]

        if fz_type_studied == fs.FUZZY_SETS.t1:
            aux = fs.FS(str(value), mem_function, [0, len(possible_fuzzy_values)])
        elif fz_type_studied == fs.FUZZY_SETS.t2 or fz_type_studied == fs.FUZZY_SETS.gt2:
            aux = fs.IVFS(str(value), mem_function, mem_function, [0, len(possible_fuzzy_values)])

        fuzzy_sets.append(aux)

    return fs.fuzzyVariable(name, fuzzy_sets)
    

def construct_conditional_frequencies(X: np.array, discrete_time_labels: list[int], initial_ffss: list[fs.FS]):
    '''
    Computes the conditional temporal function for a set of fuzzy sets according to their variation in time.

    :param X: numpy array, shape samples x features.
    :param discrete_time_labels: discrete time labels.
    :param initial_fs: initial fuzzy set list.
    :return: conditional frequencies. Array shape (time steps, initial fuzzy sets)
    '''
    obs = X.shape[0]
    discrete_time_labels = np.array(discrete_time_labels)
    memberships = np.zeros((obs, len(initial_ffss)))
    for ix, fset in enumerate(initial_ffss):
        if fset.type() == fs.FUZZY_SETS.t2:
            memberships[:, ix] = np.mean(fset.membership(X), axis=1)
        elif fset.type() == fs.FUZZY_SETS.gt2:
            memberships[:, ix] = np.mean(np.squeeze(fset._alpha_reduction(fset.membership(X))), axis=1)
        else:
            memberships[:, ix] = fset.membership(X)
    
    max_memberships = np.argmax(memberships, axis=1)
    res = np.zeros((len(np.unique(discrete_time_labels)), len(initial_ffss)))

    for time in range(len(np.unique(discrete_time_labels))):

        relevant_memberships = max_memberships[time == discrete_time_labels]
        fs_winner_counter = np.zeros(len(initial_ffss))
        for ix, fset in enumerate(initial_ffss):
            fs_winner_counter[ix] = np.sum(relevant_memberships == ix)
        
        res[time, :] = fs_winner_counter
    
    return res / (np.max(res, axis=0) + 1e-6)


def classify_temp(dates: pd.DataFrame, cutpoints: tuple[str, str], time: str) -> np.array:
    '''
    Classifies a set of dates according to the temporal cutpoints. Uses {time} as a the time resolution.
    Returns an array where true values are those values contained between those two date points.

    :param dates: data observations to cut. 
    :param cutpoints: points to check.
    :param time: time field to use as the criteria.
    :return: boolean array. True values are those contained between the cutpoints.
    '''

    def extract_hour(row):
        return row.__getattribute__(time)
    
    hours = pd.to_datetime(dates['date']).apply(extract_hour)

    cutpoint_series_0 =  pd.to_datetime(pd.Series([cutpoints[0]] * len(dates)))
    cutpoint_series_0.index = dates.index
    hours0 = cutpoint_series_0.apply(extract_hour)

    cutpoint_series_1 =  pd.to_datetime(pd.Series([cutpoints[1]] * len(dates)))
    cutpoint_series_1.index = dates.index
    hours1 = cutpoint_series_1.apply(extract_hour)

    condicion1 = hours >= hours0
    condicion2 = hours <= hours1

    return np.array(np.logical_and(condicion1, condicion2))


def assign_time(a: np.array, observations: list[np.array]) -> int:
    '''
    Assigns a temporal moment to a set of observations.
    
    :param a: array of boolean values.
    :param observations: list of boolean arrays with the corresponding timestamps.
    :return: the index of the correspondent time moment for the a-th observation.
    :raises: ValueError if a is not timestamped in any of the observation arrays.'''
    for ix, obs in enumerate(observations):
        if obs[a]:
            return ix
    
    raise ValueError('No temporal moment assigned')
        

def create_tempVariables(X_train: np.array, time_moments: np.array, precomputed_partitions: list[fs.fuzzyVariable]) -> list[temporal.temporalFS]:
    '''
    Creates a list of temporal fuzzy variables.

    :param X_train: numpy array, shape samples x features.
    :param time_moments: time moments. Array shape (samples,). Each value is an integer denoting the n-th time moment of that observation.
    :param precomputed_partitions: precomputed partitions for each feature.
    :return: list of temporal fuzzy variables.
    '''
    temp_partitions = []
    for ix in range(X_train.shape[1]):
        feat_conditional = construct_conditional_frequencies(X_train[:, ix], time_moments, initial_ffss=precomputed_partitions[ix])
        temp_fs_list = []
        for vl in range(feat_conditional.shape[1]):
            vl_temp_fs = temporal.temporalFS(precomputed_partitions[ix][vl], feat_conditional[:, vl])
            temp_fs_list.append(vl_temp_fs)

        temp_fs_variable = temporal.temporalFuzzyVariable(precomputed_partitions[ix].name, temp_fs_list)
        temp_partitions.append(temp_fs_variable)

    return temp_partitions


def create_multi_tempVariables(X_train: np.array, time_moments: np.array, fuzzy_type: fs.FUZZY_SETS) -> list[list[temporal.temporalFS]]:
    '''
    Creates a of list of lists of temporal fuzzy variables. Each corresponds to a fuzzy partition in a different moment in time.
    (So, instead of having one vl for all time moments, you have one different for each time moment that represents the same idea)

    :param X_train: numpy array, shape samples x features.
    :param time_moments: time moments. Array shape (samples,). Each value is an integer denoting the n-th time moment of that observation.
    :param precomputed_partitions: precomputed partitions for each feature.
    :return: list of lists of temporal fuzzy variables.
    '''
    temp_partitions = []

    unique_time_moments = np.unique(time_moments)
    for time in unique_time_moments:
        X_obs = X_train[time_moments == time, :]
        precomputed_partitions = construct_partitions(X_obs, fuzzy_type)

        temp_partitions.append(create_tempVariables(X_obs, time_moments[time_moments == time], precomputed_partitions))
    
    return temp_partitions


def temporal_cuts(X: pd.DataFrame, cutpoints: list[tuple[str, str]], time_resolution: str='hour') -> list[np.array]:
    '''
    Returns a list of boolean indexes for each temporal moment. Performs the cuts between time steps using the cutpoints list.

    :param X: data observations to cut in temrporal moments.
    :param temporal_moments: list of temporal moments to cut.
    :param cutpoints: list of tuples with the cutpoints for each temporal moment.
    :param time_resolution: time field to use as the criteria.
    :return: list of boolean arrays. True values are those contained between the cutpoints in each moment.
    '''

    res = []
    for ix, cutpoint in enumerate(cutpoints):
        observations = classify_temp(X, cutpoint, time=time_resolution)
        res.append(observations)
    
    return res


def temporal_assemble(X: np.array, y:np.array, temporal_moments: list[np.array]):
    '''
    Assembles the data in the temporal moments in order to have partitions with balanced time moments in each one.
    
    :param X: data observations.
    :param y: labels.
    :param temporal_moments: list of boolean arrays. True values are those contained between the cutpoints in each moment.
    :return: tuple of lists of data and labels for each temporal moment.
        First tuple is: X_train, X_test, y_train, y_test
        Second tuple is: train temporal moments, test temporal moments.
    '''
    moments_partitions = []
    train_temporal_boolean_markers = []
    test_temporal_boolean_markers = []
    train_counter = 0
    test_counter = 0

    for ix, temporal_moment in enumerate(temporal_moments):
        X_train, X_test, y_train, y_test = train_test_split(X[temporal_moment], y[temporal_moment], test_size=0.33, random_state=0)
        moments_partitions.append((X_train, X_test, y_train, y_test))
    
    if isinstance(X_train,(pd.core.series.Series,pd.DataFrame)):
        X_train = pd.concat([moments_partitions[ix][0] for ix in range(len(moments_partitions))])
        X_test =  pd.concat([moments_partitions[ix][1] for ix in range(len(moments_partitions))])
        y_train = np.concatenate([moments_partitions[ix][2] for ix in range(len(moments_partitions))])
        y_test = np.concatenate([moments_partitions[ix][3] for ix in range(len(moments_partitions))])
    else:
        X_train = np.concatenate([moments_partitions[ix][0] for ix in range(len(moments_partitions))])
        X_test =  np.concatenate([moments_partitions[ix][1] for ix in range(len(moments_partitions))])
        y_train = np.concatenate([moments_partitions[ix][2] for ix in range(len(moments_partitions))])
        y_test = np.concatenate([moments_partitions[ix][3] for ix in range(len(moments_partitions))])

    for ix, temporal_moment in enumerate(temporal_moments):
        # Believe, this makes sense to avoid rounding errrors in the size of the final vector
        _, _, y_train0, y_test0 = train_test_split(X[temporal_moment], y[temporal_moment], test_size=0.33, random_state=0)

        train_moment_observations = np.zeros((X_train.shape[0]))
        train_moment_observations[train_counter:train_counter+len(y_train0)] = 1
        train_counter += len(y_train0)
        train_temporal_boolean_markers.append(train_moment_observations)

        test_moment_observations = np.zeros((X_test.shape[0]))
        test_moment_observations[test_counter:test_counter+len(y_test0)] = 1
        test_counter += len(y_test0)
        test_temporal_boolean_markers.append(test_moment_observations)

    
    return [X_train, X_test, y_train, y_test], [train_temporal_boolean_markers, test_temporal_boolean_markers]


def extend_fuzzy_sets_enum(new_fuzzy_sets_enum: fs.FUZZY_SETS) -> list[fs.FUZZY_SETS]:
    '''
    Extends the fuzzy sets enum with additional types.

    :param fuzzy_sets_enum: fuzzy sets enum.
    :return: extended fuzzy sets enum.
    '''
    import enum
    NEW_FUZZY_SETS = enum.Enum(
        "FUZZY_SETS",
        [(es.name, es.value) for es in fs.FUZZY_SETS] + [(es.name, es.value) for es in new_fuzzy_sets_enum]
        )
    fs.FUZZY_SETS = NEW_FUZZY_SETS


def mcc_loss(ruleBase: rules.RuleBase, X:np.array, y:np.array, tolerance:float, alpha:float=0.99, beta:float=0.0125, gamma:float=0.0125) -> float:

        '''
        Fitness function for the optimization problem. Uses only the MCC, ignores the size penalization terms.


        :param ruleBase: RuleBase object
        :param X: array of train samples. X shape = (n_samples, n_features)
        :param y: array of train labels. y shape = (n_samples,)
        :param tolerance: float. Tolerance for the size evaluation.
        :param alpha: ignored.
        :param beta: ignored.
        :param gamma: ignored.
        :return: float. Fitness value.
        '''
        ev_object = evr.evalRuleBase(ruleBase, X, y)
        ev_object.add_rule_weights()

        score_acc = ev_object.classification_eval()
        
        return score_acc