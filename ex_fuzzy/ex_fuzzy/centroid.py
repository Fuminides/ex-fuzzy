# -*- coding: utf-8 -*-
"""
This is the source file that contains functions to compute centroids for the case of IV fuzzy sets,
which are commonly used when using the IV-T2 fuzzy sets.
"""
import numpy as np


def center_of_masses(z: np.array, w: np.array) -> np.array:
    '''
    Computes the ponderated centroid with normalized weighting.

    :param z: Vector of the referencial values.
    :param w: Vector of the fuzzy memberships.
    :return: The ponderated sum.
    '''
    return z @ w / np.sum(w)


def compute_centroid_t2_l(z: np.array, memberships: np.array) -> float:
    '''
    Computes the Karnik and Mendel algorithm to find the centroid of an IV fuzzy set.

    :param z: Vector of the referencial values.
    :param memberships: vector of the fuzzy memberships.
    :return: The centroid.
    '''
    centros = np.mean(memberships, axis=1)
    w = centros

    yhat = center_of_masses(z, w)

    yhat_2 = None

    while(yhat != yhat_2):
        try:
            k = np.argwhere((z - yhat) >= 0)[-1][0]
            k = min(len(centros)-1, k)
        except IndexError:
            k = 0

        # k_vector = np.argwhere((z - yhat) > 0)
        # k = k_vector[0][0] + 1
        w[0:k] = memberships[:, 1][:k]
        w[k:] = memberships[:, 0][k:]

        yhat_2 = yhat
        yhat = center_of_masses(z, w)

    return yhat


def compute_centroid_t2_r(z: np.array, memberships: np.array) -> float:
    '''
    Computes the Karnik and Mendel algorithm to find the right component of a centroid of an IV fuzzy set.

    :param z: Vector of the referencial values.
    :param memberships: vector of the fuzzy memberships.
    :return: The lowest membership of the centroid.
    '''
    centros = np.mean(memberships, axis=1)
    w = centros

    yhat = center_of_masses(z, w)

    yhat_2 = None

    while(yhat != yhat_2):
        try:
            k = np.argwhere((z - yhat) >= 0)[-1][0]
            k = min(len(centros)-1, k)
        except IndexError:
            k = 0

        w[0:k+1] = memberships[:, 0][:k+1]
        w[k+1:] = memberships[:, 1][k+1:]

        yhat_2 = yhat
        yhat = center_of_masses(z, w)

    return yhat


def consequent_centroid_r(antecedent_memberships: np.array, centroids_r: np.array) -> float:
    '''
    Computes the Karnik and Mendel algorithm to find the right component of a centroid of an IV fuzzy set
    using the centroids of the consequents and the antecedents memeberships.

    :param antecedent_memberships: M x 2 matrix. M rules and iv dimension (2). Vector of memberships.
    :param centroids_r: M sized vector. Contains the referencial values.
    :return: the IV centroid.
    '''

    memberships_left = antecedent_memberships[:, 0]
    memberships_right = antecedent_memberships[:, 1]
    initial_f = (memberships_left + memberships_right) / 2

    yhat = center_of_masses(centroids_r, initial_f)
    yhat2 = None

    while(yhat != yhat2):
        try:
            r_R = np.argwhere((yhat - centroids_r) >= 0)[-1][0]
            r_R = min(len(centroids_r)-1, r_R)
        except IndexError:
            r_R = 0

        new_memberships = initial_f
        new_memberships[0:r_R+1] = memberships_left[0:r_R+1]
        new_memberships[r_R+1:] = memberships_right[r_R+1:]

        yhat2 = yhat
        if np.sum(new_memberships) == 0:
            yhat = 0
        else:
            yhat = center_of_masses(centroids_r, new_memberships)

    return yhat2


def consequent_centroid_l(antecedent_memberships: np.array, centroids_l: np.array) -> float:
    '''
    Computes the Karnik and Mendel algorithm to find the left component of a centroid of an IV fuzzy set
    using the centroids of the consequents and the antecedent memberships.

    :param antecedent_memberships: M x 2 matrix. M rules and iv dimension (2). Vector of memberships.
    :param centroids_r: M sized vector. Contains the referencial values.
    :return: the IV centroid.
    '''

    memberships_left = antecedent_memberships[:, 0]
    memberships_right = antecedent_memberships[:, 1]
    initial_f = (memberships_left + memberships_right) / 2

    # (memberships_right * centroids_r) / np.sum(memberships_right)
    yhat = center_of_masses(centroids_l, initial_f)
    yhat2 = -100

    #R = np.where(yhat - centroids_r)[0]
    while(yhat != yhat2):
        try:
            r_R = np.argwhere((yhat - centroids_l) >= 0)[-1][0]
            r_R = min(len(centroids_l)-1, r_R)
        except IndexError:
            r_R = 0

        new_memberships = initial_f
        new_memberships[0:r_R+1] = memberships_right[0:r_R+1]
        new_memberships[r_R+1:] = memberships_left[r_R+1:]
        yhat2 = yhat
        if np.sum(new_memberships) == 0:
            yhat = 0
        else:
            yhat = center_of_masses(centroids_l, new_memberships)

    return yhat2


def compute_centroid_iv(z: np.array, memberships: np.array) -> np.array:
    '''
    Computes Karnik and Mendel algorithm to find the centroid of a iv fuzzy set. 

    :param z: Vector of the referencial values.
    :param memberships: vector of the fuzzy memberships.
    :return: The centroid.
    '''
    res = np.zeros((2,))
    res[0] = compute_centroid_t2_l(z, memberships)
    res[1] = compute_centroid_t2_r(z, memberships)
    return res


def consequent_centroid(antecedent_memberships: np.array, centroids: np.array) -> np.array:
    '''
    Computes Karnik and Mendel algorithm to find the centroid of a consequent iv fuzzy set given the centroids of the consequents
    and the antecedents memberships. 

    :param antecedent_memberships: M x 2 matrix. M rules and iv dimension (2). Vector of memberships.
    :param centroids: M x 2 matrix. M rules and iv dimension (2). Vector of centroids.
    :return: The centroid.
    '''
    centroids_l = centroids[:, 0]
    centroids_r = centroids[:, 1]
    res = np.zeros((2,))

    res[0] = consequent_centroid_l(antecedent_memberships, centroids_l)
    res[1] = consequent_centroid_r(antecedent_memberships, centroids_r)

    return res



