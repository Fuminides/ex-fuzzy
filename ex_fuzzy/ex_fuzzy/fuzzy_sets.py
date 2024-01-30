"""
This is a the source file that contains the class of GT2 fuzzy set and its most direct applications, 
like computing the FM function, etc.

"""
import enum

import numpy as np

try:
    from . import maintenance as mnt
except:
    import maintenance as mnt


''' Enum that defines the fuzzy set types.'''
class FUZZY_SETS(enum.Enum):
    t1 = 'Type 1'
    t2 = 'Type 2'
    gt2 = 'General Type 2'


    def __eq__(self, __value: object) -> bool:
        return self.value == __value.value

def trapezoidal_membership(x: np.array, params: list[float], epsilon=10E-5) -> np.array:
    '''
    Trapezoidal membership functions.

    :param x: input values in the fuzzy set referencial domain.
    :param params:  four numbers that comprises the start and end of the trapezoid.
    :param epsilon: small float number for numerical stability. Adjust accordingly only if there are NaN issues.
    '''
    a, b, c, d = params

    # Special case: a singleton trapezoid
    if a == d:
        return np.equal(a, x).astype(float)

    if b == a:
        b += epsilon
    if c == d:
        d += epsilon

    aux1 = (x - a) / (b - a)
    aux2 = (d - x) / (d - c)

    return np.maximum(np.minimum(np.minimum(aux1, aux2), 1), 0)


def __gaussian2(x, params: list[float]) -> np.array:
    '''
    Gaussian membership functions.

    :param mean:  real number, mean of the gaussian function.
    :param amplitude: real number.
    :param standard_deviation: std of the gaussian function.
    '''
    mean, amplitude, standard_deviation = params
    return amplitude * np.exp(- ((x - mean) / standard_deviation) ** 2)


class FS():
    '''
    Class that defines the most basic fuzzy sets (also known as Type 1 fuzzy sets or Zadeh sets).
    '''

    def __init__(self, name: str, membership_parameters: list[float], domain: list[float]) -> None:
        '''
        Creates a fuzzy set.

        :param name: string.
        :param secondMF_lower: four real numbers. Parameters of the lower trapezoid/gaussian function.
        :param secondMF_upper: four real numbers. Parameters of the upper trapezoid/gaussian function.
        :param domain: list of two numbers. Limits of the domain of the fuzzy set.
        '''
        if mnt.save_usage_flag:
            mnt.usage_data[mnt.usage_categories.FuzzySets][self.type().name] += 1
        self.name = name
        self.domain = domain
        self.membership_parameters = membership_parameters


    def membership(self, x: np.array) -> np.array:
        '''
        Computes the membership of a point or a vector.

        :param x: input values in the fuzzy set referencial domain.
        '''
        return trapezoidal_membership(x, self.membership_parameters)


    def type(self) -> FUZZY_SETS:
        '''
        Returns the corresponding fuzzy set type according to FUZZY_SETS enum.

        :return: FUZZY_SETS enum.
        '''
        return FUZZY_SETS.t1


    def __call__(self,  x: np.array) -> np.array:
        '''
        Calling the Fuzzy set returns its membership.

        :param x: input values in the fuzzy set referencial domain.
        :return: membership of the fuzzy set.
        '''
        return self.membership(x)


class IVFS(FS):
    '''
    Class to define a iv fuzzy set.
    '''

    def __init__(self, name: str, secondMF_lower: list[float], secondMF_upper: list[float], 
                domain: list[float], lower_height=1.0) -> None:
        '''
        Creates a IV fuzzy set.

        :param name: string.
        :param secondMF_lower: four real numbers. Parameters of the lower trapezoid/gaussian function.
        :param secondMF_upper: four real numbers. Parameters of the upper trapezoid/gaussian function.
        :param domain: list of two numbers. Limits of the domain if the fuzzy set.
        '''
        self.name = name
        self.domain = domain

        if mnt.save_usage_flag:
            mnt.usage_data[mnt.usage_categories.FuzzySets][self.type().name] += 1

        assert secondMF_lower[0] >= secondMF_upper[0], 'First term membership incoherent'
        assert secondMF_lower[0] <= secondMF_lower[1] and secondMF_lower[
            1] <= secondMF_lower[2] and secondMF_lower[2] <= secondMF_lower[3], 'Lower memberships incoherent. '
        assert secondMF_upper[0] <= secondMF_upper[1] and secondMF_upper[
            1] <= secondMF_upper[2] and secondMF_upper[2] <= secondMF_upper[3], 'Upper memberships incoherent.'
        assert secondMF_lower[3] <= secondMF_upper[3], 'Final term memberships incoherent.'

        self.secondMF_lower = secondMF_lower
        self.secondMF_upper = secondMF_upper
        self.lower_height = lower_height


    def membership(self, x: np.array) -> np.array:
        '''
        Computes the iv-membership of a point or a vector.

        :param x: input values in the fuzzy set referencial domain.
        :return: iv-membership of the fuzzy set.
        '''
        lower = trapezoidal_membership(
            x, self.secondMF_lower) * self.lower_height
        upper = trapezoidal_membership(x, self.secondMF_upper)

        try:
            assert np.all(lower <= upper)
        except AssertionError:
            np.argwhere(lower > upper)

        return np.stack([lower, upper], axis=-1)


    def type(self) -> FUZZY_SETS:
        '''
        Returns the corresponding fuzzy set type according to FUZZY_SETS enum: (t2)
        '''
        return FUZZY_SETS.t2


class GT2(FS):
    '''
    Class to define a gt2 fuzzy set.
    '''

    MAX_RES_SUPPORT = 4  # Number of decimals supported in the secondary

    def __init__(self, name: str, secondary_memberships: dict[float, FS], 
                domain: list[float], significant_decimals: int, 
                alpha_cuts: list[float] = [0.2, 0.4, 0.5, 0.7, 0.9, 1.0], 
                unit_resolution: float = 0.2) -> None:
        '''
        Creates a GT2 fuzzy set.

        :param name: string.
        :param secondary_memberships: list of fuzzy sets. Secondary membership that maps original domain to [0, 1]
        :param secondMF_upper: four real numbers. Parameters of the upper trapezoid/gaussian function.
        :param domain: list of two numbers. Limits of the domain if the fuzzy set.
        :param alpha_cuts: list of real numbers. Alpha cuts of the fuzzy set.
        :param unit_resolution: real number. Resolution of the primary membership function.
        '''
        if mnt.save_usage_flag:
            mnt.usage_data[mnt.usage_categories.FuzzySets][self.type().name] += 1

        self.name = name
        self.domain = domain
        self.secondary_memberships = secondary_memberships
        self.alpha_cuts = alpha_cuts
        self.iv_secondary_memberships = {}
        self.unit_resolution = unit_resolution
        og_keys = list(self.secondary_memberships.keys())
        self.significant_decimals = significant_decimals
        self.domain_init = int(
            float(og_keys[0]) * 10**self.significant_decimals)

        formatted_keys = [('%.' + str(self.significant_decimals) + 'f') %
                          xz for xz in np.array(list(self.secondary_memberships.keys()))]
        for og_key, formatted_key in zip(og_keys, formatted_keys):
            secondary_memberships[formatted_key] = self.secondary_memberships.pop(
                og_key)

        self.sample_unit_domain = np.arange(
            0, 1 + self.unit_resolution, self.unit_resolution)

        for ix, alpha_cut in enumerate(alpha_cuts):
            level_memberships = {}
            array_level_memberships = np.zeros(
                (len(secondary_memberships.items()), 2))

            for jx, (x, fs) in enumerate(secondary_memberships.items()):
                alpha_primary_memberships = fs.membership(
                    self.sample_unit_domain)
                alpha_membership = alpha_primary_memberships >= alpha_cut

                try:
                    b = self.sample_unit_domain[np.argwhere(
                        alpha_membership)[0]][0]
                    c = self.sample_unit_domain[np.argwhere(
                        alpha_membership)[-1]][0]
                except IndexError:  # If no numbers are bigger than alpha, then it is because of rounding errors near 0. So, we fix this manually.
                    b = 0
                    c = 0

                alpha_cut_interval = [b, c]

                level_memberships[x] = alpha_cut_interval
                array_level_memberships[jx, :] = np.array(alpha_cut_interval)

            self.iv_secondary_memberships[alpha_cut] = array_level_memberships


    def membership(self, x: np.array) -> np.array:
        '''
        Computes the alpha cut memberships of a point.

        :param x: input values in the fuzzy set referencial domain.
        :return: np array samples x alpha_cuts x 2
        '''
        x = np.array(x)
        # locate the x in the secondary membership
        formatted_x = (
            x * 10**self.significant_decimals).astype(int) - self.domain_init

        # Once the x is located we compute the function for each alpha cut
        alpha_cut_memberships = []
        for ix, alpha in enumerate(self.alpha_cuts):
            ivfs = np.squeeze(
                np.array(self.iv_secondary_memberships[alpha][formatted_x]))
            alpha_cut_memberships.append(np.squeeze(ivfs))

        alpha_cut_memberships = np.array(alpha_cut_memberships)

        # So that the result is samples x alpha_cuts x 2
        return np.swapaxes(alpha_cut_memberships, 0, 1)


    def type(self) -> FUZZY_SETS:
        return FUZZY_SETS.gt2
    

    def _alpha_reduction(self, x) -> np.array:
        '''
        Computes the type reduction to reduce the alpha cuts to one value.

        :param x: array with the values of the inputs.
        :return: array with the memberships of the consequents for each sample.
        '''
        if len(x.shape) == 3:
            formatted = np.expand_dims(np.expand_dims(
                np.array(self.alpha_cuts), axis=1), axis=0)
        else:
            formatted = self.alpha_cuts
        return np.sum(formatted * x, axis=-1) / np.sum(self.alpha_cuts)
    

    def alpha_reduction(self, x) -> np.array:
        '''
        Computes the type reduction to reduce the alpha cuts to one value.

        :param x: array with the values of the inputs.
        :return: array with the memberships of the consequents for each sample.
        '''
        return self._alpha_reduction(x)
        

class gaussianIVFS(IVFS):
    '''
    Class to define a iv fuzzy set with gaussian membership.
    '''

    def membership(self, input: np.array) -> np.array:
        '''
        Computes the gaussian iv-membership of a point or a vector.

        :param input: input values in the fuzzy set referencial domain.
        :return: np array samples x 2
        '''
        lower = __gaussian2(input, self.secondMF_lower)
        upper = __gaussian2(input, self.secondMF_upper)

        return np.array(np.concatenate([lower, upper])).T


    def type(self) -> FUZZY_SETS:
        '''
        Returns the type of the fuzzy set. (t1)
        '''
        return FUZZY_SETS.t1


class fuzzyVariable():
    '''
    Class to implement a fuzzy Variable. Contains a series of fuzzy sets and computes the memberships to all of them.
    '''

    def __init__(self, name: str, fuzzy_sets: list[FS]) -> None:
        '''
        Creates a fuzzy variable.

        :param name: string. Name of the fuzzy variable.
        :param fuzzy_sets: list of IVFS. Each of the fuzzy sets that comprises the linguist variables of the fuzzy variable.
        '''
        self.linguistic_variables = []
        self.name = name

        for ix, fs in enumerate(fuzzy_sets):
            self.linguistic_variables.append(fs)

        self.fs_type = self.linguistic_variables[0].type()


    def linguistic_variable_names(self) -> list:
        '''
        Returns the name of the linguistic variables.

        :return: list of strings.
        '''
        return [fs.name for fs in self.linguistic_variables]


    def get_linguistic_variables(self) -> list[FS]:
        '''
        Returns the name of the linguistic variables.

        :return: list of strings.
        '''
        return self.linguistic_variables


    def compute_memberships(self, x: np.array) -> list:
        '''
        Computes the membership to each of the FS in the fuzzy variables.

        :param x: numeric value or array. Computes the membership to each of the FS in the fuzzy variables.
        :return: list of floats. Membership to each of the FS in the fuzzy variables.
        '''
        res = []

        for fuzzy_set in self.linguistic_variables:
            res.append(fuzzy_set.membership(x))

        return res


    def domain(self) -> list[float]:
        '''
        Returns the domain of the fuzzy variable.

        :return: list of floats.
        '''
        return self.linguistic_variables[0].domain


    def fuzzy_type(self) -> FUZZY_SETS:
        '''
        Returns the fuzzy type of the domain

        :return: the type of the fuzzy set present in the fuzzy variable.
        '''
        return self.fs_type


    def __getitem__(self, item) -> FS:
        '''
        Returns the corresponding fs.

        :param item: int. Index of the FS.
        :return: FS. The corresponding FS.
        '''
        return self.linguistic_variables[item]


    def __setitem__(self, item: int, elem: FS) -> None:
        '''
        Sets the corresponding fs.

        :param item: int. Index of the FS.
        :param elem: FS. The FS to set.
        '''
        self.linguistic_variables[item] = elem
    

    def __iter__(self) -> FS:
        '''
        Returns the corresponding fs.

        :param item: int. Index of the FS.
        :return: FS. The corresponding FS.
        '''
        for fs in self.linguistic_variables:
            yield fs


    def __len__(self) -> int:
        '''
        Returns the number of linguistic variables.

        :return: int. Number of linguistic variables.
        '''
        return len(self.linguistic_variables)
    

    def __call__(self, x: np.array) -> list:
        '''
        Computes the membership to each of the FS in the fuzzy variables.

        :param x: numeric value or array.
        :return: list of floats. Membership to each of the FS in the fuzzy variables.
        '''
        return self.compute_memberships(x)


