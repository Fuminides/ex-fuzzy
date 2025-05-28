"""
This is a the source file that contains the class of GT2 fuzzy set and its most direct applications, 
like computing the FM function, etc.

"""
import enum
from typing import Generator

import numpy as np
import pandas as pd

try:
    from . import maintenance as mnt
except:
    import maintenance as mnt

# You dont require torch to use this module, however, we need to import it to give support in case you feed these methods with torch tensors.
try:
    import torch
    torch_available = True
except:
    torch_available = False


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
        # If they are numpy arrays, we need to use the numpy function
        if isinstance(x, np.ndarray):
            return np.equal(x, a).astype(float)
        if torch_available:
            if isinstance(x, torch.Tensor):
                return torch.eq(x, a).float()
            

    if b == a:
        b += epsilon
    if c == d:
        d += epsilon

    aux1 = (x - a) / (b - a)
    aux2 = (d - x) / (d - c)
    try:
        if isinstance(x, torch.Tensor):
            return torch.clamp(torch.min(aux1, aux2), 0.0, 1.0) 
    except NameError:
        pass

    if isinstance(x, np.ndarray):
        return np.clip(np.minimum(aux1, aux2), 0.0, 1.0)        
    elif isinstance(x, list):
        return [np.clip(min(aux1, aux2), 0.0, 1.0) for elem in x]
    elif isinstance(x, pd.Series):
        return np.clip(np.minimum(aux1, aux2), 0.0, 1.0)
    else: # Single value
        return np.clip(min(aux1, aux2), 0.0, 1.0)



def _gaussian2(x, params: list[float]) -> np.array:
    '''
    Gaussian membership functions.

    :param mean:  real number, mean of the gaussian function.
    :param amplitude: real number.
    :param standard_deviation: std of the gaussian function.
    '''
    mean, standard_deviation = params
    return np.exp(- ((x - mean) / standard_deviation) ** 2)


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
    

    def __str__(self) -> str:
        '''
        Returns the name of the fuzzy set, its type and its parameters.
        
        :return: string.
        '''
        return f'{self.name} ({self.type().name}) - {self.membership_parameters}'
    

    def shape(self) -> str:
        '''
        Returns the shape of the fuzzy set.

        :return: string.
        '''
        return 'trapezoid'


class triangularFS(FS):
    def __init__(self, name: str, membership_parameters: list[float], domain: list[float]) -> None:
        super().__init__(name, membership_parameters, domain)

    def shape(self) -> str:
        return 'triangular'


class categoricalFS(FS):

    def __init__(self, name: str, category) -> None:
        '''
        Creates a categorical fuzzy set. It gives 1 to the category and 0 to the rest.
        Use it when the variable is categorical and the categories are known, so that rule inference systems
        can naturally support both crisp and fuzzy variables.

        :param name: string.
        :param categories: list of strings. Possible categories.
        '''
        self.name = name
        self.category = category

        if mnt.save_usage_flag:
            mnt.usage_data[mnt.usage_categories.FuzzySets][self.type().name] += 1


    def membership(self, x: np.array) -> np.array:
        '''
        Computes the membership of a point or vector of elements.

        :param x: input values in the referencial domain.
        '''
        if isinstance(x, np.ndarray):
            res = np.equal(x, self.category).astype(float)
        elif isinstance(x, torch.Tensor):
            res = torch.eq(x, self.category).float()
        elif isinstance(x, list):
            res = [1.0 if elem == self.category else 0.0 for elem in x]
        elif isinstance(x, float) or isinstance(x, int):
            res = 1.0 if x == self.category else 0.0
        elif isinstance(x, pd.Series):
            res = x.apply(lambda elem: 1.0 if elem == self.category else 0.0)
            
        return res


    def type(self) -> FUZZY_SETS:
        '''
        Returns the corresponding fuzzy set type according to FUZZY_SETS enum.
        '''
        return FUZZY_SETS.t1
    

    def __str__(self) -> str:
        '''
        Returns the name of the fuzzy set, its type and its parameters.
        
        :return: string.
        '''
        return f'Categorical set: {self.name}, type 1 output'
    

    def shape(self) -> str:
        '''
        Returns the shape of the fuzzy set.

        :return: string.
        '''
        return 'categorical'


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
    

    def __str__(self) -> str:
        '''
        Returns the name of the fuzzy set, its type and its parameters.
        
        :return: string.
        '''
        return f'{self.name} ({self.type().name}) - {self.secondMF_lower} - {self.secondMF_upper}'


class categoricalIVFS(IVFS):
    '''
    Class to define a iv fuzzy set with categorical membership.
    '''

    def __init__(self, name: str, category) -> None:
        '''
        Creates a categorical iv fuzzy set. It gives 1 to the category and 0 to the rest.
        Use it when the variable is categorical and the categories are known, so that rule inference systems
        can naturally support both crisp and fuzzy variables.

        :param name: string.
        :param categories: list of strings. Possible categories.
        :param domain: list of two numbers. Limits of the domain if the fuzzy set.
        '''
        self.name = name
        self.category = category

        if mnt.save_usage_flag:
            mnt.usage_data[mnt.usage_categories.FuzzySets][self.type().name] += 1


    def membership(self, x: np.array) -> np.array:
        '''
        Computes the membership of a point or vector of elements.

        :param x: input values in the referencial domain.
        '''
        if isinstance(x, np.ndarray):
            res = np.equal(x, self.category).astype(float)
            res = np.stack([res, res], axis=-1)
        elif isinstance(x, torch.Tensor):
            res = torch.eq(x, self.category).float()
            res = torch.stack([res, res], axis=-1)
        elif isinstance(x, list):
            res = [1.0 if elem == self.category else 0.0 for elem in x]
            res = np.stack([res, res], axis=-1)
        elif isinstance(x, float) or isinstance(x, int):
            res = 1.0 if x == self.category else 0.0
            res = np.array([res, res])
        elif isinstance(x, pd.Series):
            res = x.apply(lambda elem: 1.0 if elem == self.category else 0.0)
            res = np.stack([res, res], axis=-1)
        
        return res


    def type(self) -> FUZZY_SETS:
        '''
        Returns the corresponding fuzzy set type according to FUZZY_SETS enum.
        '''
        return FUZZY_SETS.t2


    def __str__(self) -> str:
        '''
        Returns the name of the fuzzy set, its type and its parameters.
        
        :return: string.
        '''
        return f'Categorical set: {self.name}, type 2 output'
    

    def shape(self) -> str:
        '''
        Returns the shape of the fuzzy set.

        :return: string.
        '''
        return 'categorical'


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
        lower = _gaussian2(input, self.secondMF_lower)
        upper = _gaussian2(input, self.secondMF_upper)

        return np.array(np.concatenate([lower, upper])).T


    def type(self) -> FUZZY_SETS:
        '''
        Returns the type of the fuzzy set. (t1)
        '''
        return FUZZY_SETS.t2


    def shape(self) -> str:
        '''
        Returns the shape of the fuzzy set.

        :return: string.
        '''
        return 'gaussian'
    

class gaussianFS(FS):
    '''
    Class to define a gaussian fuzzy set.
    '''

    def membership(self, input: np.array) -> np.array:
        '''
        Computes the gaussian membership of a point or a vector.

        :param input: input values in the fuzzy set referencial domain.
        :return: np array samples
        '''
        return _gaussian2(input, self.membership_parameters)


    def type(self) -> FUZZY_SETS:
        '''
        Returns the type of the fuzzy set. (t1)
        '''
        return FUZZY_SETS.t1
    
    
    def shape(self) -> str:
        '''
        Returns the shape of the fuzzy set.

        :return: string.
        '''
        return 'gaussian'


class fuzzyVariable():
    '''
    Class to implement a fuzzy Variable. Contains a series of fuzzy sets and computes the memberships to all of them.
    '''

    def __init__(self, name: str, fuzzy_sets: list[FS], units:str=None) -> None:
        '''
        Creates a fuzzy variable.

        :param name: string. Name of the fuzzy variable.
        :param fuzzy_sets: list of IVFS. Each of the fuzzy sets that comprises the linguist variables of the fuzzy variable.
        :param units: string. Units of the fuzzy variable. Only for printings purposes.
        '''
        self.linguistic_variables = []
        self.name = name
        self.units = units

        for ix, fs in enumerate(fuzzy_sets):
            self.linguistic_variables.append(fs)

        self.fs_type = self.linguistic_variables[0].type()


    def append(self, fs: FS) -> None:
        '''
        Appends a fuzzy set to the fuzzy variable.

        :param fs: FS. Fuzzy set to append.
        '''
        self.linguistic_variables.append(fs)


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
        x = np.clip(x, self.linguistic_variables[0].domain[0], self.linguistic_variables[0].domain[1])

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


    def _wilcoxon_validation(self, mu_A, mu_B) -> bool:
        '''
        Validates the fuzzy variable using Earth Mover's Distance (EMD) to check if the fuzzy sets are statistically different.

        :param memberships: np.array. Memberships of the fuzzy sets.
        :return: bool. True if the fuzzy sets are statistically different, False otherwise.
        '''
        from scipy.stats import wilcoxon, ttest_1samp
        import scipy.stats as stats
        # Observed EMD
        differences = np.abs(mu_A - mu_B)

        # Check if the differences are normally distributed
        if stats.shapiro(differences)[1] > 0.05:
            # If normally distributed, use the t-test
            t_stat, p_value = ttest_1samp(differences, 0)
        else:
            # If not normally distributed, use the Wilcoxon signed-rank test
            w_stat, p_value = wilcoxon(differences)  

        return p_value < 0.05  # If p-value is greater than 0.05, the null hypothesis is accepted, meaning the distributions are not statistically different.
    

    def _permutation_validation(self, mu_A:np.array, mu_B:np.array, p_value_need:float=0.05) -> bool:
        '''
        Validates the fuzzy variable using permutation test to check if the fuzzy sets are statistically different.

        :param mu_A: np.array. Memberships of the first fuzzy set.
        :param mu_B: np.array. Memberships of the second fuzzy set.
        :return: bool. True if the fuzzy sets are statistically different, False otherwise.
        '''
        from scipy.stats import permutation_test

        # Perform permutation test
        result = permutation_test((mu_A, mu_B), lambda x, y: np.mean(np.abs(x - y)), n_resamples=1000)
        statistic, p_value, null_distribution = result.statistic, result.pvalue, result.null_distribution

        return p_value < p_value_need
    

    def validate(self, X, verbose:bool=False) -> bool:
        '''
        Validates the fuzzy variable. Checks that all the fuzzy sets have the same type and domain.

        :param X: np.array. Input data to validate the fuzzy variable.
        :return: bool. True if the fuzzy variable is valid, False otherwise.
        '''
        if len(self.linguistic_variables) == 0:
            return False
        
        # Get the fuzzy sets memberships
        memberships = self.compute_memberships(X)
        memberships = np.array(memberships)

        cond1 = True
        # Property 1: Only one of the fuzzy sets memberships can be 1 at the same time
        if np.equal(memberships, 1).sum(axis=0).max() > 1:
            cond1 = False
        if not cond1 and verbose:
            print('Property 1 violated: More than one fuzzy set has a membership of 1 at the same time.')

        # Property 2: All fuzzy sets are fuzzy numbers is fullfilled if they are trapezoidal or gaussian
        cond2 = all([fs.shape() in ['trapezoid', 'triangular', 'gaussian'] for fs in self.linguistic_variables])
        if not cond2 and verbose:
            print('Property 2 violated: Not all fuzzy sets are fuzzy numbers (trapezoidal or gaussian).')

        # Property 3: At least one fuzzy set must non zero in every point of the domain
        cond3 = np.any(memberships > 0, axis=0).all()
        if not cond3 and verbose:
            print('Property 3 violated: At least one fuzzy set must be non-zero in every point of the domain.')

        # Property 4: Given any two points of the domain, if a < b, the membership f_n+1(b)>f_n+1(a) can only hold if f_n(a)>f_n(b). So, a fuzzy set can only grow if the previous fuzzy set is decreasing.
        cond4 = True
        for i in range(len(self.linguistic_variables) - 1):
            if np.any(memberships[i, :] < memberships[i + 1, :]) and np.any(memberships[i, :] > memberships[i + 1, :]):
                valid = False
                break
        
        if not cond4 and verbose:
            print('Property 4 violated: Fuzzy sets must be non-decreasing in the domain. If a fuzzy set grows, the previous fuzzy set must be decreasing.')

        
        # Property 5: The smallest fuzzy set must be the first one and the biggest fuzzy set must be the last one. The smallest should start at the left of the domain and the biggest should end at the right of the domain.
        cond5 = (self.compute_memberships(self[0].domain[0])[0] >= 0.99) and (self.compute_memberships(self[0].domain[1])[-1] >= 0.99)
        if not cond5 and verbose:
            print('Property 5 violated: The smallest fuzzy set must be the first one and the biggest fuzzy set must be the last one. The smallest should start at the left of the domain and the biggest should end at the right of the domain.')

        # Property 6: The population of the fuzzy sets-induced memberships must be statistically different from each other
        cond6 = True
        if len(self.linguistic_variables) > 1:
            from scipy.stats import wilcoxon
            for i in range(len(self.linguistic_variables) - 1):
                # Wilcoxon signed-rank test is a paired difference test
                if not self._permutation_validation(memberships[i, :], memberships[i + 1, :], p_value_needed=0.05):
                    cond6 = False
                    break
        if not cond6 and verbose:
            print('Property 6 violated: The fuzzy sets must be statistically different from each other. Use permutation test to check this. (' + str(i) + ',' + str(i+1) + ')')

        valid = cond1 and cond2 and cond3 and cond4 and cond5 and cond6

        if verbose and valid:
            print('Fuzzy variable ' + self.name + ' is valid.')

        return valid


    def __str__(self) -> str:
        '''
        Returns the name of the fuzzy variable, its type and its parameters.
        
        :return: string.
        '''
        return f'{self.name} ({self.fs_type.name}) - {self.linguistic_variable_names()}'


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
    

    def __iter__(self) -> Generator[FS, None, None]:
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



