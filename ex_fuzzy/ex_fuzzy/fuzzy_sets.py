"""
Fuzzy Sets Module for Ex-Fuzzy Library

This module contains the core fuzzy set classes and functionality for the ex-fuzzy library.
It implements Type-1, Type-2, and General Type-2 fuzzy sets with their associated 
membership functions and operations.

Main Components:
    - FUZZY_SETS enum: Defines the types of fuzzy sets supported
    - Membership function implementations (trapezoidal, triangular, gaussian)
    - Fuzzy set classes: FS, IVFS, gaussianFS, gaussianIVFS, categoricalFS, etc.
    - fuzzyVariable class: Container for linguistic variables with multiple fuzzy sets
    - Validation and statistical testing methods for fuzzy variables

The module supports both numerical computation and provides interfaces for
PyTorch tensors when available, making it suitable for both traditional fuzzy
logic applications and modern machine learning workflows.
"""
import enum
from typing import Generator

import numpy as np
import pandas as pd

try:
    pass  # Removed deprecated maintenance module
except:
    pass

# You dont require torch to use this module, however, we need to import it to give support in case you feed these methods with torch tensors.
# Torch is imported lazily only when needed to avoid expensive imports

def _get_torch():
    """Lazy import of torch to avoid expensive imports when not needed."""
    try:
        import torch
        return torch
    except ImportError:
        return None

def _is_torch_tensor(x):
    """Check if x is a torch tensor without importing torch unless necessary."""
    # First check the type name to avoid torch import if not needed
    if type(x).__name__ != 'Tensor':
        return False
    
    # Only import torch if we have a potential tensor
    torch = _get_torch()
    if torch is None:
        return False
    
    return isinstance(x, torch.Tensor)


class FUZZY_SETS(enum.Enum):
    """
    Enumeration defining the types of fuzzy sets supported by the library.
    
    This enum is used throughout the library to specify which type of fuzzy set
    should be created or used in operations.
    
    Attributes:
        t1: Type-1 fuzzy sets with crisp membership functions
        t2: Type-2 interval fuzzy sets with upper and lower membership bounds  
        gt2: General Type-2 fuzzy sets with full secondary membership functions
        
    Example:
        >>> fz_type = FUZZY_SETS.t1
        >>> if fz_type == FUZZY_SETS.t1:
        ...     print("Using Type-1 fuzzy sets")
    """
    t1 = 'Type 1'
    t2 = 'Type 2'
    gt2 = 'General Type 2'


    def __eq__(self, __value: object) -> bool:
        return self.value == __value.value


def trapezoidal_membership(x: np.array, params: list[float], epsilon=10E-5) -> np.array:
    """
    Compute trapezoidal membership function values.
    
    This function computes the membership degree for input values using a trapezoidal
    membership function defined by four parameters representing the trapezoid vertices.
    
    Args:
        x (np.array): Input values in the fuzzy set domain for which to compute membership
        params (list[float]): Four numbers [a, b, c, d] defining the trapezoid:
            - a: left foot (membership starts rising)
            - b: left shoulder (membership reaches 1.0)  
            - c: right shoulder (membership starts falling)
            - d: right foot (membership reaches 0.0)
        epsilon (float, optional): Small value for numerical stability. Defaults to 10E-5.
            
    Returns:
        np.array: Membership values in [0, 1] for each input value
        
    Example:
        >>> x = np.array([0, 1, 2, 3, 4, 5])
        >>> params = [1, 2, 3, 4]  # trapezoid from 1 to 4, flat from 2 to 3
        >>> membership = trapezoidal_membership(x, params)
        >>> print(membership)  # [0, 0, 1, 1, 0, 0]
        
    Note:
        Special case: if a == d (singleton), returns 1.0 for exact matches, 0.0 otherwise.
        This handles degenerate trapezoids that collapse to single points.
    """
    a, b, c, d = params

    # Special case: a singleton trapezoid
    if a == d:
        # If they are numpy arrays, we need to use the numpy function
        if isinstance(x, np.ndarray):
            return np.equal(x, a).astype(float)
        if _is_torch_tensor(x):
            torch = _get_torch()
            return torch.eq(x, a).float()
            

    if b == a:
        b += epsilon
    if c == d:
        d += epsilon

    aux1 = (x - a) / (b - a)
    aux2 = (d - x) / (d - c)
    
    if _is_torch_tensor(x):
        torch = _get_torch()
        return torch.clamp(torch.min(aux1, aux2), 0.0, 1.0)

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
    """
    Base class for Type-1 fuzzy sets (Zadeh fuzzy sets).
    
    This class implements the fundamental Type-1 fuzzy set with crisp membership functions.
    It serves as the base class for more specialized fuzzy set types like triangular,
    gaussian, and categorical fuzzy sets.
    
    Attributes:
        name (str): The linguistic name of the fuzzy set (e.g., "low", "medium", "high")
        membership_parameters (list[float]): Parameters defining the membership function
        domain (list[float]): Two-element list defining the universe of discourse [min, max]
        
    Example:
        >>> fs = FS("medium", [1, 2, 3, 4], [0, 5])  # Trapezoidal fuzzy set
        >>> membership = fs.membership(2.5)
        >>> print(membership)  # Should be 1.0 (fully in the set)
        
    Note:
        This class uses trapezoidal membership functions by default. For other shapes,
        use specialized subclasses like gaussianFS or triangularFS.
    """

    def __init__(self, name: str, membership_parameters: list[float], domain: list[float]=None) -> None:
        """
        Initialize a Type-1 fuzzy set.

        Args:
            name (str): Linguistic name for the fuzzy set
            membership_parameters (list[float]): Four parameters [a, b, c, d] defining 
                the trapezoidal membership function where:
                - a: left foot (membership starts rising from 0)
                - b: left shoulder (membership reaches 1.0)
                - c: right shoulder (membership starts falling from 1.0)
                - d: right foot (membership reaches 0)
            domain (list[float]): Two-element list [min, max] defining the universe
                of discourse for this fuzzy set
                
        Example:
            >>> fs = FS("medium", [2, 3, 7, 8], [0, 10])
            >>> # Creates a trapezoidal set: rises from 2-3, flat 3-7, falls 7-8
        """
        self.name = name
        self.domain = domain
        self.membership_parameters = membership_parameters


    def membership(self, x: np.array) -> np.array:
        """
        Compute membership degrees for input values.

        This method calculates the membership degree(s) for the given input value(s)
        using the trapezoidal membership function defined by this fuzzy set's parameters.

        Args:
            x (np.array): Input value(s) for which to compute membership degrees.
                Can be a single value, list, or numpy array.

        Returns:
            np.array: Membership degree(s) in the range [0, 1]. Shape matches input.

        Example:
            >>> fs = FS("medium", [2, 3, 7, 8], [0, 10])
            >>> print(fs.membership(5))    # 1.0 (in flat region)
            >>> print(fs.membership(2.5)) # 0.5 (on rising slope)
            >>> print(fs.membership([1, 5, 9])) # [0.0, 1.0, 0.0]
        """
        return trapezoidal_membership(x, self.membership_parameters)


    def type(self) -> FUZZY_SETS:
        """
        Return the fuzzy set type identifier.

        Returns:
            FUZZY_SETS: The type identifier (FUZZY_SETS.t1 for Type-1 fuzzy sets)
        """
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


    def membership(self, x: np.array) -> np.array:
        '''
        Computes the membership of a point or vector of elements.

        :param x: input values in the referencial domain.
        '''
        if isinstance(x, np.ndarray):
            res = np.equal(x, self.category).astype(float)
        elif _is_torch_tensor(x):
            torch = _get_torch()
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
    """
    Interval-Valued Fuzzy Set (Type-2 Fuzzy Set) class.
    
    This class implements Type-2 fuzzy sets using interval-valued membership functions.
    Each point in the domain has an interval [lower, upper] representing the uncertainty
    in the membership degree, providing a more flexible representation than Type-1 sets.
    
    Attributes:
        name (str): Linguistic name of the fuzzy set
        domain (list[float]): Universe of discourse [min, max]
        secondMF_lower (list[float]): Parameters for the lower membership function
        secondMF_upper (list[float]): Parameters for the upper membership function  
        lower_height (float): Height of the lower membership function
        
    Example:
        >>> ivfs = IVFS("medium", [2,3,7,8], [1,2,8,9], [0,10], 0.8)
        >>> membership = ivfs.membership(5)  
        >>> print(membership)  # Returns [lower_membership, upper_membership]
        
    Note:
        The upper membership function should be contained within the lower membership
        function to maintain mathematical consistency of Type-2 fuzzy sets.
    """

    def __init__(self, name: str, secondMF_lower: list[float], secondMF_upper: list[float], 
                domain: list[float], lower_height=1.0) -> None:
        """
        Initialize an Interval-Valued Fuzzy Set.

        Args:
            name (str): Linguistic name for the fuzzy set
            secondMF_lower (list[float]): Four parameters [a,b,c,d] for lower trapezoidal 
                membership function (outer boundary)
            secondMF_upper (list[float]): Four parameters [a,b,c,d] for upper trapezoidal
                membership function (inner boundary)
            domain (list[float]): Two-element list [min, max] defining universe of discourse
            lower_height (float, optional): Maximum height of lower membership function.
                Defaults to 1.0.
                
        Raises:
            AssertionError: If membership function parameters are inconsistent or invalid
            
        Example:
            >>> ivfs = IVFS("high", [6,7,9,10], [6.5,7.5,8.5,9.5], [0,10], 0.9)
        """
        self.name = name
        self.domain = domain

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


    def membership(self, x: np.array) -> np.array:
        '''
        Computes the membership of a point or vector of elements.

        :param x: input values in the referencial domain.
        '''
        if isinstance(x, np.ndarray):
            res = np.equal(x, self.category).astype(float)
            res = np.stack([res, res], axis=-1)
        elif _is_torch_tensor(x):
            torch = _get_torch()
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
    """
    Gaussian Interval-Valued (Type-2) Fuzzy Set Implementation.
    
    This class implements a Gaussian interval-valued fuzzy set with two Gaussian
    membership functions (upper and lower) representing the uncertainty boundaries.
    This allows for modeling higher-order uncertainty in fuzzy systems.
    
    Attributes:
        secondMF_lower (list): Parameters [mean, std] for the lower membership function
        secondMF_upper (list): Parameters [mean, std] for the upper membership function
        name (str): Human-readable name for the fuzzy set
        universe_size (int): Size of the universe of discourse
        
    Example:
        >>> lower_params = [5.0, 1.0]
        >>> upper_params = [5.0, 1.5]
        >>> iv_gauss = gaussianIVFS(lower_params, upper_params, "Medium", 10)
        >>> membership = iv_gauss.membership(np.array([4.0, 5.0, 6.0]))
        >>> print(membership.shape)  # (3, 2) - lower and upper bounds
        
    Note:
        The interval-valued membership function provides both lower and upper
        bounds for each input, enabling Type-2 fuzzy reasoning.
    """

    def membership(self, input: np.array) -> np.array:
        """
        Computes the Gaussian interval-valued membership values for input points.
        
        Returns both lower and upper membership bounds for each input value,
        enabling Type-2 fuzzy set operations.
        
        Args:
            input (np.array): Input values in the fuzzy set domain
            
        Returns:
            np.array: Array of shape (n, 2) with [lower, upper] bounds for each input
            
        Example:
            >>> iv_gauss = gaussianIVFS([0, 1], [0, 1.5], "Zero", 5)
            >>> values = iv_gauss.membership(np.array([0.0, 1.0]))
            >>> print(values.shape)  # (2, 2) - 2 inputs, 2 bounds each
        """
        lower = _gaussian2(input, self.secondMF_lower)
        upper = _gaussian2(input, self.secondMF_upper)

        return np.array(np.concatenate([lower, upper])).T

    def type(self) -> FUZZY_SETS:
        """
        Returns the type of the fuzzy set.
        
        Returns:
            FUZZY_SETS: Always returns FUZZY_SETS.t2 for Type-2 fuzzy sets
        """
        return FUZZY_SETS.t2

    def shape(self) -> str:
        """
        Returns the shape of the fuzzy set.
        
        Returns:
            str: Always returns 'gaussian'
        """
        return 'gaussian'
    

class gaussianFS(FS):
    """
    Gaussian Type-1 Fuzzy Set Implementation.
    
    This class implements a Gaussian fuzzy set with bell-shaped membership function.
    Gaussian fuzzy sets are characterized by their mean and standard deviation,
    providing smooth membership transitions ideal for continuous variables.
    
    Attributes:
        membership_parameters (list): [mean, standard_deviation] defining the Gaussian curve
        name (str): Human-readable name for the fuzzy set
        universe_size (int): Size of the universe of discourse
        
    Example:
        >>> params = [5.0, 1.5]  # mean=5, std=1.5
        >>> gauss_fs = gaussianFS(params, "Medium", 10)
        >>> membership = gauss_fs.membership(np.array([4.0, 5.0, 6.0]))
        >>> print(membership)  # [0.606, 1.0, 0.606]
        
    Note:
        The membership function follows the formula:
        μ(x) = exp(-0.5 * ((x - mean) / std)^2)
    """

    def membership(self, input: np.array) -> np.array:
        """
        Computes the Gaussian membership values for input points.
        
        The membership function implements the Gaussian curve formula using
        the mean and standard deviation parameters.
        
        Args:
            input (np.array): Input values in the fuzzy set domain
            
        Returns:
            np.array: Membership values in range [0, 1]
            
        Example:
            >>> gauss_fs = gaussianFS([0, 1], "Zero", 5)
            >>> values = gauss_fs.membership(np.array([-1, 0, 1]))
            >>> print(values)  # [0.606, 1.0, 0.606]
        """
        return _gaussian2(input, self.membership_parameters)

    def type(self) -> FUZZY_SETS:
        """
        Returns the type of the fuzzy set.
        
        Returns:
            FUZZY_SETS: Always returns FUZZY_SETS.t1 for Type-1 fuzzy sets
        """
        return FUZZY_SETS.t1

    def shape(self) -> str:
        """
        Returns the shape of the fuzzy set.
        
        Returns:
            str: Always returns 'gaussian'
        """
        return 'gaussian'


class fuzzyVariable():
    """
    Fuzzy Variable Container and Management Class.
    
    This class represents a fuzzy variable composed of multiple fuzzy sets
    (linguistic variables). It provides methods to compute memberships across
    all fuzzy sets and manage the linguistic terms of the variable.
    
    Attributes:
        linguistic_variables (list): List of fuzzy sets that define the variable
        name (str): Name of the fuzzy variable
        units (str): Units of measurement (optional, for display purposes)
        fs_type (FUZZY_SETS): Type of fuzzy sets (t1 or t2)
        
    Example:
        >>> # Create fuzzy sets for temperature
        >>> low_temp = gaussianFS([15, 5], "Low", 100)
        >>> medium_temp = gaussianFS([25, 5], "Medium", 100)
        >>> high_temp = gaussianFS([35, 5], "High", 100)
        >>> 
        >>> # Create fuzzy variable
        >>> temp_var = fuzzyVariable("Temperature", [low_temp, medium_temp, high_temp], "°C")
        >>> memberships = temp_var.membership([20, 25, 30])
        >>> print(memberships.shape)  # (3, 3) - 3 inputs, 3 fuzzy sets
        
    Note:
        All fuzzy sets in the variable must be of the same type (t1 or t2).
    """

    def __init__(self, name: str, fuzzy_sets: list[FS], units: str = None) -> None:
        """
        Creates a fuzzy variable with the specified fuzzy sets.
        
        Args:
            name (str): Name of the fuzzy variable
            fuzzy_sets (list[FS]): List of fuzzy sets that comprise the linguistic variables
            units (str, optional): Units of the fuzzy variable for display purposes
            
        Raises:
            ValueError: If fuzzy_sets is empty or contains mixed types
            
        Example:
            >>> fs1 = gaussianFS([0, 1], "Low", 10)
            >>> fs2 = gaussianFS([5, 1], "High", 10)
            >>> var = fuzzyVariable("Speed", [fs1, fs2], "m/s")
        """
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
        try:
            x = np.clip(x, self.linguistic_variables[0].domain[0], self.linguistic_variables[0].domain[1])
        except Exception as e:
            pass

        for fuzzy_set in self.linguistic_variables:
            res.append(fuzzy_set.membership(x))

        return np.array(res)


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
    

    def _permutation_validation(self, mu_A:np.array, mu_B:np.array, p_value_need:float=0.05) -> bool:
        '''
        Validates the fuzzy variable using permutation test to check if the fuzzy sets are statistically different.

        :param mu_A: np.array. Memberships of the first fuzzy set.
        :param mu_B: np.array. Memberships of the second fuzzy set.
        :return: bool. True if the fuzzy sets are statistically different, False otherwise.
        '''
        from scipy.stats import permutation_test

        # Perform permutation test
        result = permutation_test(
            (mu_A, mu_B), 
            lambda x, y: np.mean(np.abs(x - y)) / (np.sqrt(np.var(x, ddof=1) * np.var(y, ddof=1)) + 1e-10), 
            n_resamples=1000
        )
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
            for i in range(len(self.linguistic_variables) - 1):
                # Wilcoxon signed-rank test is a paired difference test
                if not self._permutation_validation(memberships[i, :], memberships[i + 1, :], p_value_need=0.05):
                    cond6 = False
                    break
        if not cond6 and verbose:
            print('Property 6 violated: The fuzzy sets must be statistically different from each other. Used permutation test to check this. (' + str(i) + ',' + str(i+1) + ')')

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



