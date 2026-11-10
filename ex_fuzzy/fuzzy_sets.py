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
import warnings
from typing import Generator

import numpy as np
import pandas as pd

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
    
    Example:
        >>> fz_type = FUZZY_SETS.t1
        >>> if fz_type == FUZZY_SETS.t1:
        ...     print("Using Type-1 fuzzy sets")
    """
    #: Type-1 fuzzy sets with crisp membership functions.
    t1 = 'Type 1'
    #: Type-2 interval fuzzy sets with upper and lower membership bounds.
    t2 = 'Type 2'
    #: General Type-2 fuzzy sets with full secondary membership functions.
    gt2 = 'General Type 2'


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
        epsilon (float, optional): Width of the shift applied to degenerate edges (see Note).
            Defaults to 10E-5 (1e-4).
            
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

        Vertical edges: if a == b (the left edge is vertical) the function evaluates
        with ``a - epsilon`` instead of ``a``, and if c == d it evaluates with
        ``d + epsilon`` instead of ``d``. The membership is then exactly 1.0 at the
        edge itself (``x == a`` or ``x == d``) and falls to 0.0 over ``epsilon`` outside
        it, instead of dividing by zero. The shift only exists inside this function:
        the ``membership_parameters`` stored in the fuzzy set, and anything exported
        from them, keep the original, unshifted ``a`` and ``d``. The PyTorch objectives
        of the genetic fits apply the same 1e-4 shift, so both routes agree.
    """
    if isinstance(x, list):
        return list(trapezoidal_membership(np.array(x, dtype=float), params, epsilon))

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
        a -= epsilon
    if c == d:
        d += epsilon

    aux1 = (x - a) / (b - a)
    aux2 = (d - x) / (d - c)
    
    if _is_torch_tensor(x):
        torch = _get_torch()
        return torch.clamp(torch.min(aux1, aux2), 0.0, 1.0)

    if isinstance(x, np.ndarray):
        return np.clip(np.minimum(aux1, aux2), 0.0, 1.0)
    elif isinstance(x, pd.Series):
        return np.clip(np.minimum(aux1, aux2), 0.0, 1.0)
    else: # Single value
        return np.clip(min(aux1, aux2), 0.0, 1.0)



def _gaussian2(x, params: list[float]) -> np.array:
    """
    Gaussian membership functions.

    Args:
        mean:  real number, mean of the gaussian function.
        amplitude: real number.
        standard_deviation: std of the gaussian function.
    """
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
        """
        Calling the Fuzzy set returns its membership.

        Args:
            x: input values in the fuzzy set referencial domain.

        Returns:
            membership of the fuzzy set.
        """
        return self.membership(x)
    

    def __str__(self) -> str:
        """
        Returns the name of the fuzzy set, its type and its parameters.

        Returns:
            string.
        """
        return f'{self.name} ({self.type().name}) - {self.membership_parameters}'
    

    def shape(self) -> str:
        """
        Returns the shape of the fuzzy set.

        Returns:
            string.
        """
        return 'trapezoid'


class triangularFS(FS):
    def __init__(self, name: str, membership_parameters: list[float], domain: list[float]) -> None:
        super().__init__(name, membership_parameters, domain)

    def shape(self) -> str:
        return 'triangular'


class categoricalFS(FS):

    def __init__(self, name: str, category) -> None:
        """
        Creates a categorical fuzzy set. It gives 1 to the category and 0 to the rest.
        Use it when the variable is categorical and the categories are known, so that rule inference systems
        can naturally support both crisp and fuzzy variables.

        Args:
            name: string.
            categories: list of strings. Possible categories.
        """
        self.name = name
        self.category = category
        # A categorical set has no numerical domain or membership parameters.
        self.domain = None
        self.membership_parameters = None


    def membership(self, x: np.array) -> np.array:
        """
        Computes the membership of a point or vector of elements.

        Args:
            x: input values in the referencial domain.
        """
        if isinstance(x, np.ndarray):
            res = np.equal(x, self.category).astype(float)
        elif _is_torch_tensor(x):
            torch = _get_torch()
            res = torch.eq(x, self.category).float()
        elif isinstance(x, list):
            res = [1.0 if elem == self.category else 0.0 for elem in x]
        elif isinstance(x, pd.Series):
            res = x.apply(lambda elem: 1.0 if elem == self.category else 0.0)
        else:  # Single value
            res = 1.0 if x == self.category else 0.0

        return res


    def type(self) -> FUZZY_SETS:
        """
        Returns the corresponding fuzzy set type according to FUZZY_SETS enum.
        """
        return FUZZY_SETS.t1
    

    def __str__(self) -> str:
        """
        Returns the name of the fuzzy set, its type and its parameters.

        Returns:
            string.
        """
        return f'Categorical set: {self.name}, type 1 output'
    

    def shape(self) -> str:
        """
        Returns the shape of the fuzzy set.

        Returns:
            string.
        """
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
            ValueError: If membership function parameters are inconsistent or invalid
            
        Example:
            >>> ivfs = IVFS("high", [6,7,9,10], [6.5,7.5,8.5,9.5], [0,10], 0.9)
        """
        self.name = name
        self.domain = domain

        if not (secondMF_lower[0] >= secondMF_upper[0]):
            raise ValueError('First term membership incoherent')
        if not (secondMF_lower[0] <= secondMF_lower[1] <= secondMF_lower[2] <= secondMF_lower[3]):
            raise ValueError('Lower memberships incoherent.')
        if not (secondMF_upper[0] <= secondMF_upper[1] <= secondMF_upper[2] <= secondMF_upper[3]):
            raise ValueError('Upper memberships incoherent.')
        if not (secondMF_lower[3] <= secondMF_upper[3]):
            raise ValueError('Final term memberships incoherent.')

        self.secondMF_lower = secondMF_lower
        self.secondMF_upper = secondMF_upper
        self.lower_height = lower_height


    def membership(self, x: np.array) -> np.array:
        """
        Computes the iv-membership of a point or a vector.

        Args:
            x: input values in the fuzzy set referencial domain.

        Returns:
            iv-membership of the fuzzy set.
        """
        lower = trapezoidal_membership(
            x, self.secondMF_lower) * self.lower_height
        upper = trapezoidal_membership(x, self.secondMF_upper)

        return np.stack([lower, upper], axis=-1)


    def type(self) -> FUZZY_SETS:
        """
        Returns the corresponding fuzzy set type according to FUZZY_SETS enum: (t2)
        """
        return FUZZY_SETS.t2
    

    def __str__(self) -> str:
        """
        Returns the name of the fuzzy set, its type and its parameters.

        Returns:
            string.
        """
        return f'{self.name} ({self.type().name}) - {self.secondMF_lower} - {self.secondMF_upper}'


class categoricalIVFS(IVFS):
    """
    Class to define a iv fuzzy set with categorical membership.
    """

    def __init__(self, name: str, category) -> None:
        """
        Creates a categorical iv fuzzy set. It gives 1 to the category and 0 to the rest.
        Use it when the variable is categorical and the categories are known, so that rule inference systems
        can naturally support both crisp and fuzzy variables.

        Args:
            name: string.
            categories: list of strings. Possible categories.
        """
        self.name = name
        self.category = category
        # A categorical set has no numerical domain or membership parameters.
        self.domain = None
        self.membership_parameters = None


    def membership(self, x: np.array) -> np.array:
        """
        Computes the membership of a point or vector of elements.

        Args:
            x: input values in the referencial domain.
        """
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
        elif isinstance(x, pd.Series):
            res = x.apply(lambda elem: 1.0 if elem == self.category else 0.0)
            res = np.stack([res, res], axis=-1)
        else:  # Single value
            res = 1.0 if x == self.category else 0.0
            res = np.array([res, res])

        return res


    def type(self) -> FUZZY_SETS:
        """
        Returns the corresponding fuzzy set type according to FUZZY_SETS enum.
        """
        return FUZZY_SETS.t2


    def __str__(self) -> str:
        """
        Returns the name of the fuzzy set, its type and its parameters.

        Returns:
            string.
        """
        return f'Categorical set: {self.name}, type 2 output'
    

    def shape(self) -> str:
        """
        Returns the shape of the fuzzy set.

        Returns:
            string.
        """
        return 'categorical'


class GT2(FS):
    """
    Class to define a gt2 fuzzy set.
    """

    MAX_RES_SUPPORT = 4  # Number of decimals supported in the secondary

    def __init__(self, name: str, secondary_memberships: dict[float, FS], 
                domain: list[float], significant_decimals: int, 
                alpha_cuts: list[float] = [0.2, 0.4, 0.5, 0.7, 0.9, 1.0], 
                unit_resolution: float = 0.2) -> None:
        """
        Creates a GT2 fuzzy set.

        Args:
            name: string.
            secondary_memberships: list of fuzzy sets. Secondary membership that maps original domain to [0, 1]
            significant_decimals: number of decimals used to index the secondary memberships.
            domain: list of two numbers. Limits of the domain if the fuzzy set.
            alpha_cuts: list of real numbers. Alpha cuts of the fuzzy set.
            unit_resolution: real number. Resolution of the primary membership function.
        """
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
        """
        Computes the alpha cut memberships of a point.

        Args:
            x: input values in the fuzzy set referencial domain.

        Returns:
            np array samples x alpha_cuts x 2
        """
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
        """
        Computes the type reduction to reduce the alpha cuts to one value.

        Args:
            x: array with the values of the inputs.

        Returns:
            array with the memberships of the consequents for each sample.
        """
        if len(x.shape) == 3:
            formatted = np.expand_dims(np.expand_dims(
                np.array(self.alpha_cuts), axis=1), axis=0)
        else:
            formatted = self.alpha_cuts
        return np.sum(formatted * x, axis=-1) / np.sum(self.alpha_cuts)
    

    def alpha_reduction(self, x) -> np.array:
        """
        Computes the type reduction to reduce the alpha cuts to one value.

        Args:
            x: array with the values of the inputs.

        Returns:
            array with the memberships of the consequents for each sample.
        """
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

    def __init__(self, name: str, secondMF_lower: list[float], secondMF_upper: list[float],
                 domain: list[float], lower_height=1.0) -> None:
        """
        Initialize a Gaussian interval-valued fuzzy set.

        Args:
            name (str): Linguistic name for the fuzzy set
            secondMF_lower (list[float]): [mean, std] of the lower membership function
            secondMF_upper (list[float]): [mean, std] of the upper membership function
            domain (list[float]): Two-element list [min, max] defining universe of discourse
            lower_height (float, optional): Maximum height of lower membership function.
                Defaults to 1.0.
        """
        self.name = name
        self.domain = domain
        self.secondMF_lower = secondMF_lower
        self.secondMF_upper = secondMF_upper
        self.lower_height = lower_height

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
        lower = _gaussian2(input, self.secondMF_lower) * self.lower_height
        upper = _gaussian2(input, self.secondMF_upper)

        return np.stack([lower, upper], axis=-1)

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


#: Operators of a condition on a linguistic variable (see :class:`fuzzyVariable`): ``is`` one term,
#: ``at_least`` / ``at_most`` a term of the ordered partition, or ``any_of`` a named compound of terms.
CONDITION_OPERATORS = ('is', 'at_least', 'at_most', 'any_of')
_OPERATOR_TEXT = {'is': '', 'at_least': 'at least ', 'at_most': 'at most '}


def ordinal_alphabet_size(n_terms: int) -> int:
    """
    Number of conditions a variable with ``n_terms`` ordered terms offers when ordinal conditions are on.

    The ``n_terms`` single terms, then ``at least t_j`` and ``at most t_j`` for ``j = 1 .. n_terms - 2``:
    ``at least t_0`` is always true (don't care) and ``at least t_(n-1)`` is ``t_(n-1)`` itself, and the same
    holds for ``at most``, so those are not offered. A variable with fewer than 3 terms keeps its ``n_terms``.
    """
    return n_terms + 2 * (n_terms - 2) if n_terms >= 3 else n_terms


def decode_ordinal_condition(code: int, n_terms: int) -> tuple:
    """
    ``(operator, term_index)`` of a condition code of a variable with ``n_terms`` terms.

    Codes ``0 .. n_terms - 1`` are the single terms (``'is'``); the next ``n_terms - 2`` are ``'at_least'``
    terms ``1 .. n_terms - 2``; the last ``n_terms - 2`` are ``'at_most'`` terms ``1 .. n_terms - 2``.
    """
    if not 0 <= code < ordinal_alphabet_size(n_terms):
        raise ValueError(f'Condition code {code} is outside 0..{ordinal_alphabet_size(n_terms) - 1} '
                         f'for a variable with {n_terms} terms.')
    if code < n_terms:
        return 'is', int(code)
    extra = code - n_terms
    if extra < n_terms - 2:
        return 'at_least', int(extra + 1)
    return 'at_most', int(extra - (n_terms - 2) + 1)


def encode_ordinal_condition(operator: str, term: int, n_terms: int) -> int:
    """Inverse of :func:`decode_ordinal_condition`."""
    if operator not in CONDITION_OPERATORS[:3]:
        raise ValueError(f'Unknown operator {operator!r}; use one of {list(CONDITION_OPERATORS[:3])}.')
    if operator == 'is':
        if not 0 <= term < n_terms:
            raise ValueError(f'Term {term} is outside 0..{n_terms - 1}.')
        return int(term)
    if not 1 <= term <= n_terms - 2:
        raise ValueError(f"'{operator}' is offered for the terms 1..{n_terms - 2} of a variable with {n_terms} "
                         f'terms; got {term}.')
    return int(n_terms + (term - 1) + (0 if operator == 'at_least' else n_terms - 2))


def ordinal_memberships(terms: np.ndarray) -> np.ndarray:
    """
    Memberships of every condition of an ordinal variable, from the memberships of its terms.

    Args:
        terms: ``(n_terms, ...)`` memberships of the ordered terms; trailing axes (samples, and the
            lower/upper axis of interval Type-2 sets) are carried along.

    Returns:
        ``(ordinal_alphabet_size(n_terms), ...)``: the terms, then ``at least t_j`` and ``at most t_j`` for
        ``j = 1 .. n_terms - 2``. A union of terms has the bounded sum ``min(1, sum of their memberships)``
        (the Lukasiewicz t-conorm). On a strong partition (memberships adding up to 1) this is the exact
        complement of the other terms, a shouldered trapezoid; on a partition whose memberships add up to
        more than 1 it saturates at 1; for interval Type-2 sets the lower and upper bounds are summed apart.
    """
    n_terms = terms.shape[0]
    if n_terms < 3:
        return terms
    at_most = np.minimum(1.0, np.cumsum(terms, axis=0)[1:n_terms - 1])
    at_least = np.minimum(1.0, np.cumsum(terms[::-1], axis=0)[::-1][1:n_terms - 1])
    return np.concatenate([terms, at_least, at_most], axis=0)


def compound_memberships(terms: np.ndarray, sets: list) -> np.ndarray:
    """
    Memberships of named compound terms: the bounded sum ``min(1, sum of the listed terms)`` of each set.

    Args:
        terms: ``(n_terms, ...)`` memberships of the terms (as in :func:`ordinal_memberships`).
        sets: one sequence of term indexes per compound. The terms need not be contiguous: a union such as
            ``[last, first]`` is two shoulders of the axis, one at each end.

    Returns:
        ``(len(sets), ...)``.
    """
    return np.stack([np.minimum(1.0, terms[list(indexes)].sum(axis=0)) for indexes in sets], axis=0)


def has_extra_conditions(variable) -> bool:
    """Whether a variable offers conditions beyond its single terms (ordinal ones or named compound terms)."""
    return bool(getattr(variable, 'ordinal', False)) or bool(getattr(variable, 'compound', ()))


def condition_count(variable) -> int:
    """Number of conditions (term codes) a variable offers: its terms, plus the extra ones if it has them."""
    if has_extra_conditions(variable):
        return variable.n_conditions()
    names = getattr(variable, 'linguistic_variable_names', None)  # also for variable-like objects without __len__
    return len(names()) if names is not None else len(variable)


def condition_label(variable, code: int) -> str:
    """Text of condition ``code`` of ``variable``: the term name, ``at least <term>``, ``at most <term>`` or the
    name of a compound term."""
    if has_extra_conditions(variable):
        return variable.condition_names()[int(code)]
    return variable.linguistic_variable_names()[code]


def _term_position(fuzzy_set):
    """Where a Type-1 term sits on the axis (centre of its core or mean), or None if unknown."""
    parameters = getattr(fuzzy_set, 'membership_parameters', None)
    if parameters is None:
        return None
    parameters = np.asarray(parameters, dtype=float).ravel()
    if isinstance(fuzzy_set, FS) and not isinstance(fuzzy_set, gaussianFS) and len(parameters) == 4:
        return float((parameters[1] + parameters[2]) / 2)
    if isinstance(fuzzy_set, gaussianFS):
        return float(parameters[0])
    return None


_FORBIDDEN_NAME_WORDS = ('AND', 'IS', 'WITH', 'THEN')


def check_condition_options(ordinal_conditions, compound_terms) -> None:
    """
    Reject a malformed ``ordinal_conditions`` or ``compound_terms`` argument of an estimator.

    ``ordinal_conditions`` is a bool or a collection (list, tuple, set, array) of features, by name or
    index; ``compound_terms`` is None or a mapping from a feature to a list of ``(name, [terms])``.
    """
    if not (ordinal_conditions is None or isinstance(ordinal_conditions, (bool, np.bool_))):
        if isinstance(ordinal_conditions, (str, bytes, dict)) or not hasattr(ordinal_conditions, '__iter__'):
            raise ValueError('ordinal_conditions must be True, False or a collection of features (names or indexes), '
                             f'got {ordinal_conditions!r}.')
    if compound_terms is not None and not hasattr(compound_terms, 'items'):
        raise ValueError('compound_terms must be None or a mapping from a feature (name or index) to a list of '
                         f'(name, [terms]), got {type(compound_terms).__name__}.')


def conditions_requested(ordinal_conditions, compound_terms) -> bool:
    """Whether any ordinal or compound condition is asked for (False, None, an empty collection: none)."""
    if isinstance(ordinal_conditions, (bool, np.bool_)):
        ordinal = bool(ordinal_conditions)
    else:
        ordinal = ordinal_conditions is not None and len(list(ordinal_conditions)) > 0
    return ordinal or bool(compound_terms)


def _resolve_feature(key, names: list, what: str) -> int:
    """Index of the feature ``key`` (a name in one of the ``names`` lists, or an integer index)."""
    if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
        if not 0 <= int(key) < len(names[0]):
            raise ValueError(f'{what}: feature index {int(key)} is outside 0..{len(names[0]) - 1}.')
        return int(key)
    for candidates in names:
        if key in list(candidates):
            return list(candidates).index(key)
    raise ValueError(f'{what}: unknown feature {key!r}; use one of {list(names[0])} or an index.')


def resolve_features(selection, names: list, what: str = 'ordinal_conditions') -> set:
    """
    Indexes of the features in ``selection``, a collection of names or indexes.

    Args:
        selection: collection of feature names or indexes.
        names: lists of names to match, each with one name per feature (for example the variable names and
            the column names of the data); the first one is shown in errors.
    """
    out = set()
    for key in selection:
        out.add(_resolve_feature(key, names, what))
    return out


def resolve_compound_entries(entries, term_names: list, variable_name, ordinal_sets: dict = None,
                             existing: tuple = ()) -> tuple:
    """
    Validate the compound terms of one variable and return them as ``((name, (term indexes...)), ...)``.

    Args:
        entries: ``[(name, [terms])...]`` with each term a term name or an index.
        term_names: names of the variable's terms.
        variable_name: for the error messages.
        ordinal_sets: ``{name: set of term indexes}`` of the ordinal conditions the variable offers, if any.
        existing: compounds the variable already has (same name and set is accepted again).

    Raises:
        ValueError: for fewer than 2 distinct terms, every term of the variable, an unknown term, a name
            that is not usable or already taken, a set listed twice, or the set of an ordinal condition.
    """
    n_terms = len(term_names)
    out = list(existing)
    new_names = set()
    taken_names = set(map(str, term_names)) | set((ordinal_sets or {}).keys())
    for entry in entries:
        try:
            name, terms = entry
        except (TypeError, ValueError):
            raise ValueError(f'Variable {variable_name!r}: a compound term is (name, [terms]); got {entry!r}.') from None
        if isinstance(terms, (str, bytes)) or not hasattr(terms, '__iter__'):
            raise ValueError(f'Variable {variable_name!r}: the terms of compound {name!r} must be a list of term '
                             f'names or indexes, got {terms!r}.')
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError(f'Variable {variable_name!r}: a compound term needs a non-empty name without '
                             f'leading or trailing spaces, got {name!r}.')
        words = name.split()
        if (any(ch in name for ch in ';\n\r()') or any(word in _FORBIDDEN_NAME_WORDS for word in words)
                or name.startswith(('at least ', 'at most '))):
            raise ValueError(f"Variable {variable_name!r}: compound name {name!r} cannot contain ';', line breaks, "
                             "parentheses or the words AND, IS, WITH, THEN, nor start with 'at least' or 'at most' "
                             '(the rule text must stay readable back).')
        indexes = []
        for term in terms:
            if isinstance(term, (int, np.integer)) and not isinstance(term, (bool, np.bool_)):
                if not 0 <= int(term) < n_terms:
                    raise ValueError(f'Variable {variable_name!r}: compound {name!r} lists term index {int(term)}, '
                                     f'outside 0..{n_terms - 1}.')
                indexes.append(int(term))
            elif term in list(map(str, term_names)) or term in list(term_names):
                indexes.append([str(t) for t in term_names].index(str(term)))
            else:
                raise ValueError(f'Variable {variable_name!r}: compound {name!r} lists unknown term {term!r}; '
                                 f'the terms are {list(term_names)}.')
        if len(set(indexes)) != len(indexes):
            raise ValueError(f'Variable {variable_name!r}: compound {name!r} lists a term twice.')
        if len(indexes) < 2:
            raise ValueError(f'Variable {variable_name!r}: compound {name!r} needs at least 2 distinct terms '
                             '(one term is already a condition).')
        if len(indexes) == n_terms:
            raise ValueError(f'Variable {variable_name!r}: compound {name!r} lists every term, so it is always true '
                             '(a don\'t care).')
        key = tuple(sorted(indexes))
        if name in new_names:
            raise ValueError(f'Variable {variable_name!r}: compound name {name!r} is used twice; the name of a '
                             'compound term must be unique within its variable.')
        new_names.add(name)
        same_name = [item for item in existing if item[0] == name]
        if same_name:
            if same_name[0][1] == key:
                continue  # the same compound again (a refit over variables that already have it)
            raise ValueError(f'Variable {variable_name!r}: compound name {name!r} is already used by another '
                             'compound term of the variable; names must be unique within a variable.')
        if name in taken_names:
            raise ValueError(f'Variable {variable_name!r}: compound name {name!r} is already the name of a term or of '
                             'an ordinal condition of the variable; names must be unique within a variable.')
        for other_name, other_key in out:
            if other_key == key:
                raise ValueError(f'Variable {variable_name!r}: compounds {other_name!r} and {name!r} list the same terms.')
        for ordinal_name, ordinal_key in (ordinal_sets or {}).items():
            if tuple(sorted(ordinal_key)) == key:
                raise ValueError(f'Variable {variable_name!r}: compound {name!r} covers the same terms as the ordinal '
                                 f'condition {ordinal_name!r}, which the variable already has.')
        out.append((name, key))
    return tuple(out)


def _ordinal_sets(term_names: list) -> dict:
    """``{condition name: set of term indexes}`` of the ordinal conditions of a variable with these terms."""
    n_terms = len(term_names)
    sets = {}
    for code in range(n_terms, ordinal_alphabet_size(n_terms)):
        operator, term = decode_ordinal_condition(code, n_terms)
        sets[_OPERATOR_TEXT[operator] + str(term_names[term])] = (
            set(range(term, n_terms)) if operator == 'at_least' else set(range(term + 1)))
    return sets


def enable_conditions(variables: list, ordinal_conditions=True, compound_terms=None, names: list = None) -> list:
    """
    Copies of ``variables`` that offer ordinal and/or named compound conditions, ready for the search.

    Args:
        variables: list of :class:`fuzzyVariable`.
        ordinal_conditions: ``True`` for ``at least`` / ``at most`` conditions on every variable that can have
            them (categorical variables and variables with fewer than 3 terms keep single terms); a collection
            of features (names or indexes) for those features only, which must then be able to have them;
            ``False`` or ``None`` for none.
        compound_terms: None, or a mapping from a feature (name or index) to ``[(name, [terms])...]``, each
            entry a new condition whose membership is the bounded sum of the listed terms (names or indexes).
        names: further lists of feature names to match features against (the variable names always are).

    Returns:
        a list of the same length: variables that gain nothing are returned as they are, and the originals are
        never changed (the copies share their fuzzy sets).

    Raises:
        ValueError: for general Type-2 sets, an unknown feature or term, and every invalid compound term
            (see :func:`resolve_compound_entries`).
    """
    check_condition_options(ordinal_conditions, compound_terms)
    all_names = [[variable.name for variable in variables]] + [list(item) for item in (names or [])]
    if isinstance(ordinal_conditions, (bool, np.bool_)) or ordinal_conditions is None:
        explicit = set()
        everywhere = bool(ordinal_conditions)
    else:
        explicit = resolve_features(ordinal_conditions, all_names, 'ordinal_conditions')
        everywhere = False
    compound = {}
    for key, entries in (compound_terms or {}).items():
        feature = _resolve_feature(key, all_names, 'compound_terms')
        if feature in compound:
            raise ValueError(f'compound_terms: feature {all_names[0][feature]!r} is given twice.')
        compound[feature] = list(entries)
    out = []
    for feature, variable in enumerate(variables):
        want_ordinal = (everywhere or feature in explicit)
        if want_ordinal and feature in explicit and not variable.ordinal_eligible(strict=True):
            raise ValueError(f'ordinal_conditions: variable {variable.name!r} cannot offer at least / at most '
                             'conditions (it needs 3 or more terms that are not categorical).')
        if want_ordinal and not variable.ordinal_eligible(strict=True):
            want_ordinal = False  # a bool True skips such variables quietly
        entries = compound.get(feature, ())
        if entries and variable.fs_type == FUZZY_SETS.gt2:
            raise ValueError('Compound terms support Type-1 and interval Type-2 sets, not general Type-2 sets.')
        ordinal = bool(variable.ordinal) or want_ordinal
        if ordinal == bool(variable.ordinal) and not entries:
            out.append(variable)
            continue
        copy = fuzzyVariable(variable.name, variable.linguistic_variables, variable.units)
        copy._set_conditions(ordinal, entries, existing=variable.compound)
        out.append(copy)
    return out


def enable_ordinal_conditions(variables: list) -> list:
    """
    Copies of ``variables`` that offer ``at least`` / ``at most`` conditions (see :func:`enable_conditions`).

    Variables that cannot have them are returned as they are: categorical ones (their categories are not
    ordered) and those with fewer than 3 terms. A variable that already has them is returned as it is.

    Raises:
        ValueError: for general Type-2 sets, whose memberships are not a number or an interval per term.
    """
    return enable_conditions(variables, True)


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
        >>> low_temp = gaussianFS("Low", [15, 5], [0, 50])
        >>> medium_temp = gaussianFS("Medium", [25, 5], [0, 50])
        >>> high_temp = gaussianFS("High", [35, 5], [0, 50])
        >>> 
        >>> # Create fuzzy variable
        >>> temp_var = fuzzyVariable("Temperature", [low_temp, medium_temp, high_temp], "°C")
        >>> memberships = temp_var.compute_memberships([20, 25, 30])  # or temp_var([20, 25, 30])
        >>> print(memberships.shape)  # (3, 3) - (fuzzy sets, inputs)
        
    Note:
        All fuzzy sets in the variable must be of the same type (t1 or t2).

        The terms are ordered as they are listed, from low to high. With ``ordinal_conditions=True`` the
        variable also offers the conditions ``at least t_j`` and ``at most t_j`` (``j = 1 .. n - 2``), whose
        membership is the bounded sum ``min(1, sum of the memberships of the terms they cover)``; see
        :func:`ordinal_alphabet_size`. A condition is named by an integer code: the term index for a single
        term, then the ``at least`` codes, then the ``at most`` ones, then the named compound terms
        (:meth:`condition`). ``compound_terms=[(name, [terms])...]`` adds one condition per entry, whose
        membership is the bounded sum of the listed terms, which need not be contiguous (a union that wraps
        around, such as ``[last, first]``, is two shoulders, one at each end of the axis). Off by default,
        and then nothing about the variable changes.
    """

    #: Class default, so that variables pickled before ordinal conditions existed load as they were.
    ordinal = False
    #: Named compound terms, ``((name, (term indexes...)), ...)``; empty unless asked for.
    compound = ()

    def __init__(self, name: str, fuzzy_sets: list[FS], units: str = None, ordinal_conditions: bool = False,
                 compound_terms: list = None) -> None:
        """
        Creates a fuzzy variable with the specified fuzzy sets.
        
        Args:
            name (str): Name of the fuzzy variable
            fuzzy_sets (list[FS]): List of fuzzy sets that comprise the linguistic variables
            units (str, optional): Units of the fuzzy variable for display purposes
            ordinal_conditions (bool, optional): offer ``at least`` / ``at most`` conditions over the ordered
                terms (default False). Needs 3 or more Type-1 or interval Type-2 terms that are not categorical.
            compound_terms (list, optional): ``[(name, [terms])...]``, each entry one more condition whose
                membership is the bounded sum of the listed terms (names or indexes): at least 2 distinct
                terms and not all of them; the sets must differ from each other and from the ordinal
                conditions of the variable.
            
        Raises:
            ValueError: If fuzzy_sets is empty or contains mixed types, or ``ordinal_conditions`` is asked
                of a variable that cannot have them
            
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

        if ordinal_conditions:
            if not self.ordinal_eligible(strict=True):
                raise ValueError('ordinal_conditions needs at least 3 terms that are not categorical '
                                 f'(variable {name!r}).')
        if ordinal_conditions or compound_terms:
            self._set_conditions(bool(ordinal_conditions), compound_terms or ())


    def _set_conditions(self, ordinal: bool, compound_entries, existing: tuple = ()) -> None:
        """Validate and set the ordinal flag and the compound terms (``existing`` ones are kept)."""
        if self.fs_type == FUZZY_SETS.gt2:
            raise ValueError('Ordinal and compound conditions support Type-1 and interval Type-2 sets, not general '
                             'Type-2 sets.')
        term_names = self.linguistic_variable_names()
        ordinal_sets = _ordinal_sets(term_names) if ordinal and len(term_names) >= 3 else {}
        compound = resolve_compound_entries(compound_entries, term_names, self.name, ordinal_sets, tuple(existing))
        for compound_name, key in compound:  # the existing ones too: an ordinal flag may be new
            for ordinal_name, ordinal_key in ordinal_sets.items():
                if tuple(sorted(ordinal_key)) == key:
                    raise ValueError(f'Variable {self.name!r}: compound {compound_name!r} covers the same terms as '
                                     f'the ordinal condition {ordinal_name!r}, which the variable already has.')
        self.compound = compound
        if ordinal and not self.ordinal:
            self.ordinal = True
            self._warn_if_unordered()


    def ordinal_eligible(self, strict: bool = False) -> bool:
        """
        Whether this variable can offer ordinal conditions: 3 or more ordered, non-categorical Type-1 or
        interval Type-2 terms. With ``strict`` a general Type-2 variable raises instead of returning False.
        """
        if self.fs_type == FUZZY_SETS.gt2:
            if strict:
                raise ValueError('Ordinal conditions support Type-1 and interval Type-2 sets, not general Type-2 sets.')
            return False
        if isinstance(self.linguistic_variables[0], (categoricalFS, categoricalIVFS)):
            return False
        return len(self.linguistic_variables) >= 3


    def _warn_if_unordered(self) -> None:
        """Warn when the terms are not listed from low to high, which is what ``at least`` / ``at most`` read."""
        positions = [_term_position(term) for term in self.linguistic_variables]
        if None not in positions and any(b < a for a, b in zip(positions, positions[1:])):
            warnings.warn(f'The terms of variable {self.name!r} are not listed from low to high; '
                          "'at least' and 'at most' follow the listed order.", stacklevel=3)


    def _ordinal_size(self) -> int:
        """Codes of the terms and the ordinal conditions (the compound terms come after them)."""
        n_terms = len(self.linguistic_variables)
        return ordinal_alphabet_size(n_terms) if self.ordinal else n_terms


    def n_conditions(self) -> int:
        """Number of conditions (term codes) the variable offers: its terms, then the ordinal ones if
        enabled, then the named compound terms."""
        return self._ordinal_size() + len(self.compound)


    def condition(self, code: int) -> tuple:
        """
        ``(operator, reference)`` of a condition code.

        The operator is ``'is'`` (reference: term index), ``'at_least'`` or ``'at_most'`` (reference: term
        index ``t_j``) or ``'any_of'`` (reference: position of the compound in ``self.compound``).
        """
        code = int(code)
        size = self._ordinal_size()
        if code >= size:
            if code >= size + len(self.compound):
                raise ValueError(f'Condition code {code} is outside 0..{self.n_conditions() - 1}.')
            return 'any_of', code - size
        if not self.ordinal:
            if not 0 <= code < len(self.linguistic_variables):
                raise ValueError(f'Condition code {code} is outside 0..{len(self.linguistic_variables) - 1}.')
            return 'is', int(code)
        return decode_ordinal_condition(code, len(self.linguistic_variables))


    def condition_code(self, operator: str, term) -> int:
        """Code of the condition ``operator`` + term index (inverse of :meth:`condition`); for ``'any_of'``
        ``term`` is the name of the compound term (or its position)."""
        if operator == 'any_of':
            names = [name for name, _ in self.compound]
            position = names.index(term) if term in names else term
            if not isinstance(position, (int, np.integer)) or not 0 <= position < len(names):
                raise ValueError(f'Variable {self.name!r} has no compound term {term!r}; it has {names}.')
            return self._ordinal_size() + int(position)
        if not self.ordinal and operator != 'is':
            raise ValueError(f"Variable {self.name!r} has no ordinal conditions; cannot encode '{operator}'.")
        return encode_ordinal_condition(operator, term, len(self.linguistic_variables))


    def condition_terms(self, code: int) -> list:
        """Indexes of the terms a condition covers: one for ``is``, ``t_j .. t_(n-1)`` for ``at_least t_j``,
        ``t_0 .. t_j`` for ``at_most t_j``, the listed terms (low to high) for ``any_of``."""
        operator, term = self.condition(code)
        if operator == 'is':
            return [term]
        if operator == 'at_least':
            return list(range(term, len(self.linguistic_variables)))
        if operator == 'any_of':
            return list(self.compound[term][1])
        return list(range(term + 1))


    def condition_names(self) -> list:
        """Text of every condition, by code: the term names, then ``at least <term>``, then ``at most <term>``,
        then the names of the compound terms. Without extra conditions this is :meth:`linguistic_variable_names`."""
        names = self.linguistic_variable_names()
        if not self.ordinal and not self.compound:
            return names
        out = list(names)
        for code in range(len(names), self._ordinal_size()):
            operator, term = self.condition(code)
            out.append(_OPERATOR_TEXT[operator] + str(names[term]))
        return out + [name for name, _ in self.compound]


    def condition_memberships(self, terms: np.ndarray) -> np.ndarray:
        """Memberships of every condition from those of the terms, ``(n_terms, ...)`` to ``(n_conditions, ...)``."""
        parts = [terms]
        if self.ordinal:
            parts.append(ordinal_memberships(terms)[terms.shape[0]:])
        if self.compound:
            parts.append(compound_memberships(terms, [indexes for _, indexes in self.compound]))
        return np.concatenate(parts, axis=0) if len(parts) > 1 else terms


    def append(self, fs: FS) -> None:
        """
        Appends a fuzzy set to the fuzzy variable.

        Args:
            fs: FS. Fuzzy set to append.
        """
        self.linguistic_variables.append(fs)


    def linguistic_variable_names(self) -> list:
        """
        Returns the name of the linguistic variables.

        Returns:
            list of strings.
        """
        return [fs.name for fs in self.linguistic_variables]


    def get_linguistic_variables(self) -> list[FS]:
        """
        Returns the name of the linguistic variables.

        Returns:
            list of strings.
        """
        return self.linguistic_variables


    def compute_memberships(self, x: np.array) -> list:
        """
        Computes the membership to each of the FS in the fuzzy variables.

        Args:
            x: numeric value or array. Computes the membership to each of the FS in the fuzzy variables.

        Returns:
            list of floats. Membership to each of the FS in the fuzzy variables. With ordinal conditions
            there is one more row per extra condition, after the terms (see :func:`ordinal_memberships` and
            :func:`compound_memberships`).
        """
        res = self._term_memberships(x)
        if self.ordinal or self.compound:
            res = self.condition_memberships(res)
        return res


    def _term_memberships(self, x: np.array) -> np.ndarray:
        """Memberships to the terms only, whatever the conditions on offer."""
        res = []
        # Categorical variables, and sets without a domain, are not clipped.
        domain = getattr(self.linguistic_variables[0], 'domain', None)
        if domain is not None:
            x = np.clip(x, domain[0], domain[1])

        for fuzzy_set in self.linguistic_variables:
            res.append(fuzzy_set.membership(x))

        return np.array(res)


    def domain(self) -> list[float]:
        """
        Returns the domain of the fuzzy variable.

        Returns:
            list of floats, or None for a categorical variable.
        """
        return self.linguistic_variables[0].domain


    def fuzzy_type(self) -> FUZZY_SETS:
        """
        Returns the fuzzy type of the domain

        Returns:
            the type of the fuzzy set present in the fuzzy variable.
        """
        return self.fs_type
    

    def _permutation_validation(self, mu_A:np.array, mu_B:np.array, p_value_need:float=0.05) -> bool:
        """
        Validates the fuzzy variable using permutation test to check if the fuzzy sets are statistically different.

        Args:
            mu_A: np.array. Memberships of the first fuzzy set.
            mu_B: np.array. Memberships of the second fuzzy set.

        Returns:
            bool. True if the fuzzy sets are statistically different, False otherwise.
        """
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
        """
        Validates the fuzzy variable. Checks that all the fuzzy sets have the same type and domain.

        A categorical variable is valid by definition: it has one crisp fuzzy set per
        category, so there is nothing to check. The properties below are only defined
        for numerical Type-1 variables, over crisp memberships along a numerical
        domain, and an interval or general Type-2 variable answers with one membership
        interval per sample instead of a single value.

        Args:
            X: np.array. Input data to validate the fuzzy variable.

        Returns:
            bool. True if the fuzzy variable is valid, False otherwise.

        Raises:
            NotImplementedError: if the fuzzy variable is numerical and not Type-1.
        """
        if any(fset.shape() == 'categorical' for fset in self.linguistic_variables):
            if verbose:
                print('Fuzzy variable ' + self.name + ' is categorical: valid by definition.')
            return True

        if self.fs_type != FUZZY_SETS.t1:
            raise NotImplementedError(
                'Fuzzy variable validation is only implemented for Type-1 fuzzy sets, '
                f'but {self.name} uses {self.fs_type.name}.')

        if len(self.linguistic_variables) == 0:
            return False
        
        # Get the fuzzy sets memberships
        memberships = self._term_memberships(X)
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

        # Property 4 (a fuzzy set can only grow where the previous one decreases) is not checked:
        # between two samples that straddle a peak, a set looks like it is still growing, so the
        # sampled memberships flag well ordered partitions as violations.

        
        # Property 5: The smallest fuzzy set must be the first one and the biggest fuzzy set must be the last one. The smallest should start at the left of the domain and the biggest should end at the right of the domain.
        cond5 = (self._term_memberships(self[0].domain[0])[0] >= 0.99) and (self._term_memberships(self[0].domain[1])[-1] >= 0.99)
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

        valid = cond1 and cond2 and cond3 and cond5 and cond6

        if verbose and valid:
            print('Fuzzy variable ' + self.name + ' is valid.')

        return valid


    def __str__(self) -> str:
        """
        Returns the name of the fuzzy variable, its type and its parameters.

        Returns:
            string.
        """
        return f'{self.name} ({self.fs_type.name}) - {self.linguistic_variable_names()}'


    def __getitem__(self, item) -> FS:
        """
        Returns the corresponding fs.

        Args:
            item: int. Index of the FS.

        Returns:
            FS. The corresponding FS.
        """
        return self.linguistic_variables[item]


    def __setitem__(self, item: int, elem: FS) -> None:
        """
        Sets the corresponding fs.

        Args:
            item: int. Index of the FS.
            elem: FS. The FS to set.
        """
        self.linguistic_variables[item] = elem
    

    def __iter__(self) -> Generator[FS, None, None]:
        """
        Returns the corresponding fs.

        Args:
            item: int. Index of the FS.

        Returns:
            FS. The corresponding FS.
        """
        for fs in self.linguistic_variables:
            yield fs


    def __len__(self) -> int:
        """
        Returns the number of linguistic variables.

        Returns:
            int. Number of linguistic variables.
        """
        return len(self.linguistic_variables)
    

    def __call__(self, x: np.array) -> list:
        """
        Computes the membership to each of the FS in the fuzzy variables.

        Args:
            x: numeric value or array.

        Returns:
            list of floats. Membership to each of the FS in the fuzzy variables.
        """
        return self.compute_memberships(x)



