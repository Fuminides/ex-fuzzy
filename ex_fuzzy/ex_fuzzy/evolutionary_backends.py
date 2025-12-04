"""
Backend abstraction layer for evolutionary optimization.

This module provides a unified interface for different evolutionary computation backends,
allowing users to choose between pymoo (CPU-based) and EvoX (GPU-accelerated with JAX).

Backends:
    - PyMooBackend: Default CPU-based backend using the pymoo library
    - EvoXBackend: GPU-accelerated backend using EvoX with JAX
    
Usage:
    Users can specify the backend when creating a classifier:
    
    # Using default pymoo backend
    clf = BaseFuzzyRulesClassifier(backend='pymoo')
    
    # Using EvoX with GPU acceleration
    clf = BaseFuzzyRulesClassifier(backend='evox')
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Any
import numpy as np


class EvolutionaryBackend(ABC):
    """Abstract base class for evolutionary optimization backends."""
    
    @abstractmethod
    def optimize(self, problem: Any, n_gen: int, pop_size: int, 
                 random_state: int, verbose: bool, **kwargs) -> dict:
        """
        Run evolutionary optimization.
        
        Args:
            problem: The optimization problem to solve
            n_gen: Number of generations
            pop_size: Population size
            random_state: Random seed
            verbose: Whether to print progress
            **kwargs: Backend-specific parameters
            
        Returns:
            dict with keys:
                - 'X': Best solution found (numpy array)
                - 'F': Best fitness value
                - 'pop': Final population
                - 'algorithm': Algorithm object (backend-specific)
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available (dependencies installed)."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of this backend."""
        pass


class PyMooBackend(EvolutionaryBackend):
    """Backend using pymoo for CPU-based evolutionary optimization."""
    
    def __init__(self):
        self._available = self._check_availability()
    
    def _check_availability(self) -> bool:
        try:
            import pymoo
            return True
        except ImportError:
            return False
    
    def is_available(self) -> bool:
        return self._available
    
    def name(self) -> str:
        return "pymoo"
    
    def optimize(self, problem: Any, n_gen: int, pop_size: int, 
                 random_state: int, verbose: bool, 
                 var_prob: float = 0.3, sbx_eta: float = 3.0, 
                 mutation_eta: float = 7.0, tournament_size: int = 3,
                 sampling: Any = None, **kwargs) -> dict:
        """
        Optimize using pymoo's genetic algorithm.
        
        Args:
            problem: pymoo Problem instance
            n_gen: Number of generations
            pop_size: Population size
            random_state: Random seed
            verbose: Print progress
            var_prob: Crossover probability
            sbx_eta: SBX crossover eta parameter
            mutation_eta: Polynomial mutation eta parameter
            tournament_size: Tournament selection size
            sampling: Initial population sampling strategy
            **kwargs: Additional pymoo-specific parameters
            
        Returns:
            dict with optimization results
        """
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.optimize import minimize
        from pymoo.operators.repair.rounding import RoundingRepair
        from pymoo.operators.sampling.rnd import IntegerRandomSampling
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PolynomialMutation
        
        if sampling is None:
            sampling = IntegerRandomSampling()
        
        algorithm = GA(
            pop_size=pop_size,
            crossover=SBX(prob=var_prob, eta=sbx_eta, repair=RoundingRepair()),
            mutation=PolynomialMutation(eta=mutation_eta, repair=RoundingRepair()),
            tournament_size=tournament_size,
            sampling=sampling,
            eliminate_duplicates=False
        )
        
        res = minimize(
            problem,
            algorithm,
            ('n_gen', n_gen),
            seed=random_state,
            copy_algorithm=False,
            save_history=False,
            verbose=verbose
        )
        
        pop = res.pop
        fitness_last_gen = pop.get('F')
        best_solution = np.argmin(fitness_last_gen)
        best_individual = pop.get('X')[best_solution, :]
        
        return {
            'X': best_individual,
            'F': fitness_last_gen[best_solution],
            'pop': pop,
            'algorithm': res.algorithm,
            'res': res
        }


class EvoXBackend(EvolutionaryBackend):
    """Backend using EvoX for GPU-accelerated evolutionary optimization with JAX."""
    
    def __init__(self):
        self._available = self._check_availability()
        if self._available:
            self._setup_jax()
    
    def _check_availability(self) -> bool:
        try:
            import evox
            import jax
            return True
        except ImportError:
            return False
    
    def _setup_jax(self):
        """Setup JAX configuration for GPU usage."""
        try:
            import jax
            # Try to use GPU if available, otherwise fall back to CPU
            devices = jax.devices()
            self._device = devices[0]
            if self._device.platform == 'gpu':
                print(f"EvoX backend using GPU: {self._device}")
            else:
                print(f"EvoX backend using CPU (GPU not available)")
        except Exception as e:
            print(f"Warning: Could not setup JAX device: {e}")
            self._device = None
    
    def is_available(self) -> bool:
        return self._available
    
    def name(self) -> str:
        return "evox"
    
    def optimize(self, problem: Any, n_gen: int, pop_size: int,
                 random_state: int, verbose: bool,
                 var_prob: float = 0.3, sbx_eta: float = 20.0,
                 mutation_eta: float = 20.0, tournament_size: int = 3,
                 sampling: Any = None, **kwargs) -> dict:
        """
        Optimize using EvoX's genetic algorithm with JAX acceleration.
        
        Args:
            problem: Problem wrapper compatible with EvoX
            n_gen: Number of generations
            pop_size: Population size
            random_state: Random seed for JAX
            verbose: Print progress
            var_prob: Crossover probability
            sbx_eta: SBX crossover distribution index
            mutation_eta: Polynomial mutation distribution index
            tournament_size: Tournament selection size
            sampling: Initial population (numpy array)
            **kwargs: Additional EvoX-specific parameters
            
        Returns:
            dict with optimization results
        """
        import jax
        import jax.numpy as jnp
        from evox import algorithms, problems, workflows
        from evox.operators import mutation, crossover, selection
        
        # Extract problem information
        n_var = problem.n_var
        xl = problem.xl
        xu = problem.xu
        
        # Create JAX random key
        key = jax.random.PRNGKey(random_state)
        
        # Create EvoX problem wrapper
        evox_problem = EvoXProblemWrapper(problem)
        
        # Initialize population
        if sampling is not None:
            if isinstance(sampling, np.ndarray):
                # Initial population provided
                init_pop = jnp.array(sampling, dtype=jnp.int32)
            else:
                # sampling is a pymoo sampler, generate initial population
                key, subkey = jax.random.split(key)
                init_pop = jax.random.randint(
                    subkey, 
                    (pop_size, n_var), 
                    minval=jnp.array(xl, dtype=jnp.int32), 
                    maxval=jnp.array(xu, dtype=jnp.int32) + 1
                )
        else:
            # Random initialization
            key, subkey = jax.random.split(key)
            init_pop = jax.random.randint(
                subkey,
                (pop_size, n_var),
                minval=jnp.array(xl, dtype=jnp.int32),
                maxval=jnp.array(xu, dtype=jnp.int32) + 1
            )
        
        # Create evolutionary operators
        # Note: EvoX uses different operator interfaces than pymoo
        mutation_op = mutation.PolynomialMutation(eta=mutation_eta, prob=1.0/n_var)
        crossover_op = crossover.SimulatedBinaryCrossover(eta=sbx_eta, prob=var_prob)
        selection_op = selection.TournamentSelection(n_round=tournament_size)
        
        # Create a simple GA workflow
        best_solutions = []
        best_fitness = []
        
        population = init_pop
        key, subkey = jax.random.split(key)
        fitness = evox_problem.evaluate(population)
        
        for gen in range(n_gen):
            # Selection
            key, subkey = jax.random.split(key)
            selected_idx = selection_op(subkey, fitness, pop_size)
            selected_pop = population[selected_idx]
            
            # Crossover
            key, subkey = jax.random.split(key)
            offspring = crossover_op(subkey, selected_pop)
            
            # Mutation
            key, subkey = jax.random.split(key)
            offspring = mutation_op(subkey, offspring)
            
            # Clip to bounds
            offspring = jnp.clip(offspring, xl, xu).astype(jnp.int32)
            
            # Evaluate offspring
            offspring_fitness = evox_problem.evaluate(offspring)
            
            # Survival selection (generational replacement)
            population = offspring
            fitness = offspring_fitness
            
            # Track best solution
            best_idx = jnp.argmin(fitness)
            best_solutions.append(population[best_idx])
            best_fitness.append(fitness[best_idx])
            
            if verbose and gen % max(1, n_gen // 10) == 0:
                print(f'Gen {gen:4d} | Best fitness: {float(fitness[best_idx]):.6f} | '
                      f'Avg fitness: {float(jnp.mean(fitness)):.6f}')
        
        # Get final best solution
        best_gen = int(jnp.argmin(jnp.array(best_fitness)))
        best_individual = np.array(best_solutions[best_gen])
        best_fit = float(best_fitness[best_gen])
        
        if verbose:
            print(f'Optimization complete. Best fitness: {best_fit:.6f}')
        
        return {
            'X': best_individual,
            'F': best_fit,
            'pop': np.array(population),
            'fitness': np.array(fitness),
            'algorithm': None,  # EvoX doesn't have a single algorithm object
            'history': {
                'best_solutions': [np.array(s) for s in best_solutions],
                'best_fitness': [float(f) for f in best_fitness]
            }
        }


class EvoXProblemWrapper:
    """Wrapper to adapt ex-fuzzy Problem to EvoX interface."""
    
    def __init__(self, problem):
        self.problem = problem
        self.n_eval = 0
    
    def evaluate(self, population):
        """
        Evaluate a population using the original problem's fitness function.
        
        Args:
            population: JAX array of shape (pop_size, n_var)
            
        Returns:
            JAX array of fitness values
        """
        import jax.numpy as jnp
        
        # Convert JAX array to numpy for compatibility with existing code
        pop_np = np.array(population)
        
        # Evaluate each individual
        fitness = []
        for individual in pop_np:
            out = {}
            self.problem._evaluate(individual, out)
            fitness.append(out['F'])
            self.n_eval += 1
        
        return jnp.array(fitness)


def get_backend(backend_name: str = 'pymoo') -> EvolutionaryBackend:
    """
    Get an evolutionary backend by name.
    
    Args:
        backend_name: Name of the backend ('pymoo' or 'evox')
        
    Returns:
        EvolutionaryBackend instance
        
    Raises:
        ValueError: If backend is not available or unknown
    """
    backends = {
        'pymoo': PyMooBackend,
        'evox': EvoXBackend,
    }
    
    if backend_name not in backends:
        raise ValueError(
            f"Unknown backend '{backend_name}'. Available backends: {list(backends.keys())}"
        )
    
    backend = backends[backend_name]()
    
    if not backend.is_available():
        raise ValueError(
            f"Backend '{backend_name}' is not available. "
            f"Please install required dependencies. "
            f"For EvoX: pip install ex-fuzzy[evox]"
        )
    
    return backend


def list_available_backends() -> list[str]:
    """
    List all available backends.
    
    Returns:
        List of backend names that are currently available
    """
    all_backends = ['pymoo', 'evox']
    available = []
    
    for name in all_backends:
        try:
            backend = get_backend(name)
            if backend.is_available():
                available.append(name)
        except ValueError:
            pass
    
    return available
