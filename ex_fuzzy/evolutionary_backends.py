"""
Backend abstraction layer for evolutionary optimization.

This module provides a unified interface for different evolutionary computation backends,
allowing users to choose between pymoo (CPU-based) and EvoX (GPU-accelerated with PyTorch).

Backends:
    - PyMooBackend: Default CPU-based backend using the pymoo library
    - EvoXBackend: GPU-accelerated backend using EvoX with PyTorch
    
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

from ._problem import PYMOO_INSTALL_MESSAGE, as_pymoo_problem


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

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available (dependencies installed)."""

    @abstractmethod
    def name(self) -> str:
        """Return the name of this backend."""

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'


class PyMooBackend(EvolutionaryBackend):
    """Backend using pymoo for CPU-based evolutionary optimization."""

    def is_available(self) -> bool:
        # pymoo is a required dependency of ex-fuzzy.
        return True
    
    def name(self) -> str:
        return "pymoo"

    def _build_ga_algorithm(self, pop_size: int, var_prob: float, sbx_eta: float,
                            mutation_eta: float, tournament_size: int,
                            sampling: Any):
        """Create a configured pymoo GA instance."""
        try:
            from pymoo.algorithms.soo.nonconvex.ga import GA
            from pymoo.operators.repair.rounding import RoundingRepair
            from pymoo.operators.sampling.rnd import IntegerRandomSampling
            from pymoo.operators.crossover.sbx import SBX
            from pymoo.operators.mutation.pm import PolynomialMutation
        except ImportError as error:
            raise ImportError(PYMOO_INSTALL_MESSAGE) from error

        if sampling is None:
            sampling = IntegerRandomSampling()

        return GA(
            pop_size=pop_size,
            crossover=SBX(prob=var_prob, eta=sbx_eta, repair=RoundingRepair()),
            mutation=PolynomialMutation(eta=mutation_eta, repair=RoundingRepair()),
            tournament_size=tournament_size,
            sampling=sampling,
            eliminate_duplicates=False
        )

    def _run_ga_loop(self, problem: Any, algorithm: Any, n_gen: int,
                     random_state: int, verbose: bool,
                     checkpoint_freq: Optional[int] = None,
                     checkpoint_callback: Optional[Callable] = None,
                     patience: Optional[int] = None,
                     min_delta: float = 0.0) -> dict:
        """Run a pymoo GA loop with optional checkpoints and early stopping."""
        algorithm.setup(problem, seed=random_state, termination=('n_gen', n_gen))

        if verbose:
            print('=================================================')
            print('n_gen  |  n_eval  |     f_avg     |     f_min    ')
            print('=================================================')

        best_individual = None
        best_fitness = None
        generations_without_improvement = 0
        patience_enabled = patience is not None and patience > 0
        min_delta = max(0.0, float(min_delta))
        executed_generations = 0

        for gen in range(n_gen):
            algorithm.next()
            executed_generations = gen + 1
            pop = algorithm.pop
            fitness_last_gen = pop.get('F').reshape(-1)
            best_solution_arg = int(np.argmin(fitness_last_gen))
            current_best_fitness = float(fitness_last_gen[best_solution_arg])
            current_best_individual = pop.get('X')[best_solution_arg, :].copy()

            if verbose:
                print('%-6s | %-8s | %-8s | %-8s' % (
                    algorithm.n_gen, algorithm.evaluator.n_eval,
                    float(np.mean(fitness_last_gen)), current_best_fitness
                ))

            if best_fitness is None or current_best_fitness < (best_fitness - min_delta):
                best_fitness = current_best_fitness
                best_individual = current_best_individual
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if checkpoint_freq is not None and checkpoint_callback is not None and gen % checkpoint_freq == 0:
                checkpoint_callback(gen, best_individual.copy())

            if patience_enabled and generations_without_improvement >= patience:
                if verbose:
                    print(
                        f"Early stopping at generation {executed_generations} "
                        f"(no improvement larger than {min_delta} for {patience} generations)."
                    )
                break

        pop = algorithm.pop

        return {
            'X': best_individual,
            'F': best_fitness,
            'pop': pop,
            'algorithm': algorithm,
            'res': algorithm,
            'n_gen_run': executed_generations,
            'stopped_early': executed_generations < n_gen
        }
    
    def optimize(self, problem: Any, n_gen: int, pop_size: int, 
                 random_state: int, verbose: bool, 
                 var_prob: float = 0.3, sbx_eta: float = 3.0, 
                 mutation_eta: float = 7.0, tournament_size: int = 3,
                 sampling: Any = None, patience: Optional[int] = 10,
                 min_delta: float = 1e-4, **kwargs) -> dict:
        """
        Optimize using pymoo's genetic algorithm.
        
        Args:
            problem: Ex-Fuzzy problem, wrapped for pymoo here, or a pymoo Problem
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
        algorithm = self._build_ga_algorithm(
            pop_size=pop_size,
            var_prob=var_prob,
            sbx_eta=sbx_eta,
            mutation_eta=mutation_eta,
            tournament_size=tournament_size,
            sampling=sampling
        )

        return self._run_ga_loop(
            problem=as_pymoo_problem(problem),
            algorithm=algorithm,
            n_gen=n_gen,
            random_state=random_state,
            verbose=verbose,
            patience=patience,
            min_delta=min_delta
        )
    
    def optimize_with_checkpoints(self, problem: Any, n_gen: int, pop_size: int,
                                   random_state: int, verbose: bool,
                                   checkpoint_freq: int, checkpoint_callback: Callable,
                                   var_prob: float = 0.3, sbx_eta: float = 3.0,
                                   mutation_eta: float = 7.0, tournament_size: int = 3,
                                   sampling: Any = None, patience: Optional[int] = 10,
                                   min_delta: float = 1e-4, **kwargs) -> dict:
        """
        Optimize with checkpoint callbacks at specified intervals.
        
        Args:
            problem: Ex-Fuzzy problem, wrapped for pymoo here, or a pymoo Problem
            n_gen: Number of generations
            pop_size: Population size
            random_state: Random seed
            verbose: Print progress
            checkpoint_freq: Call checkpoint_callback every N generations
            checkpoint_callback: Callable(gen: int, best_individual: np.array) to call at checkpoints
            var_prob: Crossover probability
            sbx_eta: SBX crossover eta parameter
            mutation_eta: Polynomial mutation eta parameter
            tournament_size: Tournament selection size
            sampling: Initial population sampling strategy
            **kwargs: Additional parameters
            
        Returns:
            dict with optimization results
        """
        algorithm = self._build_ga_algorithm(
            pop_size=pop_size,
            var_prob=var_prob,
            sbx_eta=sbx_eta,
            mutation_eta=mutation_eta,
            tournament_size=tournament_size,
            sampling=sampling
        )

        return self._run_ga_loop(
            problem=as_pymoo_problem(problem),
            algorithm=algorithm,
            n_gen=n_gen,
            random_state=random_state,
            verbose=verbose,
            checkpoint_freq=checkpoint_freq,
            checkpoint_callback=checkpoint_callback,
            patience=patience,
            min_delta=min_delta
        )


class EvoXBackend(EvolutionaryBackend):
    """Backend using EvoX for GPU-accelerated evolutionary optimization with PyTorch."""
    
    def __init__(self):
        self._available = self._check_availability()
        if self._available:
            self._setup_pytorch()
    
    def _check_availability(self) -> bool:
        try:
            import evox
            import torch
            return True
        except Exception:
            # Catch all exceptions: ImportError, RuntimeError (e.g., torch.compile
            # not supported on Python 3.14+), or any other initialization errors
            return False
    
    def _setup_pytorch(self):
        """Setup PyTorch configuration for GPU usage."""
        import torch
        
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
            print(f"EvoX backend using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self._device = torch.device('cpu')
            print(f"EvoX backend using CPU (GPU not available)")
                
        
    def is_available(self) -> bool:
        return self._available
    
    def name(self) -> str:
        return "evox"
    
    

    def _evaluate_population(self, population: 'torch.Tensor', problem: Any,
                             device: 'torch.device') -> 'torch.Tensor':
        """
        Evaluate a population on ``device`` using the fastest available path.

        Problems can expose ``_evaluate_torch_population`` when their complete
        objective is implemented as a batched PyTorch operation.  Regression
        uses this hook so membership lookup, inference, and R-squared scoring
        remain on the GPU.  Problems exposing ``_evaluate_gene_population``
        receive the whole generation as one integer array and may score it on
        ``device`` themselves; classification uses it for its fitness caches,
        population batching and exact device objective.  Other problems are
        evaluated one individual at a time.  Either way the population leaves
        the device once per generation, not once per individual.
        """
        import torch

        on_device = False
        if hasattr(problem, '_evaluate_torch_population'):
            fitness = problem._evaluate_torch_population(population, device=device)
        else:
            genes = population.detach().cpu().numpy().astype(int)
            if hasattr(problem, '_evaluate_gene_population'):
                fitness_values, on_device = problem._evaluate_gene_population(
                    genes, device=device)
            else:
                fitness_values = []
                for gene in genes:
                    out = {}
                    problem._evaluate(gene, out)
                    fitness_values.append(float(np.asarray(out['F']).reshape(-1)[0]))
            fitness = torch.tensor(np.asarray(fitness_values, dtype=float),
                                   dtype=torch.float32, device=device)
        self._fitness_on_device = getattr(self, '_fitness_on_device', False) or on_device

        if not isinstance(fitness, torch.Tensor):
            fitness = torch.as_tensor(fitness, dtype=torch.float32, device=device)
        return fitness.to(device=device, dtype=torch.float32).reshape(-1)
    
    def optimize(self, problem: Any, n_gen: int, pop_size: int,
                 random_state: int, verbose: bool,
                 var_prob: float = 0.3, sbx_eta: float = 20.0,
                 mutation_eta: float = 20.0, tournament_size: int = 3,
                 sampling: Any = None, patience: Optional[int] = 10,
                 min_delta: float = 1e-4, **kwargs) -> dict:
        """
        Optimize using EvoX's genetic algorithm with PyTorch backend.
        
        Args:
            problem: Problem wrapper compatible with EvoX
            n_gen: Number of generations
            pop_size: Population size
            random_state: Random seed
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
        import torch
        from evox.operators import mutation, crossover
        
        # Extract problem information
        n_var = problem.n_var
        xl = problem.xl
        xu = problem.xu
        
        # Set random seed for PyTorch
        torch.manual_seed(random_state)
        
        # Get device (GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize population
        if isinstance(sampling, np.ndarray):
            # Initial population provided
            init_pop = torch.tensor(sampling, dtype=torch.int32, device=device)
        else:
            # Random initialization with per-variable bounds
            init_pop = torch.zeros((pop_size, n_var), dtype=torch.int32, device=device)
            for var_idx in range(n_var):
                init_pop[:, var_idx] = torch.randint(
                    low=int(xl[var_idx]),
                    high=int(xu[var_idx]) + 1,
                    size=(pop_size,),
                    dtype=torch.int32,
                    device=device
                )
        
        # Create a simple GA workflow with elitism
        best_solutions = []
        best_fitness = []
        generations_without_improvement = 0
        patience_enabled = patience is not None and patience > 0
        min_delta = max(0.0, float(min_delta))
        
        # Convert bounds to PyTorch tensors for mutation
        lb_torch = torch.tensor(xl, dtype=torch.float32, device=device)
        ub_torch = torch.tensor(xu, dtype=torch.float32, device=device)
        
        population = init_pop  # Keep as integers
        
        uses_torch_fitness = hasattr(problem, '_evaluate_torch_population')
        self._fitness_on_device = False

        # Initial evaluation
        fitness = self._evaluate_population(population, problem, device)
        
        for gen in range(n_gen):
            # Selection - VECTORIZED tournament selection (select pop_size parents for mating)
            # Generate all random tournaments at once
            tournament_candidates = torch.randint(0, pop_size, (pop_size, tournament_size), device=device)
            tournament_fitness = fitness[tournament_candidates]  # (pop_size, tournament_size)
            selected_idx = tournament_candidates[torch.arange(pop_size, device=device), torch.argmin(tournament_fitness, dim=1)]
            selected_pop = population[selected_idx].float()
            
            # Crossover using EvoX simulated_binary function
            offspring = crossover.simulated_binary(selected_pop, pro_c=var_prob, dis_c=sbx_eta)
            
            # Mutation using EvoX polynomial_mutation function
            offspring = mutation.polynomial_mutation(offspring, lb=lb_torch, ub=ub_torch, pro_m=1.0/n_var, dis_m=mutation_eta)
            
            # VECTORIZED clipping to bounds (per-variable) and round to integers (repair)
            offspring = torch.clamp(offspring, lb_torch.unsqueeze(0), ub_torch.unsqueeze(0))
            offspring = torch.round(offspring).int()
            
            # Evaluate offspring
            offspring_fitness = self._evaluate_population(offspring, problem, device)
            
            # Elitist survival selection: combine parents and offspring, select best pop_size
            combined_pop = torch.cat([population, offspring], dim=0)
            combined_fitness = torch.cat([fitness, offspring_fitness], dim=0)
            
            # Select best pop_size individuals
            sorted_indices = torch.argsort(combined_fitness)[:pop_size]
            population = combined_pop[sorted_indices]
            fitness = combined_fitness[sorted_indices]
            
            # Track best solution
            best_idx = torch.argmin(fitness)
            best_solutions.append(population[best_idx].cpu().numpy())
            best_fitness.append(float(fitness[best_idx]))

            if len(best_fitness) == 1 or best_fitness[-1] < (min(best_fitness[:-1]) - min_delta):
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            if verbose and gen % max(1, n_gen // 10) == 0:
                print(f'Gen {gen:4d} | Best fitness: {fitness[best_idx]:.6f} | '
                      f'Avg fitness: {torch.mean(fitness):.6f}')

            if patience_enabled and generations_without_improvement >= patience:
                if verbose:
                    print(
                        f"Early stopping at generation {gen + 1} "
                        f"(no improvement larger than {min_delta} for {patience} generations)."
                    )
                break
        
        # Get final best solution
        best_gen = int(np.argmin(best_fitness))
        best_individual = best_solutions[best_gen].astype(int)
        best_fit = best_fitness[best_gen]
        
        if verbose:
            print(f'Optimization complete. Best fitness: {best_fit:.6f}')
        
        return {
            'X': best_individual,
            'F': best_fit,
            'pop': population.cpu().numpy(),
            'fitness': fitness.cpu().numpy(),
            'algorithm': None,  # EvoX doesn't have a single algorithm object
            'history': {
                'best_solutions': best_solutions,
                'best_fitness': best_fitness
            },
            'n_gen_run': len(best_fitness),
            'stopped_early': len(best_fitness) < n_gen,
            'device': str(device),
            'gpu_accelerated': device.type == 'cuda' and (uses_torch_fitness
                                                          or self._fitness_on_device)
        }


def get_backend(backend_name: str = 'pymoo') -> EvolutionaryBackend:
    """
    Get an evolutionary backend by name.
    
    Args:
        backend_name: Name of the backend ('pymoo' or 'evox'), or an
            EvolutionaryBackend instance, which is returned as is.

    Returns:
        EvolutionaryBackend instance

    Raises:
        ValueError: If backend is not available or unknown
    """
    if isinstance(backend_name, EvolutionaryBackend):
        return backend_name

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
            get_backend(name)  # Raises when the backend cannot be used.
        except ValueError:
            continue
        available.append(name)
    
    return available
