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
    
    def optimize_with_checkpoints(self, problem: Any, n_gen: int, pop_size: int,
                                   random_state: int, verbose: bool,
                                   checkpoint_freq: int, checkpoint_callback: Callable,
                                   var_prob: float = 0.3, sbx_eta: float = 3.0,
                                   mutation_eta: float = 7.0, tournament_size: int = 3,
                                   sampling: Any = None, **kwargs) -> dict:
        """
        Optimize with checkpoint callbacks at specified intervals.
        
        Args:
            problem: pymoo Problem instance
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
        from pymoo.algorithms.soo.nonconvex.ga import GA
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
        
        if verbose:
            print('=================================================')
            print('n_gen  |  n_eval  |     f_avg     |     f_min    ')
            print('=================================================')
        
        algorithm.setup(problem, seed=random_state, termination=('n_gen', n_gen))
        
        for k in range(n_gen):
            algorithm.next()
            res = algorithm
            
            if verbose:
                print('%-6s | %-8s | %-8s | %-8s' % (
                    res.n_gen, res.evaluator.n_eval, 
                    res.pop.get('F').mean(), res.pop.get('F').min()
                ))
            
            if k % checkpoint_freq == 0:
                pop = algorithm.pop
                fitness_last_gen = pop.get('F')
                best_solution_arg = np.argmin(fitness_last_gen)
                best_individual = pop.get('X')[best_solution_arg, :]
                
                # Call user-provided checkpoint callback
                checkpoint_callback(k, best_individual)
        
        # Extract final results
        pop = algorithm.pop
        fitness_last_gen = pop.get('F')
        best_solution = np.argmin(fitness_last_gen)
        best_individual = pop.get('X')[best_solution, :]
        
        return {
            'X': best_individual,
            'F': fitness_last_gen[best_solution],
            'pop': pop,
            'algorithm': algorithm,
            'res': algorithm
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
        import jax
        devices = jax.devices()
        self._device = devices[0]
        
        if self._device.platform == 'gpu':
            print(f"EvoX backend using GPU: {self._device}")
        else:
            print(f"EvoX backend using CPU (GPU not available)")
                
        
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
        
        # Create problem wrapper (CPU evaluation for now - PyTorch-based GPU eval would need different approach)
        evox_problem = EvoXProblemWrapper(problem, use_gpu_eval=False)
        
        # Initialize population
        if sampling is not None:
            if isinstance(sampling, np.ndarray):
                # Initial population provided
                init_pop = torch.tensor(sampling, dtype=torch.int32, device=device)
            else:
                # Random initialization
                init_pop = torch.randint(
                    low=int(xl[0]), 
                    high=int(xu[0]) + 1,
                    size=(pop_size, n_var), 
                    dtype=torch.int32,
                    device=device
                )
        else:
            # Random initialization
            init_pop = torch.randint(
                low=int(xl[0]),
                high=int(xu[0]) + 1,
                size=(pop_size, n_var),
                dtype=torch.int32,
                device=device
            )
        
        # Create a simple GA workflow
        best_solutions = []
        best_fitness = []
        
        # Convert bounds to PyTorch tensors for mutation
        lb_torch = torch.tensor(xl, dtype=torch.float32, device=device)
        ub_torch = torch.tensor(xu, dtype=torch.float32, device=device)
        
        population = init_pop.float()  # Convert to float for operators
        
        # Initial evaluation
        fitness_list = []
        for ind in population:
            out = {}
            self.problem._evaluate(ind.cpu().numpy().astype(int), out)
            fitness_list.append(out['F'])
        fitness = torch.tensor(fitness_list, dtype=torch.float32, device=device)
        
        for gen in range(n_gen):
            # Selection - tournament selection
            selected_idx = []
            for _ in range(pop_size):
                candidates = torch.randperm(pop_size, device=device)[:tournament_size]
                best_in_tournament = candidates[torch.argmin(fitness[candidates])]
                selected_idx.append(best_in_tournament)
            selected_idx = torch.tensor(selected_idx, device=device)
            selected_pop = population[selected_idx]
            
            # Crossover using EvoX simulated_binary function
            offspring = crossover.simulated_binary(selected_pop, pro_c=var_prob, dis_c=sbx_eta)
            
            # Mutation using EvoX polynomial_mutation function
            offspring = mutation.polynomial_mutation(offspring, lb=lb_torch, ub=ub_torch, pro_m=1.0/n_var, dis_m=mutation_eta)
            
            # Clip to bounds and convert to int
            offspring = torch.clamp(offspring, lb_torch[0], ub_torch[0]).int()
            
            # Evaluate offspring
            offspring_fitness_list = []
            for ind in offspring:
                out = {}
                problem._evaluate(ind.cpu().numpy(), out)
                offspring_fitness_list.append(out['F'])
            offspring_fitness = torch.tensor(offspring_fitness_list, dtype=torch.float32, device=device)
            
            # Survival selection (generational replacement)
            population = offspring.float()
            fitness = offspring_fitness
            
            # Track best solution
            best_idx = torch.argmin(fitness)
            best_solutions.append(population[best_idx].cpu().numpy())
            best_fitness.append(float(fitness[best_idx]))
            
            if verbose and gen % max(1, n_gen // 10) == 0:
                print(f'Gen {gen:4d} | Best fitness: {fitness[best_idx]:.6f} | '
                      f'Avg fitness: {torch.mean(fitness):.6f}')
        
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
            }
        }


class EvoXProblemWrapper:
    """Wrapper to adapt ex-fuzzy Problem to EvoX interface."""
    
    def __init__(self, problem, use_gpu_eval: bool = True):
        self.problem = problem
        self.n_eval = 0
        self.use_gpu_eval = use_gpu_eval
        self._jax_eval_func = None
        
        # Try to create vectorized GPU evaluation if possible
        if use_gpu_eval:
            try:
                self._setup_gpu_evaluation()
            except Exception as e:
                print(f"⚠️  Could not setup GPU evaluation: {e}")
                print("   Falling back to CPU evaluation")
                self.use_gpu_eval = False
    
    def _setup_gpu_evaluation(self):
        """Setup vectorized GPU evaluation using JAX."""
        import jax
        import jax.numpy as jnp
        
        # Create JAX evaluation function for FitRuleBase
        problem_class_name = self.problem.__class__.__name__
        
        if problem_class_name == 'FitRuleBase':
            self._jax_eval_func = self._create_fitrulebase_jax_evaluate()
            if self._jax_eval_func is not None:
                # Vectorize the JAX evaluation function for batch processing
                self._vmap_eval = jax.vmap(self._jax_eval_func)
                print(f"✅ GPU-accelerated evaluation enabled for {problem_class_name}")
            else:
                raise RuntimeError(f"Could not create JAX evaluation for {problem_class_name}")
        else:
            # Fallback for unsupported problem types
            print(f"⚠️  GPU evaluation not implemented for {problem_class_name}")
            print("   Falling back to CPU evaluation")
            raise RuntimeError(f"GPU evaluation only supports FitRuleBase, got {problem_class_name}")
    
    def _create_fitrulebase_jax_evaluate(self):
        """Create JAX evaluation function for FitRuleBase problem."""
        import jax
        import jax.numpy as jnp
        
        # Capture necessary data and parameters in closure
        X = self.problem.X
        y = self.problem.y
        tolerance = self.problem.tolerance
        alpha = self.problem.alpha_
        beta = self.problem.beta_
        precomputed_truth = self.problem._precomputed_truth
        fuzzy_type = self.problem.fuzzy_type
        n_classes = self.problem.n_classes
        nRules = self.problem.nRules
        nAnts = self.problem.nAnts
        lvs = self.problem.lvs
        n_lv_possible = self.problem.n_lv_possible
        ds_mode = self.problem.ds_mode
        
        def jax_evaluate(x):
            """
            JAX-native evaluation function using pure tensor operations.
            Computes fitness directly from gene without constructing rule objects.
            """
            # Convert to JAX array and int
            x_jax = jnp.array(x, dtype=jnp.int32)
            
            try:
                # ============= JAX-NATIVE EVALUATION (no object construction) =============
                # Convert data to JAX arrays
                X_jax = jnp.array(X, dtype=jnp.float32)
                y_jax = jnp.array(y, dtype=jnp.int32)
                n_samples = X_jax.shape[0]
                n_features = X_jax.shape[1]
                
                # Step 1: Decode gene structure
                rule_features = x_jax[:nRules * nAnts].reshape(nRules, nAnts)
                rule_labels = x_jax[nRules * nAnts:2 * nRules * nAnts].reshape(nRules, nAnts)
                
                # Calculate fourth pointer (consequents position)
                if lvs is None:
                    fourth_pointer = 2 * nAnts * nRules
                else:
                    fourth_pointer = 2 * nAnts * nRules
                
                # Extract consequents
                rule_consequents = x_jax[fourth_pointer:fourth_pointer + nRules]
                
                # Extract weights if ds_mode == 2
                if ds_mode == 2:
                    fifth_pointer = fourth_pointer + nRules
                    rule_weights = x_jax[fifth_pointer:fifth_pointer + nRules] / 100.0
                else:
                    rule_weights = jnp.ones(nRules)
                
                # Step 2: Build membership tensor from precomputed truth
                if precomputed_truth is not None and lvs is not None:
                    max_lvs = max([len(lv) for lv in lvs])
                    
                    # Build membership matrix: (n_samples, n_features, max_linguistic_vars)
                    membership_matrix = []
                    for sample_idx in range(n_samples):
                        sample_memberships = []
                        for feat_idx in range(n_features):
                            truth_dict = precomputed_truth[sample_idx]
                            if feat_idx in truth_dict:
                                mems = list(truth_dict[feat_idx])
                                mems = mems + [0.0] * (max_lvs - len(mems))
                            else:
                                mems = [0.0] * max_lvs
                            sample_memberships.append(mems)
                        membership_matrix.append(sample_memberships)
                    
                    membership_tensor = jnp.array(membership_matrix, dtype=jnp.float32)
                else:
                    # No precomputed truth - cannot proceed with JAX-only evaluation
                    return jnp.array(1.0)
                
                # Step 3: Compute rule firing strengths using vectorized operations
                def compute_rule_firing(rule_idx):
                    """Compute firing strength for a single rule across all samples."""
                    features = rule_features[rule_idx]
                    labels = rule_labels[rule_idx]
                    
                    def get_antecedent_membership(ant_idx):
                        feat_idx = features[ant_idx]
                        label_idx = labels[ant_idx]
                        return jnp.where(
                            label_idx >= 0,
                            membership_tensor[:, feat_idx, label_idx],
                            jnp.ones(n_samples)
                        )
                    
                    ant_memberships = jax.vmap(get_antecedent_membership)(jnp.arange(nAnts))
                    firing_strength = jnp.min(ant_memberships, axis=0)
                    return firing_strength
                
                # Vectorize over all rules
                all_firing_strengths = jax.vmap(compute_rule_firing)(jnp.arange(nRules))
                
                # Step 4: Apply rule weights
                weighted_firing = all_firing_strengths * rule_weights[:, jnp.newaxis]
                
                # Step 5: Aggregate by consequent class
                class_activations = jnp.zeros((n_classes, n_samples))
                for class_idx in range(n_classes):
                    class_mask = (rule_consequents == class_idx)
                    class_activations = class_activations.at[class_idx].set(
                        jnp.sum(weighted_firing * class_mask[:, jnp.newaxis], axis=0)
                    )
                
                # Step 6: Make predictions
                predictions = jnp.argmax(class_activations, axis=0)
                
                # Step 7: Compute accuracy
                correct = jnp.sum(predictions == y_jax)
                accuracy = correct / n_samples
                
                # Step 8: Compute fitness
                fitness = 1.0 - accuracy
                
                return jnp.array(fitness)
                
            except Exception as e:
                return jnp.array(1.0)
        
        return jax_evaluate
    
    def evaluate(self, population):
        """
        Evaluate a population using the original problem's fitness function.
        
        Args:
            population: JAX array of shape (pop_size, n_var)
            
        Returns:
            JAX array of fitness values
        """
        import jax.numpy as jnp
        
        if self.use_gpu_eval:
            
            # Try GPU-accelerated vectorized evaluation
            fitness = self._vmap_eval(population)
            self.n_eval += len(population)
            return fitness

        # CPU evaluation (fallback or when GPU disabled)
        pop_np = np.array(population)
        
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
