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
        except ImportError:
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
    
    def _compute_mcc_torch(self, y_pred: 'torch.Tensor', y_true: 'torch.Tensor') -> float:
        """
        Compute Matthews Correlation Coefficient on GPU using PyTorch.
        
        :param y_pred: predicted labels (tensor)
        :param y_true: true labels (tensor)
        :return: MCC value as Python float
        """
        import torch
        
        # Ensure tensors are on same device and type
        y_pred = y_pred.long()
        y_true = y_true.long()
        
        # Get unique classes
        classes = torch.unique(torch.cat([y_true, y_pred]))
        n_classes = len(classes)
        
        if n_classes == 1:
            return 0.0
        
        # Compute confusion matrix on GPU
        confusion = torch.zeros((n_classes, n_classes), dtype=torch.float32, device=y_pred.device)
        for i, c in enumerate(classes):
            for j, k in enumerate(classes):
                confusion[i, j] = torch.sum((y_true == c) & (y_pred == k)).float()
        
        # Compute MCC components
        t_sum = confusion.sum()
        pred_sum = confusion.sum(dim=0)
        true_sum = confusion.sum(dim=1)
        
        # MCC numerator: sum of correct predictions * total - sum of (row_sum * col_sum)
        diag_sum = torch.trace(confusion)
        cov_ytyp = diag_sum * t_sum - torch.sum(pred_sum * true_sum)
        
        # MCC denominator
        cov_ypyp = t_sum * t_sum - torch.sum(pred_sum * pred_sum)
        cov_ytyt = t_sum * t_sum - torch.sum(true_sum * true_sum)
        
        if cov_ypyp == 0 or cov_ytyt == 0:
            return 0.0
        
        mcc = cov_ytyp / torch.sqrt(cov_ypyp * cov_ytyt)
        return float(mcc.cpu().item())
    
    def _batch_evaluate_torch(self, population: 'torch.Tensor', problem: Any, device: 'torch.device') -> 'torch.Tensor':
        """
        Batch evaluate population using PyTorch with GPU-accelerated MCC computation.
        
        :param population: population tensor (pop_size, n_var)
        :param problem: problem instance with _evaluate_torch_fast method
        :param device: torch device
        :return: fitness tensor (pop_size,)
        """
        import torch
        
        # TRUE BATCHED EVALUATION - evaluate entire population at once
        if hasattr(problem, '_evaluate_torch_batch'):
            # Use fully batched evaluation if available
            fitness = problem._evaluate_torch_batch(population, problem.y, problem.fuzzy_type, device=device)
            return 1.0 - fitness  # Convert MCC to minimization
        else:
            # Fallback to individual evaluation (slower but works)
            fitness_list = []
            for ind in population:
                # Get predictions and true labels as tensors
                y_pred, y_true = problem._evaluate_torch_fast(
                    ind, problem.y, problem.fuzzy_type, device=device, return_tensor=True
                )
                # Compute MCC on GPU
                mcc = self._compute_mcc_torch(y_pred, y_true)
                fitness_list.append(1 - mcc)  # Convert MCC to minimization objective
            
            return torch.tensor(fitness_list, dtype=torch.float32, device=device)
    
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
        
        # Convert bounds to PyTorch tensors for mutation
        lb_torch = torch.tensor(xl, dtype=torch.float32, device=device)
        ub_torch = torch.tensor(xu, dtype=torch.float32, device=device)
        
        population = init_pop  # Keep as integers
        
        # Check if problem has torch evaluation method
        has_torch_eval = hasattr(problem, '_evaluate_torch_fast')
        
        # Initial evaluation
        if has_torch_eval:
            # Use PyTorch evaluation for GPU acceleration with batched MCC computation
            fitness = self._batch_evaluate_torch(population, problem, device)
        else:
            # Fallback to numpy evaluation
            fitness_list = []
            for ind in population:
                out = {}
                problem._evaluate(ind.cpu().numpy().astype(int), out)
                fitness_list.append(out['F'])
            fitness = torch.tensor(fitness_list, dtype=torch.float32, device=device)
        
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
            if has_torch_eval:
                # Use PyTorch evaluation for GPU acceleration with batched MCC computation
                offspring_fitness = self._batch_evaluate_torch(offspring, problem, device)
            else:
                # Fallback to numpy evaluation
                offspring_fitness_list = []
                for ind in offspring:
                    out = {}
                    problem._evaluate(ind.cpu().numpy().astype(int), out)
                    offspring_fitness_list.append(out['F'])
                offspring_fitness = torch.tensor(offspring_fitness_list, dtype=torch.float32, device=device)
            
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
        """Setup vectorized GPU evaluation using PyTorch."""
        import torch
        
        # Create PyTorch evaluation function for FitRuleBase
        problem_class_name = self.problem.__class__.__name__
        
        if problem_class_name == 'FitRuleBase':
            self._torch_eval_func = self._create_fitrulebase_torch_evaluate()
            if self._torch_eval_func is not None:
                # Vectorize the PyTorch evaluation function for batch processing
                self._vmap_eval = torch.vmap(self._torch_eval_func)
                print(f"✅ GPU-accelerated evaluation enabled for {problem_class_name}")
            else:
                raise RuntimeError(f"Could not create PyTorch evaluation for {problem_class_name}")
        else:
            # Fallback for unsupported problem types
            print(f"⚠️  GPU evaluation not implemented for {problem_class_name}")
            print("   Falling back to CPU evaluation")
            raise RuntimeError(f"GPU evaluation only supports FitRuleBase, got {problem_class_name}")
    
    def _create_fitrulebase_torch_evaluate(self):
        """Create PyTorch evaluation function for FitRuleBase problem."""
        import torch
        
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
        
        def torch_evaluate(x):
            """
            PyTorch-native evaluation function using pure tensor operations.
            Computes fitness directly from gene without constructing rule objects.
            """
            # Convert to PyTorch tensor and int
            x_torch = torch.tensor(x, dtype=torch.int32)
            
            try:
                # ============= PYTORCH-NATIVE EVALUATION (no object construction) =============
                # Convert data to PyTorch tensors
                X_torch = torch.tensor(X, dtype=torch.float32)
                y_torch = torch.tensor(y, dtype=torch.int32)
                n_samples = X_torch.shape[0]
                n_features = X_torch.shape[1]
                
                # Step 1: Decode gene structure
                rule_features = x_torch[:nRules * nAnts].reshape(nRules, nAnts)
                rule_labels = x_torch[nRules * nAnts:2 * nRules * nAnts].reshape(nRules, nAnts)
                
                # Calculate fourth pointer (consequents position)
                if lvs is None:
                    fourth_pointer = 2 * nAnts * nRules
                else:
                    fourth_pointer = 2 * nAnts * nRules
                
                # Extract consequents
                rule_consequents = x_torch[fourth_pointer:fourth_pointer + nRules]
                
                # Extract weights if ds_mode == 2
                if ds_mode == 2:
                    fifth_pointer = fourth_pointer + nRules
                    rule_weights = x_torch[fifth_pointer:fifth_pointer + nRules] / 100.0
                else:
                    rule_weights = torch.ones(nRules)
                
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
                    
                    membership_tensor = torch.tensor(membership_matrix, dtype=torch.float32)
                else:
                    # No precomputed truth - cannot proceed with PyTorch-only evaluation
                    return torch.tensor(1.0)
                
                # Step 3: Compute rule firing strengths using vectorized operations
                def compute_rule_firing(rule_idx):
                    """Compute firing strength for a single rule across all samples."""
                    features = rule_features[rule_idx]
                    labels = rule_labels[rule_idx]
                    
                    def get_antecedent_membership(ant_idx):
                        feat_idx = features[ant_idx]
                        label_idx = labels[ant_idx]
                        return torch.where(
                            label_idx >= 0,
                            membership_tensor[:, feat_idx, label_idx],
                            torch.ones(n_samples)
                        )
                    
                    ant_memberships = torch.vmap(get_antecedent_membership)(torch.arange(nAnts))
                    firing_strength = torch.min(ant_memberships, dim=0)[0]
                    return firing_strength
                
                # Vectorize over all rules
                all_firing_strengths = torch.vmap(compute_rule_firing)(torch.arange(nRules))
                
                # Step 4: Apply rule weights
                weighted_firing = all_firing_strengths * rule_weights[:, None]
                
                # Step 5: Aggregate by consequent class
                class_activations = torch.zeros((n_classes, n_samples))
                for class_idx in range(n_classes):
                    class_mask = (rule_consequents == class_idx)
                    class_activations[class_idx] = torch.sum(
                        weighted_firing * class_mask[:, None], dim=0
                    )
                
                # Step 6: Make predictions
                predictions = torch.argmax(class_activations, dim=0)
                
                # Step 7: Compute accuracy
                correct = torch.sum(predictions == y_torch)
                accuracy = correct / n_samples
                
                # Step 8: Compute fitness
                fitness = 1.0 - accuracy
                
                return torch.tensor(fitness)
                
            except Exception as e:
                return torch.tensor(1.0)
        
        return torch_evaluate
    
    def evaluate(self, population):
        """
        Evaluate a population using the original problem's fitness function.
        
        Args:
            population: PyTorch tensor of shape (pop_size, n_var)
            
        Returns:
            PyTorch tensor of fitness values
        """
        import torch
        
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
        
        return torch.tensor(fitness)


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
