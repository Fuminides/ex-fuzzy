"""Optimizer-independent problem base for the genetic searches.

The fitting problems used to subclass pymoo's ``Problem``, so importing
Ex-Fuzzy imported pymoo even for EvoX fits that never use it.  This module
reproduces the part of that interface the problems and their callers rely on:
the problem attributes, elementwise runners and ``evaluate``.  It imports no
optimizer.

:func:`as_pymoo_problem` wraps a problem in a genuine pymoo problem, importing
pymoo only then.  The wrapper delegates every evaluation to the wrapped
problem, so a PyMoo search runs exactly as it did when the problems subclassed
pymoo directly.
"""
from typing import Any, Union

import numpy as np


class LoopedElementwiseEvaluation:
    """Default elementwise runner: evaluate each chromosome in turn."""

    def __call__(self, f, X):
        return [f(x) for x in X]


class ElementwiseEvaluationFunction:
    """Evaluate one chromosome through ``problem._evaluate``."""

    def __init__(self, problem, args, kwargs) -> None:
        self.problem = problem
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        out = dict()
        self.problem._evaluate(x, out, *self.args, **self.kwargs)
        return out


class StarmapParallelization:
    """Elementwise runner over a starmap function such as ``ThreadPool.starmap``.

    Interchangeable with pymoo's runner of the same name.  The starmap itself is
    not serialized, since pools cannot be pickled.
    """

    def __init__(self, starmap) -> None:
        self.starmap = starmap

    def __call__(self, f, X):
        return list(self.starmap(f, [[x] for x in X]))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("starmap", None)
        return state


class Integer:
    """A bounded integer decision variable; pymoo's equivalent is built on demand."""

    vtype = int

    def __init__(self, bounds=(None, None)) -> None:
        self.bounds = bounds


def _at_least_2d(X: np.ndarray) -> tuple:
    if X.ndim == 1:
        return X[None, :], True
    return X, False


class Problem:
    """Single-objective problem description shared by every optimizer backend.

    Attribute names, defaults and evaluation semantics follow pymoo's
    ``Problem`` for the features Ex-Fuzzy uses, so problems and runners written
    against that interface keep working.
    """

    def __init__(self, n_var: int = -1, n_obj: int = 1, n_ieq_constr: int = 0,
                 n_eq_constr: int = 0, xl=None, xu=None, vtype=None, vars: dict = None,
                 elementwise: bool = False,
                 elementwise_func=ElementwiseEvaluationFunction,
                 elementwise_runner=None, requires_kwargs: bool = False,
                 replace_nan_values_by=None, exclude_from_serialization=None,
                 callback=None, strict: bool = True, **kwargs) -> None:
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_ieq_constr = n_ieq_constr
        self.n_eq_constr = n_eq_constr
        self.data = dict(**kwargs)
        self.xl, self.xu = xl, xu
        self.callback = callback
        if vars is not None:
            self.vars = vars
            self.n_var = len(vars)
            if self.xl is None:
                self.xl = {name: var.bounds[0] for name, var in vars.items()}
            if self.xu is None:
                self.xu = {name: var.bounds[1] for name, var in vars.items()}
        self.vtype = vtype
        self.elementwise = elementwise
        self.elementwise_func = elementwise_func
        self.elementwise_runner = (LoopedElementwiseEvaluation() if elementwise_runner is None
                                   else elementwise_runner)
        self.requires_kwargs = requires_kwargs
        self.strict = strict
        # Like pymoo, bounds become float arrays only for an explicit n_var.
        if n_var > 0:
            if self.xl is not None:
                if not isinstance(self.xl, np.ndarray):
                    self.xl = np.ones(n_var) * xl
                self.xl = self.xl.astype(float)
            if self.xu is not None:
                if not isinstance(self.xu, np.ndarray):
                    self.xu = np.ones(n_var) * xu
                self.xu = self.xu.astype(float)
        self.replace_nan_values_by = replace_nan_values_by
        self.exclude_from_serialization = exclude_from_serialization

    def _evaluate(self, x, out: dict, *args, **kwargs) -> None:
        raise NotImplementedError

    def has_bounds(self) -> bool:
        return self.xl is not None and self.xu is not None

    def bounds(self) -> tuple:
        return self.xl, self.xu

    def evaluate(self, X, *args, return_values_of=None, return_as_dictionary: bool = False,
                 **kwargs) -> Union[np.ndarray, tuple, dict]:
        """Evaluate one chromosome or a population, as ``pymoo.Problem.evaluate``."""
        if not self.requires_kwargs:
            kwargs = dict()
        if return_values_of is None:
            return_values_of = ["F"]
            if self.n_ieq_constr > 0:
                return_values_of.append("G")
            if self.n_eq_constr > 0:
                return_values_of.append("H")
        if isinstance(X, np.ndarray) and X.dtype != object:
            X, only_single_value = _at_least_2d(X)
            assert X.shape[1] == self.n_var, (
                f"Input dimension {X.shape[1]} are not equal to n_var {self.n_var}!")
        else:
            only_single_value = not isinstance(X, (list, np.ndarray))

        evaluated = self.do(X, return_values_of, *args, **kwargs)
        out = {}
        for key, value in evaluated.items():
            value = np.array(value)
            if only_single_value:
                value = value[0]
            if self.replace_nan_values_by is not None:
                value[np.isnan(value)] = self.replace_nan_values_by
            try:
                out[key] = value.astype(np.float64)
            except Exception:
                out[key] = value
        if self.callback is not None:
            self.callback(X, out)
        if return_as_dictionary:
            return out
        if len(return_values_of) == 1:
            return out[return_values_of[0]]
        return tuple(out[name] for name in return_values_of)

    def do(self, X, return_values_of: list, *args, **kwargs) -> dict:
        out = {name: None for name in return_values_of}
        if self.elementwise:
            self._evaluate_elementwise(X, out, *args, **kwargs)
        else:
            self._evaluate_vectorized(X, out, *args, **kwargs)
        return self._format_dict(out, len(X), return_values_of)

    def _evaluate_vectorized(self, X, out: dict, *args, **kwargs) -> None:
        self._evaluate(X, out, *args, **kwargs)

    def _evaluate_elementwise(self, X, out: dict, *args, **kwargs) -> None:
        f = self.elementwise_func(self, args, kwargs)
        for element in self.elementwise_runner(f, X):
            for key, value in element.items():
                if out.get(key, None) is None:
                    out[key] = []
                out[key].append(value)
        # The None check matters: an unset key must not become an empty array.
        for key in out:
            if out[key] is not None:
                out[key] = np.array(out[key])

    def _format_dict(self, out: dict, N: int, return_values_of: list) -> dict:
        shape = dict(F=(N, self.n_obj), G=(N, self.n_ieq_constr), H=(N, self.n_eq_constr))
        formatted = {}
        for name, value in out.items():
            if value is None:
                continue
            if name in shape:
                if isinstance(value, list):
                    value = np.column_stack(value)
                value = value.reshape(shape[name])
            formatted[name] = value
        for name in return_values_of:
            if name not in formatted:
                formatted[name] = np.full(shape.get(name, N), np.inf)
        return formatted

    def __getstate__(self):
        if self.exclude_from_serialization is None:
            return self.__dict__
        state = self.__dict__.copy()
        for key in self.exclude_from_serialization:
            state[key] = None
        return state


#: Actionable error for code paths that need pymoo when it cannot be imported.
PYMOO_INSTALL_MESSAGE = ("The 'pymoo' backend and the temporal classifier need pymoo. "
                         "Install it with: pip install 'pymoo>=0.6.2'")

_PYMOO_PROBLEM_CLASS = None


def _pymoo_problem_class():
    """Build the pymoo wrapper class on first use, importing pymoo only then."""
    global _PYMOO_PROBLEM_CLASS
    if _PYMOO_PROBLEM_CLASS is not None:
        return _PYMOO_PROBLEM_CLASS
    try:
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.core.variable import Integer as PymooInteger
    except ImportError as error:
        raise ImportError(PYMOO_INSTALL_MESSAGE) from error

    class PymooProblemWrapper(PymooProblem):
        """A pymoo problem delegating every evaluation to an Ex-Fuzzy problem."""

        def __init__(self, problem: Problem) -> None:
            variables = getattr(problem, "vars", None)
            if variables is not None:
                variables = {name: PymooInteger(bounds=var.bounds) if isinstance(var, Integer)
                             else var for name, var in variables.items()}
            super().__init__(
                n_var=problem.n_var, n_obj=problem.n_obj,
                n_ieq_constr=problem.n_ieq_constr, n_eq_constr=problem.n_eq_constr,
                xl=problem.xl, xu=problem.xu, vtype=problem.vtype, vars=variables,
                elementwise=problem.elementwise, elementwise_runner=problem.elementwise_runner,
                requires_kwargs=problem.requires_kwargs,
                replace_nan_values_by=problem.replace_nan_values_by,
                callback=problem.callback, strict=problem.strict)
            self.problem = problem

        def _evaluate(self, x, out, *args, **kwargs):
            self.problem._evaluate(x, out, *args, **kwargs)

        def _evaluate_elementwise(self, X, out, *args, **kwargs):
            self.problem._evaluate_elementwise(X, out, *args, **kwargs)

        def _evaluate_vectorized(self, X, out, *args, **kwargs):
            self.problem._evaluate_vectorized(X, out, *args, **kwargs)

    # A module-level name keeps wrapped problems picklable by reference.
    PymooProblemWrapper.__qualname__ = PymooProblemWrapper.__name__
    PymooProblemWrapper.__module__ = __name__
    globals()[PymooProblemWrapper.__name__] = PymooProblemWrapper
    _PYMOO_PROBLEM_CLASS = PymooProblemWrapper
    return PymooProblemWrapper


def as_pymoo_problem(problem: Any):
    """Return ``problem`` as a pymoo problem, wrapping Ex-Fuzzy problems.

    Needed only to pass a fitting problem to pymoo directly; the ``pymoo``
    backend does this itself.  pymoo problems are returned unchanged.

    :raises ImportError: if pymoo is not installed.
    """
    wrapper = _pymoo_problem_class()
    if isinstance(problem, Problem):
        return wrapper(problem)
    return problem
