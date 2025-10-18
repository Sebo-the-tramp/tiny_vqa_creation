# This module provides access to various force-related functions, both real and mock versions.
# It's an abstraction layer to easily switch between real and mock implementations based on the context.

# force_router.py
from importlib import import_module
from functools import lru_cache
from typing import Callable, Any, Mapping, Union

Number = Union[int, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[int, float, str]
Resolver = Callable[[WorldState, QuestionPayload], Answer]


@lru_cache
def _load_impl_module(mock: bool):
    modname = ".force_mock" if mock else ".force_real"
    return import_module(modname, package=__package__)


def get_function_by_name_force(name: str, mock: bool = False) -> Resolver:
    mod = _load_impl_module(mock)
    try:
        fn = getattr(mod, name)
    except AttributeError:
        # Nice error with suggestions
        candidates = [n for n in dir(mod) if n.startswith("F_")]
        raise ValueError(
            f"Function '{name}' not found in {mod.__name__}. "
            f"Available: {', '.join(sorted(candidates))}"
        )
    if not callable(fn):
        raise TypeError(f"Attribute '{name}' in {mod.__name__} is not callable.")
    return fn


@lru_cache
def _load_gt_module(mock: bool):
    modname = ".force_mock_results"
    return import_module(modname, package=__package__)


def get_result_by_name_force(name: str, mock: bool = False) -> Any:
    mod = _load_gt_module(mock)
    try:
        fn = getattr(mod, name)
    except AttributeError:
        # Nice error with suggestions
        candidates = [n for n in dir(mod) if n.startswith("F_")]
        raise ValueError(
            f"Function '{name}' not found in {mod.__name__}. "
            f"Available: {', '.join(sorted(candidates))}"
        )
    if not callable(fn):
        raise TypeError(f"Attribute '{name}' in {mod.__name__} is not callable.")
    return fn()  # Call the function to get the result
