# This module provides access to various force-related functions.
# It's an abstraction layer to easily switch between factual and counterfactual implementations.

# spatial_router.py
from importlib import import_module
from functools import lru_cache
from typing import Callable, Any, Mapping, Union

Number = Union[int, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[int, float, str]
Resolver = Callable[[WorldState, QuestionPayload], Answer]


@lru_cache
def _load_impl_module(counterfactual: bool):
    modname = ".mechanics_questions_cf" if counterfactual else ".mechanics_questions"
    return import_module(modname, package=__package__)


def _get_callable_by_name(mod, name: str, prefix: str):
    try:
        fn = getattr(mod, name)
    except AttributeError:
        candidates = [n for n in dir(mod) if n.startswith(prefix)]
        raise ValueError(
            f"Function '{name}' not found in {mod.__name__}. "
            f"Available: {', '.join(sorted(candidates))}"
        )
    if not callable(fn):
        raise TypeError(f"Attribute '{name}' in {mod.__name__} is not callable.")
    return fn


def _get_function_by_name_mechanics(name: str, counterfactual: bool) -> Resolver:
    mod = _load_impl_module(counterfactual)
    prefix = "CF_" if counterfactual else "F_"
    return _get_callable_by_name(mod, name, prefix)


def _is_counterfactual_name(name: str) -> bool:
    return name.startswith("CF_")


def get_function_by_name_mechanics(name: str) -> Resolver:
    return _get_function_by_name_mechanics(
        name, counterfactual=_is_counterfactual_name(name)
    )
