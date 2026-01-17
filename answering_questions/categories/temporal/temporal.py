# This module provides access to various force-related functions.

# temporal_router.py
from importlib import import_module
from functools import lru_cache
from typing import Callable, Any, Mapping, Union

Number = Union[int, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[int, float, str]
Resolver = Callable[[WorldState, QuestionPayload], Answer]


@lru_cache
def _load_impl_module():
    return import_module(".temporal_questions", package=__package__)


def get_function_by_name_temporal(name: str) -> Resolver:
    mod = _load_impl_module()
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
