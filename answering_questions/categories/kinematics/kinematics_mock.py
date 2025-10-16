"""
Mock kinematics reasoning resolvers.

These helpers extract best-effort kinematics answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations


from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)


Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

## --- Resolver functions -- ##
