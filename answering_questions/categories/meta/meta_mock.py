"""
Mock meta reasoning resolvers.

These helpers extract best-effort meta answers from the provided world state.
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

from utils.decorators import with_resolved_attributes

@with_resolved_attributes
def F_META_PLAUSIBILITY_PHYSICAL_SANITY(
        world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    """Mock function always returning 1 (plausible)."""
    return "", ["A", "A", "A", "A"], 1