"""
Mock event_ordering reasoning resolvers.

These helpers extract best-effort event_ordering answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

from utils.decorators import with_resolved_attributes
from utils.bin_creation import (
    create_mc_options_around_gt,
    create_mc_object_names_from_dataset,
    uniform_labels,
)

from utils.all_objects import get_all_objects_names

from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

from utils.helpers import (
    _distance_between,
    _iter_objects,
    _fill_template,
)

from .event_ordering_helpers import (
    _get_position,
    point_to_plane_distance,
    _get_position_camera,
)

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

## --- Resolver functions -- ##
