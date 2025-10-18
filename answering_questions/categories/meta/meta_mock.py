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
from utils.frames_selection import uniformly_sample_frames

from utils.bin_creation import create_mc_object_names_from_dataset

CORRECT_ATTRIBUTES = {
    "gravity": {
        "type": "vector",
        "value": [0.0, 0.0, -9.81],
    },
    "collisions": {
        "type": "boolean",
        "value": True,
    },
    "restitution": {
        "type": "float",
        "value": 0.5,
    },
    "mass": {
        "type": "float",
        "value": 1.0,
    },
}


@with_resolved_attributes
def F_META_VIOLATED_CONSTRAINT(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    """Mock function always returning 1 (plausible)."""
    assert len(resolved_attributes) == 0

    correct_attributes = {attr: True for attr in CORRECT_ATTRIBUTES}

    # check if any of the attributes in CORRECT_ATTRIBUTES is violated in world_state
    for attr, correct_info in CORRECT_ATTRIBUTES.items():
        if attr in world_state["config"]["scene"]:
            scene_value = world_state["config"]["scene"][attr]
            if scene_value != correct_info["value"]:
                correct_attributes[attr] = False

    if all(correct_attributes.values()):
        labels, correct_index = create_mc_object_names_from_dataset(
            "None",
            present_objects=[],
            num_answers=4,
            dataset_labels=[x for x in CORRECT_ATTRIBUTES.keys()],
        )

    else:
        violated_attrs = [
            attr for attr, is_correct in correct_attributes.items() if not is_correct
        ]
        labels, correct_index = create_mc_object_names_from_dataset(
            violated_attrs[0],
            present_objects=[],
            num_answers=4,
            dataset_labels=[x for x in CORRECT_ATTRIBUTES.keys()],
        )

    return question, labels, correct_index, uniformly_sample_frames(world_state)


# @with_resolved_attributes
# def F_META_COLLISION_RESOLUTION_SANITY(
#     world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
# ) -> int:
#     """Mock function always returning 1 (plausible)."""
#     assert len(resolved_attributes) == 0
#     # TODO understand if we have some scenes like this
#     return "", ["A", "A", "A", "A"], 1
