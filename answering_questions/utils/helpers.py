from __future__ import annotations

import re
import math
import random

from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    List,
    cast,
)

# set random seed for reproducibility
rng = random.Random(42)

Number = Union[int, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[int, float, str]

MOVEMENT_TOLERANCE = 1e-3
DEFAULT_DISPLACEMENT_THRESHOLD = 2.0

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

# So far we have the following attributes:
# - <OBJECT> -> defines any unique object
#     <OBJECT_1> -> if there are multiple objects, _N will define the ID of the object
# - <OBJECT_TYPE> -> defines the category of objects (multiple unique <OBJECT>)
# - <TIME> -> the timestamp we want to measure
# - <MATERIAL> -> the material of the object
# - <MASS> -> the mass of the object.
# - <VOLUME> -> volume of the obejct
# - <SCENE> -> could be the scene itself or any segmented part of the scene
# - <VELOCITY> -> the velocity of an object
# - <CAMERA> -> the camera itself

# ----- General helpers -----


def _extract_attributes(question: Mapping[str, Any]) -> Mapping[str, Any]:
    question_text = question["question"]

    # Extract all tokens enclosed in <...>
    attributes = re.findall(r"<(.*?)>", question_text)

    # Optional: remove duplicates while preserving order
    seen = set()
    attributes = [a for a in attributes if not (a in seen or seen.add(a))]

    return {"attributes": attributes}


def _resolve_attributes(
    attributes: List[Mapping[str, Any]], world_state: Mapping[str, Any]
) -> Mapping[str, Any]:
    attribute_resolved = {}

    for attribute in attributes:
        attribute_resolved[attribute] = {}
        attribute_category = attribute.split("_")[
            0
        ]  # Get the part before any underscore
        result = resolver[attribute_category](world_state)

        attribute_resolved[attribute]["choice"] = result
        attribute_resolved[attribute]["category"] = attribute_category

    return attribute_resolved


def _fill_template(
    question: Mapping[str, Any], resolved_attributes: Mapping[str, Any]
) -> None:
    for attribute in resolved_attributes:
        if "OBJECT_TYPE" in attribute:
            question["question"] = question["question"].replace(
                f"<{attribute}>", resolved_attributes[attribute]["choice"]["type"]
            )
        elif "OBJECT" in attribute:
            question["question"] = question["question"].replace(
                f"<{attribute}>", resolved_attributes[attribute]["choice"]["name"]
            )
        else:
            question["question"] = question["question"].replace(
                f"<{attribute}>", str(resolved_attributes[attribute]["choice"])
            )


def get_camera(world_state: Mapping[str, Any]) -> Mapping[str, Any]:
    camera = world_state.get("camera")
    if not camera:
        raise ValueError("No camera found in the world state.")
    return camera


def get_random_material(world_state: Mapping[str, Any]) -> str:
    materials = set()
    for obj in _iter_objects(world_state):
        material = _as_lower(obj.get("material"))
        if material:
            materials.add(material)
    if not materials:
        raise ValueError("No materials found in the world state.")
    return rng.choice(list(materials))


def get_random_object(
    world_state: Mapping[str, Any], object_type: Optional[str] = None
) -> Mapping[str, Any]:
    objects = list(_objects_of_type(world_state, object_type))
    if not objects:
        raise ValueError(f"No objects found of type '{object_type}'")
    return rng.choice(objects)


def get_random_object_type(world_state: Mapping[str, Any]) -> str:
    object_types = set()
    for obj in _iter_objects(world_state):
        obj_type = _as_lower(obj.get("type"))
        if obj_type:
            object_types.add(obj_type)
    if not object_types:
        raise ValueError("No object types found in the world state.")
    return rng.choice(list(object_types))


def get_random_time(world_state: Mapping[str, Any]) -> float:
    timestamps = world_state.get("simulation_steps", [])
    if not timestamps:
        raise ValueError("No timestamps found in the world state")
    return rng.choice(list(timestamps.keys()))


# TODO Those random values are just hardcoded
resolver = {
    "CAMERA": get_camera,
    "DISTANCE": lambda ws: round(
        rng.uniform(1.0, 5.0), 1
    ),  # random distance between 1 and 5 meters, 1 decimal place
    "MATERIAL": get_random_material,
    "OBJECT_TYPE": get_random_object_type,
    "OBJECT": get_random_object,
    "TIME": get_random_time,
}


def _round_sig(x: float, sig: int = 3) -> float:
    """Round to `sig` significant digits, preserving sign."""
    if x == 0:
        return 0.0
    return round(x, sig - 1 - int(math.floor(math.log10(abs(x)))))


def _decimals_for_sig(x: float, sig: int = 3) -> int:
    """Number of decimal places that keeps `sig` significant digits when using round(x, decimals)."""
    if x == 0:
        # e.g., for sig=3, show 2 decimals by default
        return sig - 1
    return max(0, sig - 1 - int(math.floor(math.log10(abs(x)))))


# ____________ DON'T KNOW WHAT IS AFTER THIS LINE ____________ #


def _iter_objects(world_state: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    objects = world_state.get("objects", [])
    if isinstance(objects, Mapping):
        iterable: Iterable[Any] = objects.values()
    else:
        iterable = cast(Iterable[Any], objects)

    for obj in iterable:
        if isinstance(obj, Mapping):
            yield obj


def _objects_of_type(
    world_state: Mapping[str, Any], object_type: Optional[str]
) -> Iterator[Mapping[str, Any]]:
    if not object_type:
        yield from _iter_objects(world_state)
        return

    target = object_type.casefold()
    for obj in _iter_objects(world_state):
        obj_type = _as_lower(obj.get("type"))
        if obj_type == target:
            yield obj


def _resolve_single_object(
    world_state: Mapping[str, Any], question: Mapping[str, Any]
) -> Mapping[str, Any]:
    object_name = _extract_object_name(question)
    if not object_name:
        raise KeyError("Unable to resolve object name from the question payload.")

    target = object_name.casefold()
    for obj in _iter_objects(world_state):
        for key in ("name", "id", "label", "type"):
            value = obj.get(key)
            if isinstance(value, str) and _as_lower(value) == target:
                return obj

    raise ValueError(f"Object '{object_name}' not found in the provided world state.")


def _is_moving(object_id: str, timestep: str, world_state: Mapping[str, Any]) -> bool:
    return _get_speed(object_id, timestep, world_state) > MOVEMENT_TOLERANCE


def _get_speed(object_id: str, timestep: str, world_state: Mapping[str, Any]) -> float:
    timestamp_world = world_state["simulation_steps"][timestep]
    current_timestamp_involved_object = timestamp_world["objects"][object_id][
        "kinematics"
    ]["speed"]
    return current_timestamp_involved_object


def _get_acceleration(
    object_id: str, timestep: str, world_state: Mapping[str, Any]
) -> float:
    timestamp_world = world_state["simulation_steps"][timestep]
    current_timestamp_involved_object = timestamp_world["objects"][object_id][
        "kinematics"
    ]["normal_accel"]
    return current_timestamp_involved_object


def get_angular_velocity(
    object_id: str, timestep: str, world_state: Mapping[str, Any]
) -> float:
    timestamp_world = world_state["simulation_steps"][timestep]
    current_timestamp_involved_object = timestamp_world["objects"][object_id][
        "kinematics"
    ]["angular_velocity_world"]
    return current_timestamp_involved_object


def _get_vertical_speed(obj: Mapping[str, Any]) -> float:
    motion = _get_motion(obj)

    value = motion.get("vertical_speed") or motion.get("verticalSpeed")
    vertical_speed = _coerce_to_float(value)
    if vertical_speed is not None:
        return vertical_speed

    velocity = motion.get("velocity")
    components = _as_vector(velocity)
    if components and len(components) >= 3:
        return float(components[2])

    if isinstance(velocity, Mapping):
        for key in ("z", "vz", "vertical"):
            component = velocity.get(key)
            component_value = _coerce_to_float(component)
            if component_value is not None:
                return component_value

    return 0.0


def _get_displacement(
    obj: str, timestep_start: str, timestep_end: str, world_state: Mapping[str, Any]
) -> float:
    position_start = world_state["simulation_steps"][timestep_start]["objects"][obj][
        "transform"
    ][:3]
    position_end = world_state["simulation_steps"][timestep_end]["objects"][obj][
        "transform"
    ][:3]
    displacement = _distance_between(position_start, position_end)

    displacement = _coerce_to_float(displacement)
    if displacement is not None:
        return displacement

    return 0.0


def _get_motion(obj: Mapping[str, Any]) -> Mapping[str, Any]:
    motion = obj.get("motion")
    if isinstance(motion, Mapping):
        return motion
    return {}


def _get_motion_property(obj: Mapping[str, Any], keys: Sequence[str]) -> Any:
    motion = _get_motion(obj)
    for key in keys:
        if key in motion:
            return motion[key]
    return None


def _coerce_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, Mapping):
        for key in ("magnitude", "value", "amount", "norm", "length"):
            if key in value:
                coerced = _coerce_to_float(value[key])
                if coerced is not None:
                    return coerced

        if all(isinstance(v, (int, float)) for v in value.values()):
            numeric_values = tuple(cast(Number, v) for v in value.values())
            return _vector_magnitude(numeric_values)

        return None

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return None

        if all(isinstance(item, (int, float)) for item in value):
            numeric_values = tuple(cast(Number, item) for item in value)
            return _vector_magnitude(numeric_values)

    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None

    return None


def _vector_magnitude(components: Tuple[Number, ...]) -> float:
    if not components:
        return 0.0
    return math.sqrt(sum(float(component) ** 2 for component in components))


def _as_vector(value: Any) -> Optional[Tuple[Number, ...]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if all(isinstance(item, (int, float)) for item in value):
            return tuple(cast(Number, item) for item in value)
    if isinstance(value, Mapping):
        numeric_items = [v for v in value.values() if isinstance(v, (int, float))]
        if numeric_items:
            return tuple(cast(Number, v) for v in numeric_items)
    return None


def _extract_object_type(question: Mapping[str, Any]) -> Optional[str]:
    value = _extract_text(question, "object_type", "objectType", "type")
    if value:
        return _as_lower(value)

    return None


def _extract_object_name(question: Mapping[str, Any]) -> Optional[str]:
    value = _extract_text(question, "<OBJECT>")
    if value:
        return _as_lower(value)

    return None


def _extract_numeric(question: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = question.get(key)
        numeric = _coerce_to_float(value)
        if numeric is not None:
            return numeric
    return None


def _extract_text(question: Mapping[str, Any], *keys: str) -> Optional[str]:
    print(question)
    for key in keys:
        value = question.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _as_lower(value: Any) -> Optional[str]:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            return candidate.casefold()
    return None


# ---------------------------------------------------------------------------
# Extended helpers used by mock resolvers
# ---------------------------------------------------------------------------


def _resolve_object_by_name(
    world_state: Mapping[str, Any], object_name: str
) -> Mapping[str, Any]:
    """Resolve an object using any of its textual identifiers."""
    if not object_name:
        raise KeyError("Object identifier cannot be empty.")

    target = _as_lower(object_name)
    if not target:
        raise KeyError(f"Unable to normalise identifier '{object_name}'.")

    for obj in _iter_objects(world_state):
        for key in ("name", "id", "label", "type"):
            value = obj.get(key)
            if isinstance(value, str) and _as_lower(value) == target:
                return obj

    raise ValueError(f"Object '{object_name}' not found in the provided world state.")


def _resolve_object_from_question(
    world_state: Mapping[str, Any],
    question: Mapping[str, Any],
    *keys: str,
) -> Mapping[str, Any]:
    """Resolve an object using explicit keys from the question payload."""
    for key in keys:
        value = question.get(key)
        if isinstance(value, str) and value.strip():
            try:
                return _resolve_object_by_name(world_state, value)
            except ValueError:
                continue

    return _resolve_single_object(world_state, question)


def _ensure_vector_size(
    components: Tuple[Number, ...], size: int = 3
) -> Tuple[float, ...]:
    """Normalise a vector to the desired size by truncating or padding with zeros."""
    floats = tuple(float(component) for component in components)
    if len(floats) >= size:
        return floats[:size]
    if not floats:
        return tuple(0.0 for _ in range(size))
    padded = list(floats)
    while len(padded) < size:
        padded.append(0.0)
    return tuple(padded)


def _extract_vector_from_mapping(
    mapping: Mapping[str, Any], *keys: str
) -> Optional[Tuple[float, ...]]:
    for key in keys:
        if key not in mapping:
            continue
        vector = _as_vector(mapping[key])
        if vector:
            return tuple(float(component) for component in vector)
    return None


def _extract_position(obj: Mapping[str, Any]) -> Optional[Tuple[float, ...]]:
    position = _extract_vector_from_mapping(
        obj,
        "position",
        "pos",
        "location",
        "center",
        "centre",
        "center_of_mass",
        "centre_of_mass",
        "centroid",
    )
    if position:
        return _ensure_vector_size(position)

    pose = obj.get("pose")
    if isinstance(pose, Mapping):
        position = _extract_vector_from_mapping(
            pose, "position", "pos", "location", "translation"
        )
        if position:
            return _ensure_vector_size(position)

    motion = _get_motion(obj)
    position = _extract_vector_from_mapping(motion, "position", "pos", "location")
    if position:
        return _ensure_vector_size(position)

    transform = obj.get("transform")
    if isinstance(transform, Sequence) and transform:
        vector = _as_vector(transform[-3:])
        if vector:
            return _ensure_vector_size(vector)

    return None


def _extract_orientation(obj: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    orientation = obj.get("orientation") or obj.get("rotation")
    if isinstance(orientation, Mapping):
        return orientation
    if (
        isinstance(orientation, Sequence)
        and orientation
        and not isinstance(orientation, (str, bytes, bytearray))
    ):
        return {"vector": tuple(float(component) for component in orientation)}

    pose = obj.get("pose")
    if isinstance(pose, Mapping):
        for key in ("orientation", "rotation", "attitude"):
            value = pose.get(key)
            if isinstance(value, Mapping):
                return value
            vector = _as_vector(value)
            if vector:
                return {"vector": tuple(float(component) for component in vector)}

    motion = _get_motion(obj)
    for key in ("orientation", "rotation"):
        value = motion.get(key)
        if isinstance(value, Mapping):
            return value
        vector = _as_vector(value)
        if vector:
            return {"vector": tuple(float(component) for component in vector)}

    return None


def _extract_motion_vector(
    obj: Mapping[str, Any], *keys: str
) -> Optional[Tuple[float, ...]]:
    motion = _get_motion(obj)
    vector = _extract_vector_from_mapping(motion, *keys)
    if vector:
        return _ensure_vector_size(vector)
    return None


def _extract_velocity_vector(obj: Mapping[str, Any]) -> Optional[Tuple[float, ...]]:
    vector = _extract_motion_vector(
        obj, "velocity_vector", "velocity", "linear_velocity", "speed_vector"
    )
    if vector:
        return vector

    motion = _get_motion(obj)
    components = motion.get("velocity_components")
    if isinstance(components, Mapping):
        numeric_items = [components.get(axis) for axis in ("x", "y", "z")]
        if any(isinstance(item, (int, float)) for item in numeric_items):
            as_tuple = tuple(float(item or 0.0) for item in numeric_items)
            return _ensure_vector_size(as_tuple)

    return None


def _extract_acceleration_vector(obj: Mapping[str, Any]) -> Optional[Tuple[float, ...]]:
    return _extract_motion_vector(
        obj, "acceleration_vector", "acceleration", "linear_acceleration"
    )


def _extract_angular_velocity_vector(
    obj: Mapping[str, Any],
) -> Optional[Tuple[float, ...]]:
    return _extract_motion_vector(
        obj, "angular_velocity_vector", "angular_velocity", "rotation_rate", "spin"
    )


def _extract_numeric_attribute(
    obj: Mapping[str, Any], *keys: str, default: Optional[float] = None
) -> Optional[float]:
    for key in keys:
        if key not in obj:
            continue
        numeric = _coerce_to_float(obj[key])
        if numeric is not None:
            return numeric
    return default


def _extract_text_attribute(obj: Mapping[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _object_identifier(obj: Mapping[str, Any]) -> Optional[str]:
    for key in ("name", "id", "label", "type"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _distance_between(
    first: Optional[Sequence[Number]],
    second: Optional[Sequence[Number]],
) -> float:
    if not first or not second:
        return 0.0
    a = _ensure_vector_size(tuple(cast(Number, component) for component in first))
    b = _ensure_vector_size(tuple(cast(Number, component) for component in second))
    return math.sqrt(
        sum((a_component - b_component) ** 2 for a_component, b_component in zip(a, b))
    )
