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

from copy import deepcopy
from utils.config import get_config

from utils.my_exception import ImpossibleToAnswer

# set random seed for reproducibility
rng = random.Random(42)

Number = Union[int, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[int, float, str]

SAMPLING_RATE = get_config()["sampling_rate"]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

# So far we have the following attributes:
# - <OBJECT> -> defines any unique object
#     <OBJECT_1> -> if there are multiple objects, _N will define the ID of the object
# - <OBJECT_CATEGORY> -> defines the category of objects (multiple unique <OBJECT>)
# - <TIME> -> the timesteps we want to measure
# - <MATERIAL> -> the material of the object
# - <MASS> -> the mass of the object.
# - <VOLUME> -> volume of the obejct
# - <SCENE> -> could be the scene itself or any segmented part of the scene
# - <VELOCITY> -> the velocity of an object
# - <CAMERA> -> the camera itself

# ----- General helpers -----

units = {
    "DISTANCE": "meters",
    "MASS": "kilograms",
    "VOLUME": "cubic centimeters",
    "DENSITY": "kg/m3",
    "TIME": "seconds",
    "SPEED": "meters/second",
    "ACCELERATION": "meters/second^2",
}


def resolve_units(measurement: str) -> str:
    return units.get(measurement, "")


def get_random_integer(min_value: int, max_value: int) -> int:
    return rng.randint(min_value, max_value)


def shuffle_array(array: List[int]) -> List[int]:
    rng.shuffle(array)
    return array


def get_total_timesteps():
    # very important function for images<->state conversion
    # TODO supposing 100fps and 5 seconds of video
    return 100 * 5


def get_total_images():
    # very important function for images<->state conversion
    # TODO supposing 100fps and 5 seconds of video
    return 8


def extract_attributes(question: Mapping[str, Any]) -> Mapping[str, Any]:
    question_text = question["question"]

    # Extract all tokens enclosed in <...>
    attributes = re.findall(r"<(.*?)>", question_text)

    # Optional: remove duplicates while preserving order
    seen = set()
    attributes = [a for a in attributes if not (a in seen or seen.add(a))]

    return {"attributes": attributes}


def get_object_state_at_timestep(
    world_state: Mapping[str, Any], object_id: str, timestep: str
) -> Optional[Mapping[str, Any]]:
    """Retrieve the state of an object at a specific timestep."""
    simulation_steps = world_state.get("simulation", {})
    if not simulation_steps:
        return None

    step_data = simulation_steps.get(str(timestep), {})
    objects = step_data.get("objects", {})
    return objects.get(object_id)


def get_all_objects_state_at_time(
    world_state: Mapping[str, Any], timestep: str
) -> Optional[Mapping[str, Any]]:
    """Retrieve the state of an object at a specific timestep."""
    simulation_steps = world_state.get("simulation", {})
    if not simulation_steps:
        return None

    step_data = simulation_steps.get(str(timestep), {})
    objects = step_data.get("objects", {})
    return objects


def get_visible_timesteps_for_attributes_min_objects(
    attributes: List[Mapping[str, Any]], world_state: Mapping[str, Any], min_objects=1
) -> List[str]:
    visible_timesteps = []

    for attribute in attributes:
        attribute_category = attribute.split("_")[
            0
        ]  # Get the part before any underscore
        if attribute_category == "OBJECT" or attribute_category == "OBJECT-CATEGORY":
            for timestep in world_state.get("simulation", {}).keys():
                visible_objects_id = []
                for obj in iter_objects(world_state):
                    obj_id = obj.get("id")
                    if not obj_id:
                        continue
                    obj_state = get_object_state_at_timestep(
                        world_state, obj_id, timestep
                    )

                    if (
                        obj_state
                        and obj_state["fov_visibility"] > 0.25
                        and obj_state["infov"]
                    ):  # at least 25% visible for now cause of a bug
                        visible_objects_id.append(obj_id)
                if len(visible_objects_id) >= min_objects:
                    visible_timesteps.append(timestep)
    return visible_timesteps


def get_continuous_subsequences_min_length(
    timesteps: List[str], min_length: int
) -> List[List[str]]:
    if not timesteps:
        return []

    sorted_timesteps = sorted(int(t.replace(".", "")) for t in timesteps)
    subsequences = []
    current_subseq = [str(sorted_timesteps[0])]

    time_interval_in_milliseconds = int(
        (1 / SAMPLING_RATE) * 1000
    )  # e.g., 100ms -> 100*10 = 1000

    # we live a buffer of 1 timestep to allow for small gaps
    for i in range(2, len(sorted_timesteps)):
        if sorted_timesteps[i] == sorted_timesteps[i - 1] + (
            1 * time_interval_in_milliseconds
        ) or sorted_timesteps[i] == sorted_timesteps[i - 1] + (
            2 * time_interval_in_milliseconds
        ):
            current_subseq.append(str(timesteps[i]))
        else:
            if len(current_subseq) >= min_length:
                subsequences.append(current_subseq)
            current_subseq = [str(timesteps[i])]

    if len(current_subseq) >= min_length:
        subsequences.append(current_subseq)

    return subsequences


def resolve_attributes(
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


def resolve_attributes_visible_at_timestep(
    attributes: List[Mapping[str, Any]], world_state: Mapping[str, Any], timestep: str
) -> Mapping[str, Any]:
    attribute_resolved = {}

    copy_of_world_state = deepcopy(world_state)

    for attribute in attributes:
        attribute_resolved[attribute] = {}
        attribute_category = attribute.split("_")[
            0
        ]  # Get the part before any underscore
        result = resolver[attribute_category](
            copy_of_world_state, visible_at_timestep=timestep
        )

        attribute_resolved[attribute]["choice"] = result
        attribute_resolved[attribute]["category"] = attribute_category

    return attribute_resolved


def fill_template(
    question: Mapping[str, Any], resolved_attributes: Mapping[str, Any]
) -> None:
    for attribute in resolved_attributes:
        if "OBJECT-CATEGORY" in attribute:
            question["question"] = question["question"].replace(
                f"<{attribute}>",
                resolved_attributes[attribute]["choice"],
            )
        elif "OBJECT" in attribute:
            question["question"] = question["question"].replace(
                f"<{attribute}>",
                resolved_attributes[attribute]["choice"]["model"],
            )
        else:
            question["question"] = question["question"].replace(
                f"<{attribute}>",
                str(resolved_attributes[attribute]["choice"])
                + resolve_units(attribute),
            )

    # check if there is a single frame or multi frame task
    if question["task_splits"] == "multi":
        question["question"] = (
            "Consider all frames, but answer only based on the last frame. "
            + question["question"]
        )


def get_camera(world_state: Mapping[str, Any]) -> Mapping[str, Any]:
    # taking the first camera
    camera = world_state["simulation"]["0000.010"]["camera"]
    if not camera:
        raise ValueError("No camera found in the world state.")
    return camera


def get_random_material(world_state: Mapping[str, Any]) -> str:
    materials = set()
    for obj in iter_objects(world_state):
        material = as_lower(obj["description"]["material_group"])
        if material:
            materials.add(material)
    if not materials:
        raise ValueError("No materials found in the world state.")
    return rng.choice(list(materials))


def get_random_object_and_remove(
    world_state: Mapping[str, Any],
    OBJECT_CATEGORY: Optional[str] = None,
    visible_at_timestep: str = None,
) -> Mapping[str, Any]:
    # objects = list(_objects_of_type(world_state, OBJECT_CATEGORY))
    # if not objects:
    #     raise ValueError(f"No objects found of type '{OBJECT_CATEGORY}'")

    objects = world_state["objects"]
    if visible_at_timestep is not None:
        visible_objects = []
        for obj_id, object in objects.items():
            obj_state = get_object_state_at_timestep(
                world_state, obj_id, visible_at_timestep
            )
            if (
                obj_state["fov_visibility"] > 0.05
            ):  # at least 5% visible for now cause of a bug
                object["id"] = obj_id
                visible_objects.append(object)
        objects = {obj["id"]: obj for obj in visible_objects}

    # also if no visible objects found, we raise an error
    if not objects:
        raise ImpossibleToAnswer(f"No objects found of type '{OBJECT_CATEGORY}'")

    object_chosen = rng.choice(list(objects.values()))

    del world_state["objects"][object_chosen["id"]]

    return object_chosen


def get_random_object_visible(
    world_state: Mapping[str, Any], OBJECT_CATEGORY: Optional[str] = None
) -> Mapping[str, Any]:
    # objects = list(_objects_of_type(world_state, OBJECT_CATEGORY))
    # if not objects:
    #     raise ValueError(f"No objects found of type '{OBJECT_CATEGORY}'")

    objects = world_state.get("objects", [])

    # TODO I think we should only resolve objects that are visible at least 50%
    # This can be improved and made a function of the time also
    visible_objects = []
    for obj_id, object in objects.items():
        is_visible_everywhere = True
        for timestep in world_state.get("simulation", {}).keys():
            obj_state = get_object_state_at_timestep(world_state, obj_id, timestep)
            if obj_state.get("is_visible_from_camera", False):
                continue
            else:
                is_visible_everywhere = False
                break
        if is_visible_everywhere:
            object["id"] = obj_id
            visible_objects.append(object)
    if not visible_objects:
        raise ValueError(f"No visible objects found of type '{OBJECT_CATEGORY}'")

    return rng.choice(visible_objects)


def get_random_OBJECT_CATEGORY(world_state: Mapping[str, Any]) -> str:
    OBJECT_CATEGORYs = set()
    for obj in iter_objects(world_state):
        obj_type = as_lower(obj["description"]["category_gso"])
        if obj_type:
            OBJECT_CATEGORYs.add(obj_type)
    if not OBJECT_CATEGORYs:
        raise ValueError("No object types found in the world state.")
    return rng.choice(list(OBJECT_CATEGORYs))


def get_random_timestep(world_state: Mapping[str, Any]) -> float:
    timesteps = world_state.get("simulation", [])
    if not timesteps:
        raise ValueError("No timesteps found in the world state")
    return rng.choice(list(timesteps.keys()))


# TODO Those random values are just hardcoded
resolver = {
    "CAMERA": get_camera,
    "CATEGORY": get_random_OBJECT_CATEGORY,
    "DENSITY": lambda ws: round(
        rng.uniform(10, 600), 1
    ),  # random density between 10 and 600 kg/m3
    "DISTANCE": lambda ws: round(
        rng.uniform(1.0, 5.0), 1
    ),  # random distance between 1 and 5 meters, 1 decimal place
    "MASS": lambda ws: round(
        rng.uniform(0.1, 3.0), 1
    ),  # random mass between 0.1 and 5 kg
    "MATERIAL": get_random_material,
    "OBJECT-CATEGORY": get_random_OBJECT_CATEGORY,
    "OBJECT": get_random_object_and_remove,
    "STRESS-THRESHOLD": lambda ws: round(
        rng.uniform(0.0, 10.0), 1
    ),  # random stress threshold between 10 and 100 MPa
    "TIME": get_random_timestep,
    "VOLUME": lambda ws: round(
        rng.uniform(0.001, 0.5), 1
    ),  # random volume between 0.001 and .5 cubic meters
}


def round_sig(x: float, sig: int = 3) -> float:
    """Round to `sig` significant digits, preserving sign."""
    if x == 0:
        return 0.0
    return round(x, sig - 1 - int(math.floor(math.log10(abs(x)))))


def decimals_for_sig(x: float, sig: int = 3) -> int:
    """Number of decimal places that keeps `sig` significant digits when using round(x, decimals)."""
    if x == 0:
        # e.g., for sig=3, show 2 decimals by default
        return sig - 1
    return max(0, sig - 1 - int(math.floor(math.log10(abs(x)))))


# ____________ DON'T KNOW WHAT IS AFTER THIS LINE ____________ #


def iter_objects(world_state: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    objects = world_state.get("objects", [])
    if isinstance(objects, Mapping):
        iterable: Iterable[Any] = objects.values()
    else:
        iterable = cast(Iterable[Any], objects)

    for obj in iterable:
        if isinstance(obj, Mapping):
            yield obj


# TODO improve this and make it actually work
def iter_visible_objects(
    world_state: Mapping[str, Any],
) -> Iterator[Mapping[str, Any]]:
    for obj in iter_objects(world_state):
        obj_id = obj.get("id")
        if not obj_id:
            continue

        is_visible_everywhere = True
        for timestep in world_state.get("simulation", {}).keys():
            obj_state = get_object_state_at_timestep(world_state, obj_id, timestep)
            if obj_state.get("is_visible_from_camera", True):
                continue
            else:
                is_visible_everywhere = False
                break
        if is_visible_everywhere:
            yield obj


def iter_visible_objects_at_time(
    world_state: Mapping[str, Any], timestep: str
) -> Iterator[Mapping[str, Any]]:
    step_data = world_state.get("simulation", {}).get(str(timestep), {})
    objects_state = step_data.get("objects", {})

    for obj in iter_objects(world_state):
        obj_id = obj.get("id")
        if not obj_id or obj_id not in objects_state:
            continue

        obj_state = objects_state[obj_id]
        if obj_state.get("is_visible_from_camera", True):
            yield obj


def objects_of_type(
    world_state: Mapping[str, Any], OBJECT_CATEGORY: Optional[str]
) -> Iterator[Mapping[str, Any]]:
    if not OBJECT_CATEGORY:
        yield from iter_objects(world_state)
        return

    target = OBJECT_CATEGORY.casefold()
    for obj in iter_objects(world_state):
        obj_type = as_lower(obj["description"]["category_gso"])
        if obj_type == target:
            yield obj


def resolve_single_object(
    world_state: Mapping[str, Any], question: Mapping[str, Any]
) -> Mapping[str, Any]:
    object_name = extract_object_name(question)
    if not object_name:
        raise KeyError("Unable to resolve object name from the question payload.")

    target = object_name.casefold()
    for obj in iter_objects(world_state):
        for key in ("name", "id", "label", "type"):
            value = obj.get(key)
            if isinstance(value, str) and as_lower(value) == target:
                return obj

    raise ValueError(f"Object '{object_name}' not found in the provided world state.")


def get_acceleration(
    object_id: str, timestep: str, world_state: Mapping[str, Any]
) -> float:
    timestep_world = world_state["simulation"][timestep]
    current_timestep_involved_object = timestep_world["objects"][object_id][
        "kinematics"
    ]["normal_accel"]
    return current_timestep_involved_object


def get_angular_velocity_vector(
    object_id: str, timestep: str, world_state: Mapping[str, Any]
) -> float:
    timestep_world = world_state["simulation"][timestep]
    current_timestep_involved_object = timestep_world["objects"][object_id][
        "kinematics"
    ]["angular_velocity_world"]
    return current_timestep_involved_object


def get_vertical_velocity(obj: Mapping[str, Any]) -> float:
    motion = get_motion(obj)

    value = motion.get("vertical_velocity") or motion.get("verticalvelocity")
    vertical_velocity = coerce_to_float(value)
    if vertical_velocity is not None:
        return vertical_velocity

    velocity = motion.get("velocity")
    components = as_vector(velocity)
    if components and len(components) >= 3:
        return float(components[2])

    if isinstance(velocity, Mapping):
        for key in ("z", "vz", "vertical"):
            component = velocity.get(key)
            component_value = coerce_to_float(component)
            if component_value is not None:
                return component_value

    return 0.0


def get_displacement(
    obj: str, timestep_start: str, timestep_end: str, world_state: Mapping[str, Any]
) -> float:
    position_start = world_state["simulation"][timestep_start]["objects"][obj][
        "transform"
    ][:3]
    position_end = world_state["simulation"][timestep_end]["objects"][obj]["transform"][
        :3
    ]
    displacement = distance_between(position_start, position_end)

    displacement = coerce_to_float(displacement)
    if displacement is not None:
        return displacement

    return 0.0


def get_motion(obj: Mapping[str, Any]) -> Mapping[str, Any]:
    motion = obj.get("motion")
    if isinstance(motion, Mapping):
        return motion
    return {}


def get_motion_property(obj: Mapping[str, Any], keys: Sequence[str]) -> Any:
    motion = get_motion(obj)
    for key in keys:
        if key in motion:
            return motion[key]
    return None


def coerce_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, Mapping):
        for key in ("magnitude", "value", "amount", "norm", "length"):
            if key in value:
                coerced = coerce_to_float(value[key])
                if coerced is not None:
                    return coerced

        if all(isinstance(v, (int, float)) for v in value.values()):
            numeric_values = tuple(cast(Number, v) for v in value.values())
            return vector_magnitude(numeric_values)

        return None

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return None

        if all(isinstance(item, (int, float)) for item in value):
            numeric_values = tuple(cast(Number, item) for item in value)
            return vector_magnitude(numeric_values)

    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None

    return None


def vector_magnitude(components: Tuple[Number, ...]) -> float:
    if not components:
        return 0.0
    return math.sqrt(sum(float(component) ** 2 for component in components))


def as_vector(value: Any) -> Optional[Tuple[Number, ...]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if all(isinstance(item, (int, float)) for item in value):
            return tuple(cast(Number, item) for item in value)
    if isinstance(value, Mapping):
        numeric_items = [v for v in value.values() if isinstance(v, (int, float))]
        if numeric_items:
            return tuple(cast(Number, v) for v in numeric_items)
    return None


def extract_OBJECT_CATEGORY(question: Mapping[str, Any]) -> Optional[str]:
    value = extract_text(question, "OBJECT-CATEGORY", "objectType", "type")
    if value:
        return as_lower(value)

    return None


def extract_object_name(question: Mapping[str, Any]) -> Optional[str]:
    value = extract_text(question, "<OBJECT>")
    if value:
        return as_lower(value)

    return None


def extract_numeric(question: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = question.get(key)
        numeric = coerce_to_float(value)
        if numeric is not None:
            return numeric
    return None


def extract_text(question: Mapping[str, Any], *keys: str) -> Optional[str]:
    print(question)
    for key in keys:
        value = question.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def as_lower(value: Any) -> Optional[str]:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            return candidate.casefold()
    return None


# ---------------------------------------------------------------------------
# Extended helpers used by mock resolvers
# ---------------------------------------------------------------------------


def resolve_object_by_name(
    world_state: Mapping[str, Any], object_name: str
) -> Mapping[str, Any]:
    """Resolve an object using any of its textual identifiers."""
    if not object_name:
        raise KeyError("Object identifier cannot be empty.")

    target = as_lower(object_name)
    if not target:
        raise KeyError(f"Unable to normalise identifier '{object_name}'.")

    for obj in iter_objects(world_state):
        for key in ("name", "id", "label", "type"):
            value = obj.get(key)
            if isinstance(value, str) and as_lower(value) == target:
                return obj

    raise ValueError(f"Object '{object_name}' not found in the provided world state.")


def resolve_object_from_question(
    world_state: Mapping[str, Any],
    question: Mapping[str, Any],
    *keys: str,
) -> Mapping[str, Any]:
    """Resolve an object using explicit keys from the question payload."""
    for key in keys:
        value = question.get(key)
        if isinstance(value, str) and value.strip():
            try:
                return resolve_object_by_name(world_state, value)
            except ValueError:
                continue

    return resolve_single_object(world_state, question)


def ensure_vector_size(
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


def extract_vector_from_mapping(
    mapping: Mapping[str, Any], *keys: str
) -> Optional[Tuple[float, ...]]:
    for key in keys:
        if key not in mapping:
            continue
        vector = as_vector(mapping[key])
        if vector:
            return tuple(float(component) for component in vector)
    return None


def extract_position(obj: Mapping[str, Any]) -> Optional[Tuple[float, ...]]:
    position = extract_vector_from_mapping(
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
        return ensure_vector_size(position)

    pose = obj.get("pose")
    if isinstance(pose, Mapping):
        position = extract_vector_from_mapping(
            pose, "position", "pos", "location", "translation"
        )
        if position:
            return ensure_vector_size(position)

    motion = get_motion(obj)
    position = extract_vector_from_mapping(motion, "position", "pos", "location")
    if position:
        return ensure_vector_size(position)

    transform = obj.get("transform")
    if isinstance(transform, Sequence) and transform:
        vector = as_vector(transform[-3:])
        if vector:
            return ensure_vector_size(vector)

    return None


def extract_orientation(obj: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
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
            vector = as_vector(value)
            if vector:
                return {"vector": tuple(float(component) for component in vector)}

    motion = get_motion(obj)
    for key in ("orientation", "rotation"):
        value = motion.get(key)
        if isinstance(value, Mapping):
            return value
        vector = as_vector(value)
        if vector:
            return {"vector": tuple(float(component) for component in vector)}

    return None


def extract_motion_vector(
    obj: Mapping[str, Any], *keys: str
) -> Optional[Tuple[float, ...]]:
    motion = get_motion(obj)
    vector = extract_vector_from_mapping(motion, *keys)
    if vector:
        return ensure_vector_size(vector)
    return None


def extract_velocity_vector(obj: Mapping[str, Any]) -> Optional[Tuple[float, ...]]:
    vector = ensure_vector_size(
        obj, "velocity_vector", "velocity", "linear_velocity", "velocity_vector"
    )
    if vector:
        return vector

    motion = get_motion(obj)
    components = motion.get("velocity_components")
    if isinstance(components, Mapping):
        numeric_items = [components.get(axis) for axis in ("x", "y", "z")]
        if any(isinstance(item, (int, float)) for item in numeric_items):
            as_tuple = tuple(float(item or 0.0) for item in numeric_items)
            return ensure_vector_size(as_tuple)

    return None


def extract_acceleration_vector(obj: Mapping[str, Any]) -> Optional[Tuple[float, ...]]:
    return ensure_vector_size(
        obj, "acceleration_vector", "acceleration", "linear_acceleration"
    )


def extract_angular_velocity_vector(
    obj: Mapping[str, Any],
) -> Optional[Tuple[float, ...]]:
    return ensure_vector_size(
        obj, "angular_velocity_vector", "angular_velocity", "rotation_rate", "spin"
    )


def extract_numeric_attribute(
    obj: Mapping[str, Any], *keys: str, default: Optional[float] = None
) -> Optional[float]:
    for key in keys:
        if key not in obj:
            continue
        numeric = coerce_to_float(obj[key])
        if numeric is not None:
            return numeric
    return default


def extract_text_attribute(obj: Mapping[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def object_identifier(obj: Mapping[str, Any]) -> Optional[str]:
    for key in ("name", "id", "label", "type"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def distance_between(
    first: Optional[Sequence[Number]],
    second: Optional[Sequence[Number]],
) -> float:
    if not first or not second:
        return 0.0
    a = ensure_vector_size(tuple(cast(Number, component) for component in first))
    b = ensure_vector_size(tuple(cast(Number, component) for component in second))
    return math.sqrt(
        sum((a_component - b_component) ** 2 for a_component, b_component in zip(a, b))
    )
