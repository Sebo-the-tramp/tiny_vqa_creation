from utils.helpers import (
    _extract_attributes,
    _resolve_attributes,
)


def with_resolved_attributes(func):
    def wrapper(world_state, question, *args, **kwargs):
        attributes = _extract_attributes(question)
        resolved_attributes = _resolve_attributes(
            world_state=world_state,
            attributes=attributes.get("attributes", []),
        )

        # Useful attributes without need of recomputation every time in each function
        timestep_start = "0.01"  # as first timestep is always 0.01
        timestep_end = str(0.01 * len(world_state["simulation_steps"]) - 0.01)

        kwargs.update(
            {
                "timestep_start": timestep_start,
                "timestep_end": timestep_end,
            }
        )

        # Pass them along so the wrapped function can use them
        return func(world_state, question, resolved_attributes, *args, **kwargs)

    return wrapper
