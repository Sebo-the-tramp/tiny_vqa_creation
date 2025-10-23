from utils.helpers import _extract_attributes, _resolve_attributes, _fill_template


# original but pretty much difficult to customize
# def with_resolved_attributes(func):
#     def wrapper(world_state, question, *args, **kwargs):
#         attributes = _extract_attributes(question)
#         # print("I AM CALLED FROM function:", func.__name__)
#         resolved_attributes = _resolve_attributes(
#             world_state=world_state,
#             attributes=attributes.get("attributes", []),
#         )

#         # Useful attributes without need of recomputation every time in each function
#         timestep_start = "0.01"  # as first timestep is always 0.01
#         timestep_end = str(0.01 * len(world_state["simulation"]) - 0.01)

#         kwargs.update(
#             {
#                 "timestep_start": timestep_start,
#                 "timestep_end": timestep_end,
#             }
#         )

#         # adaptor part to original names format
#         for obj_id, object in world_state["objects"].items():
#             object["id"] = obj_id
#             object["name"] = object["model"]
#             # object.pop('model', None)

#         _fill_template(question, resolved_attributes)

#         # Pass them along so the wrapped function can use them
#         return func(world_state, question, resolved_attributes, *args, **kwargs)

#     return wrapper


def with_resolved_attributes(func):
    def wrapper(world_state, question, *args, **kwargs):
        attributes = _extract_attributes(question)

        # Useful attributes without need of recomputation every time in each function
        list_timesteps = list(world_state["simulation"].keys())
        timestep_start = list_timesteps[0]
        timestep_end = list_timesteps[-1]

        kwargs.update(
            {
                "timestep_start": timestep_start,
                "timestep_end": timestep_end,
            }
        )

        # adaptor part to original names format
        for obj_id, object in world_state["objects"].items():
            object["id"] = obj_id
            object["name"] = object["model"]
            # object.pop('model', None)

        # Pass them along so the wrapped function can use them
        return func(world_state, question, attributes["attributes"], *args, **kwargs)

    return wrapper
