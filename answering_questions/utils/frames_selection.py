MAX_FRAMES = 8


def uniformly_sample_frames(world_state):
    total_steps = len(world_state["simulation"])
    delta_fram_img = total_steps // (MAX_FRAMES - 1)  # max are 8 images
    imgs_idx = [x for x in range(0, total_steps - 1, delta_fram_img)]
    return imgs_idx


def sample_frames_at_timesteps(world_state, timesteps):
    total_steps = len(world_state["simulation"])
    imgs_idx = []
    for t in timesteps:
        idx = str(world_state["simulation"][t]["simstep"]).zfill(6)
        imgs_idx.append(idx)
    return imgs_idx


def uniformly_sample_frames_start_end_delta(start, end, delta):
    imgs_idx = [str(x).zfill(6) for x in range(start, end, delta)]
    return imgs_idx
