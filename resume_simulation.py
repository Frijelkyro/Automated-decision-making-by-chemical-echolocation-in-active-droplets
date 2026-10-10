import ast
import os
import numpy as np


def _read_parameter_file(filename):
    parameters = {}

    with open(filename) as f:
        for line in f:
            if ":" not in line:
                continue

            key, value = map(str.strip, line.split(":", 1))

            try:
                parameters[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parameters[key] = value

    return parameters


def _read_concentration(filename, n_steps):
    with np.load(filename) as data:
        concentration = data["concentration"]

    c = np.zeros((n_steps, *concentration.shape), dtype=concentration.dtype)
    c[0] = concentration
    return c


def _read_particles(filename, num_particles, n_steps):
    with np.load(filename) as data:
        p = np.zeros((num_particles, n_steps, 2), dtype=data["x"].dtype)
        v = np.zeros((num_particles, n_steps, 2), dtype=data["vx"].dtype)
        theta = np.zeros((num_particles, n_steps), dtype=data["theta"].dtype)
        omega = np.zeros((num_particles, n_steps), dtype=data["omega"].dtype)

        p[:, 0, 0] = data["x"]
        p[:, 0, 1] = data["y"]
        v[:, 0, 0] = data["vx"]
        v[:, 0, 1] = data["vy"]
        theta[:, 0] = data["theta"]
        omega[:, 0] = data["omega"]

    return p, theta, v, omega


def resume_simulation_from_file(data, param_filename, maze, n_steps):
    state_filename = param_filename.removesuffix(".txt") + "_state_checkpoint.npz"
    param_filename = param_filename.removesuffix(".txt") + "_checkpoint.txt"

    if not os.path.exists(param_filename):
        param_filename = "data/param.txt.bak"
    if not os.path.exists(state_filename):
        state_filename = "data/param_state.npz.bak"
    print(f"Using recovery parameter file: {param_filename}")

    old = _read_parameter_file(param_filename)

    with np.load(state_filename) as state:
        birth_times = state["birth_times"]
        active_mask = state["active_mask"]
        dead_tracker = state["dead_tracker"]
        exit_trigger_time = state["exit_trigger_time"]
        resume_step = int(state["resume_step"])
        resume_time = float(state["simulation_time"])

    num_particles = int(old["num_particles"])
    conc_prefix = str(old["file_prefix_conc"])
    part_prefix = str(old["file_prefix_part"])

    for prefix in (conc_prefix, part_prefix):
        prefix = os.path.basename(prefix)
        for filename in os.listdir(data):
            if filename.startswith(prefix + "_") and filename.endswith(".npz"):
                step = int(filename.rsplit("_", 1)[1][:-4])
                if step > resume_step:
                    os.remove(os.path.join(data, filename))

    conc = _read_concentration(
        f"{conc_prefix}_{resume_step}.npz",
        n_steps,
    )

    p, theta, v, omega = _read_particles(
        f"{part_prefix}_{resume_step}.npz",
        num_particles,
        n_steps,
    )
    print(f"using particle file: {part_prefix}_{resume_step}.npz")
    print(f"using conc file: {conc_prefix}_{resume_step}.npz")

    return (
        conc,
        p,
        theta,
        v,
        omega,
        active_mask,
        dead_tracker,
        exit_trigger_time,
        birth_times,
        resume_step,
        resume_time,
        float(old["dt"]),
    )
