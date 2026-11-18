import ast
import os
import random
import numpy as np
from storage import read_concentration


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


def _read_concentration(filename, n_steps, maze=None):
    concentration = read_concentration(filename, os.path.dirname(filename), maze)
    if concentration.ndim == 2:
        shape = concentration.shape
    else:
        shape = concentration.shape[-2:]

    c = np.zeros((n_steps, *shape), dtype=concentration.dtype)
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
    checkpoint_filename = os.path.join(data, "ckpt_previous.npz")
    if not os.path.exists(checkpoint_filename):
        raise FileNotFoundError(
            f"Checkpoint bundle not found: {checkpoint_filename}"
        )
    print(f"Using recovery checkpoint bundle: {checkpoint_filename}")

    with np.load(checkpoint_filename, allow_pickle=True) as state:
        birth_times = state["birth_times"].copy()
        active_mask = state["active_mask"].copy()
        dead_tracker = state["dead_tracker"].copy()
        exit_trigger_time = state["exit_trigger_time"].copy()
        resume_step = int(state["step"] if "step" in state else state["resume_step"])
        resume_time = float(state["time"] if "time" in state else state["simulation_time"])
        num_particles = int(state["num_particles"]) if "num_particles" in state else int(
            _read_parameter_file(param_filename)["num_particles"]
        )
        dt = float(state["dt"]) if "dt" in state else float(
            _read_parameter_file(param_filename)["dt"]
        )
        conc_prefix = os.path.join(
            data,
            os.path.basename(str(state["file_prefix_conc"])) if "file_prefix_conc" in state else "conc",
        )
        part_prefix = os.path.join(
            data,
            os.path.basename(str(state["file_prefix_part"])) if "file_prefix_part" in state else "part",
        )
        if "concentration" in state:
            concentration = state["concentration"].copy()
        elif "conc" in state:
            concentration = state["conc"].copy()
        else:
            concentration = None
        particles = state["particles"].copy() if "particles" in state else None
        theta = state["theta"].copy() if "theta" in state else None
        velocity = state["velocity"].copy() if "velocity" in state else None
        ang_velocity = state["ang_velocity"].copy() if "ang_velocity" in state else None
        extra_state = state["extra_state"].item() if "extra_state" in state else {}

    for prefix in (conc_prefix, part_prefix):
        prefix = os.path.basename(prefix)
        for filename in os.listdir(data):
            if filename.startswith(prefix + "_") and filename.endswith(".npz"):
                step = int(filename.rsplit("_", 1)[1][:-4])
                if step > resume_step:
                    os.remove(os.path.join(data, filename))

    if concentration is None:
        conc = _read_concentration(
            f"{conc_prefix}_{resume_step}.npz",
            n_steps,
            maze,
        )
    else:
        conc = np.zeros((n_steps, *concentration.shape), dtype=concentration.dtype)
        conc[0] = concentration

    if particles is None:
        p, theta, v, omega = _read_particles(
            f"{part_prefix}_{resume_step}.npz",
            num_particles,
            n_steps,
        )
    else:
        p = np.zeros((num_particles, n_steps, 2), dtype=particles.dtype)
        v = np.zeros((num_particles, n_steps, 2), dtype=velocity.dtype)
        theta_out = np.zeros((num_particles, n_steps), dtype=theta.dtype)
        omega = np.zeros((num_particles, n_steps), dtype=ang_velocity.dtype)
        p[:, 0] = particles
        v[:, 0] = velocity
        theta_out[:, 0] = theta
        omega[:, 0] = ang_velocity
        theta = theta_out

    if isinstance(extra_state, dict):
        numpy_state = extra_state.get("numpy_rng")
        python_state = extra_state.get("python_rng")
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        if python_state is not None:
            random.setstate(python_state)

    print(f"using particle checkpoint bundle at step {resume_step}")
    print(f"using concentration checkpoint bundle at step {resume_step}")

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
        dt,
    )
