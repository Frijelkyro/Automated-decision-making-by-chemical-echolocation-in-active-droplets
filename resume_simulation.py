import os
import re
import ast
import numpy as np
from list_of_functions import get_param_filename

def _read_parameter_file(filename):
    parameters = {}
    with open(filename) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if ":" not in lines[i]:
            i += 1
            continue

        key, value = lines[i].split(":", 1)
        value = value.strip()
        while value.count("[") > value.count("]"):
            i += 1
            value += " " + lines[i].strip()

        try:
            parameters[key.strip()] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parameters[key.strip()] = value
        i += 1

    return parameters


def _read_concentration(filename, maze, n_steps):
    """
    Read the concentration stored in conc_<timestep>.txt.

    The file contains:
        TIMESTEP:
        <number>
        DATA:c
        <flattened concentration>
    """
    nx, ny = maze.shape

    c = np.zeros((n_steps, nx, ny), dtype=np.float32)

    data = np.loadtxt(filename, skiprows=5)

    expected_size = nx * ny

    if data.size != expected_size:
        raise ValueError(
            f"Concentration file {filename} contains {data.size} values, "
            f"but maze requires {expected_size}."
        )

    c[0] = data.reshape((nx, ny))

    return c


def _read_particles(filename, num_particles, n_steps):
    """
    Read particle state from part_<timestep>.txt.

    Returns arrays with the loaded state stored at local index 0.
    """
    p = np.zeros((num_particles, n_steps, 2), dtype=np.float32)
    theta = np.zeros((num_particles, n_steps), dtype=np.float32)
    v = np.zeros((num_particles, n_steps, 2), dtype=np.float32)
    omega = np.zeros((num_particles, n_steps), dtype=np.float32)

    data = np.genfromtxt(
        filename,
        skip_header=5,
        dtype=np.float64,
    )

    if data.ndim == 1:
        data = data[np.newaxis, :]

    if data.shape[1] < 7:
        raise ValueError(
            f"Particle file {filename} has only {data.shape[1]} columns; "
            "at least 7 are required."
        )

    for row in data:
        particle_id = int(row[0])

        if particle_id < 0 or particle_id >= num_particles:
            raise ValueError(
                f"Particle ID {particle_id} in {filename} is outside "
                f"0..{num_particles - 1}."
            )

        p[particle_id, 0, 0] = row[1]
        p[particle_id, 0, 1] = row[2]

        theta[particle_id, 0] = row[3]

        v[particle_id, 0, 0] = row[4]
        v[particle_id, 0, 1] = row[5]

        omega[particle_id, 0] = row[6]

    return p, theta, v, omega


def resume_simulation_from_file(data, param_filename, maze, n_steps):
    print(f"Using parameter file: {param_filename}")
    old = _read_parameter_file(param_filename)

    state_filename = param_filename.removesuffix(".bak").removesuffix(".txt") + "_state.npz"
    state = np.load(state_filename + ".bak")

    birth_times = state["birth_times"]
    active_mask = state["active_mask"]
    dead_tracker = state["dead_tracker"]
    exit_trigger_time = state["exit_trigger_time"]
    resume_step = int(state["resume_step"])
    resume_time = float(state["simulation_time"])
    num_particles = int(old["num_particles"])

    for prefix in ("conc", "part"):
        for f in os.listdir(data):
            m = re.fullmatch(rf"{prefix}_(\d+)\.txt", f)
            if m and int(m.group(1)) > resume_step:
                os.remove(os.path.join(data, f))

    conc = _read_concentration(f"{old['file_prefix_conc']}_{resume_step}.txt", maze, n_steps)
    p, theta, v, omega = _read_particles(f"{old['file_prefix_part']}_{resume_step}.txt", num_particles, n_steps)

    return conc, p, theta, v, omega, active_mask, dead_tracker, exit_trigger_time, birth_times, resume_step, resume_time, float(old["dt"])