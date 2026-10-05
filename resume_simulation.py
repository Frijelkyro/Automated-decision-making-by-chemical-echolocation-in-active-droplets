import os
import re
import ast
import numpy as np
from list_of_functions import get_param_filename

def _read_parameter_file(data,param_filename):
    
    """
    Read the simple 'key: value' parameter file produced by write_parameters().
    """
    parameters = {}

    with open(data+"/"+param_filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            try:
                parameters[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parameters[key] = value

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

    data = np.loadtxt(filename, skiprows=3)

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
        skip_header=3,
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


def resume_simulation_from_file(
    data,
    param_filename,
    maze,
    n_steps
):
    """
    Resume a simulation from the current param.txt file.
    """
    # 1. Load params from the param.txt
    print(f"Using parameter file: {param_filename}")
    old_parameters = _read_parameter_file(data, param_filename)
    old_dt = float(old_parameters["dt"])

    dead_tracker = np.array(old_parameters["dead_tracker"], dtype=bool)
    active_mask = np.array(old_parameters["active_mask"], dtype=bool)
    birth_times = np.array(old_parameters["birth_times"], dtype=float)
    file_prefix_conc = str(old_parameters["file_prefix_conc"])
    file_prefix_part = str(old_parameters["file_prefix_part"])
    num_particles = int(old_parameters["num_particles"])
    exit_trigger_time = np.array(old_parameters["exit_trigger_time"], dtype=float)

    resume_step = int(old_parameters["resume_step"])
    # remove all simulation written files that were created on or after the resume step:
    for prefix in ("conc", "part"):
        for filename in os.listdir(data):
            match = re.fullmatch(rf"{prefix}_(\d+)\.txt", filename)
            if match and int(match.group(1)) >= resume_step:
                os.remove(os.path.join(data, filename))

    resume_time = float(old_parameters["simulation_time"])

    conc = _read_concentration(
        file_prefix_conc,
        maze,
        n_steps,
    )
    p, theta, v, omega = _read_particles(
        file_prefix_part,
        num_particles,
        n_steps,
    )

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
        old_dt,
    )