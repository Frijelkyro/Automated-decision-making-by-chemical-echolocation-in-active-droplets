import os
import re
import ast
import numpy as np


def _find_latest_file(data, prefix):
    """
    Find the file with the largest integer timestep:
        part_123.txt
        conc_123.txt
    """
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.txt$")

    candidates = []

    for filename in os.listdir(data):
        match = pattern.match(filename)
        if match:
            timestep = int(match.group(1))
            candidates.append((timestep, os.path.join(data, filename)))

    if not candidates:
        raise FileNotFoundError(
            f"No files matching '{prefix}_<timestep>.txt' found in {data}"
        )

    return max(candidates, key=lambda x: x[0])


def _read_parameter_file(param_filename):
    
    """
    Read the simple 'key: value' parameter file produced by write_parameters().
    """
    parameters = {}

    with open(param_filename, "r") as f:
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
    maze,
    num_particles,
    n_steps,
    emitter_position,
    emission_rate,
):
    """
    Resume a simulation from the latest particle/concentration files.

    Returns:
        conc
        p
        theta
        v
        omega
        active_mask
        dead_tracker
        exit_trigger_time
        birth_times
        resume_step
        resume_time
        old_dt
    """

    # ------------------------------------------------------------
    # 1. Find latest saved timestep
    # ------------------------------------------------------------

    part_step, part_filename = _find_latest_file(data, "part")
    conc_step, conc_filename = _find_latest_file(data, "conc")

    if part_step != conc_step:
        raise ValueError(
            f"Latest particle timestep ({part_step}) does not match "
            f"latest concentration timestep ({conc_step}).\n"
            f"Particle file: {part_filename}\n"
            f"Concentration file: {conc_filename}"
        )

    resume_step = part_step

    # ------------------------------------------------------------
    # 2. Read old simulation parameters
    # ------------------------------------------------------------

    param_files = []

    for filename in os.listdir(data):
        match = re.fullmatch(r"param(\d+)\.txt", filename)
        if match:
            param_files.append(
                (int(match.group(1)), os.path.join(data, filename))
            )

    if not param_files:
        raise FileNotFoundError(f"No paramXX.txt files found in {data}")

    _, param_filename = max(param_files, key=lambda x: x[0])

    print(f"Using parameter file: {param_filename}")
    
    param_filename = os.path.join(data, "param.txt")
    old_parameters = _read_parameter_file(param_filename)

    if "dt" not in old_parameters:
        raise ValueError(
            f"'dt' was not found in {param_filename}."
        )

    old_dt = float(old_parameters["dt"])

    # Physical time represented by the last saved timestep
    resume_time = resume_step * old_dt

    # ------------------------------------------------------------
    # 3. Load concentration
    # ------------------------------------------------------------

    conc = _read_concentration(
        conc_filename,
        maze,
        n_steps,
    )

    # ------------------------------------------------------------
    # 4. Load particle state
    # ------------------------------------------------------------

    p, theta, v, omega = _read_particles(
        part_filename,
        num_particles,
        n_steps,
    )

    # ------------------------------------------------------------
    # 5. Determine which particles are alive
    #
    # NaN position = particle was already reaped.
    # ------------------------------------------------------------

    particle_is_nan = np.isnan(p[:, 0, :]).any(axis=1)

    dead_tracker = particle_is_nan.copy()
    active_mask = ~particle_is_nan

    # A particle which has not yet been emitted is sitting at the
    # emitter position with zero velocity in the saved file.
    #
    # Reconstruct birth times from the ORIGINAL emission schedule.
    birth_times = np.array(
        [i / emission_rate for i in range(num_particles)],
        dtype=np.float64,
    )

    # Particles whose birth time is still in the future are inactive.
    active_mask &= birth_times <= resume_time

    # Particles not yet born are neither dead nor active.
    # They remain at the emitter position and will be activated later.
    # ------------------------------------------------------------

    exit_trigger_time = np.full(
        num_particles,
        np.inf,
        dtype=np.float64,
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