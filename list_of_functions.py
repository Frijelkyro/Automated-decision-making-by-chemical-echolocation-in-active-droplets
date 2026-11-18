import os
import re
import shutil
from dataclasses import dataclass

import numpy as np
from maze_functions import get_exit_wall_mask, leaky_exit_wall
from storage import (
    build_free_index,
    maze_hash,
    save_free_index,
    write_concentration,
)


@dataclass
class SimParams:
    dx: float
    dc: float
    dp: float
    bp: float
    moving_source_production_strength: float
    static_source_production_strength: float
    moving_source_decay_rate: float
    static_source_decay_rate: float
    sp_decay_rate: float
    global_decay_strength: float
    self_propulsion_speed: float
    self_propulsion_frequency: float
    dr: float
    particle_mass: float
    particle_moment_inertia: float
    lx: float
    ly: float
    n_xbins: int
    n_ybins: int
    n_steps: int
    dt: float
    gamma: float
    time_loop: int
    part_every: int
    conc_every: int
    num_particles: int
    param_filename: str
    grid_filename: str
    file_prefix_conc: str
    file_prefix_part: str
    advection: bool
    massive_particle: bool
    moving_source_decay: bool
    static_source_decay: bool
    epsilon_lj: float
    static_source_position: tuple
    exit_radius: float
    birth_times: np.ndarray
    active_mask: np.ndarray
    dead_tracker: np.ndarray
    death_zone_map: np.ndarray
    exit_zone_map: np.ndarray
    birth_zone_map: np.ndarray
    grim_reaper_delay: float
    exit_trigger_time: np.ndarray
    emitter_position: np.ndarray
    permeability: float
    exit_wall_mask: np.ndarray
    maze_file: str
    maze_config: str
    emitter_jitter: tuple
    run_id: object
    shot: int
    seed: int
    junction_tracker: object = None
    decision_output_dir: str = ""
    decision_flush_loops: int = 100

    def to_parameter_dict(self):
        return {
            "Dc": self.dc,
            "Dp": self.dp,
            "Bp": self.bp,
            "moving_source_production_strength": self.moving_source_production_strength,
            "sp_decay_rate": self.sp_decay_rate,
            "moving_source_decay_rate": self.moving_source_decay_rate,
            "static_source_decay_rate": self.static_source_decay_rate,
            "global_decay_strength": self.global_decay_strength,
            "self_propulsion_speed": self.self_propulsion_speed,
            "self_propulsion_frequency": self.self_propulsion_frequency,
            "Dr": self.dr,
            "M": self.particle_mass,
            "J": self.particle_moment_inertia,
            "epsilon_LJ": self.epsilon_lj,
            "static_source_position": self.static_source_position,
            "static_source_production_strength": self.static_source_production_strength,
            "advection": self.advection,
            "massive_particle": self.massive_particle,
            "dx": self.dx,
            "Lx": self.lx,
            "Ly": self.ly,
            "n_xbins": self.n_xbins,
            "n_ybins": self.n_ybins,
            "n_steps": self.n_steps,
            "dt": self.dt,
            "gamma": self.gamma,
            "part_every": self.part_every,
            "conc_every": self.conc_every,
            "num_particles": self.num_particles,
            "time_loop": self.time_loop,
            "birth_times": self.birth_times,
            "active_mask": self.active_mask,
            "dead_tracker": self.dead_tracker,
            "death_zone_map": self.death_zone_map,
            "exit_zone_map": self.exit_zone_map,
            "birth_zone_map": self.birth_zone_map,
            "grim_reaper_delay": self.grim_reaper_delay,
            "exit_trigger_time": self.exit_trigger_time,
            "emitter_position": tuple(self.emitter_position),
            "param_filename": self.param_filename,
            "grid_filename": self.grid_filename,
            "file_prefix_conc": self.file_prefix_conc,
            "file_prefix_part": self.file_prefix_part,
            "exit_radius": self.exit_radius,
            "exit_wall_mask": self.exit_wall_mask,
            "permeability": self.permeability,
            "maze_file": self.maze_file,
            "maze_config": self.maze_config,
            "emitter_jitter": self.emitter_jitter,
            "run_id": self.run_id,
            "shot": self.shot,
            "seed": self.seed,
        }


@dataclass
class SolverResult:
    concentration: np.ndarray
    position: np.ndarray
    theta: np.ndarray
    velocity: np.ndarray
    ang_velocity: np.ndarray
    forces_self_propulsion: np.ndarray
    forces_chemotaxis: np.ndarray
    forces_interaction: np.ndarray
    forces_wall: np.ndarray
    finished: bool
    exit_timestep: float
    exit_trigger_time: np.ndarray

# Define the initial conditions
def initialize_c(c_initial, n_steps, maze):
    # Initialize solution: the grid of u(k, i, j)
    nx = maze.shape[0]
    ny = maze.shape[1]
    c = np.full((n_steps, nx, ny), c_initial, dtype=np.float32)

    return c


def active_particle_mask(simulation_time, birth_times):
    return simulation_time >= birth_times


# define the source function for a moving point source
def moving_point_source(
    s,
    production_strength,
    local_step,
    simulation_time,
    particle_positions,   # (n_active, 2)
    particle_birth_times, # (n_active,)
    dx,
    moving_source_decay_rate,
    n_steps,
):
    t = local_step % n_steps
    bins = np.rint(particle_positions / dx).astype(int)
    age = simulation_time - particle_birth_times
    strength = (production_strength / dx**2) * np.exp(-moving_source_decay_rate * age)
    np.add.at(s[t], (bins[:, 0], bins[:, 1]), strength)
    return s

# define the source function for a static point source
def static_point_source(
    s,
    production_strength,
    local_step,
    simulation_time,
    source_position,
    dx,
    static_source_decay_rate,
    n_steps,
):
    t = local_step % n_steps
    x_bin = int(np.rint(source_position[0] / dx))
    y_bin = int(np.rint(source_position[1] / dx))
    s[t, x_bin, y_bin] += (production_strength / (dx**2)) * np.exp(
        -simulation_time * static_source_decay_rate
    )
    return s


# update the flow field based on the distance from the particle
def update_flow_field(
    ux, uy, timestep, particle_positions, dx, dt, moving_source_decay_rate, n_steps
):
    t = timestep % n_steps
    # Calculate the shape of the flow velocity arrays
    nx, ny = ux[t, :, :].shape

    # Create mesh grids of indices
    x_indices, y_indices = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")

    for particle_id in range(particle_positions.shape[0]):
        # Calculate the distances from the particle coordinates to grid points
        dist_x = x_indices * dx - particle_positions[particle_id, 0]
        dist_y = y_indices * dx - particle_positions[particle_id, 1]

        # Calculate the distances using the 1/r^n dependency, avoiding division by zero
        r = np.sqrt(dist_x**2 + dist_y**2)
        # r[r < 1e-6] = 1e-6  # Avoid division by zero by adding a small value

        # Calculate the flow field based on the 1/r^n dependency
        # n=1
        # ux_update = 10**(-1) * dist_x / r**(n+1)
        # uy_update = 10**(-1) * dist_y / r**(n+1)

        # Calculate the flow field based on the re^(-r) dependency
        ux_update = (
            2.3
            * 1e4
            * dist_x
            * np.exp(-3.5 * r)
            * np.exp(-timestep * dt * moving_source_decay_rate)
        )
        uy_update = (
            2.3
            * 1e4
            * dist_y
            * np.exp(-3.5 * r)
            * np.exp(-timestep * dt * moving_source_decay_rate)
        )

        # Update the flow field
        ux[t, :, :] += ux_update
        uy[t, :, :] += uy_update

    return ux, uy


# Define the no-flux boundary condition for a maze
def noflux_maze(
    conc,
    t,
    x_diff_indices,
    y_diff_indices,
    x_diff_neg_indices,
    y_diff_neg_indices,
    maze,
    regular_wall_mask=None,
):
    conc[t, x_diff_indices[:, 0], x_diff_indices[:, 1] + 1] = conc[
        t, x_diff_indices[:, 0], x_diff_indices[:, 1] + 2
    ]
    conc[t, x_diff_neg_indices[:, 0], x_diff_neg_indices[:, 1]] = conc[
        t, x_diff_neg_indices[:, 0], x_diff_neg_indices[:, 1] - 1
    ]

    conc[t, y_diff_indices[:, 0] + 1, y_diff_indices[:, 1]] = conc[
        t, y_diff_indices[:, 0] + 2, y_diff_indices[:, 1]
    ]
    conc[t, y_diff_neg_indices[:, 0], y_diff_neg_indices[:, 1]] = conc[
        t, y_diff_neg_indices[:, 0] - 1, y_diff_neg_indices[:, 1]
    ]

    if regular_wall_mask is not None:
        conc[t, regular_wall_mask] = 0.0
    else:
        conc[t, maze == 0] = 0.0

    return conc


def chemotaxis_force(position, t, c, maze, dx, bp, active_mask=None):
    num_particles = position.shape[0]
    all_forces = np.zeros((num_particles, 2))
    if active_mask is None:
        active_mask = np.ones(num_particles, dtype=bool)

    for particle_id in np.where(active_mask)[0]:
        x_bin = int(np.rint(position[particle_id, t, 0] / dx))
        y_bin = int(np.rint(position[particle_id, t, 1] / dx))

        particle_force = np.zeros(2)

        if maze[x_bin, y_bin] != 0:
            particle_force[0] = (
                bp
                * (c[t + 1, x_bin + 1, y_bin] - c[t + 1, x_bin - 1, y_bin])
                / (2.0 * dx)
            )
            particle_force[1] = (
                bp
                * (c[t + 1, x_bin, y_bin + 1] - c[t + 1, x_bin, y_bin - 1])
                / (2.0 * dx)
            )

        if maze[x_bin + 1, y_bin] == 0 or maze[x_bin - 1, y_bin] == 0:
            particle_force[0] = 0.0
            particle_force[1] = (
                bp
                * (c[t + 1, x_bin, y_bin + 1] - c[t + 1, x_bin, y_bin - 1])
                / (2.0 * dx)
            )

        if maze[x_bin, y_bin + 1] == 0 or maze[x_bin, y_bin - 1] == 0:
            particle_force[0] = (
                bp
                * (c[t + 1, x_bin + 1, y_bin] - c[t + 1, x_bin - 1, y_bin])
                / (2.0 * dx)
            )
            particle_force[1] = 0.0

        if (maze[x_bin + 1, y_bin] == 0 or maze[x_bin - 1, y_bin] == 0) and (
            maze[x_bin, y_bin + 1] == 0 or maze[x_bin, y_bin - 1] == 0
        ):
            particle_force[0] = 0.0
            particle_force[1] = 0.0

        all_forces[particle_id] = particle_force

    return all_forces


def interaction_force(position, t, dx, epsilon_lj, active_mask=None):
    num_particles = position.shape[0]
    interaction_forces = np.zeros((num_particles, num_particles, 2))
    if active_mask is None:
        active_mask = np.ones(num_particles, dtype=bool)

    active_indices = np.where(active_mask)[0]
    sigma = dx

    for i_index in range(len(active_indices)):
        i = active_indices[i_index]
        for j_index in range(i_index + 1, len(active_indices)):
            j = active_indices[j_index]
            disp = position[j, t, :] - position[i, t, :]
            dist = np.linalg.norm(disp)

            if dist != 0:
                lj_force = (
                    epsilon_lj
                    * (12.0 * (sigma / dist) ** 12 - 6.0 * (sigma / dist) ** 6)
                    * disp
                    / dist**2
                )
                interaction_forces[i, j] = -lj_force
                interaction_forces[j, i] = lj_force

    return interaction_forces


def wall_force(position, t, wall_coords, dx, active_mask=None):
    num_particles = position.shape[0]
    if active_mask is None:
        active_mask = np.ones(num_particles, dtype=bool)

    r_cut = 2.0 * dx
    sigma = r_cut / 2.0 ** (1.0 / 6.0)
    forces = np.zeros((num_particles, 2))

    for i in np.where(active_mask)[0]:
        disp_from_wall = position[i, t, :] - wall_coords
        dist = np.linalg.norm(disp_from_wall, axis=1)
        if min(dist) < r_cut:
            index = np.argmin(dist)

            # WCA potential
            epsilon_wca = 10.0
            force_magnitude = (
                4.0
                * epsilon_wca
                * (
                    12.0 * (sigma / dist[index]) ** 12
                    - 6.0 * (sigma / dist[index]) ** 6
                )
            )
            forces[i] = force_magnitude * disp_from_wall[index] / (dist[index] ** 2)

            # Soft potential
            # epsilon_soft = 10000.0
            # force_magnitude = epsilon_soft * (np.pi / r_cut) * np.sin(np.pi * dist[index] / r_cut)
            # forces[i] = force_magnitude * disp_from_wall[index] / dist[index]

    return forces


def self_propulsion_force(
    position,
    local_step,
    simulation_time,
    theta,
    sp_decay_rate,
    n_steps,
    self_propulsion_speed,
    active_mask=None,
):
    t = local_step % n_steps
    num_particles = position.shape[0]

    forces = np.zeros((num_particles, 2), dtype=np.float32)

    if active_mask is None:
        active_mask = np.ones(num_particles, dtype=bool)

    decay = np.exp(-simulation_time * sp_decay_rate)

    active_indices = np.where(active_mask)[0]

    forces[active_indices, 0] = (
        self_propulsion_speed * np.cos(theta[active_indices, t]) * decay
    )
    forces[active_indices, 1] = (
        self_propulsion_speed * np.sin(theta[active_indices, t]) * decay
    )

    return forces


def write_parameters(**parameters):
    param_filename = parameters.get("param_filename", "parameters.txt")

    directory = os.path.dirname(param_filename)
    basename = os.path.basename(param_filename)

    if directory:
        os.makedirs(directory, exist_ok=True)

    # Find existing paramXX.txt files
    existing = []

    for filename in os.listdir(directory or "."):
        match = re.fullmatch(r"param(\d+)\.txt", filename)
        if match:
            existing.append(int(match.group(1)))

    # First file is param00.txt
    next_number = max(existing, default=-1) + 1
    numbered_filename = os.path.join(
        directory or ".",
        f"param{next_number:02d}.txt",
    )
    with open(numbered_filename, "w") as param_file:
        for key, value in parameters.items():
            param_file.write(f"{key}: {value}\n")

    return numbered_filename


def write_param_snapshot(
    parameters,
    simulation_time,
    resume_step,
    concentration=None,
    particles=None,
    theta=None,
    velocity=None,
    ang_velocity=None,
    extra_state=None,
):
    """Atomically write the current and previous checkpoint generations."""
    param_filename = parameters.get("param_filename", "parameters.txt")
    data_dir = os.path.dirname(param_filename) or "."
    os.makedirs(data_dir, exist_ok=True)
    current = os.path.join(data_dir, "ckpt_current.npz")
    previous = os.path.join(data_dir, "ckpt_previous.npz")
    temp = os.path.join(data_dir, f".ckpt_current.{os.getpid()}.tmp.npz")

    if concentration is None:
        concentration = np.empty((0,), dtype=np.float32)
    if particles is None:
        particles = np.empty((0,), dtype=np.float32)
    if theta is None:
        theta = np.empty((0,), dtype=np.float32)
    if velocity is None:
        velocity = np.empty((0, 2), dtype=np.float32)
    if ang_velocity is None:
        ang_velocity = np.empty((0,), dtype=np.float32)

    active_mask = parameters["active_mask"]
    dead_tracker = parameters["dead_tracker"]
    masks = np.stack((active_mask, dead_tracker), axis=0)

    extra_state = {} if extra_state is None else extra_state
    if isinstance(extra_state, dict):
        extra_state_array = np.empty((), dtype=object)
        extra_state_array[()] = extra_state
    else:
        extra_state_array = extra_state

    try:
        with open(temp, "wb") as checkpoint_file:
            np.savez(
                checkpoint_file,
                conc=concentration,
                concentration=concentration,
                particles=particles,
                theta=theta,
                velocity=velocity,
                ang_velocity=ang_velocity,
                masks=masks,
                active_mask=active_mask,
                dead_tracker=dead_tracker,
                birth_times=parameters["birth_times"],
                exit_times=parameters["exit_trigger_time"],
                exit_trigger_time=parameters["exit_trigger_time"],
                step=resume_step,
                resume_step=resume_step,
                time=simulation_time,
                simulation_time=simulation_time,
                extra_state=extra_state_array,
                dt=float(parameters["dt"]),
                num_particles=int(parameters["num_particles"]),
                file_prefix_conc=str(parameters["file_prefix_conc"]),
                file_prefix_part=str(parameters["file_prefix_part"]),
                maze_config=str(parameters["maze_config"]),
                static_source_position=np.asarray(parameters["static_source_position"], dtype=float),
                exit_radius=float(parameters["exit_radius"]),
            )
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())

        if os.path.exists(current):
            os.replace(current, previous)
        else:
            previous_temp = os.path.join(
                data_dir, f".ckpt_previous.{os.getpid()}.tmp.npz"
            )
            shutil.copy2(temp, previous_temp)
            os.replace(previous_temp, previous)
        os.replace(temp, current)
    finally:
        if os.path.exists(temp):
            os.remove(temp)


def get_param_filename(data):
    param_files = []

    for filename in os.listdir(data):
        match = re.fullmatch(r"param(\d+)\.txt", filename)
        if match:
            number = int(match.group(1))
            param_files.append((number, filename))

    if not param_files:
        return None

    # Select the file with the highest integer
    _, filename = max(param_files)
    filename = str(filename)
    return filename


# Load dt from the parameter file with the highest integer suffix
def get_dt_from_params(data):

    filename = get_param_filename(data)

    with open(data + "/" + filename, "r") as file:
        for line in file:
            parts = line.strip().split(":")
            if len(parts) == 2 and parts[0].strip() == "dt":
                return float(parts[1].strip())

    return None


def write_grid(grid_filename, nx, ny):
    with open(grid_filename, "w") as f:
        f.write("BOX:\n")
        f.write(f"0 {nx}\n")
        f.write(f"0 {ny}\n")
        f.write("SHAPE:\n")
        f.write(f"{nx} {ny}\n")
        f.write("COORDINATES: X Y\n")

        for i in range(nx):
            for j in range(ny):
                f.write(f"{i} {j}\n")


def save_snapshot(
    file_prefix_conc,
    file_prefix_part,
    concentration,
    particles,
    theta,
    velocity,
    ang_velocity,
    forces_self_propulsion,
    forces_chemotaxis,
    forces_interaction,
    forces_wall,
    timestep,
    simulation_time,
    n_steps,
    save_forces=True,
    maze=None,
):
    conc_filename = f"{file_prefix_conc}_{timestep}.npz"
    conc_snapshot = concentration[timestep % n_steps]
    if maze is None:
        np.savez(
            conc_filename,
            timestep=timestep,
            simulation_time=simulation_time,
            concentration=conc_snapshot,
        )
    else:
        directory = os.path.dirname(file_prefix_conc) or "."
        save_free_index(directory, maze)
        free_index = build_free_index(maze)
        write_concentration(
            conc_filename,
            conc_snapshot,
            free_index,
            maze_hash_value=maze_hash(maze),
            timestep=timestep,
            simulation_time=simulation_time,
        )
    t = timestep % particles.shape[1]
    data = {
        "timestep": timestep,
        "simulation_time": simulation_time,
        "x": particles[:, t, 0],
        "y": particles[:, t, 1],
        "vx": velocity[:, t, 0],
        "vy": velocity[:, t, 1],
        "theta": theta[:, t],
        "omega": ang_velocity[:, t],
    }

    if save_forces:
        data["f_self_propulsion"] = forces_self_propulsion
        data["f_chemotaxis"] = forces_chemotaxis
        data["f_interaction"] = forces_interaction.sum(axis=1)
        data["f_wall"] = forces_wall

    np.savez(f"{file_prefix_part}_{timestep}.npz", **data)


def save_concentration_snapshot(
    file_prefix_conc,
    concentration,
    timestep,
    simulation_time,
    n_steps,
    maze=None,
):
    filename = f"{file_prefix_conc}_{timestep}.npz"
    snapshot = concentration[timestep % n_steps]
    if maze is None:
        np.savez(
            filename,
            timestep=timestep,
            simulation_time=simulation_time,
            concentration=snapshot,
        )
        return
    directory = os.path.dirname(file_prefix_conc) or "."
    save_free_index(directory, maze)
    write_concentration(
        filename,
        snapshot,
        build_free_index(maze),
        maze_hash_value=maze_hash(maze),
        timestep=timestep,
        simulation_time=simulation_time,
    )


def save_particle_snapshot(
    file_prefix_part,
    particles,
    theta,
    velocity,
    ang_velocity,
    forces_self_propulsion,
    forces_chemotaxis,
    forces_interaction,
    forces_wall,
    timestep,
    simulation_time,
    save_forces=True,
):
    t = timestep % particles.shape[1]
    data = {
        "timestep": timestep,
        "simulation_time": simulation_time,
        "x": particles[:, t, 0],
        "y": particles[:, t, 1],
        "vx": velocity[:, t, 0],
        "vy": velocity[:, t, 1],
        "theta": theta[:, t],
        "omega": ang_velocity[:, t],
    }
    if save_forces:
        data["f_self_propulsion"] = forces_self_propulsion
        data["f_chemotaxis"] = forces_chemotaxis
        data["f_interaction"] = forces_interaction.sum(axis=1)
        data["f_wall"] = forces_wall
    np.savez(f"{file_prefix_part}_{timestep}.npz", **data)



def chemical_solver(
    c,
    position,
    theta,
    velocity,
    ang_velocity,
    maze,
    params,
    start_step=0,
    start_time=0.0,
):
    dx = params.dx
    dc = params.dc
    dp = params.dp
    bp = params.bp
    moving_source_production_strength = params.moving_source_production_strength
    static_source_production_strength = params.static_source_production_strength
    moving_source_decay_rate = params.moving_source_decay_rate
    static_source_decay_rate = params.static_source_decay_rate
    sp_decay_rate = params.sp_decay_rate
    global_decay_strength = params.global_decay_strength
    self_propulsion_speed = params.self_propulsion_speed
    self_propulsion_frequency = params.self_propulsion_frequency
    dr = params.dr
    particle_mass = params.particle_mass
    particle_moment_inertia = params.particle_moment_inertia
    lx = params.lx
    ly = params.ly
    n_xbins = params.n_xbins
    n_ybins = params.n_ybins
    n_steps = params.n_steps
    dt = params.dt
    gamma = params.gamma
    time_loop = params.time_loop
    part_every = params.part_every
    conc_every = params.conc_every
    num_particles = params.num_particles
    param_filename = params.param_filename
    grid_filename = params.grid_filename
    file_prefix_conc = params.file_prefix_conc
    file_prefix_part = params.file_prefix_part
    advection = params.advection
    massive_particle = params.massive_particle
    moving_source_decay = params.moving_source_decay
    static_source_decay = params.static_source_decay
    epsilon_lj = params.epsilon_lj
    static_source_position = params.static_source_position
    exit_radius = params.exit_radius
    birth_times = params.birth_times
    active_mask = params.active_mask
    dead_tracker = params.dead_tracker
    death_zone_map = params.death_zone_map
    birth_zone_map = params.birth_zone_map
    exit_zone_map = params.exit_zone_map
    grim_reaper_delay = params.grim_reaper_delay
    exit_trigger_time = params.exit_trigger_time
    emitter_position = np.array(params.emitter_position, dtype=np.float32)
    permeability = params.permeability
    exit_wall_mask = params.exit_wall_mask
    grim_reaper_delay_time = grim_reaper_delay

    nt, nx, ny = c.shape
    wall = maze == 0
    x_diff = np.diff(maze, axis=1)
    y_diff = np.diff(maze, axis=0)

    # Precalculate x_diff and y_diff indices
    x_diff_indices = np.transpose(np.where(x_diff == 1))
    y_diff_indices = np.transpose(np.where(y_diff == 1))
    x_diff_neg_indices = np.transpose(np.where(x_diff == -1))
    y_diff_neg_indices = np.transpose(np.where(y_diff == -1))

    # Filter boundary-transition indices and wall coordinates for leaky walls
    if permeability > 0.0 and exit_wall_mask is not None:
        in_exit_wall_x = exit_wall_mask[x_diff_indices[:, 0], x_diff_indices[:, 1]]
        x_diff_indices_reg = x_diff_indices[~in_exit_wall_x]

        in_exit_wall_x_neg = exit_wall_mask[
            x_diff_neg_indices[:, 0], x_diff_neg_indices[:, 1] + 1
        ]
        x_diff_neg_indices_reg = x_diff_neg_indices[~in_exit_wall_x_neg]

        in_exit_wall_y = exit_wall_mask[y_diff_indices[:, 0], y_diff_indices[:, 1]]
        y_diff_indices_reg = y_diff_indices[~in_exit_wall_y]

        in_exit_wall_y_neg = exit_wall_mask[
            y_diff_neg_indices[:, 0] + 1, y_diff_neg_indices[:, 1]
        ]
        y_diff_neg_indices_reg = y_diff_neg_indices[~in_exit_wall_y_neg]

        regular_wall_mask = (maze == 0) & ~exit_wall_mask
        wall_coords = np.transpose(np.where((maze == 0) & ~exit_wall_mask)) * dx
    else:
        x_diff_indices_reg = x_diff_indices
        x_diff_neg_indices_reg = x_diff_neg_indices
        y_diff_indices_reg = y_diff_indices
        y_diff_neg_indices_reg = y_diff_neg_indices
        regular_wall_mask = maze == 0
        wall_coords = np.transpose(np.where(maze == 0)) * dx

    source = np.full((nt, nx, ny), 0.0, dtype=np.float32)
    ux = np.full((nt, nx, ny), 0.0, dtype=np.float32)
    uy = np.full((nt, nx, ny), 0.0, dtype=np.float32)

    f_sp = np.full((num_particles, n_steps, 2), 0.0, dtype=np.float32)
    f_chem = np.full((num_particles, n_steps, 2), 0.0, dtype=np.float32)
    f_int = np.full((num_particles, n_steps, 2), 0.0, dtype=np.float32)
    f_wall = np.full((num_particles, n_steps, 2), 0.0, dtype=np.float32)

    finished = False
    exit_timestep = np.inf

    px_bins = np.zeros(num_particles, dtype=int)
    py_bins = np.zeros(num_particles, dtype=int)

    for local_step in range(nt - 1):

        timestep = start_step + local_step
        simulation_time = start_time + local_step * dt

        t = local_step
        t_next = local_step + 1  # goes from 1 to 49

        # Solver calls are separate rolling-buffer loops. Recompute the bins
        # from the particles' current positions before the spawn check.
        px_bins[:] = 0
        py_bins[:] = 0
        finite_positions = np.isfinite(position[:, t, 0]) & np.isfinite(position[:, t, 1])
        px_bins[finite_positions] = np.rint(position[finite_positions, t, 0] / dx).astype(int)
        py_bins[finite_positions] = np.rint(position[finite_positions, t, 1] / dx).astype(int)

        spawning_queue = np.where(
            (birth_times <= simulation_time) & ~active_mask & ~dead_tracker
        )[0]

        if len(spawning_queue) and not np.any(
            birth_zone_map[px_bins[active_mask], py_bins[active_mask]]
        ):
            i = spawning_queue[0]
            active_mask[i] = True
            birth_times[i] = simulation_time

        inactive_mask = ~active_mask
        position[inactive_mask, t_next, :] = position[inactive_mask, t, :]
        velocity[inactive_mask, t_next, :] = 0.0
        theta[inactive_mask, t_next] = theta[inactive_mask, t]
        ang_velocity[inactive_mask, t_next] = ang_velocity[inactive_mask, t]

        A = c[t, 2:, 1:-1]  # c[k, i+1, j]
        B = c[t, :-2, 1:-1]  # c[k, i-1, j]
        C = c[t, 1:-1, 2:]  # c[k, i, j+1]
        D = c[t, 1:-1, :-2]  # c[k, i, j-1]
        E = c[t, 1:-1, 1:-1]  # c[k, i, j]

        particle_positions = position[:, t, :]
        source = moving_point_source(
            source,
            moving_source_production_strength,
            local_step,
            simulation_time,
            particle_positions[active_mask],
            birth_times[active_mask],   # <-- new: per-particle birth times
            dx,
            moving_source_decay_rate,
            n_steps,
        )
        if static_source_production_strength:
            source = static_point_source(
                source,
                static_source_production_strength,
                local_step,
                simulation_time,
                static_source_position,
                dx,
                static_source_decay_rate,
                n_steps,
            )
        S = source[t, 1:-1, 1:-1]
        result = (
            gamma * (A + B + C + D - 4 * E)
            + E
            + S * dt
            - global_decay_strength * dt * E
        )

        if advection == True:
            # Calculate fluid velocity
            ux, uy = update_flow_field(
                ux,
                uy,
                timestep,
                particle_positions[active_mask],
                dx,
                dt,
                moving_source_decay_rate,
                n_steps,
            )

            # Calculate advection term
            adv = -(dt / (2.0 * dx)) * (
                np.multiply(ux[t, 1:-1, 1:-1], (A - B))
                + np.multiply(uy[t, 1:-1, 1:-1], (C - D))
            )

            # Add advection term to the result
            result += adv

        # Set the newly computed heatmap at time k+1
        c[t + 1, 1:-1, 1:-1] = result

        # Noflux boundary condition for maze
        noflux_maze(
            c,
            t + 1,
            x_diff_indices_reg,
            y_diff_indices_reg,
            x_diff_neg_indices_reg,
            y_diff_neg_indices_reg,
            maze,
            regular_wall_mask,
        )

        # Leaky boundary condition for exit walls
        if permeability > 0.0 and exit_wall_mask is not None:
            c = leaky_exit_wall(c, t + 1, exit_wall_mask, permeability, dt)

        # Forces
        forces_interaction = interaction_force(
            position, t, dx, epsilon_lj, active_mask=active_mask
        )
        forces_chemotaxis = chemotaxis_force(
            position, t, c, maze, dx, bp, active_mask=active_mask
        )
        forces_wall = wall_force(position, t, wall_coords, dx, active_mask=active_mask)
        forces_self_propulsion = self_propulsion_force(
            position,
            local_step,
            simulation_time,
            theta,
            sp_decay_rate,
            n_steps,
            self_propulsion_speed,
            active_mask=active_mask,
        )

        forces_interaction_sum = forces_interaction.sum(axis=1)

        f_int[:, t, :] = forces_interaction_sum
        f_chem[:, t, :] = forces_chemotaxis
        f_wall[:, t, :] = forces_wall
        f_sp[:, t, :] = forces_self_propulsion

        dt_noise = np.sqrt(2.0 * dp * dt)
        dr_noise = np.sqrt(2.0 * dr * dt)
        # Generate random noise for ALL particles at once (faster than generating it inside a loop)
        noise_x = dt_noise * np.random.normal(0, 1, size=num_particles)
        noise_y = dt_noise * np.random.normal(0, 1, size=num_particles)
        noise_theta = dr_noise * np.random.normal(0, 1, size=num_particles)

        if massive_particle:
            # Update Active particles simultaneously using the mask ---
            # Position
            position[active_mask, t + 1, 0] = (
                position[active_mask, t, 0] + dt * velocity[active_mask, t, 0]
            )
            position[active_mask, t + 1, 1] = (
                position[active_mask, t, 1] + dt * velocity[active_mask, t, 1]
            )

            # Velocity
            velocity[active_mask, t + 1, 0] = velocity[active_mask, t, 0] + (1 / particle_mass) * (
                -dt * velocity[active_mask, t, 0]
                + dt * forces_self_propulsion[active_mask, 0]
                + dt * forces_chemotaxis[active_mask, 0]
                + dt * forces_interaction_sum[active_mask, 0]
                + dt * forces_wall[active_mask, 0]
                + noise_x[active_mask]
            )
            velocity[active_mask, t + 1, 1] = velocity[active_mask, t, 1] + (1 / particle_mass) * (
                -dt * velocity[active_mask, t, 1]
                + dt * forces_self_propulsion[active_mask, 1]
                + dt * forces_chemotaxis[active_mask, 1]
                + dt * forces_interaction_sum[active_mask, 1]
                + dt * forces_wall[active_mask, 1]
                + noise_y[active_mask]
            )

            # Angular
            theta[active_mask, t + 1] = (
                theta[active_mask, t] + dt * ang_velocity[active_mask, t]
            )
            ang_velocity[active_mask, t + 1] = ang_velocity[active_mask, t] + (
                1 / particle_moment_inertia
            ) * (
                -dt * ang_velocity[active_mask, t]
                + dt * self_propulsion_frequency
                + noise_theta[active_mask]
            )
        else:
            # --- Update Remaining Active Particles ---
            position[active_mask, t + 1, 0] = position[active_mask, t, 0] + (
                dt * forces_self_propulsion[active_mask, 0]
                + dt * forces_chemotaxis[active_mask, 0]
                + dt * forces_interaction_sum[active_mask, 0]
                + dt * forces_wall[active_mask, 0]
                + noise_x[active_mask]
            )
            position[active_mask, t + 1, 1] = position[active_mask, t, 1] + (
                dt * forces_self_propulsion[active_mask, 1]
                + dt * forces_chemotaxis[active_mask, 1]
                + dt * forces_interaction_sum[active_mask, 1]
                + dt * forces_wall[active_mask, 1]
                + noise_y[active_mask]
            )

            theta[active_mask, t + 1] = theta[active_mask, t] + (
                dt * self_propulsion_frequency + noise_theta[active_mask]
            )

            # Calculate velocities based on position change
            velocity[active_mask, t + 1, 0] = (
                position[active_mask, t + 1, 0] - position[active_mask, t, 0]
            ) / dt
            velocity[active_mask, t + 1, 1] = (
                position[active_mask, t + 1, 1] - position[active_mask, t, 1]
            ) / dt
            ang_velocity[active_mask, t + 1] = (
                theta[active_mask, t + 1] - theta[active_mask, t]
            ) / dt

        if np.any(active_mask):
            px_bins[active_mask] = np.rint(position[active_mask, t + 1, 0] / dx).astype(
                int
            )
            py_bins[active_mask] = np.rint(position[active_mask, t + 1, 1] / dx).astype(
                int
            )

            # # Clip indices safely within maze boundaries
            # px_bins[active_mask] = np.clip(px_bins[active_mask], 0, maze.shape[0] - 1)
            # py_bins[active_mask] = np.clip(py_bins[active_mask], 0, maze.shape[1] - 1)

        ## out_of_bounds_check()

        # ---------------------------------------------------------
        # Safety check: particles must stay away from grid boundaries
        # ---------------------------------------------------------
        # Requires at least one-cell padding because force calculations
        # access neighboring cells.
        particle_positions_now = position[active_mask, t + 1, :]

        if particle_positions_now.size > 0:
            particle_bins_x = np.rint(particle_positions_now[:, 0] / dx).astype(int)
            particle_bins_y = np.rint(particle_positions_now[:, 1] / dx).astype(int)

            unsafe = (
                (particle_bins_x < 1)
                | (particle_bins_x >= maze.shape[0] - 1)
                | (particle_bins_y < 1)
                | (particle_bins_y >= maze.shape[1] - 1)
            )

            if np.any(unsafe):
                bad_particles = np.where(active_mask)[0][unsafe]

                bad_positions = ""
                forces = np.array([f_chem, f_int, f_sp, f_wall])
                for i in bad_particles:
                    bad_positions += f"forces(chem,inter,selfprop,wall): {forces[:,i,t-3:t+1]}\n"
                    bad_positions += f"positions: x0: {position[i,t-2,0]} y0: {position[i,t-2,1]}\n"
                    bad_positions += f"positions: x1: {position[i,t-1,0]} y1: {position[i,t-1,1]}\n"
                    bad_positions += f"positions: x2: {position[i,t,0]} y2: {position[i,t,1]}\n"
                    bad_positions += f"positions: x3: {position[i,t+1,0]} y3: {position[i,t+1,1]}\n"
                print(
                    f"\n OUT-OF-BOUNDS ERROR\nParticle out of safe grid bounds at timestep {timestep}. "
                    f"Particle id(s): {bad_particles.tolist()}\n"
                    + bad_positions 
                )

                # Save the current state/output before terminating
                save_snapshot(
                    file_prefix_conc,
                    file_prefix_part,
                    c,
                    position,
                    theta,
                    velocity,
                    ang_velocity,
                    forces_self_propulsion,
                    forces_chemotaxis,
                    forces_interaction,
                    forces_wall,
                    timestep,
                    simulation_time,
                    n_steps,
                    save_forces=True,
                    maze=maze,
                )
                raise RuntimeError(
                    f"OUT-OF-BOUNDS ERROR at timestep {timestep}; "
                    f"particle id(s): {bad_particles.tolist()}"
                )
            ## ----------------------------------------------------------------------------

        if params.junction_tracker is not None:
            params.junction_tracker.update(
                timestep,
                simulation_time,
                px_bins,
                py_bins,
                active_mask,
            )

        # Write concentration and particle data
        if local_step == 0 and start_step == 0:
            write_parameters(**params.to_parameter_dict())
            write_grid(grid_filename, nx, ny)

        if timestep % conc_every == 0:
            save_concentration_snapshot(
                file_prefix_conc,
                c,
                timestep,
                simulation_time,
                n_steps,
                maze=maze,
            )
        if timestep % part_every == 0:
            save_particle_snapshot(
                file_prefix_part,
                position,
                theta,
                velocity,
                ang_velocity,
                forces_self_propulsion,
                forces_chemotaxis,
                forces_interaction,
                forces_wall,
                timestep,
                simulation_time,
                save_forces=True,
            )

        # exit condition

        hit_exit_zone = (
            active_mask & death_zone_map[px_bins, py_bins] & np.isinf(exit_trigger_time)
        )

        if np.any(hit_exit_zone):
            exit_trigger_time[hit_exit_zone] = simulation_time

        ready_to_reap = (
            ~dead_tracker
            & np.isfinite(exit_trigger_time)
            & ((simulation_time - exit_trigger_time) >= grim_reaper_delay_time)
        )

        if np.any(ready_to_reap):
            dead_tracker[ready_to_reap] = True
            position[ready_to_reap, t:, :] = np.nan
            velocity[ready_to_reap, t:, :] = np.nan

        active_mask[dead_tracker] = False

        # Check if all particles have successfully exited or died
        has_exited_or_died = np.isfinite(exit_trigger_time) | dead_tracker
        if np.all(has_exited_or_died):
            save_snapshot(
                file_prefix_conc,
                file_prefix_part,
                c,
                position,
                theta,
                velocity,
                ang_velocity,
                forces_self_propulsion,
                forces_chemotaxis,
                forces_interaction,
                forces_wall,
                timestep,
                simulation_time,
                n_steps,
                save_forces=True,
                maze=maze,
            )
            finished = True
            exit_timestep = timestep
            break

    return SolverResult(
        concentration=c,
        position=position,
        theta=theta,
        velocity=velocity,
        ang_velocity=ang_velocity,
        forces_self_propulsion=f_sp,
        forces_chemotaxis=f_chem,
        forces_interaction=f_int,
        forces_wall=f_wall,
        finished=finished,
        exit_timestep=exit_timestep,
        exit_trigger_time=exit_trigger_time,
    )
