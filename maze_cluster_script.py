# %reload_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from maze_functions import *
from resume_simulation import *
from list_of_functions import *

from time import perf_counter
from tqdm import tqdm

# This script is used to run the chemical solver on a maze with one particle.
# It initializes the parameters, generates a maze, and runs the simulation.


init_t0 = perf_counter()  # time tracking
beta = -8

Dc = 2.0 * 10 ** (2)  # diffusion coefficient of the chemical (100)
Dp = 1.0 * 10 ** (0)  # noise strength for the particle (0.1)
Bp = beta * 10 ** (4)  # chemotactic sensitivity (CR: -16000, CA:16000)
moving_source_production_strength = 1.0  # strength of the source
moving_source_decay_rate = 1 / 70  # characteristic decay rate of the source
global_decay_strength = 0.0  # evaporation of chemical from environment
self_propulsion_speed = 0.0 * 10 ** (1)  # self-propulsion speed of the particle (1.0)
sp_decay_rate = 0.0  # characteristic decay rate of the self-propulsion
self_propulsion_frequency = 0.0  # self-propulsion angular velocity of the particle
Dr = 3.0  # rotational diffusion coefficient of the particle (1.0)
M = 1.0 * 10 ** (-1)  # mass of the particle (0.001, 0.2)
J = 4.0 * 10 ** (-2)  # moment of inertia of the particle (0.001, 0.06)

epsilon_LJ = 0.10  # Lennard-Jones potential parameter for interaction between particles
static_source_position = (90.2, 10.5)  # Position of the static source
static_source_production_strength = 0.0  # Strength of the static source
static_source_decay_rate = 0.0  # characteristic decay rate of the source

advection = False  # whether to include advection term in the chemical equation
massive_particle = True  # whether to include mass in the particle equation

# Simulation parameters
dx = 1.0  # grid spacing
Lx = 100.0  # domain size
Ly = 100.0  # domain size
n_xbins = int(Lx / dx)  # number of bins in x direction
n_ybins = int(Ly / dx)  # number of bins in y direction
n_steps = 5000  # number of time steps 40000
dt = 0.09 * 10 ** (-3)  # time step size
gamma = (Dc * dt) / (dx**2)  # gamma parameter
time_loop = 100  # number of time loops

# global_step = 0      # integer, mainly for output filenames
# local_step = 0       # integer, position within the current solver call
# buffer_index = 0     # index into p/v/theta/c rolling arrays
simulation_time = (
    0.0  # physical time in seconds (will be overweritten if resume_simulation)
)

# start_time = 0.0
# start_step = 0

time = np.arange(0, time_loop * n_steps, 1) * dt
time = time[np.newaxis, :]

write_every = 100  # write output after every this many time steps

num_particles = 350  # Number of particles
emission_rate = 3.5  # droplets per second
emitter_position = np.array([4.1, 82.1], dtype=np.float32)
drops_added_incremental = True

test_run = True

# Data directory
data = "data"  # for linux
# data = 'D:\maze_data' # for windows
# Check if data directory exists, if not, create it
if not os.path.exists(data):
    os.makedirs(data)
param_filename = data + "/param.txt"
grid_filename = data + "/grid.txt"
file_prefix_conc = data + "/conc"
file_prefix_part = data + "/part"


# Generate a maze
# maze = box_maze(n_xbins, n_ybins)

# maze = maze_from_file('different_mazes/empty_box.tsv')
maze = maze_from_file("different_mazes/Ran_maze_size_prop_to_droplet.tsv")
if test_run:
    maze = maze_from_file("different_mazes/Ran_maze_size_prop_to_droplet_testrun.tsv")
# maze = maze_from_file('different_mazes/Maass_maze_1x.tsv')
wall = np.transpose(np.where(maze == 0))

exit_radius = 20.0  # radius of the exit around the target (static source)
grim_reaper_delay = 0.1
exit_wall_radius = 20.0  # radius for the leaky exit wall (this also removes particles when they get <2 pixels close)
permeability = 0.0  # permeability of the exit wall (0 = no-flux, >0 = leaky)

if test_run:
    # num_particles = int(num_particles * 0.1 // 1)
    n_steps = int(n_steps * 0.01 // 1)  # preferably 600
    # time_loop = int(time_loop * 0.1 // 1)  # preferably 10
    write_every = 100
    static_source_position = (42.5, 10.5)  # Position of the static source
    emitter_position = np.array([2.1, 14.8], dtype=np.float32)

# death zone and reaper timer
death_zone_map = np.zeros_like(maze, dtype=bool)
X, Y = np.indices(maze.shape)
cx, cy = np.rint(
    np.array(static_source_position) / dx
)  # static source position hold exit position
exit_zone_map = ((X - cx) ** 2 + (Y - cy) ** 2) <= (exit_radius / dx) ** 2
death_zone_map = ((X - cx) ** 2 + (Y - cy) ** 2) <= (exit_radius * 0.9 / dx) ** 2
# open walls (if leaky)
exit_wall_mask = get_exit_wall_mask(maze, static_source_position, dx, exit_wall_radius)

# Initial condition everywhere inside the grid
c_initial = 0.0
# Create new map and display the result of chemical diffusion
conc = initialize_c(c_initial, n_steps, maze)
# conc = initialize_c_from_file(c_initial, n_steps, maze, data +'/conc_Ran_maze_1x.txt')

# Calculate arrays safely using the master num_particles variable
p = np.full((num_particles, n_steps, 2), 0.0, dtype=np.float32)
v = np.full((num_particles, n_steps, 2), 0.0, dtype=np.float32)
theta = np.full((num_particles, n_steps), 0.0, dtype=np.float32)
omega = np.full((num_particles, n_steps), 0.0, dtype=np.float32)

active_mask = np.zeros(num_particles, dtype=bool)
dead_tracker = np.zeros(num_particles, dtype=bool)
exit_trigger_time = np.full(num_particles, np.inf)


# Default: All particles start at the same emitter location and activate at delayed birth times.
p[:, 0, :] = emitter_position
v[:, 0, :] = 0.0
theta[:, 0] = np.random.uniform(0, 2.0 * np.pi, size=num_particles)
omega[:, 0] = 0.0

birth_times = np.array(
    [i / emission_rate for i in range(num_particles)],
    dtype=np.float64,
)

if not drops_added_incremental:
    max_attempts = 1000  # Prevent infinite loops
    min_separation = 0.8
    initial_spread = 1.3
    placed_positions = np.empty((0, 2), dtype=np.float32)
    print_nearest_wall(maze, emitter_position[0], emitter_position[1])
    for particle_id in range(num_particles):
        attempts = 0
        placed_successfully = False

        while attempts < max_attempts:
            candidate = np.random.uniform(
                emitter_position - initial_spread, emitter_position + initial_spread
            ).astype(np.float32)

            if placed_positions.shape[0] == 0:
                placed_successfully = True
                break

            diffs = placed_positions - candidate
            dists = np.hypot(diffs[:, 0], diffs[:, 1])

            if np.all(dists >= min_separation):
                placed_successfully = True
                break

            attempts += 1

        if not placed_successfully:
            raise ValueError(
                f"Could not fit particle {particle_id}. Increase initial_spread or decrease min_separation."
            )

        p[particle_id, 0] = candidate  # type: ignore
        placed_positions = np.vstack([placed_positions, candidate])  # type: ignore
    birth_times = np.array([0 for i in range(num_particles)], dtype=int)
    active_mask[:] = True

# Resume settings
resume_simulation = True

resume_step = 0  # this will be read from the last sim
resume_old_dt = np.inf  # 0.0001 this will be read from the last sim
resume_new_dt = dt
full_traj = np.empty((num_particles, 0, 15), dtype=np.float32)

if resume_simulation:
    (
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
        simulation_time,
        resume_old_dt,
    ) = resume_simulation_from_file(data, param_filename+".bak", maze, n_steps)

    print(
        f"Resuming from timestep {resume_step} "
        f"(t = {simulation_time:.6f} s, old dt = {resume_old_dt})"
    )


# build a parameter dictionary
parameter_dict = {
    "Dc": Dc,
    "Dp": Dp,
    "Bp": Bp,
    "moving_source_production_strength": moving_source_production_strength,
    "sp_decay_rate": sp_decay_rate,
    "moving_source_decay_rate": moving_source_decay_rate,
    "static_source_decay_rate": static_source_decay_rate,
    "global_decay_strength": global_decay_strength,
    "self_propulsion_speed": self_propulsion_speed,
    "self_propulsion_frequency": self_propulsion_frequency,
    "Dr": Dr,
    "M": M,
    "J": J,
    "epsilon_LJ": epsilon_LJ,
    "static_source_position": static_source_position,
    "static_source_production_strength": static_source_production_strength,
    "advection": advection,
    "massive_particle": massive_particle,
    "dx": dx,
    "Lx": Lx,
    "Ly": Ly,
    "n_xbins": n_xbins,
    "n_ybins": n_ybins,
    "n_steps": n_steps,
    "dt": dt,
    "gamma": gamma,
    "write_every": write_every,
    "num_particles": num_particles,
    "time_loop": time_loop,
    "birth_times": birth_times,
    "active_mask": active_mask,
    "dead_tracker": dead_tracker,
    "death_zone_map": death_zone_map,
    "exit_zone_map": exit_zone_map,
    "grim_reaper_delay": grim_reaper_delay,
    "exit_trigger_time": exit_trigger_time,
    "emitter_position": tuple(emitter_position),
    "param_filename": param_filename,
    "grid_filename": grid_filename,
    "file_prefix_conc": file_prefix_conc,
    "file_prefix_part": file_prefix_part,
    "exit_radius": exit_radius,
    "exit_wall_mask": exit_wall_mask,
    "permeability": permeability,
    "drops_added_incremental": drops_added_incremental,
}

# --------------- time tracking ---------------------
init_duration_perf_metric = perf_counter() - init_t0  # time tracking
print(
    f"Initialization time: {init_duration_perf_metric:.3f} s | particles: {num_particles} | n_steps/loop: {n_steps} | maze.shape: {maze.shape} | concentration shape: {conc.shape} | emission_rate: {emission_rate} | dt: {dt}"
)
simulation_t0 = perf_counter()  # time tracking
n_active = active_mask.sum()
pbar = tqdm(range(time_loop), desc="Simulation", unit="loop")

# ---------------- Simulation loop ------------------
for i in pbar:
    # for i in range(time_loop):
    loop_t0 = perf_counter()  # time tracking

    (
        conc,
        p,
        theta,
        v,
        omega,
        f_sp,
        f_chem,
        f_int,
        f_wall,
        exit,
        exit_timestep,
        exit_trigger_time,
    ) = chemical_solver(
        conc,
        p,
        theta,
        v,
        omega,
        maze,
        start_step=resume_step + i * n_steps,
        start_time=simulation_time,
        **parameter_dict,
    )

    write_param_snapshot(parameter_dict, simulation_time, resume_step + i * n_steps)

    simulation_time += (n_steps - 1) * dt

    if exit:
        # current_time = np.repeat(
        #     time[:, i * n_steps : exit_timestep + 1, np.newaxis], num_particles, axis=0
        # )
        # current_traj = np.concatenate(
        #     (
        #         current_time,
        #         p[:, 0 : exit_timestep % n_steps + 1, :],
        #         theta[:, 0 : exit_timestep % n_steps + 1, np.newaxis],
        #         v[:, 0 : exit_timestep % n_steps + 1, :],
        #         omega[:, 0 : exit_timestep % n_steps + 1, np.newaxis],
        #         f_sp[:, 0 : exit_timestep % n_steps + 1, :],
        #         f_chem[:, 0 : exit_timestep % n_steps + 1, :],
        #         f_int[:, 0 : exit_timestep % n_steps + 1, :],
        #         f_wall[:, 0 : exit_timestep % n_steps + 1, :],
        #     ),
        #     axis=-1,
        # )
        # full_traj = np.append(full_traj, current_traj, axis=1)
        conc[-1, :, :] = conc[exit_timestep % n_steps, :, :]
        break
    # current_time = np.repeat(
    #    time[:, i * n_steps : (i + 1) * n_steps, np.newaxis], num_particles, axis=0
    # )
    # current_traj = np.concatenate(
    #    (
    #        current_time,
    #        p,
    #        theta[:, :, np.newaxis],
    #        v,
    #        omega[:, :, np.newaxis],
    #        f_sp,
    #        f_chem,
    #        f_int,
    #        f_wall,
    #    ),
    #    axis=-1,
    # )
    # full_traj = np.append(full_traj, current_traj, axis=1)

    # set the first particles parameters for the next loop to the most recent from the current loop
    conc[0, :, :] = conc[-1, :, :]
    p[:, 0, :] = p[:, -1, :]
    theta[:, 0] = theta[:, -1]
    omega[:, 0] = omega[:, -1]
    v[:, 0, :] = v[:, -1, :]

    # time tracking
    n_active = active_mask.sum()
    pbar.set_postfix(
        timestep=(i + 1) * n_steps,
        active=n_active,
        simulation_time=f"{simulation_time:.0f}s",
    )

column_names = [
    "Time",
    "X",
    "Y",
    "Theta",
    "VX",
    "VY",
    "Omega",
    "f_spx",
    "f_spy",
    "f_chemx",
    "f_chemy",
    "f_intx",
    "f_inty",
    "f_wallx",
    "f_wally",
]


# Save the full trajectory with a header
# np.savetxt(data + '/full_traj.txt', full_traj[0], fmt='%.8f', header=' '.join(column_names), comments='')


# Get the job ID from the command line arguments
job_id = 1  # sys.argv[1]

# Create the filename using the job ID
# filename1 = data + f"/full_traj_{job_id}.txt"
# Save the full trajectory with a header
# full_traj_write_every = 1
# if exit:
#     np.savetxt(
#         filename1,
#         full_traj[0][::full_traj_write_every],
#         fmt="%.4f",
#         header=" ".join(column_names),
#         comments="",
#     )
#

filename2 = data + "/exit_times.txt"
# Check if the file exists
if not os.path.isfile(filename2):
    # If the file doesn't exist, write the header
    with open(filename2, "w") as f:
        f.write("ExitTime Beta JobID ParticleID\n")

# Append the data to the file
with open(filename2, "a") as f:
    for particle_id in range(num_particles):
        if not np.isfinite(exit_trigger_time[particle_id]):
            f.write(f"{-1} {beta} {job_id} {particle_id}\n")
        else:
            f.write(
                f"{max((exit_trigger_time [particle_id]-birth_times[particle_id]), -1)} {beta} {job_id} {particle_id}\n"
            )


sim_duration_perf_metric = perf_counter() - simulation_t0

print(param_filename)
with open(param_filename, "a") as f:  # type: ignore
    f.write(f"emission_rate: {emission_rate:.3f}\n")
    f.write(f"init_duration_perf_metric: {init_duration_perf_metric:.3f} s\n")
    f.write(f"sim_duration_perf_metric: {sim_duration_perf_metric:.3f} s\n")

print(
    f"This simulation duration (performance metric): {sim_duration_perf_metric:.3f} s"
)
