# %reload_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from maze_functions import *
from list_of_functions import *

# This script is used to run the chemical solver on a maze with one particle.
# It initializes the parameters, generates a maze, and runs the simulation.

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
exit_radius = 20.0  # radius of the exit aroudn the target (static source)
exit_wall_radius = 20.0  # radius for the leaky exit wall
permeability = 0.0  # permeability of the exit wall (0 = no-flux, >0 = leaky)
drops_added_incremental = True
test_run = False

# Simulation parameters
dx = 1.0  # grid spacing
Lx = 100.0  # domain size
Ly = 100.0  # domain size
n_xbins = int(Lx / dx)  # number of bins in x direction
n_ybins = int(Ly / dx)  # number of bins in y direction
n_steps = 16000  # number of time steps 40000
dt = 0.12 * 10 ** (-3)  # time step size
gamma = (Dc * dt) / (dx**2)  # gamma parameter
time_loop = 100  # number of time loops
time = np.arange(0, time_loop * n_steps, 1) * dt
time = time[np.newaxis, :]
total_time = dt * time_loop * n_steps  # total time of the simulation
write_every = 100  # write output after every this many time steps
if test_run:
    n_steps = 300
    time_loop = 100

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
death_zone_map = np.zeros_like(maze, dtype=bool)
# exit at [90.2, 10.5]
# death_zone_map[94:98, 0:35] = True
# death_zone_map[50:98, 0:4] = True
X, Y = np.indices(maze.shape)
cx, cy = np.rint(np.array(static_source_position) / dx)
exit_zone_map = ((X - cx) ** 2 + (Y - cy) ** 2) <= (exit_radius / dx) ** 2
death_zone_map = ((X - cx) ** 2 + (Y - cy) ** 2) <= (exit_radius * 0.9 / dx) ** 2


# Initial condition everywhere inside the grid
c_initial = 0.0

num_particles = 200 # Number of particles
emission_rate = 2 # droplets per second

exit_wall_mask = get_exit_wall_mask(maze, static_source_position, dx, exit_wall_radius)
active_mask = np.zeros(num_particles, dtype=bool)
dead_tracker = np.zeros(num_particles, dtype=bool)

# Calculate arrays safely using the master num_particles variable
p = np.full((num_particles, n_steps, 2), 0.0, dtype=np.float32)
v = np.full((num_particles, n_steps, 2), 0.0, dtype=np.float32)
theta = np.full((num_particles, n_steps), 0.0, dtype=np.float32)
omega = np.full((num_particles, n_steps), 0.0, dtype=np.float32)

emitter_position = np.array([4.1, 82.1], dtype=np.float32)
if test_run:
    emitter_position = np.array([48.1, 14.8], dtype=np.float32)

if drops_added_incremental:
    # All particles start at the same emitter location and activate at delayed birth times.
    p[:, 0, :] = emitter_position
    v[:, 0, :] = 0.0
    theta[:, 0] = np.random.uniform(0, 2.0 * np.pi, size=num_particles)
    omega[:, 0] = 0.0

    birth_steps = np.array(
        [int(round(i / emission_rate / dt)) for i in range(num_particles)], dtype=int
    )

else:
    max_attempts = 1000 # Prevent infinite loops
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
            raise ValueError(f"Could not fit particle {particle_id}. Increase initial_spread or decrease min_separation.")
            
        p[particle_id, 0] = candidate
        placed_positions = np.vstack([placed_positions, candidate])
    birth_steps = np.array([0 for i in range(num_particles)], dtype=int)

# Create new map and display the result of chemical diffusion
conc = initialize_c(c_initial, n_steps, maze)
# conc = initialize_c_from_file(c_initial, n_steps, maze, data +'/conc_Ran_maze_1x.txt')

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
    "total_time": total_time,
    "write_every": write_every,
    "num_particles": num_particles,
    "time_loop": time_loop,
    "birth_steps": birth_steps,
    "active_mask": active_mask,
    "dead_tracker": dead_tracker,
    "death_zone_map": death_zone_map,
    "exit_zone_map": exit_zone_map,
    "emitter_position": tuple(emitter_position),
    "param_filename": param_filename,
    "grid_filename": grid_filename,
    "file_prefix_conc": file_prefix_conc,
    "file_prefix_part": file_prefix_part,
    "exit_radius": exit_radius,
    "exit_wall_mask": exit_wall_mask,
    "permeability": permeability,
    "drops_added_incremental": drops_added_incremental
}

full_traj = np.empty((num_particles, 0, 15), dtype=np.float32)
exit_times = np.zeros(num_particles)
for i in range(time_loop):
    conc, p, theta, v, omega, f_sp, f_chem, f_int, f_wall, exit, exit_timestep = (
        chemical_solver(
            conc,
            p,
            theta,
            v,
            omega,
            maze,
            exit_times,
            start_step=i * n_steps,
            **parameter_dict,
        )
    )
    if exit:
        current_time = np.repeat(
            time[:, i * n_steps : exit_timestep + 1, np.newaxis], num_particles, axis=0
        )
        current_traj = np.concatenate(
            (
                current_time,
                p[:, 0 : exit_timestep % n_steps + 1, :],
                theta[:, 0 : exit_timestep % n_steps + 1, np.newaxis],
                v[:, 0 : exit_timestep % n_steps + 1, :],
                omega[:, 0 : exit_timestep % n_steps + 1, np.newaxis],
                f_sp[:, 0 : exit_timestep % n_steps + 1, :],
                f_chem[:, 0 : exit_timestep % n_steps + 1, :],
                f_int[:, 0 : exit_timestep % n_steps + 1, :],
                f_wall[:, 0 : exit_timestep % n_steps + 1, :],
            ),
            axis=-1,
        )
        # full_traj = np.append(full_traj, current_traj, axis=1)
        conc[-1, :, :] = conc[exit_timestep % n_steps, :, :]
        break
    current_time = np.repeat(
        time[:, i * n_steps : (i + 1) * n_steps, np.newaxis], num_particles, axis=0
    )
    current_traj = np.concatenate(
        (
            current_time,
            p,
            theta[:, :, np.newaxis],
            v,
            omega[:, :, np.newaxis],
            f_sp,
            f_chem,
            f_int,
            f_wall,
        ),
        axis=-1,
    )
    # full_traj = np.append(full_traj, current_traj, axis=1)
    conc[0, :, :] = conc[-1, :, :]
    p[:, 0, :] = p[:, -1, :]
    theta[:, 0] = theta[:, -1]
    omega[:, 0] = omega[:, -1]
    v[:, 0, :] = v[:, -1, :]


# Assuming you have column names
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
        f.write(
            f"{max((exit_times[particle_id]-birth_steps[particle_id]) *dt, -1)} {beta} {job_id} {particle_id}\n"
        )
