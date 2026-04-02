#%reload_ext autoreload
#%autoreload 2

import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from maze_functions import *
from list_of_functions import *

# This script is used to run the chemical solver on a maze with one particle.
# It initializes the parameters, generates a maze, and runs the simulation.


#Simulation parameters
#beta = np.random.uniform(1,25)
beta = -1

Dc = 1.0*10**(2) #diffusion coefficient of the chemical
Dp = 1.0*10**(0) #noise strength for the particle
Bp = beta*10**(4) #chemotactic sensitivity
moving_source_production_strength = 1.0 #strength of the source
moving_source_decay_rate = 0.0 #characteristic decay rate of the source
global_decay_strength = 0.0 #evaporation of chemical from environment
self_propulsion_speed = 0.0 #self-propulsion speed of the particle 
sp_decay_rate = 0.0 #characteristic decay rate of the self-propulsion
self_propulsion_frequency = 0.0 #self-propulsion angular velocity of the particle
Dr = 3.0 #rotational diffusion coefficient of the particle 
M = 6.0*10**(-2) #mass of the particle
J = 2.0*10**(-2) #moment of inertia of the particle

epsilon_LJ = 0.10  # Lennard-Jones potential parameter for interaction between particles
static_source_position = (90.2, 10.5)  # Position of the static source
static_source_production_strength = 0.0  # Strength of the static source
static_source_decay_rate = 0.0 #characteristic decay rate of the source

advection=False #whether to include advection term in the chemical equation
massive_particle = False #whether to include mass in the particle equation
exit_radius = 20.0 #radius of the exit aroudn the target (static source)

# Simulation parameters
dx = 1.0 #grid spacing
Lx = 100.0 #domain size
Ly = 100.0 #domain size
n_xbins = int(Lx/dx) #number of bins in x direction
n_ybins = int(Ly/dx) #number of bins in y direction
n_steps = 4000 #number of time steps
dt = 10**(-3) #time step size
gamma = (Dc * dt) / (dx ** 2) #gamma parameter
time_loop = 100 #number of time loops
time = np.arange(0, time_loop*n_steps, 1)*dt
time = time[np.newaxis,:]
total_time = dt*time_loop*n_steps #total time of the simulation
write_every = 100 #write output after every this many time steps

# Data directory
data = 'data' # for linux
#data = 'D:\maze_data' # for windows
# Check if data directory exists, if not, create it
if not os.path.exists(data):
    os.makedirs(data)
param_filename = data + '/param.txt'
grid_filename = data + '/grid.txt'
file_prefix_conc = data + '/conc'
file_prefix_part = data + '/part'


# Generate a maze
#maze = box_maze(n_xbins, n_ybins)
maze = maze_from_file('different_mazes/Ran_maze_size_prop_to_droplet.tsv')
#maze = maze_from_file('different_mazes/Maass_maze_1x.tsv')
wall = np.transpose(np.where(maze == 0))


# Initial condition everywhere inside the grid
c_initial = 0.0

num_particles = 1  # Number of particles
p = np.full((num_particles, n_steps, 2), 0.0, dtype=np.float32)
v = np.full((num_particles, n_steps, 2), 0.0, dtype=np.float32)
theta = np.full((num_particles, n_steps), 0.0, dtype=np.float32)
omega = np.full((num_particles, n_steps), 0.0, dtype=np.float32)
for particle_id in range(num_particles):
    p[particle_id, 0, 0] = np.random.uniform(2, 4) # Initial x-coordinate
    p[particle_id, 0, 1] = 81.05 # Initial y-coordinate
    v[particle_id, 0, 0] = 0.0  # Initial x-velocity
    v[particle_id, 0, 1] = 0.0  # Initial y-velocity
    theta[particle_id, 0] = np.random.uniform(0, 2.0 * np.pi)  # Initial angle
    omega[particle_id, 0] = 0.0  # Initial angular velocity

# Create new map and display the result of chemical diffusion
conc = initialize_c(c_initial, n_steps, maze)
#conc = initialize_c_from_file(c_initial, n_steps, maze, data +'/conc_Ran_maze_1x.txt')

# build a parameter dictionary
parameter_dict={'Dc': Dc, 'Dp': Dp, 'Bp': Bp, 'moving_source_production_strength': moving_source_production_strength, 'sp_decay_rate': sp_decay_rate,
                'moving_source_decay_rate': moving_source_decay_rate, 'static_source_decay_rate': static_source_decay_rate, 'global_decay_strength': global_decay_strength,
                'self_propulsion_speed': self_propulsion_speed, 'self_propulsion_frequency': self_propulsion_frequency,
                'Dr': Dr, 'M': M, 'J': J, 'epsilon_LJ': epsilon_LJ, 'static_source_position': static_source_position,
                'static_source_production_strength': static_source_production_strength, 'advection': advection, 'massive_particle': massive_particle,
                'dx': dx, 'Lx': Lx, 'Ly': Ly, 'n_xbins': n_xbins, 'n_ybins': n_ybins, 'n_steps': n_steps, 'dt': dt, 'gamma': gamma,
                'total_time': total_time, 'write_every': write_every, 'num_particles': num_particles, 'time_loop': time_loop,
                'param_filename': param_filename, 'grid_filename': grid_filename, 'file_prefix_conc': file_prefix_conc,
                'file_prefix_part': file_prefix_part, 'exit_radius': exit_radius}

full_traj = np.empty((num_particles,0,15), dtype=np.float32)
for i in range(time_loop):
    conc, p, theta, v, omega, f_sp, f_chem, f_int, f_wall, exit, exit_timestep = chemical_solver(conc, p, theta, v, omega, maze, start_step=i*n_steps, **parameter_dict)
    if exit:
        #current_traj = np.concatenate((time[:, i*n_steps: exit_timestep+1 , np.newaxis],p[:,0: exit_timestep%n_steps+1,:], theta[:, 0: exit_timestep%n_steps+1, np.newaxis], v[:,0: exit_timestep%n_steps+1,:], omega[:, 0: exit_timestep%n_steps+1, np.newaxis]), axis=-1)
        current_traj = np.concatenate((time[:, i*n_steps: exit_timestep+1 , np.newaxis], p[:,0: exit_timestep%n_steps+1,:], 
                                       theta[:, 0: exit_timestep%n_steps+1, np.newaxis], v[:,0: exit_timestep%n_steps+1,:], 
                                       omega[:, 0: exit_timestep%n_steps+1, np.newaxis], f_sp[:,0: exit_timestep%n_steps+1,:],
                                       f_chem[:,0: exit_timestep%n_steps+1,:], f_int[:,0: exit_timestep%n_steps+1,:],
                                       f_wall[:,0: exit_timestep%n_steps+1,:]), axis=-1)
        full_traj = np.append(full_traj, current_traj, axis=1)
        conc[-1,:,:] = conc[exit_timestep % n_steps,:,:]
        break
    #current_traj = np.concatenate((time[:, i*n_steps: (i+1)*n_steps, np.newaxis],p, theta[:, :, np.newaxis], v, omega[:, :, np.newaxis]), axis=-1)
    current_traj = np.concatenate((time[:, i*n_steps: (i+1)*n_steps, np.newaxis],p, theta[:, :, np.newaxis], v, omega[:, :, np.newaxis],
                                   f_sp, f_chem, f_int, f_wall), axis=-1)
    full_traj = np.append(full_traj, current_traj, axis=1)
    conc[0, :, :] = conc[-1, :, :]
    p[:, 0, :] = p[:,-1,:]
    theta[:, 0] = theta[:,-1]
    omega[:, 0] = omega[:,-1]
    v[:, 0, :] = v[:,-1,:]


# Assuming you have column names
column_names = ['Time', 'X', 'Y', 'Theta', 'VX', 'VY', 'Omega', 'f_spx', 'f_spy', 'f_chemx', 'f_chemy', 'f_intx', 'f_inty', 'f_wallx', 'f_wally']


# Save the full trajectory with a header
#np.savetxt(data + '/full_traj.txt', full_traj[0], fmt='%.8f', header=' '.join(column_names), comments='')


# Get the job ID from the command line arguments
job_id = 1 #sys.argv[1]

# Create the filename using the job ID
filename1 = data + f'/full_traj_{job_id}.txt'
# Save the full trajectory with a header
full_traj_write_every = 1
if exit:
    np.savetxt(filename1, full_traj[0][::full_traj_write_every], fmt='%.4f', header=' '.join(column_names), comments='')

filename2 = data + '/exit_times.txt'
# Check if the file exists
if not os.path.isfile(filename2):
    # If the file doesn't exist, write the header
    with open(filename2, 'w') as f:
        f.write("ExitTime Beta JobID\n")

# Append the data to the file
with open(filename2, 'a') as f:
    f.write(f"{exit_timestep*dt} {beta} {job_id}\n")
    
