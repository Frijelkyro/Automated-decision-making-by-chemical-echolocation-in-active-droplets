import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend suitable for saving files
import matplotlib.pyplot as plt
import seaborn as sns

# Turn off interactive mode
plt.ioff()
plt.rcParams['interactive'] = False

# Basic clean styling
sns.set_style("ticks")

# Font and tick settings (LaTeX removed)
plt.rc('axes', titlesize=30)
plt.rc('axes', labelsize=30)
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
plt.rc('legend', fontsize=30)
plt.rc('font', family='serif')
plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')

plt.rcParams['xtick.major.pad'] = 10
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.major.size'] = 7
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 2

# Animation for Large maze
import matplotlib.animation as animation
import sys
import os
import re
from matplotlib.colors import Normalize
import glob

sys.path.append(os.path.join(os.getcwd(), 'trajectory_video'))
from maze_functions import maze_from_file, load_c_from_file, load_traj_from_file

# Data directory
data = './data/'

# Load the maze
maze = maze_from_file('./different_mazes/Ran_maze_size_prop_to_droplet.tsv')
wall = np.transpose(np.where(maze == 0))

# Get all available timestamps by listing files
traj_files = sorted(glob.glob(data + 'part_*.txt'))
timestamps = sorted([int(re.search(r'part_(\d+)\.txt$', f).group(1)) for f in traj_files])

# Load dt from parameter file
def get_dt_from_params(filename):
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            if len(parts) == 2 and parts[0].strip() == 'dt':
                return float(parts[1].strip())
    return None

t_unit = 60 / 60  # min
r_unit = 1e-04    # cm
dt = get_dt_from_params(data + 'param.txt')
times = np.array(timestamps) * dt * t_unit
formatted_times = [f'Time: {t:.2f}min' for t in times]

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Pre-load concentration data
print("Loading all concentration data...")
concentration_data = []
vmin, vmax = float('inf'), float('-inf')

for ts in timestamps:
    conc_data = load_c_from_file(maze, data + f'conc_{ts}.txt')
    concentration_data.append(conc_data)
    if np.any(conc_data):
        vmax = max(vmax, np.max(conc_data))
        min_nonzero = np.min(conc_data[conc_data > 0])
        vmin = min(vmin, min_nonzero)

vmin = 0.0 if vmin == float('inf') else vmin
vmax = 1.0 if vmax == float('-inf') else vmax
norm = Normalize(vmin=vmin, vmax=vmax)

# Initialize plot
conc_plot = ax.imshow(concentration_data[0].T, interpolation='None',
                      origin='lower', cmap=plt.cm.inferno, alpha=1.0, norm=norm)

# Plot wall
ax.plot(wall[:, 0], wall[:, 1], 's', markersize=6, color='#B8C7E5')

# Annotations
source = np.array([90.2, 10.5])
ax.text(source[0] - 4, source[1] - 2, 'No source', color='red', fontsize=15, ha='right', va='bottom')
ax.text(source[0] + 4, source[1] + 8, 'Exit', color='k', fontsize=15, ha='right', va='bottom', backgroundcolor='white')

start = np.array([5, 86])
ax.text(start[0], start[1], 'Start', color='k', fontsize=15, ha='right', va='bottom', backgroundcolor='white', rotation='vertical')

# Trajectory lines
trajectory_line, = ax.plot([], [], 'w-', linewidth=3.0, alpha=1.0)
current_point, = ax.plot([], [], 'o', color='cyan', markersize=7)

# Timestamp text
timestamp_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         color='white', fontsize=20, verticalalignment='top')

# Remove ticks
plt.xticks([])
plt.yticks([])
fig.tight_layout()

# Pre-load trajectory data
print("Loading and preprocessing all trajectory data...")
all_x_points, all_y_points = [], []
trajectory_indices = {}

current_index = 0
for ts in timestamps:
    traj_data = np.loadtxt(data + f'part_{ts}.txt', skiprows=3)
    x_traj = np.atleast_1d(traj_data[1])
    y_traj = np.atleast_1d(traj_data[2])
    all_x_points.extend(x_traj)
    all_y_points.extend(y_traj)
    current_index += len(x_traj)
    trajectory_indices[ts] = current_index

all_x_points = np.array(all_x_points, dtype=np.float32)
all_y_points = np.array(all_y_points, dtype=np.float32)

# Update function
def update(frame):
    timestamp = timestamps[frame]
    conc_plot.set_array(concentration_data[frame].T)
    end_idx = trajectory_indices[timestamp]
    trajectory_line.set_data(all_x_points[:end_idx], all_y_points[:end_idx])
    if end_idx > 0:
        current_point.set_data([all_x_points[end_idx - 1]], [all_y_points[end_idx - 1]])
    timestamp_text.set_text(formatted_times[frame])
    return conc_plot, trajectory_line, current_point

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(timestamps), interval=100, blit=True)

# Save animation
print("Saving animation...")
Writer = animation.writers['ffmpeg']
writer = Writer(fps=40, bitrate=2000, codec='libx264',
                extra_args=['-crf', '17', '-threads', '16', '-preset', 'ultrafast', '-tune', 'film'])
ani.save(data + 'particle_trajectory.mp4', writer=writer)
print("Animation saved successfully.")
