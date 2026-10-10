import os, sys, heapq
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt

root_folder = "/home/ecasual/Documents/Uni/droplet-maze/simulation-code/output_npz/reaper_timer/0.5_emission_rate/6.0_until_death"
data_folder = root_folder + "/data"
maze_file = "/home/ecasual/Documents/Uni/droplet-maze/simulation-code/different_mazes/Ran_maze_size_prop_to_droplet.tsv"
output_file = os.path.join(root_folder, "particle_success.txt")
tolerance = 4.0
start_point, end_point = (2, 81), (73, 20)

sys.path.extend([os.path.join(os.getcwd(), "strategy_comparison"), ".."])
from maze_functions import maze_from_file

maze = maze_from_file(maze_file)


def centerline_path(maze, start, end, wall_weight=20.0, wall_power=2.0):
    rows, cols = maze.shape
    wd = distance_transform_edt(maze)
    wd[wd < 0.5] = 0.5
    dist = np.full(maze.shape, np.inf)
    parent = {}
    pq = [(0.0, start)]
    dist[start] = 0
    while pq:
        cost, node = heapq.heappop(pq)
        if node == end:
            break
        if cost > dist[node]:
            continue
        i, j = node
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols and maze[ni, nj]:
                new = cost + 1 + wall_weight / (wd[ni, nj] ** wall_power)
                if new < dist[ni, nj]:
                    dist[ni, nj] = new
                    parent[ni, nj] = node
                    heapq.heappush(pq, (new, (ni, nj)))
    if end not in parent:
        return []
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return path[::-1]


path = centerline_path(maze, start_point, end_point, 30, 2)
print("Shortest path length:", len(path))
path_tree = KDTree(np.asarray(path))

plt.figure(figsize=(8, 8))
plt.imshow(maze.T, cmap="gray_r", origin="lower")
path = np.asarray(path)
plt.plot(path[:, 0], path[:, 1], "r", linewidth=2, label="Chosen path")
plt.fill_betweenx(
    path[:, 1],
    path[:, 0] - tolerance,
    path[:, 0] + tolerance,
    color="green",
    alpha=0.8,
    label=f"Tolerance ±{tolerance}",
)
plt.fill_between(
    path[:, 0],
    path[:, 1] - tolerance,
    path[:, 1] + tolerance,
    color="green",
    alpha=0.8,
    label=f"Tolerance ±{tolerance}",
)
plt.scatter(*start_point, color="lime", s=80, label="Start")
plt.scatter(*end_point, color="dodgerblue", s=80, label="Exit")
plt.axis("equal")
plt.legend()
plt.show()

files = sorted(
    (
        f
        for f in os.listdir(data_folder)
        if f.endswith(".npz") and f.startswith("part_")
    ),
    key=lambda f: int(f.split("_")[-1].split(".")[0]),
)
print("Number of particle files:", len(files))

first = np.load(os.path.join(data_folder, files[0]))
max_particles = len(first["x"])
first.close()

correct_time = np.zeros(max_particles)
wrong_time = np.zeros(max_particles)
previous_time = None

for filename in tqdm(files):
    with np.load(os.path.join(data_folder, filename)) as data:
        timestep = int(data["timestep"])
        simulation_time = float(data["simulation_time"])
        positions = np.column_stack((data["x"], data["y"]))

    if previous_time is None:
        previous_time = simulation_time
        continue

    dt = simulation_time - previous_time
    previous_time = simulation_time

    valid = np.isfinite(positions).all(axis=1)
    distances = np.full(len(positions), np.nan)
    if np.any(valid):
        distances[valid] = path_tree.query(positions[valid])[0]

    correct = valid & (distances <= tolerance)
    wrong = valid & (distances > tolerance)

    np.add.at(correct_time, np.arange(len(positions))[correct], dt)
    np.add.at(wrong_time, np.arange(len(positions))[wrong], dt)

total_time = correct_time + wrong_time
valid = total_time > 0
particle_ids = np.arange(max_particles)[valid]
success_fraction = np.divide(
    correct_time, total_time, out=np.zeros_like(total_time), where=valid
)[valid]

np.savetxt(
    output_file,
    np.column_stack((particle_ids, success_fraction)),
    header="particle_id success_fraction",
    fmt="%d %.6f",
)

print("\nSaved:", output_file)
print("Particles analysed:", len(particle_ids))

plt.figure(figsize=(10, 5))
plt.plot(particle_ids, success_fraction, "o", markersize=3)
plt.xlabel("Particle ID")
plt.ylabel("Success fraction")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(root_folder + "/particle_success.png", dpi=300, bbox_inches="tight")
plt.show()
