import os, sys, heapq
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

root = "/home/ecasual/Documents/Uni/droplet-maze/simulation-code"
data = root + "/data"
sys.path.extend([root + "/strategy_comparison", ".."])
from maze_functions import maze_from_file

maze = maze_from_file(root + "/different_mazes/Ran_maze_size_prop_to_droplet.tsv")
start, end, tol = (2, 81), (73, 20), 4.0

def centerline(maze, start, end, ww=30., wp=2.):
    wd = np.maximum(distance_transform_edt(maze), .5)
    dist = np.full(maze.shape, np.inf); dist[start] = 0
    parent, pq = {}, [(0., start)]
    while pq:
        cost, (i, j) = heapq.heappop(pq)
        if cost > dist[i, j]: continue
        if (i, j) == end: break
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            p = (i+di, j+dj)
            if 0 <= p[0] < maze.shape[0] and 0 <= p[1] < maze.shape[1] and maze[p]:
                c = cost + 1 + ww / wd[p]**wp
                if c < dist[p]:
                    dist[p], parent[p] = c, (i,j)
                    heapq.heappush(pq, (c,p))
    if start != end and end not in parent: raise ValueError("No path found")
    path = [end]
    while path[-1] != start: path.append(parent[path[-1]])
    return np.array(path[::-1])

tree = KDTree(centerline(maze, start, end))

path = tree.data

plt.figure(figsize=(8, 8))
plt.imshow(maze.T, cmap="gray_r", origin="lower")
plt.plot(path[:, 0], path[:, 1], "r", lw=2, label="Chosen path")
plt.fill_betweenx(path[:, 1], path[:, 0]-tol, path[:, 0]+tol,
                  color="green", alpha=.35, label=f"Tolerance ±{tol}")
plt.fill_between(path[:, 0], path[:, 1]-tol, path[:, 1]+tol,
                 color="green", alpha=.35)
plt.scatter(*start, color="lime", s=80, label="Start")
plt.scatter(*end, color="dodgerblue", s=80, label="Exit")
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()

with np.load(data + "/param_state.npz") as s:
    birth, death = s["birth_times"].copy(), s["exit_trigger_time"].copy()
death = np.where(np.isfinite(death), death, np.inf)

files = sorted((f for f in os.listdir(data) if f.startswith("part_") and f.endswith(".npz")),
               key=lambda f: int(f[5:-4]))
if len(files) < 2: raise ValueError("Need at least two snapshots")

with np.load(data + "/" + files[0]) as d: n = len(d["x"])
if len(birth) != n or len(death) != n: raise ValueError("State/snapshot particle counts differ")

good = np.zeros(n); bad = np.zeros(n)
prev_t = None
for f in tqdm(files, desc="Processing particles", unit="file"):
    with np.load(data + "/" + f) as d:
        t = float(d["simulation_time"])
        pos = np.column_stack((d["x"], d["y"]))
    if prev_t is not None:
        dt = t - prev_t
        if dt < 0: raise ValueError("Snapshots are not chronological")
        if dt:
            live = np.maximum(0., np.minimum(t, death) - np.maximum(prev_t, birth))
            valid = np.isfinite(pos).all(axis=1)
            ok = np.zeros(n, dtype=bool)
            ok[valid] = tree.query(pos[valid], k=1, workers=-1)[0] <= tol
            good += live * (valid & ok)
            bad += live * (valid & ~ok)
    prev_t = t

total = good + bad
ids = np.flatnonzero(total)
frac = good[ids] / total[ids]
np.savetxt(root + "/particle_success.txt", np.c_[ids, frac],
           header="particle_id success_fraction", fmt="%d %.6f")
plt.figure(figsize=(10,5)); plt.plot(ids, frac, "o", ms=3)
plt.xlabel("Particle ID"); plt.ylabel("Success fraction")
plt.ylim(0,1); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(root + "/particle_success.png", dpi=200); plt.show()
print(f"Analysed {len(ids)}/{n} particles; saved results.")
