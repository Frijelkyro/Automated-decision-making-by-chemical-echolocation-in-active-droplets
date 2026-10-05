import os, glob, shutil, ast
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from tqdm import tqdm
from time import perf_counter
from maze_functions import maze_from_file

print("started video rendering")
t0 = perf_counter()

DATA = "./data"
MAZE = "./different_mazes/Ran_maze_size_prop_to_droplet.tsv"
PARAM = os.path.join(DATA, "param.txt")
if not os.path.isfile(PARAM):
    PARAM += ".bak"
GRID = os.path.join(DATA, "grid.txt")
SHOW_TRAJECTORIES = False

# Parameters
params = {}
with open(PARAM) as f:
    for line in f:
        if ":" not in line:
            continue
        k, v = map(str.strip, line.split(":", 1))
        try:
            params[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            params[k] = v

# Geometry
with open(GRID) as f:
    lines = f.read().splitlines()
i, j = lines.index("BOX:"), lines.index("SHAPE:")
box = np.array([x.split() for x in lines[i + 1:j] if x.strip()], dtype=float)
maze = maze_from_file(MAZE)
wall = np.argwhere(maze == 0)
source = np.asarray(params["static_source_position"], float)
radius = float(params.get("exit_radius", 20))

# Files
part_files = sorted(
    glob.glob(os.path.join(DATA, "part_*.npz")),
    key=lambda f: int(os.path.basename(f).split("_")[-1][:-4])
)
timestamps = [int(os.path.basename(f).split("_")[-1][:-4]) for f in part_files]
conc_files = {
    int(os.path.basename(f).split("_")[-1][:-4]): f
    for f in glob.glob(os.path.join(DATA, "conc_*.npz"))
}

if not timestamps:
    raise RuntimeError("No particle NPZ files found.")

# Concentration range
vmin, vmax = np.inf, -np.inf
for ts, f in tqdm(conc_files.items(), desc="Scanning concentrations", unit="frame"):
    with np.load(f) as d:
        c = d["concentration"]
        c = c[np.isfinite(c) & (c > 0)]
        if c.size:
            vmin, vmax = min(vmin, c.min()), max(vmax, c.max())

if not np.isfinite(vmin):
    vmin = 0
if not np.isfinite(vmax) or vmax <= vmin:
    vmax = vmin + 1

# Figure
fig, ax = plt.subplots(figsize=(10, 8))
with np.load(part_files[0]) as d:
    n = len(d["x"])
with np.load(conc_files[timestamps[0]]) as d:
    first_c = d["concentration"]

image = ax.imshow(
    first_c.T, origin="lower", interpolation="None", cmap="inferno",
    norm=Normalize(vmin, vmax),
    extent=[box[0, 0], box[0, 1], box[1, 0], box[1, 1]],
)

if wall.size:
    ax.plot(wall[:, 0] + .5, wall[:, 1] + .5, "s", ms=6, color="#B8C7E5")

ax.add_patch(plt.Circle(source, radius, color="green", fill=False))
ax.add_patch(plt.Circle(source, radius * .9, color="red", fill=False))
ax.set(xlim=box[0], ylim=box[1])
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])

scatter = ax.scatter([], [], s=49)
trail_lines = []

if SHOW_TRAJECTORIES:
    trail = np.full((n, len(timestamps), 2), np.nan)
    trail_lines = [ax.plot([], [], "-", lw=1, alpha=.5)[0] for _ in range(n)]

time_text = ax.text(.02, .98, "", transform=ax.transAxes, color="white",
                    fontsize=20, va="top")
fig.tight_layout()


def update(i):
    ts = timestamps[i]

    with np.load(part_files[i]) as d:
        x, y, time = d["x"], d["y"], float(d["simulation_time"])

    scatter.set_offsets(np.column_stack((x, y)))

    if ts in conc_files:
        with np.load(conc_files[ts]) as d:
            image.set_array(d["concentration"].T)

    if SHOW_TRAJECTORIES:
        trail[:, i] = np.column_stack((x, y))
        for p, line in enumerate(trail_lines):
            line.set_data(trail[p, :i + 1, 0], trail[p, :i + 1, 1])

    time_text.set_text(f"Time: {time:.3f}s")
    return [image, scatter, time_text, *trail_lines]


writer = animation.FFMpegWriter(
    fps=40, bitrate=2000, codec="libx264",
    extra_args=["-crf", "17", "-threads", "16", "-preset", "ultrafast"],
)

out = os.path.join(DATA, "particle_trajectory.mp4")
with writer.saving(fig, out, dpi=fig.dpi):
    for i in tqdm(range(len(timestamps)), desc="Rendering video",
                  unit="frame", mininterval=1):
        writer.grab_frame()

shutil.copy(out, "particle_trajectory.mp4")
elapsed = perf_counter() - t0
print(f"\nFinished in {elapsed:.1f} s")
print(f"Average: {len(timestamps) / elapsed:.2f} frames/s")
print(f"Output: {out}")