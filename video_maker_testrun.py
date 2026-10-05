import os, re, glob, ast, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from tqdm import tqdm
from time import perf_counter
from maze_functions import maze_from_file, load_c_from_file

print("started video rendering")
t0 = perf_counter()

DATA = "./data"
MAZE = "./different_mazes/Ran_maze_size_prop_to_droplet_testrun.tsv"
PARAM = os.path.join(DATA, "param.txt")
if not os.path.isfile(PARAM):
    PARAM = os.path.join(DATA, "param.txt.bak")
GRID = os.path.join(DATA, "grid.txt")
SHOW_TRAJECTORIES = False

# Read parameters
params, key, value = {}, None, ""
for line in open(PARAM):
    if ":" in line:
        if key:
            try:
                params[key] = ast.literal_eval(value)
            except:
                params[key] = value.strip()
        key, value = map(str.strip, line.split(":", 1))
    elif key:
        value += " " + line.strip()

if key:
    try:
        params[key] = ast.literal_eval(value)
    except:
        params[key] = value.strip()

# Grid geometry
grid = open(GRID).read()
lines = grid.splitlines()
i, j = lines.index("BOX:"), lines.index("SHAPE:")
box = np.array([
    list(map(float, x.split()))
    for x in lines[i + 1:j]
    if x.strip()
])
shape = tuple(map(int, lines[j + 1].split()))

maze = maze_from_file(MAZE)
wall = np.transpose(np.where(maze == 0))
source = np.asarray(params["static_source_position"], float)
radius = float(params.get("exit_radius", 20))

timestamps = sorted(
    int(re.search(r"part_(\d+)\.txt$", f).group(1))
    for f in glob.glob(os.path.join(DATA, "part_*.txt"))
)

def part(ts):
    with open(os.path.join(DATA, f"part_{ts}.txt")) as f:
        lines = f.readlines()
    return float(lines[3]), np.atleast_2d(np.loadtxt(lines[5:]))

npart = max(part(ts)[1].shape[0] for ts in timestamps)

# Find concentration range
vmin, vmax = np.inf, -np.inf

for ts in tqdm(timestamps, desc="Scanning concentrations", unit="frame"):
    f = os.path.join(DATA, f"conc_{ts}.txt")
    if os.path.exists(f):
        c = load_c_from_file(maze, f)
        c = c[np.isfinite(c) & (c > 0)]
        if c.size:
            vmin = min(vmin, c.min())
            vmax = max(vmax, c.max())

if not np.isfinite(vmin):
    vmin = 0
if not np.isfinite(vmax) or vmax <= vmin:
    vmax = vmin + 1

# Figure
fig, ax = plt.subplots(figsize=(10, 8))

first = load_c_from_file(
    maze,
    os.path.join(DATA, f"conc_{timestamps[0]}.txt")
)

image = ax.imshow(
    first.T,
    origin="lower",
    interpolation="None",
    cmap="inferno",
    norm=Normalize(vmin, vmax),
    extent=[box[0, 0], box[0, 1], box[1, 0], box[1, 1]],
)

if wall.size:
    ax.plot(
        wall[:, 0] + 0.5,
        wall[:, 1] + 0.5,
        "s",
        ms=6,
        color="#B8C7E5",
    )

ax.add_patch(plt.Circle(source, radius, color="green", fill=False))
ax.add_patch(plt.Circle(source, radius * .9, color="red", fill=False))

ax.set(xlim=box[0], ylim=box[1])
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])

colors = [
    "red", "blue", "green", "orange", "purple", "brown", "pink",
    "gray", "olive", "cyan", "magenta", "yellow", "black", "white",
    "lime", "teal"
]

points = [
    ax.plot(
        [], [], "o", ms=7,
        color=colors[p % len(colors)]
    )[0]
    for p in range(npart)
]

if SHOW_TRAJECTORIES:
    trails = [
        ax.plot(
            [], [], "-", lw=2, alpha=.7,
            color=colors[p % len(colors)]
        )[0]
        for p in range(npart)
    ]
    trail = np.full((npart, len(timestamps), 2), np.nan)

time_text = ax.text(
    .02, .98, "",
    transform=ax.transAxes,
    color="white",
    fontsize=20,
    va="top",
)

fig.tight_layout()

def update(i):
    ts = timestamps[i]
    time, traj = part(ts)

    f = os.path.join(DATA, f"conc_{ts}.txt")

    if os.path.exists(f):
        image.set_array(load_c_from_file(maze, f).T)

    for p in range(npart):
        if p < len(traj):
            x, y = traj[p, 1:3]
            points[p].set_data([x], [y])

            if SHOW_TRAJECTORIES:
                trail[p, i] = x, y
        else:
            points[p].set_data([], [])

        if SHOW_TRAJECTORIES:
            trails[p].set_data(
                trail[p, :i + 1, 0],
                trail[p, :i + 1, 1],
            )

    time_text.set_text(f"Time: {time:.3f}s")

    return (
        [image, *points, time_text, *trails]
        if SHOW_TRAJECTORIES
        else [image, *points, time_text]
    )

# Render video with progress bar
writer = animation.FFMpegWriter(
    fps=40,
    bitrate=2000,
    codec="libx264",
    extra_args=[
        "-crf", "17",
        "-threads", "16",
        "-preset", "ultrafast",
    ],
)

out = os.path.join(DATA, "particle_trajectory.mp4")

with writer.saving(fig, out, dpi=fig.dpi):
    for i in tqdm(
        range(len(timestamps)),
        desc="Rendering video",
        unit="frame",
        mininterval=1.0,
    ):
        update(i)
        writer.grab_frame()

shutil.copy(out, "particle_trajectory.mp4")

elapsed = perf_counter() - t0

print(f"\nFinished in {elapsed:.1f} s")
print(f"Average: {len(timestamps) / elapsed:.2f} frames/s")
print(f"Output: {out}")