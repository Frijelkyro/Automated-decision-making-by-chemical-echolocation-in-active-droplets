import os, glob, ast, argparse, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from tqdm import tqdm
from time import perf_counter
from maze_functions import maze_from_file
from storage import read_concentration

print("started video rendering")
t0 = perf_counter()

parser = argparse.ArgumentParser(description="Render a simulation shot to MP4.")
parser.add_argument("--data", default="./data", help="Shot data directory.")
parser.add_argument("--maze", default="./different_mazes/Ran_maze_size_prop_to_droplet.tsv")
parser.add_argument("--frames", type=int, default=400, help="Maximum frames; 0 renders all frames.")
parser.add_argument("--output", help="Output MP4 path; defaults to particle_trajectory.mp4 in the shot directory.")
args = parser.parse_args()
if args.frames < 0:
    parser.error("--frames must be zero or a positive integer")

DATA = args.data
MAZE = args.maze
PARAM = os.path.join(DATA, "param.txt")
if not os.path.isfile(PARAM):
    numbered_params = []
    for path in glob.glob(os.path.join(DATA, "param*.txt")):
        match = re.fullmatch(r"param(\d+)\.txt", os.path.basename(path))
        if match:
            numbered_params.append((int(match.group(1)), path))
    if numbered_params:
        PARAM = max(numbered_params)[1]
GRID = os.path.join(DATA, "grid.txt")
SHOW_TRAJECTORIES = False

# Parameters
params = {}
if os.path.isfile(PARAM):
    with open(PARAM) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = map(str.strip, line.split(":", 1))
            try:
                params[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                params[k] = v
else:
    checkpoint = os.path.join(DATA, "ckpt_previous.npz")
    if not os.path.isfile(checkpoint):
        raise RuntimeError(
            "No parameter text file or ckpt_previous.npz found."
        )
    with np.load(checkpoint, allow_pickle=True) as data:
        params["static_source_position"] = tuple(
            data["static_source_position"].tolist()
        )
        params["exit_radius"] = float(data["exit_radius"])

# Grid / maze
with open(GRID) as f:
    lines = f.read().splitlines()

i, j = lines.index("BOX:"), lines.index("SHAPE:")
box = np.array(
    [x.split() for x in lines[i + 1:j] if x.strip()],
    dtype=float,
)

maze = maze_from_file(MAZE)
wall = np.argwhere(maze == 0)
source = np.asarray(params["static_source_position"], float)
radius = float(params.get("exit_radius", 20))

# Output files
part_snapshots = []
for path in glob.glob(os.path.join(DATA, "part_*.npz")):
    match = re.fullmatch(r"part_(\d+)\.npz", os.path.basename(path))
    if match:
        part_snapshots.append((int(match.group(1)), path))
part_snapshots.sort()
timestamps = [timestep for timestep, _ in part_snapshots]
part_files = [path for _, path in part_snapshots]

conc_files = {}
for path in glob.glob(os.path.join(DATA, "conc_*.npz")):
    match = re.fullmatch(r"conc_(\d+)\.npz", os.path.basename(path))
    if match:
        conc_files[int(match.group(1))] = path

if args.frames:
    part_files = part_files[-args.frames:]
    timestamps = timestamps[-args.frames:]

if not timestamps:
    raise RuntimeError("No particle NPZ files found.")

# Concentration range
vmin, vmax = np.inf, -np.inf

for ts, filename in tqdm(
    conc_files.items(),
    desc="Scanning concentrations",
    unit="frame",
):
    c = read_concentration(filename, DATA, maze)
    if c.ndim == 3:
        c = c[0]
    else:
        c = c[np.isfinite(c) & (c > 0)]
    c = c[np.isfinite(c) & (c > 0)]

    if c.size:
        vmin = min(vmin, c.min())
        vmax = max(vmax, c.max())

if not np.isfinite(vmin):
    vmin = 0

if not np.isfinite(vmax) or vmax <= vmin:
    vmax = vmin + 1

# Initial data
with np.load(part_files[0]) as d:
    x0 = d["x"]
    y0 = d["y"]
    time0 = float(d["simulation_time"])

npart = len(x0)

first_conc = None
earlier_conc_steps = [ts for ts in conc_files if ts <= timestamps[0]]
if earlier_conc_steps:
    first_conc_step = max(earlier_conc_steps)
    first_conc = read_concentration(conc_files[first_conc_step], DATA, maze)
    if first_conc.ndim == 3:
        first_conc = first_conc[0]

# Figure
fig, ax = plt.subplots(figsize=(10, 8))

if first_conc is None:
    raise RuntimeError(
        f"No concentration data found for timestep {timestamps[0]}"
    )

image = ax.imshow(
    first_conc.T,
    origin="lower",
    interpolation="none",
    cmap="inferno",
    norm=Normalize(vmin=vmin, vmax=vmax),
    extent=(
        box[0, 0],
        box[0, 1],
        box[1, 0],
        box[1, 1],
    ),
)


if wall.size:
    ax.plot(
        wall[:, 0] + 0.5,
        wall[:, 1] + 0.5,
        "s",
        ms=6,
        color="#B8C7E5",
    )

ax.add_patch(
    plt.Circle(source, radius, color="green", fill=False)
)
ax.add_patch(
    plt.Circle(source, radius * 0.9, color="red", fill=False)
)

ax.set(
    xlim=box[0],
    ylim=box[1],
)

ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])

# One scatter object for all particles
scatter = ax.scatter(
    x0,
    y0,
    s=49,
)

# Optional trajectories
trail_lines = []

if SHOW_TRAJECTORIES:
    trail = np.full(
        (npart, len(timestamps), 2),
        np.nan,
    )

    trail_lines = [
        ax.plot(
            [],
            [],
            "-",
            lw=1,
            alpha=0.5,
        )[0]
        for _ in range(npart)
    ]

time_text = ax.text(
    0.02,
    0.98,
    f"Time: {time0:.3f}s",
    transform=ax.transAxes,
    color="white",
    fontsize=20,
    va="top",
)

fig.tight_layout()


def update(i):
    ts = timestamps[i]

    # Particles
    with np.load(part_files[i]) as d:
        x = d["x"]
        y = d["y"]
        simulation_time = float(d["simulation_time"])

    scatter.set_offsets(
        np.column_stack((x, y))
    )

    # Concentration
    earlier_conc_steps = [conc_ts for conc_ts in conc_files if conc_ts <= ts]
    if earlier_conc_steps:
        conc_ts = max(earlier_conc_steps)
        concentration = read_concentration(conc_files[conc_ts], DATA, maze)
        if concentration.ndim == 3:
            concentration = concentration[0]
        image.set_data(concentration.T)

    # Trajectories
    if SHOW_TRAJECTORIES:
        trail[:, i, 0] = x
        trail[:, i, 1] = y

        for p, line in enumerate(trail_lines):
            line.set_data(
                trail[p, :i + 1, 0],
                trail[p, :i + 1, 1],
            )

    time_text.set_text(
        f"Time: {simulation_time:.3f}s"
    )

    return [
        image,
        scatter,
        time_text,
        *trail_lines,
    ]


# Video writer
writer = animation.FFMpegWriter(
    fps=40,
    bitrate=2000,
    codec="libx264",
    extra_args=[
        "-crf",
        "17",
        "-threads",
        "16",
        "-preset",
        "ultrafast",
    ],
)

out = args.output or os.path.join(DATA, "particle_trajectory.mp4")
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

# Render
with writer.saving(
    fig,
    out,
    dpi=fig.dpi,
):
    for i in tqdm(
        range(len(timestamps)),
        desc="Rendering video",
        unit="frame",
        mininterval=1,
    ):
        update(i)
        writer.grab_frame()

elapsed = perf_counter() - t0

print(f"\nFinished in {elapsed:.1f} s")
print(
    f"Average: "
    f"{len(timestamps) / elapsed:.2f} frames/s"
)
print(f"Output: {out}")