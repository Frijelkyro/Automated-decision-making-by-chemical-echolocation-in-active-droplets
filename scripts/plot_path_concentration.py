#!/usr/bin/env python3
"""Plot concentration along the maze path (+ heatmap) for one run or a whole experiment series.

Examples:
  python plot_path_concentration.py --series-dir data/reaper_emission_scan \
      --jobs output/reaper_emission_scan/jobs.txt --out-dir plots/reaper_emission_scan
  python plot_path_concentration.py --series-dir data/reaper_emission_scan --pattern 'quadratic_c000*' --show
"""
import argparse, re, shlex
from fnmatch import fnmatch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tomllib
from collections import deque
from scipy.ndimage import distance_transform_edt
import heapq

def bfs_path(maze, start, goal, wall=1, wall_repulsion=5.0):
    """Shortest path with wall repulsion via Dijkstra on a distance-transform potential."""
    open_mask = maze != wall
    dist_to_wall = distance_transform_edt(open_mask)          # 0 at walls, grows inward
    cost_map = 1.0 + wall_repulsion / (dist_to_wall + 0.5)   # high near walls, ~1 in corridors

    INF = float("inf")
    best = np.full(maze.shape, INF)
    best[start] = 0.0
    prev = {start: None}
    heap = [(0.0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if u == goal:
            break
        if d > best[u]:
            continue
        r, c = u
        for n in ((r+1,c),(r-1,c),(r,c+1),(r,c-1)):
            if 0 <= n[0] < maze.shape[0] and 0 <= n[1] < maze.shape[1] and open_mask[n]:
                nd = d + cost_map[n]
                if nd < best[n]:
                    best[n] = nd; prev[n] = u
                    heapq.heappush(heap, (nd, n))

    if goal not in prev:
        raise ValueError(f"No open path from {start} to {goal}")
    path, node = [], goal
    while node is not None:
        path.append(node); node = prev[node]
    return path[::-1]


def load_conc(f, shape, k=1):
    """conc_*.npz['concentration'] lives in the sim frame (= rot90(TSV, -1)); rotate back into TSV frame."""
    c = np.rot90(np.load(f)["concentration"], k)
    return c if c.shape == shape else None

def read_jobs(p):
    """Parse jobs.txt -> {(run_id, shot): {param: value}}."""
    if not p or not Path(p).is_file():
        return {}
    jobs = {}
    for line in Path(p).read_text().splitlines():
        t = shlex.split(line)
        d = {a[2:]: v for a, v in zip(t, t[1:]) if a.startswith("--")}
        if "run-id" in d:
            jobs[(d["run-id"], int(d.get("shot", 0)))] = d
    return jobs


def plot_run(d, maze, path, args, jobs):
    m = re.fullmatch(r"shot_(\d+)", d.name)
    rid, shot = (d.parent if m else d).name, int(m[1]) if m else 0
    j = jobs.get((rid, shot), {})
    dash = lambda v: str(float(v)).replace(".", "-")  # 12 -> "12-0", 0.01 -> "0-01"
    if {"grim-reaper-delay", "emission-rate", "desired-time"} <= j.keys():
        name = f"rt{dash(j['grim-reaper-delay'])}_er{dash(j['emission-rate'])}_{float(j['desired-time']):g}s" + (f"_shot{shot:03d}" if shot else "")
    else:  # no jobs.txt entry -> fall back to run id
        name = f"{rid}_shot{shot:03d}"
    out = Path(args.out_dir) / f"{name}.png"
    if args.skip_existing and out.exists():
        return print(f"skip  {out}")

    steps = sorted((int(g[1]), f) for f in d.glob("conc_*.npz")
                   if (g := re.fullmatch(r"conc_(\d+)\.npz", f.name)) and int(g[1]) <= args.max_step)
    if not steps:
        return print(f"WARN  no conc_*.npz in {d}")
    idx = np.unique(np.linspace(0, len(steps) - 1, min(args.n_curves, len(steps)), dtype=int))
    data = [(steps[i][0], c) for i in idx if (c := load_conc(steps[i][1], maze.shape, args.rot90)) is not None]
    if not data:
        return print(f"WARN  no concentration file in {d} matches maze shape {maze.shape}")

    info = ", ".join(f"{lbl}={j[k]}" for k, lbl in (("emission-rate", "emission rate"), ("grim-reaper-delay", "reaper delay")) if k in j)
    py, px = np.array(path).T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"{rid} (shot {shot})" + (f" - {info}" if info else ""), fontsize=16)

    for (ts, c), col in zip(data, plt.cm.viridis(np.linspace(0, 1, len(data)))):
        ax1.plot(c[py, px], color=col, lw=2, label=f"Timestep {ts}")
    ax1.set(title="Concentration Profile Along Path Over Time", xlabel="Cell Number (Step along path)", ylabel="Concentration")
    ax1.grid(True, ls="--", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=10, title="Timesteps")

    im = ax2.imshow(data[-1][1], cmap="inferno")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Concentration")
    wy, wx = np.where(maze == args.wall)
    ax2.plot(wx, wy, "s", ms=4, color="#B8C7E5", label="Wall")
    ax2.plot(px, py, "r-", lw=1.5, label="Path to Goal")
    ax2.plot(px[0], py[0], "go", ms=8, label="Start")
    ax2.plot(px[-1], py[-1], "ro", ms=8, label="End")
    ax2.set(title=f"Concentration Heatmap (Timestep: {data[-1][0]})", xlabel="X coordinate", ylabel="Y coordinate", aspect="equal")
    ax2.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"saved {out}")
    if args.show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series-dir", required=True, help="Series data root (e.g. data/reaper_emission_scan) or a single run/shot dir")
    ap.add_argument("--maze-meta", default="different_mazes/Ran_maze_size_prop_to_droplet_analysis/maze_metadata.toml",
                    help="maze_metadata.toml (maze_file, wall_value, emitter_grid, goal_grid)")
    ap.add_argument("--out-dir", required=True, help="Where to save the PNGs")
    ap.add_argument("--jobs", help="Optional jobs.txt to put emission rate / reaper delay into titles")
    ap.add_argument("--pattern", default="*", help="fnmatch filter on run dir path relative to --series-dir")
    ap.add_argument("--n-curves", type=int, default=10, help="Number of evenly spaced timesteps per plot")
    ap.add_argument("--max-step", type=int, default=2_000_000, help="Ignore conc files above this timestep")
    ap.add_argument("--rot90", type=int, default=1, help="rotations applied to conc to get into TSV frame (sim frame is rot90(-1) of TSV)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--skip-existing", action="store_true", help="Skip runs whose PNG already exists")
    ap.add_argument("--show", action="store_true", help="Also open an interactive window per plot")
    ap.add_argument("--wall-repulsion", type=float, default=5.0,
                help="Wall repulsion strength for path planning (0 = pure shortest path, try 5-20)")
    args = ap.parse_args()
    if not args.show:
        plt.switch_backend("Agg")

    root = Path(args.series_dir)
    meta = tomllib.loads(Path(args.maze_meta).read_text())
    maze, args.wall = np.loadtxt(meta["maze_file"], delimiter="\t"), meta["wall_value"]
    path = bfs_path(maze, tuple(meta["emitter_grid"]), tuple(meta["goal_grid"]), args.wall, args.wall_repulsion)
    jobs = read_jobs(args.jobs)
    print(f"path: {len(path)} cells (metadata path_length: {meta.get('path_length')})")
    runs = [d for d in sorted({f.parent for f in root.rglob("conc_*.npz")})
            if fnmatch(str(d.relative_to(root)), args.pattern)]
    if not runs:
        raise SystemExit(f"No conc_*.npz found under {root} (pattern '{args.pattern}')")
    print(f"{len(runs)} run(s) found")
    for d in runs:
        try:
            plot_run(d, maze, path, args, jobs)
        except Exception as e:
            print(f"ERROR {d}: {e}")


if __name__ == "__main__":
    main()
