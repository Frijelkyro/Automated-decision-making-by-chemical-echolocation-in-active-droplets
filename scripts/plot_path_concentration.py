#!/usr/bin/env python3
"""Concentration along the optimal maze path, for every parameter configuration of an experiment TOML.

Writes to output/<series>/concentration_analysis/:
  optimal_path.{png,csv,md}   the single path used for all plots + how it was computed
  <run_id>/shot_NNN.png       10 profiles (label = simulation time) over distance from start,
                              each shifted so the mean of its last --tail cells is 0

Everything is in simulation coordinates (x, y) = the frame of conc_*.npz; no rotation needed.
Example: python scripts/plot_path_concentration.py --config experiments/simulation_series_parameters.toml
"""
import argparse, heapq, itertools, re, sys, tomllib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from maze_functions import maze_from_file
from storage import read_concentration

DX = 1.0  # grid spacing used by maze_cluster_script.py


def load_toml(p):
    p = Path(p)
    return tomllib.loads((p if p.is_absolute() else ROOT / p).read_text())


def optimal_path(open_, start, goal, rep):
    """Dijkstra on 4-neighbours; cost of entering a cell = 1 + rep / (distance_to_wall + 0.5)."""
    cost = 1 + rep / (distance_transform_edt(open_) + 0.5)
    best = np.full(open_.shape, np.inf)
    best[start] = 0.0
    prev, heap = {start: None}, [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == goal:
            break
        if d > best[u]:
            continue
        for n in ((u[0] + 1, u[1]), (u[0] - 1, u[1]), (u[0], u[1] + 1), (u[0], u[1] - 1)):
            if 0 <= n[0] < open_.shape[0] and 0 <= n[1] < open_.shape[1] and open_[n] and d + cost[n] < best[n]:
                best[n] = d + cost[n]
                prev[n] = u
                heapq.heappush(heap, (best[n], n))
    if goal not in prev:
        sys.exit(f"No open path from {start} to {goal}")
    path, n = [], goal
    while n is not None:
        path.append(n)
        n = prev[n]
    return np.array(path[::-1])


def sim_time(f, ts, par):
    """simulation_time stored in the npz, else rebuilt from the timestep (each solver loop advances n_steps ids but (n_steps-1)*dt)."""
    try:
        return float(np.load(f)["simulation_time"])
    except Exception:
        n = par.get("n_steps", 100)
        k, loc = divmod(ts, n)
        return (k * (n - 1) + loc) * par.get("dt", 5e-4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "experiments" / "simulation_series_parameters.toml")
    ap.add_argument("--n-curves", type=int, default=10, help="Linspaced profiles per plot")
    ap.add_argument("--tail", type=int, default=8, help="Last N path cells whose mean is shifted to 0")
    ap.add_argument("--wall-repulsion", type=float, default=5.0, help="0 = pure shortest path")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    cfg = load_toml(a.config)
    s, defaults = cfg["series"], cfg.get("defaults", {})
    series_dir = ROOT / s.get("data_root", "data") / s["name"]
    out = ROOT / s.get("output_root", "output") / s["name"] / "concentration_analysis"
    out.mkdir(parents=True, exist_ok=True)

    # ---- the one optimal path: emitter_position -> static_source_position of the maze profile ----
    name = "Testrun" if defaults.get("test_run") else defaults.get("maze_config", "Ran_maze_size_prop_to_droplet")
    prof = load_toml(ROOT / "maze_startpoints_configuration.toml")["configurations"][name]
    maze_file = ROOT / prof["maze_file"]
    maze = maze_from_file(str(maze_file))  # simulation frame, 0 = wall
    open_ = maze != 0
    snap = distance_transform_edt(~open_, return_indices=True)[1]  # nearest open cell for every cell
    cell = lambda k: tuple(int(v) for v in snap[:, int(np.rint(prof[k][0] / DX)), int(np.rint(prof[k][1] / DX))])
    start, goal = cell("emitter_position"), cell("static_source_position")
    path = optimal_path(open_, start, goal, a.wall_repulsion)
    px, py = path.T
    dist = np.r_[0.0, np.cumsum(np.hypot(np.diff(px), np.diff(py)))] * DX

    np.savetxt(out / "optimal_path.csv", np.c_[path, dist], delimiter=",", header="x_cell,y_cell,distance_from_start", fmt="%g")
    (out / "optimal_path.md").write_text(
        f"# Optimal path\n\nMaze: `{maze_file}` (profile `{name}`). Start = `emitter_position` {prof['emitter_position']}, "
        f"goal = `static_source_position` {prof['static_source_position']}, both divided by dx={DX}, rounded to a grid cell "
        f"and snapped to the nearest open cell if they fall into a wall: start {start}, goal {goal}.\n\n"
        f"Computed with Dijkstra on the 4-connected grid of open cells. Entering a cell costs\n"
        f"`1 + w / (d + 0.5)`, with d = Euclidean distance of the cell to the nearest wall "
        f"(scipy `distance_transform_edt`) and w = {a.wall_repulsion} (`--wall-repulsion`).\n"
        f"The path is therefore short but keeps away from walls instead of cutting corners.\n\n"
        f"Length: {len(path)} cells, {dist[-1]:.1f} length units. `distance_from_start` is the cumulative Euclidean length along the path.\n"
        f"Cells are (x, y) indices in the simulation frame, the same frame as `conc_*.npz`.\n")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(open_.T, cmap="gray", origin="lower")
    ax.plot(px, py, "r-", lw=1.5, label="Optimal path")
    ax.plot(*start, "go", label="Start")
    ax.plot(*goal, "ro", label="Goal")
    ax.set(title=f"Optimal path ({len(path)} cells)", xlabel="x", ylabel="y")
    ax.legend()
    fig.savefig(out / "optimal_path.png", dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"path: {len(path)} cells -> {out}")

    # ---- one folder per parameter configuration, one plot per shot ----
    for reg in cfg["regimes"]:
        keys = list(reg["grid"])
        for i, vals in enumerate(itertools.product(*reg["grid"].values())):
            par, run_id = defaults | dict(zip(keys, vals)), f"{reg['name']}_c{i:04d}"
            label = ", ".join(f"{k}={par[k]}" for k in keys)
            for shot_dir in sorted((series_dir / run_id).glob("shot_*")):
                png = out / run_id / f"{shot_dir.name}.png"
                if a.skip_existing and png.exists():
                    continue
                files = sorted((int(m[1]), f) for f in shot_dir.glob("conc_*.npz") if (m := re.fullmatch(r"conc_(\d+)\.npz", f.name)))
                if not files:
                    print(f"WARN  no conc files in {shot_dir}")
                    continue
                sel = [files[j] for j in np.unique(np.linspace(0, len(files) - 1, min(a.n_curves, len(files))).astype(int))]
                fig, ax = plt.subplots(figsize=(10, 6))
                for (ts, f), col in zip(sel, plt.cm.viridis(np.linspace(0, 1, len(sel)))):
                    c = read_concentration(f, f.parent)
                    c = (c[0] if c.ndim == 3 else c)[px, py]
                    ax.plot(dist, c - c[-a.tail:].mean(), color=col, lw=2, label=f"t = {sim_time(f, ts, par):.2f}")
                ax.set(title=f"{run_id} / {shot_dir.name}\n{label}", xlabel="Distance from start along optimal path",
                       ylabel=f"Concentration - mean of last {a.tail} path cells")
                ax.grid(True, ls="--", alpha=0.6)
                ax.legend(title="Simulation time", fontsize=9)
                png.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(png, dpi=a.dpi, bbox_inches="tight")
                plt.close(fig)
                print(f"saved {png}")


if __name__ == "__main__":
    main()
