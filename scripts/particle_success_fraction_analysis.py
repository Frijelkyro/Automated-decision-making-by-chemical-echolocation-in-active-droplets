"""Batch analysis of particle success fractions from part_*.npz files.

This module implements the corrected pipeline described in the engineering task.
It loads a maze, constructs a centerline path, indexes particle snapshots by
simulation time, scores particles by streaming over each shot, and aggregates the
per-particle results into batch-level summaries and plots.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import logging
import math
import re
import shlex
from concurrent.futures import ProcessPoolExecutor
from heapq import heappop, heappush
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tomllib
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree
from tqdm import tqdm

START_DEFAULT = (2, 81)
END_DEFAULT = (73, 20)
DEFAULT_TOLERANCE = 4.0
DEFAULT_WALL_WEIGHT = 30.0
DEFAULT_WALL_POWER = 2.0


logger = logging.getLogger("particle_success_analysis")


def _make_kdtree(points: np.ndarray) -> KDTree:
    points = np.asarray(points, dtype=float)
    try:
        return KDTree(points, workers=1)
    except TypeError:
        return KDTree(points)


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve() if str(path) else Path()


def _resolve_maze_file(metadata_path: Path, maze_file: str | Path) -> Path:
    path = Path(maze_file).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (Path.cwd() / path, metadata_path.resolve().parent / path, Path(__file__).resolve().parents[1] / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Maze file {maze_file!s} from {metadata_path} was not found.")


def _maze_coordinate_to_sim(coordinate: Iterable[int], raw_shape: tuple[int, int]) -> tuple[int, int]:
    row, column = (int(value) for value in coordinate)
    return column, raw_shape[0] - 1 - row


def _read_jobs(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    jobs: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = shlex.split(line)
        parameters: dict[str, str] = {}
        for index, token in enumerate(tokens[:-1]):
            if token.startswith("--"):
                parameters[token[2:].replace("-", "_")] = tokens[index + 1]
        if "run_id" in parameters:
            jobs[parameters["run_id"]] = parameters
    return jobs


def _parameters_from_grid(config: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    if config is None:
        return {}
    defaults = config.get("defaults", {})
    run_parameters: dict[str, dict[str, str]] = {}
    for regime in config.get("regimes", []):
        grid = regime.get("grid", {})
        keys = list(grid)
        combinations = product(*(grid[key] for key in keys)) if keys else [()]
        for config_index, values in enumerate(combinations):
            parameters = defaults | dict(zip(keys, values))
            run_id = f"{regime['name']}_c{config_index:04d}"
            run_parameters[run_id] = {key: str(value) for key, value in parameters.items()}
    return run_parameters


def _configuration_plot_name(run_id: str, parameters: dict[str, str], maze_path: Path) -> str:
    required = {"grim_reaper_delay", "emission_rate", "desired_time"}
    if required.issubset(parameters):
        dash = lambda value: str(float(value)).replace(".", "-")
        return (
            f"rt{dash(parameters['grim_reaper_delay'])}_"
            f"er{dash(parameters['emission_rate'])}_"
            f"{float(parameters['desired_time']):g}s_{maze_path.stem}.png"
        )
    return f"{run_id}_{maze_path.stem}.png"


def _ensure_logger(logger_obj: logging.Logger | None) -> logging.Logger:
    if logger_obj is not None:
        return logger_obj
    return logger


def load_maze_oriented(
    file_path: str | Path,
    start: tuple[int, int] = START_DEFAULT,
    end: tuple[int, int] = END_DEFAULT,
    wall_value: int = 1,
) -> np.ndarray:
    """Load a maze TSV and apply the single, authoritative orientation fix.

    The maze file stores walls as wall_value. The raw file is rotated and inverted
    to match the simulation coordinate convention used
    elsewhere in the project: maze indices are treated as (i, j), path plotting
    uses maze.T with origin='lower', and particle coordinates are compared directly
    against path vertices in index space.
    """
    maze_path = Path(file_path)
    raw = np.loadtxt(maze_path, delimiter="\t", dtype=int)
    maze = np.rot90(raw != wall_value, k=-1).astype(np.int8)

    start_i, start_j = start
    end_i, end_j = end
    if maze[start_i, start_j] != 1:
        raise ValueError(f"Maze start {start} is not walkable in {maze_path}.")
    if maze[end_i, end_j] != 1:
        raise ValueError(f"Maze end {end} is not walkable in {maze_path}.")
    return maze


def centerline_path(
    maze: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    wall_weight: float = DEFAULT_WALL_WEIGHT,
    wall_power: float = DEFAULT_WALL_POWER,
) -> np.ndarray:
    """Compute the weighted shortest centerline path via Dijkstra's algorithm."""
    maze = np.asarray(maze, dtype=np.int8)
    if not np.isin(maze, [0, 1]).all():
        raise ValueError("Maze must contain only binary walkability values 0/1.")
    if maze[start] != 1:
        raise ValueError(f"Maze start {start} must be walkable.")
    if maze[end] != 1:
        raise ValueError(f"Maze end {end} must be walkable.")

    rows, cols = maze.shape
    wall_dist = distance_transform_edt(maze)
    wall_dist[wall_dist < 0.5] = 0.5

    dist = np.full(maze.shape, np.inf, dtype=float)
    dist[start] = 0.0
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]

    while heap:
        current_cost, node = heappop(heap)
        if node == end:
            break
        if current_cost > dist[node]:
            continue
        i, j = node
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols and maze[ni, nj] == 1:
                step_cost = 1.0 + wall_weight / (wall_dist[ni, nj] ** wall_power)
                candidate_cost = current_cost + step_cost
                if candidate_cost < dist[ni, nj]:
                    dist[ni, nj] = candidate_cost
                    parent[(ni, nj)] = node
                    heappush(heap, (candidate_cost, (ni, nj)))

    if not np.isfinite(dist[end]):
        raise ValueError(f"No valid path exists between {start} and {end}.")

    path: list[tuple[int, int]] = [end]
    while path[-1] != start:
        previous = parent.get(path[-1])
        if previous is None:
            raise ValueError(f"Failed to reconstruct path from {start} to {end}.")
        path.append(previous)
    path.reverse()
    return np.asarray(path, dtype=float)


def plot_path_tolerance(
    maze: np.ndarray,
    path: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
    start: tuple[int, int] = START_DEFAULT,
    end: tuple[int, int] = END_DEFAULT,
    output_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the maze, centerline path, and Euclidean tolerance band."""
    if path.size == 0:
        raise ValueError("Cannot plot an empty path.")

    path = np.asarray(path, dtype=float)
    tree = _make_kdtree(path)
    grid_i, grid_j = np.indices(maze.shape)
    coords = np.column_stack((grid_i.ravel(), grid_j.ravel()))
    distances = tree.query(coords)[0].reshape(maze.shape)
    tolerance_mask = distances <= tolerance

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(maze.T, cmap="gray_r", origin="lower")
    ax.imshow(tolerance_mask.T, cmap="Greens", alpha=0.35, origin="lower")
    ax.plot(path[:, 0], path[:, 1], color="red", linewidth=2, label="Centerline path")
    ax.scatter([start[0]], [start[1]], color="lime", s=80, label="Start")
    ax.scatter([end[0]], [end[1]], color="dodgerblue", s=80, label="End")
    ax.set_title(f"Maze path with tolerance {tolerance}")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()

    if output_path is not None:
        figure_path = Path(output_path)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    return fig, ax


def index_shot(
    shot_dir: str | Path,
    output_dir: str | Path | None = None,
    logger_obj: logging.Logger | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """Create a metadata index for the shot's part_*.npz files.

    The index is ordered by actual simulation_time, not by filename number. The
    filenames serve only as a cross-check against the scalar timestep metadata.
    """
    if logger_obj is None:
        logger_obj = logger
    log = _ensure_logger(logger_obj)
    shot_dir = Path(shot_dir)
    entries: list[dict[str, Any]] = []

    for file_path in sorted(shot_dir.glob("part_*.npz"), key=lambda p: p.name):
        match = re.fullmatch(r"part_(\d+)\.npz", file_path.name)
        if match is None:
            continue
        try:
            with np.load(file_path) as data:
                required = ["simulation_time", "timestep"]
                missing = [name for name in required if name not in data]
                if missing:
                    raise KeyError(f"Missing keys {missing} in {file_path}")
                simulation_time = float(np.asarray(data["simulation_time"]).item())
                timestep = int(np.asarray(data["timestep"]).item())
            filename_timestep = int(match.group(1))
            if filename_timestep != timestep:
                log.warning("%s: filename timestep %d differs from stored timestep %d.", file_path, filename_timestep, timestep)
            entries.append({"simulation_time": simulation_time, "timestep": timestep, "path": file_path})
        except (OSError, ValueError, KeyError, TypeError) as error:
            log.warning("Skipping unreadable particle snapshot %s: %s", file_path, error)

    if not entries:
        return []

    entries.sort(key=lambda item: (float(item["simulation_time"]), int(item["timestep"])))
    timesteps = np.asarray([int(item["timestep"]) for item in entries], dtype=int)
    if len(timesteps) > 1 and np.any(np.diff(timesteps) < 0):
        log.warning("%s: timesteps are not monotonic in simulation-time-sorted ordering.", shot_dir)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path / "part_index.npz",
            simulation_time=np.asarray([float(item["simulation_time"]) for item in entries], dtype=float),
            timestep=np.asarray([int(item["timestep"]) for item in entries], dtype=int),
            path=np.asarray([str(item["path"]) for item in entries], dtype=str),
        )

    return entries


def _load_state_arrays(shot_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    state_path = Path(shot_dir) / "param_state.npz"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state file: {state_path}")
    with np.load(state_path) as data:
        if "birth_times" not in data or "exit_trigger_time" not in data:
            raise KeyError(f"State file {state_path} is missing required arrays 'birth_times' and 'exit_trigger_time'.")
        birth_times = np.asarray(data["birth_times"], dtype=float).ravel()
        exit_trigger_time = np.asarray(data["exit_trigger_time"], dtype=float).ravel()
    return birth_times, exit_trigger_time


def _compute_sample_count(snapshot_times: np.ndarray, birth_times: np.ndarray, exit_trigger_time: np.ndarray) -> np.ndarray:
    sample_counts = np.zeros_like(birth_times, dtype=int)
    for idx, (birth_time, exit_time) in enumerate(zip(birth_times, exit_trigger_time)):
        if np.isfinite(exit_time):
            upper = exit_time
        else:
            upper = np.inf
        sample_counts[idx] = int(np.count_nonzero((snapshot_times >= birth_time) & (snapshot_times <= upper)))
    return sample_counts


def score_shot(
    shot_dir: str | Path,
    maze: np.ndarray | None = None,
    path: np.ndarray | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    wall_weight: float = DEFAULT_WALL_WEIGHT,
    wall_power: float = DEFAULT_WALL_POWER,
    logger_obj: logging.Logger | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Score one shot by streaming over particle snapshots in time order."""
    if logger_obj is None:
        logger_obj = logger
    log = _ensure_logger(logger_obj)
    shot_dir = Path(shot_dir)
    if maze is None:
        raise ValueError("A maze array is required for scoring.")
    if path is None:
        raise ValueError("A centerline path is required for scoring.")

    index = index_shot(shot_dir, output_dir=shot_dir, logger_obj=log)
    if not index:
        raise ValueError(f"No matching part_*.npz snapshots found in {shot_dir}.")

    birth_times, exit_trigger_time = _load_state_arrays(shot_dir)
    sim_times = np.asarray([float(item["simulation_time"]) for item in index], dtype=float)

    n_particles = None
    first_snapshot_time = sim_times[0]
    first_snapshot_walkable_fraction = np.nan
    first_snapshot_positions = None

    for idx, item in enumerate(index):
        try:
            with np.load(item["path"]) as data:
                x = np.asarray(data["x"], dtype=float).ravel()
                y = np.asarray(data["y"], dtype=float).ravel()
                if x.shape != y.shape:
                    raise ValueError(f"x and y arrays have different shapes in {item['path']}.")
                if n_particles is None:
                    n_particles = len(x)
                elif len(x) != n_particles:
                    raise ValueError(
                        f"Shot {shot_dir} has mismatched particle count: expected {n_particles}, found {len(x)} in {item['path']}."
                    )
                if idx == 0:
                    finite = np.isfinite(x) & np.isfinite(y)
                    if finite.any():
                        rounded_x = np.rint(x[finite]).astype(int)
                        rounded_y = np.rint(y[finite]).astype(int)
                        in_bounds = (rounded_x >= 0) & (rounded_x < maze.shape[0]) & (rounded_y >= 0) & (rounded_y < maze.shape[1])
                        walkable = np.zeros(np.count_nonzero(finite), dtype=bool)
                        if np.any(in_bounds):
                            walkable[in_bounds] = maze[rounded_x[in_bounds], rounded_y[in_bounds]] == 1
                        first_snapshot_walkable_fraction = float(np.mean(walkable)) if walkable.size else 0.0
                    else:
                        first_snapshot_walkable_fraction = 0.0
                if birth_times.size != n_particles or exit_trigger_time.size != n_particles:
                    raise ValueError(
                        f"Shot {shot_dir} particle count mismatch: state arrays have {birth_times.size} and {exit_trigger_time.size}, expected {n_particles}."
                    )
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Unreadable or invalid snapshot {item['path']}: {error}") from error

    if n_particles is None:
        raise ValueError(f"No valid particle data found in {shot_dir}.")

    if birth_times.size != n_particles or exit_trigger_time.size != n_particles:
        raise ValueError(
            f"State arrays in {shot_dir} have lengths {birth_times.size} and {exit_trigger_time.size}, expected {n_particles}."
        )

    if len(sim_times) == 0:
        raise ValueError(f"No particle snapshots at all in {shot_dir}.")

    path_tree = _make_kdtree(path)
    num = np.zeros(n_particles, dtype=np.float64)
    den = np.zeros(n_particles, dtype=np.float64)
    s_prev = np.full(n_particles, np.nan, dtype=float)
    case3_count = 0
    timesteps_mismatch_count = 0

    if len(index) > 1:
        for snapshot_index, item in enumerate(index):
            current_time = float(item["simulation_time"])
            prev_time = float(index[snapshot_index - 1]["simulation_time"]) if snapshot_index > 0 else -np.inf
            next_time = float(index[snapshot_index + 1]["simulation_time"]) if snapshot_index + 1 < len(index) else np.inf
            with np.load(item["path"]) as data:
                x = np.asarray(data["x"], dtype=float).ravel()
                y = np.asarray(data["y"], dtype=float).ravel()
            if x.shape != (n_particles,) or y.shape != (n_particles,):
                raise ValueError(f"Mismatched particle array lengths in {item['path']}.")

            finite = np.isfinite(x) & np.isfinite(y)
            s_curr = np.full(n_particles, np.nan, dtype=float)
            valid_rows = np.flatnonzero(finite)
            if valid_rows.size:
                eligible = np.full(n_particles, False, dtype=bool)
                eligible[valid_rows] = (birth_times[valid_rows] < next_time) & ((prev_time < exit_trigger_time[valid_rows]) | ~np.isfinite(exit_trigger_time[valid_rows]))
                query_rows = valid_rows[eligible[valid_rows]]
                if query_rows.size:
                    query_points = np.column_stack((x[query_rows], y[query_rows]))
                    distances = path_tree.query(query_points)[0]
                    s_curr[query_rows] = (distances <= tolerance).astype(float)

            if snapshot_index == 0:
                s_prev = s_curr.copy()
                continue

            left = s_prev
            right = s_curr
            lo = np.maximum(prev_time, birth_times)
            hi = np.minimum(current_time, exit_trigger_time)
            interval_length = np.maximum(0.0, hi - lo)

            left_finite = np.isfinite(left)
            right_finite = np.isfinite(right)
            case1 = left_finite & right_finite & (interval_length > 0.0)
            case2 = left_finite & (~right_finite) & np.isfinite(exit_trigger_time) & (exit_trigger_time < current_time) & (interval_length > 0.0)
            case3 = ((~left_finite) | (~right_finite)) & (~case2)
            case3_count += int(np.count_nonzero(case3))

            interval_length_case = interval_length.copy()
            interval_length_case[case3] = 0.0
            if np.any(case1):
                num[case1] += interval_length_case[case1] * (0.5 * (left[case1] + right[case1]))
                den[case1] += interval_length_case[case1]
            if np.any(case2):
                num[case2] += interval_length_case[case2] * left[case2]
                den[case2] += interval_length_case[case2]

            s_prev = s_curr.copy()
    else:
        with np.load(index[0]["path"]) as data:
            x = np.asarray(data["x"], dtype=float).ravel()
            y = np.asarray(data["y"], dtype=float).ravel()
        s_prev = np.full(n_particles, np.nan, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            query_rows = np.flatnonzero(finite)
            query_points = np.column_stack((x[query_rows], y[query_rows]))
            distances = path_tree.query(query_points)[0]
            s_prev[query_rows] = (distances <= tolerance).astype(float)

    row_records: list[dict[str, Any]] = []
    for particle_id in range(n_particles):
        birth_time = float(birth_times[particle_id])
        exit_time = float(exit_trigger_time[particle_id])
        exited = np.isfinite(exit_time)
        traversal_time = float(den[particle_id])
        success_fraction = float(num[particle_id] / den[particle_id]) if den[particle_id] > 0.0 else math.nan
        n_samples = int(np.count_nonzero((sim_times >= birth_time) & (sim_times <= (exit_time if np.isfinite(exit_time) else np.inf))))
        row_records.append(
            {
                "particle_id": int(particle_id),
                "success_fraction": success_fraction,
                "traversal_time": traversal_time,
                "n_samples": n_samples,
                "birth_time": birth_time,
                "exit_time": exit_time,
                "exited": bool(exited),
            }
        )

    diagnostics = {
        "birth_after_final_snapshot": int(np.count_nonzero(birth_times > sim_times[-1])),
        "exit_less_than_or_equal_birth": int(np.count_nonzero(np.isfinite(exit_trigger_time) & (exit_trigger_time <= birth_times))),
        "case3_count": int(case3_count),
        "timestep_warnings": int(timesteps_mismatch_count),
        "mean_n_samples_exited": float(np.mean([row["n_samples"] for row in row_records if row["exited"]])) if any(row["exited"] for row in row_records) else 0.0,
        "min_n_samples_exited": int(np.min([row["n_samples"] for row in row_records if row["exited"]])) if any(row["exited"] for row in row_records) else 0,
        "first_snapshot_walkable_fraction": float(first_snapshot_walkable_fraction),
    }

    log.info("%s: birth after final snapshot = %d; e_i <= b_i = %d; case3_count = %d; walkable first snapshot fraction = %.4f", shot_dir, diagnostics["birth_after_final_snapshot"], diagnostics["exit_less_than_or_equal_birth"], diagnostics["case3_count"], diagnostics["first_snapshot_walkable_fraction"])
    return {"rows": row_records, "diagnostics": diagnostics}


def write_shot_outputs(
    shot_dir: str | Path,
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path | None = None,
) -> Path:
    """Write per-shot particle success output and return the file path."""
    shot_dir = Path(shot_dir)
    target_dir = Path(output_dir) if output_dir is not None else shot_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / "particle_success.txt"

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            success_value = row["success_fraction"]
            success_text = "nan" if not math.isfinite(success_value) else f"{success_value:.6f}"
            birth_time = float(row["birth_time"])
            exit_time = float(row["exit_time"])
            birth_text = "nan" if not math.isfinite(birth_time) else f"{birth_time:.6f}"
            exit_text = "inf" if np.isinf(exit_time) else ("nan" if not math.isfinite(exit_time) else f"{exit_time:.6f}")
            file.write(
                f"{int(row['particle_id']):d} {success_text} "
                f"{float(row['traversal_time']):.6f} {int(row['n_samples']):d} "
                f"{birth_text} {exit_text} {1 if bool(row['exited']) else 0:d}\n"
            )
    return output_path


def _parse_particle_success_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 7:
                continue
            particle_id, success_fraction, traversal_time, n_samples, birth_time, exit_time, exited = parts
            rows.append(
                {
                    "particle_id": int(particle_id),
                    "success_fraction": float(success_fraction) if success_fraction.lower() != "nan" else math.nan,
                    "traversal_time": float(traversal_time),
                    "n_samples": int(n_samples),
                    "birth_time": float(birth_time) if birth_time.lower() != "nan" else math.nan,
                    "exit_time": float(exit_time) if exit_time.lower() not in {"nan", "inf"} else (math.inf if exit_time.lower() == "inf" else math.nan),
                    "exited": bool(int(exited)),
                }
            )
    return rows


def _summarize_birth_times(
    rows: Iterable[dict[str, Any]], bin_width: float = 1.0
) -> list[dict[str, Any]]:
    if not math.isfinite(bin_width) or bin_width <= 0.0:
        raise ValueError("Birth-time bin width must be positive and finite.")
    groups: dict[int, dict[str, Any]] = {}
    for row in rows:
        birth_time = float(row["birth_time"])
        success_fraction = float(row["success_fraction"])
        if not math.isfinite(birth_time) or not math.isfinite(success_fraction):
            continue
        bin_index = math.floor(birth_time / bin_width)
        group = groups.setdefault(bin_index, {"values": [], "shots": set()})
        group["values"].append(success_fraction)
        if row.get("shot") is not None:
            group["shots"].add(row["shot"])

    summaries = []
    for bin_index, group in sorted(groups.items()):
        values = np.asarray(group["values"], dtype=float)
        count = len(values)
        standard_deviation = float(np.std(values, ddof=1)) if count > 1 else math.nan
        summaries.append(
            {
                "birth_time_bin": bin_index,
                "birth_time": (bin_index + 0.5) * bin_width,
                "mean_success_fraction": float(np.mean(values)),
                "standard_deviation_success_fraction": standard_deviation,
                "particle_count": count,
                "shot_count": len(group["shots"]),
            }
        )
    return summaries


def plot_configuration(
    run_id: str,
    shot_rows: dict[int, list[dict[str, Any]]],
    output_path: str | Path,
    logger_obj: logging.Logger | None = None,
    birth_bin_width: float = 1.0,
    show_individual_points: bool = False,
) -> None:
    """Plot mean success fraction by birth time for one configuration."""
    log = _ensure_logger(logger_obj)
    fig, ax = plt.subplots(figsize=(10, 5))
    rows = [dict(row, shot=shot_index) for shot_index, shot_rows in shot_rows.items() for row in shot_rows]
    summary = _summarize_birth_times(rows, bin_width=birth_bin_width)
    if summary:
        if show_individual_points:
            ax.scatter(
                [float(row["birth_time"]) for row in rows if math.isfinite(float(row["birth_time"])) and math.isfinite(float(row["success_fraction"]))],
                [float(row["success_fraction"]) for row in rows if math.isfinite(float(row["birth_time"])) and math.isfinite(float(row["success_fraction"]))],
                alpha=0.16,
                s=12,
                color="tab:gray",
                label="Individual particles",
                zorder=1,
            )
        means = [row["mean_success_fraction"] for row in summary]
        errors = [row["standard_deviation_success_fraction"] for row in summary]
        ax.errorbar(
            [row["birth_time"] for row in summary],
            means,
            yerr=[error if math.isfinite(error) else 0.0 for error in errors],
            fmt="o-",
            markersize=3,
            capsize=3,
            linewidth=1.2,
            label=f"Binned mean ± 1σ (SD), width {birth_bin_width:g}",
            zorder=2,
        )
        ax.legend()

    ax.set_title(f"{run_id} ({len(shot_rows)} shots)")
    ax.set_xlabel("Particle birth time")
    ax.set_ylabel("Mean success fraction across shots")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    log.info("Saved configuration plot to %s", path)


def aggregate(
    series_root: str | Path, output_root: str | Path, birth_bin_width: float = 1.0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate shot-level outputs into a long-form table and per-configuration summary."""
    series_root = Path(series_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    birth_time_rows: list[dict[str, Any]] = []

    for run_dir in sorted(series_root.glob("*")):
        if not run_dir.is_dir():
            continue
        if not re.fullmatch(r".*", run_dir.name):
            continue
        run_id = run_dir.name
        shot_dirs = sorted(run_dir.glob("shot_*"))
        shot_count = 0
        config_particles: list[dict[str, Any]] = []
        for shot_dir in shot_dirs:
            particle_file = shot_dir / "particle_success.txt"
            if not particle_file.exists():
                continue
            shot_count += 1
            shot_name = shot_dir.name
            for row in _parse_particle_success_file(particle_file):
                long_rows.append(
                    {
                        "series": series_root.name,
                        "run_id": run_id,
                        "shot": shot_name,
                        "particle_id": row["particle_id"],
                        "success_fraction": row["success_fraction"],
                        "traversal_time": row["traversal_time"],
                        "n_samples": row["n_samples"],
                        "birth_time": row["birth_time"],
                        "exit_time": row["exit_time"],
                        "exited": row["exited"],
                    }
                )
                config_particles.append({
                    "run_id": run_id,
                    "shot": shot_name,
                    "particle_id": row["particle_id"],
                    "birth_time": row["birth_time"],
                    "success_fraction": row["success_fraction"],
                    "traversal_time": row["traversal_time"],
                    "exited": row["exited"],
                })

        if shot_count == 0 or not config_particles:
            continue

        birth_time_rows.extend(
            {"series": series_root.name, "run_id": run_id, **row}
            for row in _summarize_birth_times(config_particles, bin_width=birth_bin_width)
        )
        valid = np.asarray([float(item["success_fraction"]) for item in config_particles if np.isfinite(item["success_fraction"]) and float(item["traversal_time"]) > 0.0], dtype=float)
        mean_success = float(np.mean(valid)) if valid.size else math.nan
        sem_success = float(np.std(valid, ddof=1) / np.sqrt(valid.size)) if valid.size > 1 else math.nan
        fraction_exited = float(np.mean([1 if item["exited"] else 0 for item in config_particles])) if config_particles else math.nan
        summary_rows.append(
            {
                "run_id": run_id,
                "n_shots": int(shot_count),
                "n_particles": int(len(config_particles)),
                "mean_success_fraction": mean_success,
                "standard_error_success_fraction": sem_success,
                "fraction_exited": fraction_exited,
            }
        )

    all_path = output_root / "particle_success_all.csv"
    with all_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["series", "run_id", "shot", "particle_id", "success_fraction", "traversal_time", "n_samples", "birth_time", "exit_time", "exited"],
        )
        writer.writeheader()
        writer.writerows(long_rows)

    summary_path = output_root / "particle_success_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["run_id", "n_shots", "n_particles", "mean_success_fraction", "standard_error_success_fraction", "fraction_exited"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    birth_time_path = output_root / "particle_success_by_birth_time.csv"
    with birth_time_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["series", "run_id", "birth_time_bin", "birth_time", "mean_success_fraction", "standard_deviation_success_fraction", "particle_count", "shot_count"],
        )
        writer.writeheader()
        writer.writerows(birth_time_rows)

    return long_rows, summary_rows


def _discover_run_ids(series_dir: Path, only_glob: str | None = None) -> list[str]:
    run_dirs = sorted(series_dir.glob("*"))
    run_ids = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("."):
            continue
        if not any(part.is_dir() for part in run_dir.iterdir()):
            continue
        if only_glob is not None and not fnmatch.fnmatch(run_dir.name, only_glob):
            continue
        run_ids.append(run_dir.name)
    return run_ids


def _run_single_shot(args: tuple[str, Path, np.ndarray, np.ndarray, float, float, float, logging.Logger]) -> tuple[str, dict[str, Any], Path | None]:
    shot_path, maze_path, maze_arr, path_arr, tolerance, wall_weight, wall_power, logger_obj = args
    shot_dir = Path(shot_path)
    try:
        result = score_shot(shot_dir, maze=maze_arr, path=path_arr, tolerance=tolerance, wall_weight=wall_weight, wall_power=wall_power, logger_obj=logger_obj)
        output_dir = shot_dir
        write_shot_outputs(output_dir, result["rows"], output_dir)
        return (str(shot_dir), result, output_dir / "particle_success.txt")
    except Exception as error:  # pragma: no cover - used in multiprocessing where exceptions need to be captured.
        return (str(shot_dir), {"error": str(error)}, None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze particle success fractions across a series of shots.")
    parser.add_argument("--config", type=Path, help="TOML configuration file with series.data_root and series.output_root.")
    parser.add_argument("--series-dir", type=Path, help="Direct path to a series directory: <data_root>/<series>.")
    parser.add_argument("--maze-file", type=Path, help="Path to the maze TSV file.")
    parser.add_argument("--maze-meta", type=Path, help="Maze metadata TOML with maze_file, wall_value, emitter_grid, and goal_grid.")
    parser.add_argument("--jobs", type=Path, help="Generated jobs.txt used to label plots with simulation parameters.")
    parser.add_argument("--out-dir", type=Path, help="Single directory for all PNG plots; defaults to output_root/<series>.")
    parser.add_argument("--birth-bin-width", type=float, default=1.0, help="Birth-time bin width in simulation-time units.")
    parser.add_argument("--show-individual-points", action="store_true", help="Overlay transparent per-particle points on binned means.")
    parser.add_argument("--start", type=int, nargs=2, default=None, metavar=("I", "J"), help="Override maze start coordinate in simulation frame.")
    parser.add_argument("--end", type=int, nargs=2, default=None, metavar=("I", "J"), help="Override maze end coordinate in simulation frame.")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--wall-weight", type=float, default=DEFAULT_WALL_WEIGHT)
    parser.add_argument("--wall-power", type=float, default=DEFAULT_WALL_POWER)
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for parallel shot scoring. Set to -1 for all available CPUs.")
    parser.add_argument("--only", type=str, default=None, help="Glob filter for run_id names.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-shot particle_success.txt outputs.")
    parser.add_argument("--path-only", action="store_true", help="Create the path-tolerance plot and exit before analysis.")
    parser.add_argument("--show-path", action="store_true", help="Display the path-tolerance plot interactively (allowed only for path display).")
    parser.add_argument("--no-plots", action="store_true", help="Disable all figure generation.")
    args = parser.parse_args()

    if not math.isfinite(args.birth_bin_width) or args.birth_bin_width <= 0.0:
        parser.error("--birth-bin-width must be positive and finite")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    config_dict: dict[str, Any] | None = None
    if args.config is not None:
        with args.config.open("rb") as file:
            config_dict = tomllib.load(file)
        series_cfg = config_dict.get("series", {})
        series_name = str(series_cfg.get("name"))
        data_root = Path(series_cfg.get("data_root", "data"))
        output_root = Path(series_cfg.get("output_root", "output"))
        if not data_root.is_absolute():
            data_root = (args.config.parent / data_root).resolve()
        if not output_root.is_absolute():
            output_root = (args.config.parent / output_root).resolve()
        if not series_name:
            raise ValueError("Config file must provide series.name.")
    elif args.series_dir is not None:
        series_dir = args.series_dir.resolve()
        data_root = series_dir.parent
        output_root = data_root / "output"
        series_name = series_dir.name
    else:
        raise ValueError("Provide either --config or --series-dir.")

    maze_metadata: dict[str, Any] = {}
    metadata_path = args.maze_meta.resolve() if args.maze_meta is not None else None
    if metadata_path is not None:
        with metadata_path.open("rb") as file:
            maze_metadata = tomllib.load(file)
    if args.maze_file is not None:
        maze_path = args.maze_file.resolve()
    elif "maze_file" in maze_metadata:
        maze_path = _resolve_maze_file(metadata_path, maze_metadata["maze_file"])
    elif config_dict is not None and "maze_file" in config_dict:
        maze_path = Path(config_dict["maze_file"]).resolve()
    else:
        raise ValueError("Provide --maze-file or --maze-meta with a maze_file entry.")

    series_root = data_root / series_name
    if not series_root.exists():
        raise FileNotFoundError(f"Series directory does not exist: {series_root}")

    raw_shape = np.loadtxt(maze_path, delimiter="\t", dtype=int).shape
    metadata_start = _maze_coordinate_to_sim(maze_metadata["emitter_grid"], raw_shape) if "emitter_grid" in maze_metadata else START_DEFAULT
    metadata_end = _maze_coordinate_to_sim(maze_metadata["goal_grid"], raw_shape) if "goal_grid" in maze_metadata else END_DEFAULT
    start = tuple(args.start) if args.start is not None else metadata_start
    end = tuple(args.end) if args.end is not None else metadata_end

    maze = load_maze_oriented(maze_path, start=start, end=end, wall_value=int(maze_metadata.get("wall_value", 1)))
    path = centerline_path(maze, start, end, wall_weight=args.wall_weight, wall_power=args.wall_power)
    plot_output_dir = args.out_dir.resolve() if args.out_dir is not None else output_root / series_name
    jobs_candidates = (
        output_root / series_name / "jobs.txt",
        data_root.parent / "output" / series_name / "jobs.txt",
        Path(__file__).resolve().parents[1] / "output" / series_name / "jobs.txt",
    )
    jobs_path = args.jobs.resolve() if args.jobs is not None else next(
        (candidate for candidate in jobs_candidates if candidate.is_file()), jobs_candidates[0]
    )
    run_parameters = _parameters_from_grid(config_dict)
    run_parameters.update(_read_jobs(jobs_path))
    if not run_parameters:
        logger.warning("No parameter grid or jobs file found; plot filenames will use run IDs.")
    if not args.no_plots:
        path_plot_path = plot_output_dir / f"maze_{maze_path.stem}_path_tolerance.png"
        plot_path_tolerance(maze, path, tolerance=args.tolerance, start=start, end=end, output_path=path_plot_path, show=args.show_path)
        if args.path_only:
            return 0

    run_ids = _discover_run_ids(series_root, only_glob=args.only)
    if not run_ids:
        logger.warning("No run_id directories matched under %s", series_root)
        return 0

    all_aggregated_rows: list[dict[str, Any]] = []
    skipped_rows: list[tuple[str, str]] = []
    shot_jobs: list[tuple[str, Path, np.ndarray, np.ndarray, float, float, float, logging.Logger]] = []

    for run_id in run_ids:
        run_dir = series_root / run_id
        shot_dirs = sorted(run_dir.glob("shot_*"))
        for shot_dir in shot_dirs:
            if not shot_dir.is_dir():
                continue
            if (shot_dir / "particle_success.txt").exists() and not args.overwrite:
                continue
            try:
                shot_index = index_shot(shot_dir, output_dir=shot_dir, logger_obj=logger)
                if not shot_index:
                    raise ValueError("No part_*.npz snapshots found.")
                state_path = shot_dir / "param_state.npz"
                if not state_path.exists():
                    raise FileNotFoundError(f"Missing {state_path}")
                shot_jobs.append((str(shot_dir), shot_dir, maze, path, args.tolerance, args.wall_weight, args.wall_power, logger))
            except Exception as error:
                skipped_rows.append((str(shot_dir), str(error)))
                logger.warning("Skipping %s: %s", shot_dir, error)

    if not shot_jobs:
        logger.warning("No eligible shots to score under %s", series_root)
    else:
        worker_count = args.workers
        if worker_count == -1:
            worker_count = None
        if worker_count == 1:
            results = []
            for job in tqdm(shot_jobs, desc="Scoring shots", unit="shot"):
                shot_dir, _, maze_arr, path_arr, tolerance, wall_weight, wall_power, log_handle = job
                try:
                    result = score_shot(shot_dir, maze=maze_arr, path=path_arr, tolerance=tolerance, wall_weight=wall_weight, wall_power=wall_power, logger_obj=log_handle)
                    write_shot_outputs(shot_dir, result["rows"], shot_dir)
                    results.append((shot_dir, result))
                except Exception as error:
                    logger.warning("Skipping %s: %s", shot_dir, error)
                    skipped_rows.append((str(shot_dir), str(error)))
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_run_single_shot, (job[0], job[1], job[2], job[3], job[4], job[5], job[6], job[7])) for job in shot_jobs]
                for future in tqdm(futures, desc="Scoring shots", unit="shot"):
                    shot_path, result, output_file = future.result()
                    if output_file is None:
                        if "error" in result:
                            skipped_rows.append((shot_path, result["error"]))
                        continue
                    if not result.get("rows"):
                        logger.warning("No rows produced for %s", shot_path)
                    else:
                        all_aggregated_rows.extend(result["rows"])

    if not args.no_plots:
        for run_id in run_ids:
            run_dir = series_root / run_id
            shot_rows: dict[int, list[dict[str, Any]]] = {}
            for shot_dir in sorted(run_dir.glob("shot_*")):
                particle_file = shot_dir / "particle_success.txt"
                if not particle_file.exists():
                    continue
                shot_key = int(shot_dir.name.split("_")[-1])
                records = _parse_particle_success_file(particle_file)
                if records:
                    shot_rows[shot_key] = records
            if shot_rows:
                plot_path = plot_output_dir / _configuration_plot_name(run_id, run_parameters.get(run_id, {}), maze_path)
                plot_configuration(
                    run_id,
                    shot_rows,
                    plot_path,
                    logger_obj=logger,
                    birth_bin_width=args.birth_bin_width,
                    show_individual_points=args.show_individual_points,
                )

    aggregate_root = output_root / series_name
    aggregate(series_root, aggregate_root, birth_bin_width=args.birth_bin_width)

    skipped_path = output_root / series_name / "skipped_shots.tsv"
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    with skipped_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["shot_path", "failure_reason"])
        writer.writerows(skipped_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
