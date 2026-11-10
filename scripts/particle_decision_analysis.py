
"""Track particle junction decisions and plot per-particle success."""

import argparse
import csv
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, cast

import matplotlib.pyplot as plt
import h5py
import numpy as np
from tqdm import tqdm


def load_maze(directory):
    import tomllib

    directory = Path(directory)
    meta = tomllib.loads((directory / "maze_metadata.toml").read_text())
    shape, dx = tuple(meta["grid_shape"]), float(meta["dx"])
    zones = defaultdict(dict)

    with np.load(directory / "zone_masks.npz") as data:
        for name in data.files:
            for suffix in ("before", "correct", "wrong"):
                ending = "_" + suffix
                if name.endswith(ending):
                    zones[name[:-len(ending)]][suffix] = data[name].astype(bool)
                    break

    for junction, masks in zones.items():
        if not all(k in masks for k in ("before", "correct", "wrong")):
            raise ValueError(f"Incomplete masks for junction {junction}")
        if any(mask.shape != shape for mask in masks.values()):
            raise ValueError(f"Incorrect mask shape for junction {junction}")

    if not zones:
        raise ValueError("No junction masks found")
    return shape, dx, dict(zones)


def load_birth_times(directory):
    candidates = []
    for path in Path(directory).glob("param*_state*.npz"):
        try:
            with np.load(path) as data:
                if "birth_times" not in data:
                    continue
                time = float(data["simulation_time"]) if "simulation_time" in data else -math.inf
                step = int(data["resume_step"]) if "resume_step" in data else -1
                candidates.append((time, step, path, data["birth_times"].copy()))
        except (OSError, ValueError, KeyError):
            continue

    if not candidates:
        warnings.warn("No particle birth times found")
        return {}

    birth_times = max(candidates, key=lambda item: item[:2])[3]
    return {i: float(t) for i, t in enumerate(birth_times)}


def analyze(data_dir, maze_dir=None, shot_id=None, maze_data=None, show_progress=False):
    shape, dx, zones = maze_data if maze_data is not None else load_maze(maze_dir)
    birth_times = load_birth_times(data_dir)
    hdf5_candidates = (
        Path(data_dir) / "particles_analysis.h5",
        Path(data_dir) / "particles.h5",
    )
    hdf5_path = next((path for path in hdf5_candidates if path.is_file()), hdf5_candidates[0])
    hdf5_handle: h5py.File | None = None
    h5_x: h5py.Dataset | None = None
    h5_y: h5py.Dataset | None = None
    h5_ids: h5py.Dataset | None = None
    h5_timestep: h5py.Dataset | None = None
    h5_time: h5py.Dataset | None = None
    static_particle_ids = None
    paths = []
    for path in Path(data_dir).glob("part_*.npz"):
        match = re.fullmatch(r"part_(\d+)\.npz", path.name)
        if match:
            paths.append((int(match.group(1)), path))
    paths = [path for _, path in sorted(paths)]
    hdf5_handle = None
    if hdf5_path.is_file():
        hdf5_handle = h5py.File(hdf5_path, "r")
        if hdf5_handle.attrs.get("format") not in {
            "particle-decision-analysis-v1",
            "full-shot-hdf5-v1",
        }:
            hdf5_handle.close()
            raise ValueError(f"Unsupported HDF5 particle format in {hdf5_path}")
        if not hdf5_handle.attrs.get("complete", False):
            hdf5_handle.close()
            raise ValueError(f"HDF5 particle file is incomplete: {hdf5_path}")
        required = {"x", "y", "particle_ids", "timestep", "simulation_time"}
        if not required.issubset(hdf5_handle.keys()):
            hdf5_handle.close()
            raise ValueError(f"HDF5 particle file is missing datasets: {', '.join(sorted(required - set(hdf5_handle.keys())))}")
        h5_x = cast(h5py.Dataset, hdf5_handle["x"])
        h5_y = cast(h5py.Dataset, hdf5_handle["y"])
        h5_ids = cast(h5py.Dataset, hdf5_handle["particle_ids"])
        h5_timestep = cast(h5py.Dataset, hdf5_handle["timestep"])
        h5_time = cast(h5py.Dataset, hdf5_handle["simulation_time"])
        frame_count = len(h5_time)
        if any(len(dataset) != frame_count for dataset in (h5_x, h5_y, h5_timestep)):
            hdf5_handle.close()
            raise ValueError(f"HDF5 datasets have inconsistent frame counts in {hdf5_path}")
        ids_shape = h5_ids.shape
        if len(ids_shape) not in (1, 2) or ids_shape[-1] != h5_x.shape[-1]:
            hdf5_handle.close()
            raise ValueError(f"Invalid particle_ids dataset shape in {hdf5_path}")
        if len(ids_shape) == 2 and ids_shape[0] != frame_count:
            hdf5_handle.close()
            raise ValueError(f"Per-frame particle IDs have an inconsistent frame count in {hdf5_path}")
        static_particle_ids = h5_ids[:] if len(ids_shape) == 1 else None
        paths = range(frame_count)
        if not frame_count:
            hdf5_handle.close()
            raise ValueError(f"HDF5 particle file contains no snapshots: {hdf5_path}")

        def read_hdf5_frame(frame_index: int):
            return (
                h5_x[frame_index],
                h5_y[frame_index],
                static_particle_ids if static_particle_ids is not None else h5_ids[frame_index],
                float(h5_time[frame_index]),
                int(h5_timestep[frame_index]),
            )

    else:
        def read_npz_snapshot(path: Path):
            with np.load(path) as data:
                x = np.asarray(data["x"])
                y = np.asarray(data["y"])
                ids = np.asarray(data["particle_ids"] if "particle_ids" in data else np.arange(len(x)))
                time = float(data["simulation_time"])
                timestep = int(data["timestep"]) if "timestep" in data else int(path.stem[5:])
                return x, y, ids, time, timestep


    def read_snapshot(locator: Path | int):
        if hdf5_handle is not None:
            if not isinstance(locator, int):
                raise TypeError(f"Expected an HDF5 frame index, got {locator!r}")
            return read_hdf5_frame(locator)
        if not isinstance(locator, Path):
            raise TypeError(f"Expected an NPZ path, got {locator!r}")
        return read_npz_snapshot(locator)

    if not paths:
        raise ValueError(f"No part_<timestep>.npz snapshots found in {data_dir}")

    junction_ids = list(zones)
    junction_count = len(junction_ids)
    before_masks = np.stack([zones[junction]["before"] for junction in junction_ids])
    correct_masks = np.stack([zones[junction]["correct"] for junction in junction_ids])
    wrong_masks = np.stack([zones[junction]["wrong"] for junction in junction_ids])
    def process_paths(ordered_paths, verify_order, description, read_snapshot):
        particles = []
        id_to_slot = {}
        state = {
            "birth_times": np.empty(0, dtype=float),
            "retired": np.empty(0, dtype=bool),
            "open": np.empty((0, junction_count), dtype=bool),
            "previous_before": np.empty((0, junction_count), dtype=bool),
            "previous_correct": np.empty((0, junction_count), dtype=bool),
            "previous_wrong": np.empty((0, junction_count), dtype=bool),
            "first_decision": np.empty((0, junction_count), dtype=np.int8),
        }
        local_events = []
        previous_key = None

        def ensure_ids(ids):
            new_ids = sorted(set(map(int, ids)) - id_to_slot.keys())
            if not new_ids:
                return
            old_count = len(particles)
            particles.extend(new_ids)
            id_to_slot.update({pid: old_count + offset for offset, pid in enumerate(new_ids)})
            state["birth_times"] = np.concatenate(
                (state["birth_times"], np.array([birth_times.get(pid, math.nan) for pid in new_ids]))
            )
            state["retired"] = np.concatenate((state["retired"], np.zeros(len(new_ids), dtype=bool)))
            for name in ("open", "previous_before", "previous_correct", "previous_wrong"):
                state[name] = np.vstack((state[name], np.zeros((len(new_ids), junction_count), dtype=bool)))
            state["first_decision"] = np.vstack(
                (state["first_decision"], np.zeros((len(new_ids), junction_count), dtype=np.int8))
            )

        snapshot_progress = tqdm(
            ordered_paths,
            total=len(ordered_paths),
            desc=description,
            unit="file",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        )
        for path in snapshot_progress:
            if not isinstance(path, (Path, int)):
                raise TypeError(f"Unexpected snapshot locator: {path!r}")
            x, y, ids, time, timestep = read_snapshot(path)
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            ids = np.asarray(ids)
            if x.ndim != 1 or y.shape != x.shape:
                raise ValueError(f"Invalid position arrays in {path}")
            if ids.shape != x.shape:
                raise ValueError(f"Particle ID count does not match positions in {path}")
            sort_key = (time, timestep)
            if verify_order and previous_key is not None and sort_key < previous_key:
                return None
            previous_key = sort_key

            ensure_ids(ids)
            slots = np.fromiter((id_to_slot[int(pid)] for pid in ids), dtype=int, count=len(ids))
            finite = np.isfinite(x) & np.isfinite(y)
            state["retired"][slots[~finite]] = True
            born = state["birth_times"][slots] <= time
            eligible = finite & born & ~state["retired"][slots]
            eligible_indices = np.flatnonzero(eligible)
            if not len(eligible_indices):
                continue

            eligible_slots = slots[eligible_indices]
            cols = np.rint(x[eligible_indices] / dx).astype(int)
            rows = shape[0] - 1 - np.rint(y[eligible_indices] / dx).astype(int)
            inside_grid = (
                (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
            )
            inside_before = np.zeros((len(eligible_indices), junction_count), dtype=bool)
            if inside_grid.any():
                inside_before[inside_grid] = before_masks[:, rows[inside_grid], cols[inside_grid]].T
            was_open = state["open"][eligible_slots].copy()
            entered_before = inside_before & ~state["previous_before"][eligible_slots]
            state["previous_before"][eligible_slots] = inside_before
            state["open"][eligible_slots] |= (~was_open & entered_before)

            query_targets = was_open & inside_grid[:, np.newaxis]
            target_particles, target_junctions = np.where(query_targets)
            previous_correct = state["previous_correct"][eligible_slots].copy()
            previous_wrong = state["previous_wrong"][eligible_slots].copy()
            previous_correct[~query_targets] = False
            previous_wrong[~query_targets] = False
            state["previous_correct"][eligible_slots] = previous_correct
            state["previous_wrong"][eligible_slots] = previous_wrong

            if len(target_particles):
                target_slots = eligible_slots[target_particles]
                target_rows = rows[target_particles]
                target_cols = cols[target_particles]
                inside_correct = correct_masks[target_junctions, target_rows, target_cols]
                inside_wrong = wrong_masks[target_junctions, target_rows, target_cols]
                entered_correct = inside_correct & ~previous_correct[target_particles, target_junctions]
                entered_wrong = inside_wrong & ~previous_wrong[target_particles, target_junctions]
                ambiguous = inside_correct & inside_wrong
                succeeded = entered_correct & ~entered_wrong & ~ambiguous
                failed = entered_wrong & ~entered_correct & ~ambiguous
                state["previous_correct"][target_slots, target_junctions] = inside_correct
                state["previous_wrong"][target_slots, target_junctions] = inside_wrong

                for event_index in np.flatnonzero(succeeded | failed):
                    slot = int(target_slots[event_index])
                    junction_index = int(target_junctions[event_index])
                    decision = "correct" if succeeded[event_index] else "wrong"
                    local_events.append(
                        {
                            "shot": shot_id,
                            "particle_id": particles[slot],
                            "birth_time": float(state["birth_times"][slot]),
                            "junction_id": junction_ids[junction_index],
                            "decision_time": time,
                            "decision": decision,
                        }
                    )
                    if state["first_decision"][slot, junction_index] == 0:
                        state["first_decision"][slot, junction_index] = 1 if decision == "correct" else -1
                    state["open"][slot, junction_index] = False

        summary = []
        for slot, particle_id in enumerate(particles):
            correct = int(np.count_nonzero(state["first_decision"][slot] == 1))
            decided = int(np.count_nonzero(state["first_decision"][slot]))
            summary.append(
                {
                    "shot": shot_id,
                    "particle_id": particle_id,
                    "birth_time": float(state["birth_times"][slot]),
                    "total_junctions": junction_count,
                    "decided_junctions": decided,
                    "correct_junctions": correct,
                    "wrong_or_unpassed_junctions": junction_count - correct,
                    "success_fraction": correct / junction_count if junction_count else math.nan,
                }
            )
        return local_events, summary

    try:
        if show_progress:
            source_format = "HDF5" if hdf5_handle is not None else "NPZ"
            tqdm.write(
                f"[start] {shot_id or Path(data_dir).name}: {len(paths):,} {source_format} snapshots, "
                f"{junction_count} junctions, {len(birth_times):,} saved birth times"
            )
        result = process_paths(
            paths,
            verify_order=True,
            description=f"{shot_id or Path(data_dir).name} snapshots",
            read_snapshot=read_snapshot,
        )
        if result is None:
            tqdm.write(
                f"[retry] {shot_id or Path(data_dir).name}: snapshot times are not ordered by timestep; "
                "restarting in physical-time order"
            )
            if hdf5_handle is not None:
                times = cast(h5py.Dataset, hdf5_handle["simulation_time"])[:]
                timesteps = cast(h5py.Dataset, hdf5_handle["timestep"])[:]
                ordered_paths = sorted(range(len(times)), key=lambda index: (times[index], timesteps[index]))
            else:
                metadata = []
                for path in paths:
                    if not isinstance(path, Path):
                        raise TypeError(f"Expected an NPZ path, got {path!r}")
                    try:
                        with np.load(path) as data:
                            time = float(data["simulation_time"])
                            timestep = int(data["timestep"]) if "timestep" in data else int(path.stem[5:])
                            metadata.append((time, timestep, path))
                    except (OSError, ValueError, KeyError) as error:
                        warnings.warn(f"Skipping incomplete snapshot {path}: {error}", RuntimeWarning)
                ordered_paths = [item[2] for item in sorted(metadata)]
            result = process_paths(
                ordered_paths,
                verify_order=False,
                description=f"{shot_id or Path(data_dir).name} time-order retry",
                read_snapshot=read_snapshot,
            )
    finally:
        if hdf5_handle is not None:
            hdf5_handle.close()
    if result is None:
        raise RuntimeError(f"Could not order particle snapshots in {data_dir}")
    events, summary = result
    return events, summary


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class BirthTimeBins:
    """Compute an equal-particle mean of each shot's all-junction fraction."""

    def __init__(self, bin_width):
        self.bin_width = bin_width
        self.groups = defaultdict(lambda: {"sum_fraction": 0.0, "particles": 0, "shots": set()})

    def add(self, summary_rows):
        for row in summary_rows:
            birth_time = float(row["birth_time"])
            fraction = float(row["success_fraction"])
            if not math.isfinite(birth_time) or not math.isfinite(fraction):
                continue
            index = math.floor(birth_time / self.bin_width)
            group = self.groups[index]
            group["sum_fraction"] += fraction
            group["particles"] += 1
            group["shots"].add(row.get("shot"))

    def rows(self):
        return [
            {
                "birth_time_bin": index,
                "birth_time": (index + 0.5) * self.bin_width,
                "mean_success_fraction": values["sum_fraction"] / values["particles"],
                "particle_count": values["particles"],
                "shot_count": len(values["shots"]),
            }
            for index, values in sorted(self.groups.items())
            if values["particles"]
        ]


def plot_birth_times(binned_summary, output):
    if not binned_summary:
        warnings.warn("No valid particle birth-time bins to plot")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    times = [row["birth_time"] for row in binned_summary]
    means = [row["mean_success_fraction"] for row in binned_summary]
    ax.plot(times, means, "o-", markersize=4, linewidth=1.5)
    ax.set(
        xlabel="Particle birth time",
        ylabel="Mean per-particle fraction of junctions chosen correctly",
        title=f"Particle decision success vs birth time (n={sum(row['particle_count'] for row in binned_summary)})",
        ylim=(-0.05, 1.05),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        required=True,
        help="One shot directory or an experiment run directory containing shot_* directories",
    )
    parser.add_argument("--maze-analysis-dir", required=True, help="Directory containing maze masks")
    parser.add_argument("--output-dir", default="decision_results")
    parser.add_argument("--birth-bin-width", type=float, default=1.0)
    args = parser.parse_args()

    if not math.isfinite(args.birth_bin_width) or args.birth_bin_width <= 0:
        parser.error("--birth-bin-width must be positive and finite")

    data_root = Path(args.data_dir)
    has_hdf5 = any((data_root / name).is_file() for name in ("particles_analysis.h5", "particles.h5"))
    has_npz = any(re.fullmatch(r"part_\d+\.npz", path.name) for path in data_root.glob("part_*.npz"))
    if has_hdf5 or has_npz:
        shot_dirs = [data_root]
    else:
        shot_dirs = sorted(
            path for path in data_root.glob("shot_*")
            if path.is_dir()
            and (
                any((path / name).is_file() for name in ("particles_analysis.h5", "particles.h5"))
                or any(re.fullmatch(r"part_\d+\.npz", item.name) for item in path.glob("part_*.npz"))
            )
        )
    if not shot_dirs:
        parser.error(f"No shot snapshots found under {data_root}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print(f"[init] Data directory: {data_root.resolve()}", flush=True)
    print(f"[init] Found {len(shot_dirs):,} shot director{'y' if len(shot_dirs) == 1 else 'ies'}", flush=True)
    print(f"[init] Loading maze metadata and masks from {Path(args.maze_analysis_dir).resolve()}", flush=True)
    maze_data = load_maze(args.maze_analysis_dir)
    print(
        f"[ready] Maze loaded: {maze_data[0][0]}x{maze_data[0][1]} grid, "
        f"dx={maze_data[1]:g}, {len(maze_data[2])} junctions",
        flush=True,
    )
    print(f"[ready] Output directory: {output.resolve()}", flush=True)
    print("[start] Processing snapshots; progress bars show files/s and ETA.", flush=True)
    event_fields = ["run_id", "shot", "particle_id", "birth_time", "junction_id", "decision_time", "decision"]
    summary_fields = [
        "run_id", "shot", "particle_id", "birth_time", "total_junctions", "decided_junctions",
        "correct_junctions", "wrong_or_unpassed_junctions", "success_fraction",
    ]
    event_count = particle_count = 0
    bins = BirthTimeBins(args.birth_bin_width)
    with (output / "junction_decisions.csv").open("w", newline="", encoding="utf-8") as event_file, (
        output / "particle_success.csv"
    ).open("w", newline="", encoding="utf-8") as particle_file:
        event_writer = csv.DictWriter(event_file, fieldnames=event_fields)
        particle_writer = csv.DictWriter(particle_file, fieldnames=summary_fields)
        event_writer.writeheader()
        particle_writer.writeheader()
        for shot_dir in tqdm(shot_dirs, desc="Shots", unit="shot", dynamic_ncols=True):
            shot_id = shot_dir.name if re.fullmatch(r"shot_\d+", shot_dir.name) else "shot_000"
            try:
                events, summary = analyze(
                    shot_dir,
                    shot_id=shot_id,
                    maze_data=maze_data,
                    show_progress=True,
                )
            except (OSError, ValueError, KeyError) as error:
                tqdm.write(f"[skip] Incomplete shot {shot_dir}: {error}")
                continue
            run_id = shot_dir.parent.name if shot_dir.name.startswith("shot_") else shot_dir.name
            for row in events:
                row["run_id"] = run_id
            for row in summary:
                row["run_id"] = run_id
            event_writer.writerows(events)
            particle_writer.writerows(summary)
            event_file.flush()
            particle_file.flush()
            bins.add(summary)
            event_count += len(events)
            particle_count += len(summary)
            tqdm.write(
                f"[done] {shot_id}: {len(summary):,} particles, {len(events):,} decisions"
            )

    binned_summary = bins.rows()
    write_csv(
        output / "birth_time_summary.csv",
        binned_summary,
        ["birth_time_bin", "birth_time", "mean_success_fraction", "particle_count", "shot_count"],
    )
    plot_birth_times(binned_summary, output / "success_vs_birth_time.png")

    print(f"Shots: {len(shot_dirs)}")
    print(f"Particles: {particle_count}")
    print(f"Decision events: {event_count}")
    print(f"Results saved to {output.resolve()}")


if __name__ == "__main__":
    main()
