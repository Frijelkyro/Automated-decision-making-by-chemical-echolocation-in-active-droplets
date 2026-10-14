
"""Track particle junction decisions and plot per-particle success."""

import argparse
import csv
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def analyze(data_dir, maze_dir, shot_id=None):
    shape, dx, zones = load_maze(maze_dir)
    birth_times = load_birth_times(data_dir)
    paths = []
    for path in Path(data_dir).glob("part_*.npz"):
        match = re.fullmatch(r"part_(\d+)\.npz", path.name)
        if match:
            paths.append((int(match.group(1)), path))
    paths = [path for _, path in sorted(paths)]
    if not paths:
        raise ValueError(f"No part_<timestep>.npz snapshots found in {data_dir}")

    junction_ids = list(zones)
    junction_count = len(junction_ids)
    before_masks = np.stack([zones[junction]["before"] for junction in junction_ids])
    correct_masks = np.stack([zones[junction]["correct"] for junction in junction_ids])
    wrong_masks = np.stack([zones[junction]["wrong"] for junction in junction_ids])
    def process_paths(ordered_paths, verify_order):
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

        for path in ordered_paths:
            with np.load(path) as data:
                x = np.asarray(data["x"], dtype=float)
                y = np.asarray(data["y"], dtype=float)
                if x.ndim != 1 or y.shape != x.shape:
                    raise ValueError(f"Invalid position arrays in {path}")
                ids = np.asarray(data["particle_ids"] if "particle_ids" in data else np.arange(len(x)))
                if ids.shape != x.shape:
                    raise ValueError(f"Particle ID count does not match positions in {path}")
                time = float(data["simulation_time"])
                timestep = int(data["timestep"]) if "timestep" in data else int(path.stem[5:])
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

    result = process_paths(paths, verify_order=True)
    if result is None:
        warnings.warn(
            f"Stored simulation times are not ordered by timestep in {data_dir}; "
            "retrying with physical-time sorting.",
            RuntimeWarning,
        )
        metadata = []
        for path in paths:
            try:
                with np.load(path) as data:
                    time = float(data["simulation_time"])
                    timestep = int(data["timestep"]) if "timestep" in data else int(path.stem[5:])
                    metadata.append((time, timestep, path))
            except (OSError, ValueError, KeyError) as error:
                warnings.warn(f"Skipping incomplete snapshot {path}: {error}", RuntimeWarning)
        result = process_paths([item[2] for item in sorted(metadata)], verify_order=False)
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
    if any(re.fullmatch(r"part_\d+\.npz", path.name) for path in data_root.glob("part_*.npz")):
        shot_dirs = [data_root]
    else:
        shot_dirs = sorted(
            path for path in data_root.glob("shot_*")
            if path.is_dir() and any(re.fullmatch(r"part_\d+\.npz", item.name) for item in path.glob("part_*.npz"))
        )
    if not shot_dirs:
        parser.error(f"No shot snapshots found under {data_root}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
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
        for shot_dir in shot_dirs:
            shot_id = shot_dir.name if re.fullmatch(r"shot_\d+", shot_dir.name) else "shot_000"
            try:
                events, summary = analyze(shot_dir, args.maze_analysis_dir, shot_id=shot_id)
            except (OSError, ValueError, KeyError) as error:
                warnings.warn(f"Skipping incomplete shot {shot_dir}: {error}", RuntimeWarning)
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
