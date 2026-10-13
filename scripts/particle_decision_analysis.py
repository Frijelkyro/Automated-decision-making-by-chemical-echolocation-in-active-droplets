
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


def analyze(data_dir, maze_dir):
    shape, dx, zones = load_maze(maze_dir)
    birth_times = load_birth_times(data_dir)
    paths = sorted(
        Path(data_dir).glob("part_*.npz"),
        key=lambda p: int(re.fullmatch(r"part_(\d+)\.npz", p.name).group(1))
        if re.fullmatch(r"part_(\d+)\.npz", p.name) else -1,
    )
    paths = [p for p in paths if re.fullmatch(r"part_(\d+)\.npz", p.name)]
    if not paths:
        raise ValueError("No part_<timestep>.npz snapshots found")

    # Current state for each particle and junction.
    state = {}
    particles = set()
    events = []

    for path in paths:
        with np.load(path) as data:
            x, y = data["x"], data["y"]
            ids = data["particle_ids"] if "particle_ids" in data else np.arange(len(x))
            time = float(data["simulation_time"])
            positions = np.column_stack((x, y))
            finite = np.isfinite(positions).all(axis=1)

            cols = np.zeros(len(x), dtype=int)
            rows = np.zeros(len(x), dtype=int)
            cols[finite] = np.rint(x[finite] / dx).astype(int)
            rows[finite] = shape[0] - 1 - np.rint(y[finite] / dx).astype(int)
            valid = finite & (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])

            for i, particle_id in enumerate(ids):
                pid = int(particle_id)
                particles.add(pid)
                for junction, masks in zones.items():
                    key = (pid, junction)
                    old = state.setdefault(key, {
                        "before": False, "correct": False, "wrong": False,
                        "open": False, "first_decision": None,
                    })

                    inside = {
                        name: bool(valid[i] and masks[name][rows[i], cols[i]])
                        for name in ("before", "correct", "wrong")
                    }
                    entered_before = inside["before"] and not old["before"]
                    entered_correct = inside["correct"] and not old["correct"]
                    entered_wrong = inside["wrong"] and not old["wrong"]

                    ambiguous = inside["correct"] and inside["wrong"]
                    decision = None
                    if old["open"] and not ambiguous:
                        if entered_correct and not entered_wrong:
                            decision = "correct"
                        elif entered_wrong and not entered_correct:
                            decision = "wrong"

                    if decision is not None:
                        event = {
                            "particle_id": pid,
                            "birth_time": birth_times.get(pid, math.nan),
                            "junction_id": junction,
                            "decision_time": time,
                            "decision": decision,
                        }
                        events.append(event)
                        if old["first_decision"] is None:
                            old["first_decision"] = decision
                        old["open"] = False
                    elif not old["open"] and entered_before and not (
                        entered_correct or entered_wrong or ambiguous
                    ):
                        old["open"] = True

                    old.update(inside)

    # Score each junction once, using its first completed decision.
    summary = []
    for pid in sorted(particles):
        decisions = [
            state[(pid, junction)]["first_decision"] for junction in zones
        ]
        correct = sum(decision == "correct" for decision in decisions)
        summary.append({
            "particle_id": pid,
            "birth_time": birth_times.get(pid, math.nan),
            "total_junctions": len(zones),
            "correct_junctions": correct,
            "wrong_or_unpassed_junctions": len(zones) - correct,
            "success_fraction": correct / len(zones),
        })

    return events, summary


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_birth_times(summary, output, bin_width):
    rows = [
        r for r in summary
        if math.isfinite(r["birth_time"])
    ]
    if not rows:
        warnings.warn("No valid birth times to plot")
        return

    groups = defaultdict(list)
    for row in rows:
        index = math.floor(row["birth_time"] / bin_width)
        groups[index].append(row["success_fraction"])

    times, means, counts = [], [], []
    for index, fractions in sorted(groups.items()):
        times.append((index + 0.5) * bin_width)
        means.append(np.mean(fractions))
        counts.append(len(fractions))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(times, means, "o-", markersize=4, linewidth=1.5)
    ax.set(
        xlabel="Particle birth time",
        ylabel="Mean fraction of all junctions passed correctly",
        title="Particle decision success vs birth time",
        ylim=(-0.05, 1.05),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory for one simulation shot")
    parser.add_argument("--maze-analysis-dir", required=True, help="Directory containing maze masks")
    parser.add_argument("--output-dir", default="decision_results")
    parser.add_argument("--birth-bin-width", type=float, default=1.0)
    args = parser.parse_args()

    if not math.isfinite(args.birth_bin_width) or args.birth_bin_width <= 0:
        parser.error("--birth-bin-width must be positive and finite")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    events, summary = analyze(args.data_dir, args.maze_analysis_dir)
    write_csv(output / "junction_decisions.csv", events)
    write_csv(output / "particle_success.csv", summary)
    plot_birth_times(
        summary,
        output / "success_vs_birth_time.png",
        args.birth_bin_width,
    )

    print(f"Particles: {len(summary)}")
    print(f"Decision events: {len(events)}")
    print(f"Junctions per particle score: {len(summary[0]['total_junctions']) if False else (summary[0]['total_junctions'] if summary else 0)}")
    print(f"Results saved to {output.resolve()}")


if __name__ == "__main__":
    main()
