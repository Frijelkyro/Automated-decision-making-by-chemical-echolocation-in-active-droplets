#!/usr/bin/env python3
"""Analyze junction decisions independently for each experiment configuration.

For a series TOML, each ``<regime>_cNNNN`` configuration is handled separately.
All shots belonging to one configuration contribute to that configuration's
plot and CSVs; data from different configurations are never pooled.

Default behavior is deliberately observed-event based: decision logs can show
which junctions a particle actually recorded, but they cannot establish that a
particle should have visited every junction. Therefore missing particle/junction
pairs are *not* inferred by default. To audit missing events, provide an explicit
``--expected-pairs`` CSV, or opt in to the stronger (and potentially false)
assumption with ``--audit-all-junctions``.

When a checkpoint is available, its simulation time, birth times, and spawned
state are used per shot. Particles that have not spawned by the last completed
simulation step are excluded from expected-pair audits and from any event rows
that fall beyond the checkpoint's progress time.

Outputs (default):
    <output_root>/<series>/decision_analysis/<run_id>/
        particle_decision_average_vs_junction.png
        particle_decision_averages.csv (including per-junction standard deviation)
        particle_success_summary.csv (overall success fraction per particle)
        decision_integrity_audit.csv
        decision_audit_summary.json
    <output_root>/<series>/decision_analysis/configurations_overview.csv
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import itertools
import json
import math
import re
import sys
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DECISION_RE = re.compile(r"decisions_(\d+)\.npz")
ANALYSIS_DIRNAME = "decision_analysis"


# ---------------------------------------------------------------------------
# Series / configuration discovery
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShotSpec:
    regime: str
    run_id: str
    config_index: int
    shot: int
    grid: dict[str, Any]
    params: dict[str, Any]
    out_dir: Path
    data_dir: Path

    @property
    def name(self) -> str:
        return f"{self.run_id}/shot_{self.shot:03d}"


def load_series_config(config_path: Path, root: Path) -> tuple[dict, Path]:
    config_path = config_path.expanduser()
    if not config_path.is_absolute() and not config_path.exists():
        config_path = root / config_path
    config_path = config_path.resolve()
    with config_path.open("rb") as fh:
        return tomllib.load(fh), config_path


def series_dirs(config: dict, root: Path) -> tuple[Path, Path]:
    series = config["series"]
    resolved = []
    for key in ("data_root", "output_root"):
        path = Path(series[key]).expanduser()
        resolved.append((path if path.is_absolute() else root / path) / series["name"])
    return resolved[0], resolved[1]


def expand_shots(
    config: dict,
    series_data: Path,
    series_out: Path,
    regime_filter: str | None = None,
    run_ids: set[str] | None = None,
) -> list[ShotSpec]:
    """Mirror make_jobs.generate_jobs' regime/grid/shot expansion exactly."""
    defaults = config.get("defaults", {})
    specs: list[ShotSpec] = []
    for regime in config["regimes"]:
        regime_name = regime["name"]
        if regime_filter and regime_name != regime_filter:
            continue
        grid = regime["grid"]
        keys = list(grid)
        for config_index, values in enumerate(itertools.product(*(grid[k] for k in keys))):
            run_id = f"{regime_name}_c{config_index:04d}"
            if run_ids and run_id not in run_ids:
                continue
            grid_values = dict(zip(keys, values))
            # Keep the exact effective parameter dictionary for provenance.
            params = defaults | grid_values
            for shot in range(int(regime["shots"])):
                shot_name = f"shot_{shot:03d}"
                specs.append(
                    ShotSpec(
                        regime=regime_name,
                        run_id=run_id,
                        config_index=config_index,
                        shot=shot,
                        grid=grid_values,
                        params=params,
                        out_dir=series_out / run_id / shot_name,
                        data_dir=series_data / run_id / shot_name,
                    )
                )
    return specs


def find_shot_decision_files(
    specs: list[ShotSpec], recursive: bool = False
) -> tuple[dict[str, list[Path]], list[str]]:
    found: dict[str, list[Path]] = {}
    notes: list[str] = []
    for spec in specs:
        if not spec.out_dir.is_dir():
            found[spec.name] = []
            notes.append(f"{spec.name}: output directory missing ({spec.out_dir})")
            continue
        iterator = spec.out_dir.rglob("decisions_*.npz") if recursive else spec.out_dir.glob("decisions_*.npz")
        files = sorted(
            (path for path in iterator if DECISION_RE.fullmatch(path.name)),
            key=lambda path: (int(DECISION_RE.fullmatch(path.name).group(1)), str(path)),
        )
        found[spec.name] = files
        chunk_numbers = {int(DECISION_RE.fullmatch(path.name).group(1)) for path in files}
        if chunk_numbers:
            gaps = sorted(set(range(max(chunk_numbers) + 1)) - chunk_numbers)
            if gaps:
                notes.append(f"{spec.name}: decision chunk numbering has gaps {gaps}")
    return found, notes


# ---------------------------------------------------------------------------
# Decision event loading
# ---------------------------------------------------------------------------
def _first(npz: Any, names: tuple[str, ...]):
    for name in names:
        if name in npz.files:
            return npz[name]
    return None


def _as_str(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype.kind == "S":
        return np.char.decode(array, "utf-8")
    return array.astype(str)


def load_decision_data(files: list[Path]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Read and concatenate decision chunks belonging to one shot only."""
    steps: list[np.ndarray] = []
    times: list[np.ndarray] = []
    pids: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    decisions: list[np.ndarray] = []
    problems: list[str] = []

    for path in files:
        try:
            with np.load(path, allow_pickle=False) as data:
                step = _first(data, ("global_step", "global_steps"))
                sim_time = _first(data, ("sim_time", "simulation_time"))
                pid = _first(data, ("particle_id", "particle_ids"))
                decision = _first(data, ("decision", "correct"))
                label = data["junction_ids"] if "junction_ids" in data.files else None
                if label is None:
                    junction_index = data["junction_id"] if "junction_id" in data.files else None
                    if junction_index is None:
                        raise KeyError("neither junction_ids nor junction_id is present")
                    label = junction_index
                    problems.append(f"{path}: junction_ids labels absent; using numeric junction indices")
                if step is None or pid is None or decision is None:
                    raise KeyError("missing global_step, particle_id, or decision array")
                if sim_time is None:
                    # Older logs may not store sim_time. Keep their events, but
                    # checkpoint-time filtering cannot be applied to this chunk.
                    sim_time = np.full(len(step), np.nan, dtype=np.float64)
                    problems.append(f"{path}: sim_time absent; checkpoint-time filtering unavailable for this chunk")
                if not (len(step) == len(sim_time) == len(pid) == len(decision) == len(label)):
                    raise ValueError("event array lengths differ")
                if len(step) == 0:
                    continue
                steps.append(np.asarray(step, dtype=np.int64))
                times.append(np.asarray(sim_time, dtype=np.float64))
                pids.append(np.asarray(pid, dtype=np.int64))
                labels.append(_as_str(np.asarray(label)))
                decisions.append(np.asarray(decision))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"{path}: could not be read ({exc})")

    if not steps:
        return {
            "step": np.empty(0, np.int64),
            "time": np.empty(0, np.float64),
            "pid": np.empty(0, np.int64),
            "jlab": np.empty(0, dtype="<U1"),
            "dec": np.empty(0, np.int8),
        }, problems

    events = {
        "step": np.concatenate(steps),
        "time": np.concatenate(times),
        "pid": np.concatenate(pids),
        "jlab": np.concatenate(labels),
        "dec": np.concatenate(decisions),
    }
    bad_decisions = ~np.isin(events["dec"], (0, 1))
    if bad_decisions.any():
        problems.append(f"{int(bad_decisions.sum())} events have decision values outside {{0,1}}")
    return events, problems


# ---------------------------------------------------------------------------
# Per-shot checkpoint / birth-time state
# ---------------------------------------------------------------------------
def _config_particle_count(params: dict) -> int:
    desired_time = params.get("desired_time", 200)
    if desired_time:
        return int(math.ceil(float(params.get("emission_rate", 0.5)) * float(desired_time)))
    return int(params.get("num_particles", 60))


def checkpoint_meta(data_dir: Path) -> dict[str, Any]:
    """Read small checkpoint arrays needed for birth/progress filtering."""
    result: dict[str, Any] = {
        "exists": False,
        "n": None,
        "simulation_time": None,
        "progress_time": None,
        "dt": None,
        "birth_times": None,
        "spawned": None,
        "exited": None,
    }
    path = data_dir / "ckpt_current.npz"
    if not path.exists():
        return result
    try:
        with np.load(path, allow_pickle=False) as checkpoint:
            result["exists"] = True
            if "num_particles" in checkpoint.files:
                result["n"] = int(checkpoint["num_particles"])
            sim_time = _first(checkpoint, ("simulation_time", "time"))
            if sim_time is not None:
                result["simulation_time"] = float(np.asarray(sim_time).reshape(-1)[0])
            if "dt" in checkpoint.files:
                result["dt"] = float(np.asarray(checkpoint["dt"]).reshape(-1)[0])
            if result["simulation_time"] is not None:
                # chemical_solver's last tracker update is one dt before the
                # wrapper's saved end_time (see maze_cluster_script.py).
                dt = result["dt"] or 0.0
                result["progress_time"] = result["simulation_time"] - dt
            if "birth_times" in checkpoint.files:
                result["birth_times"] = np.asarray(checkpoint["birth_times"], dtype=np.float64)
            if "exit_trigger_time" in checkpoint.files:
                result["exited"] = np.isfinite(np.asarray(checkpoint["exit_trigger_time"], dtype=np.float64))
            elif "exit_times" in checkpoint.files:
                result["exited"] = np.isfinite(np.asarray(checkpoint["exit_times"], dtype=np.float64))
            active = np.asarray(checkpoint["active_mask"], dtype=bool) if "active_mask" in checkpoint.files else None
            dead = np.asarray(checkpoint["dead_tracker"], dtype=bool) if "dead_tracker" in checkpoint.files else None
            if active is not None and dead is not None and active.shape == dead.shape:
                result["spawned"] = active | dead
            elif "masks" in checkpoint.files:
                masks = np.asarray(checkpoint["masks"], dtype=bool)
                if masks.ndim == 2 and masks.shape[0] >= 2:
                    result["spawned"] = masks[0] | masks[1]
    except (OSError, ValueError, TypeError, IndexError):
        result["exists"] = False
    return result


def shot_particle_eligibility(spec: ShotSpec, meta: dict[str, Any]) -> tuple[np.ndarray | None, list[str]]:
    """Return particles known to have spawned by the checkpoint's last step.

    None means eligibility cannot be determined reliably (e.g. no checkpoint).
    """
    notes: list[str] = []
    if not meta["exists"] or meta["progress_time"] is None:
        notes.append("checkpoint/progress time unavailable; no missing-pair expectations inferred")
        return None, notes

    n = meta["n"] if meta["n"] is not None else _config_particle_count(spec.params)
    if meta["n"] is None:
        notes.append(f"checkpoint has no num_particles; using configuration-derived count {n}")
    if meta["n"] is not None and meta["n"] != _config_particle_count(spec.params):
        notes.append(
            f"checkpoint num_particles={meta['n']} differs from config-derived "
            f"{_config_particle_count(spec.params)}; using checkpoint count"
        )

    progress_time = float(meta["progress_time"])
    tolerance = max(float(meta["dt"] or 0.0) * 1e-6, 1e-12)
    birth_times = meta["birth_times"]
    born_by_progress = np.ones(n, dtype=bool)
    if birth_times is not None:
        if len(birth_times) < n:
            notes.append(f"checkpoint birth_times has {len(birth_times)} entries; expected {n}")
            born_by_progress[len(birth_times):] = False
            born_by_progress[:len(birth_times)] = birth_times[:n] <= progress_time + tolerance
        else:
            born_by_progress = birth_times[:n] <= progress_time + tolerance
    else:
        notes.append("checkpoint has no birth_times; eligibility inferred from spawned mask only")

    spawned = meta["spawned"]
    if spawned is not None and len(spawned) >= n:
        eligible = born_by_progress & spawned[:n]
    elif spawned is not None:
        eligible = np.zeros(n, dtype=bool)
        eligible[:len(spawned)] = born_by_progress[:len(spawned)] & spawned
        notes.append(f"checkpoint spawned mask has {len(spawned)} entries; expected {n}")
    else:
        eligible = born_by_progress
        notes.append("checkpoint has no spawned mask; using birth-time eligibility")

    not_ready = int(n - eligible.sum())
    if not_ready:
        notes.append(
            f"excluded {not_ready}/{n} particle slots not yet spawned by progress time "
            f"{progress_time:.6g}"
        )
    return eligible, notes


def event_progress_mask(events: dict[str, np.ndarray], meta: dict[str, Any], eligible: np.ndarray | None) -> np.ndarray:
    """Drop log rows beyond checkpoint progress or for not-yet-spawned slots."""
    keep = np.ones(len(events["pid"]), dtype=bool)
    if meta.get("progress_time") is not None:
        known_time = np.isfinite(events["time"])
        tolerance = max(float(meta.get("dt") or 0.0) * 1e-6, 1e-12)
        keep &= ~known_time | (events["time"] <= float(meta["progress_time"]) + tolerance)
        keep &= np.isin(events["dec"], (0, 1))
    if eligible is not None and len(events["pid"]):
        valid_pid = (events["pid"] >= 0) & (events["pid"] < len(eligible))
        allowed = np.zeros(len(events["pid"]), dtype=bool)
        allowed[valid_pid] = eligible[events["pid"][valid_pid]]
        keep &= allowed
    return keep


# ---------------------------------------------------------------------------
# Zone labels and optional expected-pair declarations
# ---------------------------------------------------------------------------
_ZONE_CACHE: dict[Path, list[str] | None] = {}


def resolve_zones_file(params: dict, root: Path) -> Path | None:
    zones_file = params.get("zones_file")
    if zones_file is None:
        try:
            with (root / "maze_startpoints_configuration.toml").open("rb") as fh:
                configurations = tomllib.load(fh)["configurations"]
            profile = "Testrun" if params.get("test_run") else (
                params.get("maze_config") or "Ran_maze_size_prop_to_droplet"
            )
            zones_file = configurations[profile].get("zones_file")
        except (OSError, KeyError, tomllib.TOMLDecodeError, TypeError):
            return None
    if zones_file is None:
        return None
    path = Path(zones_file).expanduser()
    return path if path.is_absolute() else root / path


def zone_labels(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    path = path.resolve()
    if path not in _ZONE_CACHE:
        try:
            with np.load(path, allow_pickle=False) as data:
                _ZONE_CACHE[path] = _as_str(data["junction_ids"]).tolist()
        except (OSError, KeyError, ValueError):
            _ZONE_CACHE[path] = None
    return _ZONE_CACHE[path]


def load_expected_pairs(path: Path) -> dict[str, set[tuple[int, str]]]:
    """CSV columns: particle_id,junction_id[,shot]; shot is shot_NNN or *."""
    pairs: dict[str, set[tuple[int, str]]] = defaultdict(set)
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            shot = (row.get("shot") or "*").strip() or "*"
            pairs[shot].add((int(row["particle_id"]), str(row["junction_id"]).strip()))
    return pairs


def expected_pairs_for_shot(spec: ShotSpec, pairs_by_shot: dict[str, set[tuple[int, str]]]):
    # Accept both the concise shot_000 form and the original script's
    # run_id/shot_000 form for backward compatibility.
    return (
        set(pairs_by_shot.get("*", set()))
        | set(pairs_by_shot.get(f"shot_{spec.shot:03d}", set()))
        | set(pairs_by_shot.get(spec.name, set()))
    )


# ---------------------------------------------------------------------------
# Audit and aggregation
# ---------------------------------------------------------------------------
def observed_pair_counts(events: dict[str, np.ndarray], keep: np.ndarray) -> Counter:
    return Counter(
        (int(pid), str(label))
        for pid, label in zip(events["pid"][keep], events["jlab"][keep])
    )


def audit_shot(
    spec: ShotSpec,
    events: dict[str, np.ndarray],
    event_keep: np.ndarray,
    labels_from_zones: list[str] | None,
    eligible: np.ndarray | None,
    explicit_pairs: set[tuple[int, str]] | None,
    audit_all_junctions: bool,
    writer: csv.writer,
) -> dict[str, Any]:
    """Audit only pairs whose expectation is actually defined.

    Without an explicit pair list or --audit-all-junctions, missing pairs are
    not assessable. This avoids silently assuming every particle visits every
    junction. Observed duplicate events are still reported.
    """
    counts = observed_pair_counts(events, event_keep)
    expected: set[tuple[int, str]] | None = None
    audit_basis = "observed-only"

    if explicit_pairs is not None:
        if eligible is None:
            # Without reliable checkpoint progress, avoid fabricating a
            # missing event; the pair's absence cannot be assessed.
            expected = None
            audit_basis = "explicit expected-pairs CSV, but checkpoint progress unavailable"
        else:
            expected = set()
            for pid, label in explicit_pairs:
                if 0 <= pid < len(eligible) and eligible[pid]:
                    expected.add((pid, label))
            audit_basis = "explicit expected-pairs CSV"
    elif audit_all_junctions:
        if eligible is not None and labels_from_zones is not None:
            expected = {
                (pid, label)
                for pid in np.flatnonzero(eligible)
                for label in labels_from_zones
            }
            audit_basis = "all eligible particles x every junction in zones file"
        else:
            audit_basis = "all-junction audit requested but checkpoint/zone labels unavailable"

    # No expected set: report only observed counts and duplicates, never zeroes.
    missing = duplicate = unexpected = 0
    if expected is None:
        for (pid, label), count in sorted(counts.items()):
            status = "DUPLICATE" if count > 1 else "OBSERVED"
            duplicate += int(count > 1)
            writer.writerow([spec.name, pid, label, count, "not assessed", status])
    else:
        all_pairs = expected | set(counts)
        for pid, label in sorted(all_pairs):
            count = counts.get((pid, label), 0)
            is_expected = (pid, label) in expected
            if is_expected:
                status = "OK" if count == 1 else ("MISSING" if count == 0 else "DUPLICATE")
                missing += int(count == 0)
                duplicate += int(count > 1)
                expected_count: int | str = 1
            else:
                status = "UNEXPECTED"
                unexpected += 1
                expected_count = 0
            writer.writerow([spec.name, pid, label, count, expected_count, status])

    return {
        "observed_pairs": len(counts),
        "observed_events": int(event_keep.sum()),
        "expected_pairs": None if expected is None else len(expected),
        "missing_pairs": missing if expected is not None else None,
        "duplicate_pairs": duplicate,
        "unexpected_pairs": unexpected,
        "audit_basis": audit_basis,
        "audit_assessable": expected is not None,
    }


def add_events_to_aggregate(
    events: dict[str, np.ndarray],
    keep: np.ndarray,
    sums: dict[tuple[int, str], float],
    sums_sq: dict[tuple[int, str], float],
    counts: Counter,
    label_order: dict[str, int],
) -> None:
    for pid, label, decision in zip(
        events["pid"][keep], events["jlab"][keep], events["dec"][keep]
    ):
        key = (int(pid), str(label))
        value = float(decision)
        sums[key] = sums.get(key, 0.0) + value
        sums_sq[key] = sums_sq.get(key, 0.0) + value * value
        counts[key] += 1
        label_order.setdefault(str(label), len(label_order))


def _sample_std_from_moments(total: float, total_sq: float, count: int) -> float:
    """Sample standard deviation of recorded decision values (ddof=1).

    This is computed from the pooled event-level first and second moments,
    not from the already-averaged shot means. For a single observation there
    is no empirical spread, so the displayed standard deviation is 0.
    """
    if count <= 1:
        return 0.0
    mean = total / count
    variance = (total_sq - count * mean * mean) / (count - 1)
    return math.sqrt(max(float(variance), 0.0))


def wilson_ci(successes: int, trials: int) -> tuple[float, float]:
    if not trials:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / trials
    d = 1 + z*z/trials
    c = (p + z*z/(2*trials)) / d
    m = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials**2)) / d
    return max(0.0, c-m), min(1.0, c+m)


def particle_success_fractions(
    sums: dict[tuple[int, str], float], counts: Counter
) -> dict[int, float]:
    """Overall success fraction per particle across all recorded events."""
    totals: dict[int, float] = defaultdict(float)
    totals_n: Counter = Counter()
    for (pid, _label), n in counts.items():
        totals[pid] += sums[(pid, _label)]
        totals_n[pid] += n
    return {
        pid: totals[pid] / totals_n[pid]
        for pid in sorted(totals_n)
        if totals_n[pid] > 0
    }


def write_average_csv(path, sums, sums_sq, counts, label_order):
    pids = sorted({pid for pid, _ in counts})
    labels = sorted(label_order, key=label_order.get)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "particle_id", "junction_id", "mean_decision",
            "std_decision_1sigma", "ci95_lower", "ci95_upper",
            "number_of_decisions",
        ])
        for pid in pids:
            for label in labels:
                key = (pid, label)
                n = counts.get(key, 0)
                if n:
                    total = sums[key]
                    mean = total / n
                    std = _sample_std_from_moments(total, sums_sq[key], n)
                    lo, hi = wilson_ci(int(round(total)), n)
                    writer.writerow([
                        pid, label, f"{mean:.8g}", f"{std:.8g}",
                        f"{lo:.8g}", f"{hi:.8g}", n,
                    ])


def write_particle_success_csv(path: Path, sums: dict, counts: Counter) -> None:
    success = particle_success_fractions(sums, counts)
    total_counts: Counter = Counter()
    for (pid, _label), count in counts.items():
        total_counts[pid] += count
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "particle_id", "average_success_fraction", "number_of_decisions"
        ])
        for pid, fraction in success.items():
            writer.writerow([pid, f"{fraction:.8g}", total_counts[pid]])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_particle_decisions(
    sums: dict[tuple[int, str], float],
    sums_sq: dict[tuple[int, str], float],
    counts: Counter,
    label_order: dict[str, int],
    title: str,
    subtitle: str,
    output_path: Path,
    n_shots: int,
    n_shots_with_events: int,
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    labels = sorted(label_order, key=label_order.get)
    pids = sorted({pid for pid, _ in counts})
    x = np.arange(len(labels))
    label_to_col = {label: i for i, label in enumerate(labels)}
    success_by_particle = particle_success_fractions(sums, counts)

    # A continuous heatmap palette maps lower particle IDs to the low end of
    # viridis and higher IDs to the high end; sorting above guarantees ascending
    # particle order regardless of which IDs are absent in the event logs.
    if len(pids) == 1:
        norm = Normalize(vmin=pids[0] - 0.5, vmax=pids[0] + 0.5)
    else:
        norm = Normalize(vmin=min(pids), vmax=max(pids))
    cmap = plt.get_cmap("viridis")

    ncol = max(1, math.ceil(len(pids) / 25))
    fig, ax = plt.subplots(figsize=(10 + 0.6 * ncol, 7))
    for pid in pids:
        y = np.full(len(labels), np.nan)
        yerr = np.full((2, len(labels)), np.nan)

        for label in labels:
            key = (pid, label)
            n = counts.get(key, 0)
            if n:
                mean = sums[key] / n
                lo, hi = wilson_ci(int(round(sums[key])), n)
                col = label_to_col[label]
                y[col] = mean
                yerr[:, col] = mean - lo, hi - mean

        color = cmap(norm(pid))
        fraction = success_by_particle[pid]
        ax.errorbar(
            x, y, yerr=yerr,
            marker="o", markersize=3, linewidth=1,
            elinewidth=0.8, capsize=2, capthick=0.8,
            color=color,
            label=f"{pid} (success={fraction:.1%})",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45 if len(labels) > 10 else 0,
        ha="right" if len(labels) > 10 else "center",
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Junction ID")
    ax.set_ylabel("Mean decision (fraction correct)")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)

    # Reserve the right side for the per-particle legend and the bottom for a
    # color scale showing the ascending particle-ID heatmap mapping.
    if pids:
        ax.legend(
            title="Particle ID (overall success fraction)",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            ncol=ncol,
            fontsize=6,
            title_fontsize=7,
            frameon=False,
            handlelength=1.5,
            columnspacing=0.8,
        )
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=ax, orientation="horizontal", pad=0.18, fraction=0.06, aspect=40
        )
        cbar.set_label("Particle ID — ascending heatmap color scale", fontsize=8)
        if len(pids) <= 12:
            cbar.set_ticks(pids)
        else:
            tick_indices = np.unique(np.linspace(0, len(pids) - 1, 7).round().astype(int))
            cbar.set_ticks([pids[i] for i in tick_indices])

    fig.text(
        0.01, 0.005,
        f"Pooled recorded events: {n_shots_with_events}/{n_shots} shots. "
        "Points = mean; error bars = 95% Wilson binomial confidence intervals.\n"
        "Legend success = correct decisions / all recorded decisions for that particle; "
        "gaps = no recorded event. Particle IDs are simulation slot indices.",
        fontsize=7, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.09, 0.78, 1))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def json_safe(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# One configuration: the isolation boundary
# ---------------------------------------------------------------------------
def evaluate_configuration(
    run_id: str,
    specs: list[ShotSpec],
    files_by_shot: dict[str, list[Path]],
    args: argparse.Namespace,
    root: Path,
    series_name: str,
    config_dir: Path,
    expected_pairs_by_shot: dict[str, set[tuple[int, str]]] | None,
) -> dict[str, Any]:
    if not specs:
        raise ValueError(f"No shots supplied for configuration {run_id}")
    # A hard guard against accidental cross-configuration pooling.
    if any(spec.run_id != run_id for spec in specs):
        raise ValueError(f"Internal grouping error: mixed run IDs in {run_id}")

    config_dir.mkdir(parents=True, exist_ok=True)
    grid = specs[0].grid
    params = specs[0].params
    grid_text = ", ".join(f"{key}={value}" for key, value in grid.items()) or "(no grid parameters)"
    notes: list[str] = []
    load_problems: list[str] = []
    total_events = 0
    shots_with_events = 0
    shots_without_files = 0
    shots_without_checkpoint = 0
    excluded_event_rows = 0
    spawned_total = 0
    eligible_total = 0
    exited_total = 0
    sums: dict[tuple[int, str], float] = {}
    sums_sq: dict[tuple[int, str], float] = {}
    counts: Counter = Counter()
    label_order: dict[str, int] = {}
    audit_totals = Counter()
    audit_assessable_shots = 0
    audit_bases = Counter()

    audit_path = config_dir / "decision_integrity_audit.csv"
    with audit_path.open("w", newline="") as audit_fh:
        writer = csv.writer(audit_fh)
        writer.writerow([
            "shot", "particle_id", "junction_id", "number_of_decisions",
            "expected_decisions", "status",
        ])

        for spec in specs:
            files = files_by_shot.get(spec.name, [])
            if not files:
                shots_without_files += 1
            events, problems = load_decision_data(files)
            load_problems.extend(f"{spec.name}: {problem}" for problem in problems)

            meta = checkpoint_meta(spec.data_dir)
            if not meta["exists"]:
                shots_without_checkpoint += 1
            eligible, eligibility_notes = shot_particle_eligibility(spec, meta)
            notes.extend(f"{spec.name}: {note}" for note in eligibility_notes)
            if eligible is not None:
                eligible_total += int(eligible.sum())
                if meta["spawned"] is not None:
                    spawned_total += int(np.asarray(meta["spawned"], dtype=bool).sum())
                if meta["exited"] is not None:
                    exited_total += int(np.asarray(meta["exited"], dtype=bool).sum())

            event_keep = event_progress_mask(events, meta, eligible)
            excluded_event_rows += int((~event_keep).sum())
            if event_keep.any():
                shots_with_events += 1
            total_events += int(event_keep.sum())

            zones_path = resolve_zones_file(spec.params, root)
            labels = zone_labels(zones_path)
            if labels is None:
                notes.append(f"{spec.name}: zone labels unavailable; only observed junctions can be analyzed")
            else:
                for label in labels:
                    label_order.setdefault(label, len(label_order))

            explicit_pairs = None
            if expected_pairs_by_shot is not None:
                explicit_pairs = expected_pairs_for_shot(spec, expected_pairs_by_shot)
                if not explicit_pairs:
                    notes.append(f"{spec.name}: no rows in expected-pairs CSV; missing-pair audit unavailable for this shot")
                    explicit_pairs = None

            shot_audit = audit_shot(
                spec=spec,
                events=events,
                event_keep=event_keep,
                labels_from_zones=labels,
                eligible=eligible,
                explicit_pairs=explicit_pairs,
                audit_all_junctions=args.audit_all_junctions,
                writer=writer,
            )
            audit_totals.update({
                "observed_pairs": shot_audit["observed_pairs"],
                "observed_events": shot_audit["observed_events"],
                "duplicate_pairs": shot_audit["duplicate_pairs"],
                "unexpected_pairs": shot_audit["unexpected_pairs"],
            })
            if shot_audit["audit_assessable"]:
                audit_assessable_shots += 1
                audit_totals["expected_pairs"] += int(shot_audit["expected_pairs"] or 0)
                audit_totals["missing_pairs"] += int(shot_audit["missing_pairs"] or 0)
            audit_bases[shot_audit["audit_basis"]] += 1

            add_events_to_aggregate(
                events, event_keep, sums, sums_sq, counts, label_order
            )

    avg_csv = config_dir / "particle_decision_averages.csv"
    write_average_csv(avg_csv, sums, sums_sq, counts, label_order)
    success_csv = config_dir / "particle_success_summary.csv"
    write_particle_success_csv(success_csv, sums, counts)

    if expected_pairs_by_shot is not None:
        requested_audit = "explicit expected-pairs CSV"
    elif args.audit_all_junctions:
        requested_audit = "all eligible particles x every zone junction (assumption enabled)"
    else:
        requested_audit = "observed-only; missing pairs not inferred"

    duplicate_pairs = int(audit_totals["duplicate_pairs"])
    unexpected_pairs = int(audit_totals["unexpected_pairs"])
    if duplicate_pairs or unexpected_pairs:
        status = "FAILED"
    elif requested_audit == "observed-only; missing pairs not inferred":
        status = "INCOMPLETE" if load_problems else "OBSERVED_ONLY"
    elif audit_assessable_shots < len(specs) or load_problems:
        status = "INCOMPLETE"
    else:
        status = "FAILED" if audit_totals["missing_pairs"] else "PASSED"

    if status == "FAILED":
        status_line = (
            f"Audit {status}: {duplicate_pairs} duplicate, {unexpected_pairs} unexpected, "
            f"{audit_totals['missing_pairs']} missing expected pairs"
        )
    elif requested_audit == "observed-only; missing pairs not inferred":
        status_line = "Observed-event plot; missing pairs not assessed without a route expectation"
    else:
        status_line = f"Audit {status}: {audit_assessable_shots}/{len(specs)} shots assessable"

    plot_path = config_dir / "particle_decision_average_vs_junction.png"
    if counts:
        plot_particle_decisions(
            sums=sums,
            sums_sq=sums_sq,
            counts=counts,
            label_order=label_order,
            title=f"{series_name} | {run_id} | {grid_text}",
            subtitle=status_line,
            output_path=plot_path,
            n_shots=len(specs),
            n_shots_with_events=shots_with_events,
            dpi=args.dpi,
        )
    else:
        plot_path = None
        notes.append("no decision events remained after per-shot progress filtering; no plot written")

    summary = {
        "series": series_name,
        "run_id": run_id,
        "config_index": specs[0].config_index,
        "grid": grid,
        "effective_parameters": params,
        "status": status,
        "shots": len(specs),
        "shots_with_decision_events": shots_with_events,
        "shots_without_decision_files": shots_without_files,
        "shots_without_checkpoints": shots_without_checkpoint,
        "eligible_spawned_particles_sum": eligible_total,
        "spawned_particles_sum": spawned_total,
        "exited_particles_sum": exited_total,
        "recorded_events_used": total_events,
        "event_rows_excluded_by_progress_or_invalidity": excluded_event_rows,
        "unique_particle_junction_pairs_observed": len(counts),
        "audit_expectation": requested_audit,
        "audit_assessable_shots": audit_assessable_shots,
        "expected_pairs": audit_totals.get("expected_pairs") if audit_assessable_shots else None,
        "missing_pairs": audit_totals.get("missing_pairs") if audit_assessable_shots else None,
        "duplicate_pairs": duplicate_pairs,
        "unexpected_pairs": unexpected_pairs,
        "audit_basis_by_shot": dict(audit_bases),
        "load_problems": len(load_problems),
        "notes": notes,
        "files": {
            "plot": str(plot_path) if plot_path else None,
            "averages_csv": str(avg_csv),
            "particle_success_csv": str(success_csv),
            "audit_csv": str(audit_path),
        },
    }
    (config_dir / "decision_audit_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )

    print(f"\n=== {run_id} | {grid_text} | {len(specs)} shots ===")
    print(f"Shots with events:                 {shots_with_events}/{len(specs)}")
    print(f"Shots without decision files:      {shots_without_files}")
    print(f"Shots without checkpoints:         {shots_without_checkpoint}")
    print(f"Eligible spawned particles:        {eligible_total} (summed over shots)")
    print(f"Recorded events used in mean:      {total_events}")
    print(f"Event rows excluded (progress/invalid): {excluded_event_rows}")
    print(f"Unique particle/junction pairs:    {len(counts)}")
    print(f"Audit expectation:                 {requested_audit}")
    if audit_assessable_shots:
        print(f"Expected pairs:                    {audit_totals['expected_pairs']}")
        print(f"Pairs with 0 decisions:             {audit_totals['missing_pairs']}")
    else:
        print("Pairs with 0 decisions:             NOT ASSESSED (no valid route expectation)")
    print(f"Pairs with multiple decisions:      {duplicate_pairs}")
    print(f"Unexpected pairs:                  {unexpected_pairs}")
    print(f"AUDIT STATUS:                      {status}")
    for problem in load_problems:
        print(f"WARNING (load): {problem}")
    for note in notes:
        print(f"NOTE: {note}")
    print(f"Audit CSV: {audit_path}")
    print(f"Averages CSV: {avg_csv}")
    print(f"Particle success CSV: {success_csv}")
    if plot_path:
        print(f"Plot: {plot_path}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze junction decisions independently for every parameter configuration."
    )
    parser.add_argument("--config", required=True, type=Path, help="experiment-series TOML")
    parser.add_argument("--root", type=Path, default=ROOT, help="project root for relative paths")
    parser.add_argument(
        "--run-id", nargs="+", metavar="RUN_ID",
        help="only analyze these configuration IDs, e.g. rate-sweep_c0003",
    )
    parser.add_argument("--regime", help="only analyze configurations from this regime")
    parser.add_argument(
        "--decisions-dir", type=Path,
        help="override the decision-log base directory (normally <output_root>/<series>)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help=f"analysis output base (default: <output_root>/<series>/{ANALYSIS_DIRNAME})",
    )
    parser.add_argument("--recursive", action="store_true", help="search shot output folders recursively")
    parser.add_argument(
        "--expected-pairs", type=Path,
        help="CSV particle_id,junction_id[,shot] defining pairs that should have a decision",
    )
    parser.add_argument(
        "--audit-all-junctions", action="store_true",
        help="opt in to expecting every eligible spawned particle at every junction in the zones file; "
             "this can overcount missing pairs when routes bypass junctions",
    )
    parser.add_argument("--strict", action="store_true", help="return exit status 1 if any configuration audit fails or is incomplete")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    try:
        config, config_path = load_series_config(args.config, root)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        parser.error(f"Cannot load experiment config: {exc}")
    series_name = config["series"]["name"]
    series_data, series_out = series_dirs(config, root)

    if args.decisions_dir:
        decisions_base = args.decisions_dir.expanduser()
        series_out = decisions_base if decisions_base.is_absolute() else (root / decisions_base)
    base_dir = args.output_dir.expanduser() if args.output_dir else (series_dirs(config, root)[1] / ANALYSIS_DIRNAME)
    if not base_dir.is_absolute():
        base_dir = root / base_dir

    selected_run_ids = set(args.run_id) if args.run_id else None
    specs = expand_shots(config, series_data, series_out, args.regime, selected_run_ids)
    if not specs:
        parser.error("No matching configurations/shots (check --run-id / --regime)")

    # Stable configuration ordering follows TOML regime/grid order, not filesystem order.
    groups: dict[str, list[ShotSpec]] = {}
    for spec in specs:
        groups.setdefault(spec.run_id, []).append(spec)

    files_by_shot, discovery_notes = find_shot_decision_files(specs, args.recursive)
    for note in discovery_notes:
        print(f"NOTE: {note}")
    expected_pairs_by_shot = load_expected_pairs(args.expected_pairs) if args.expected_pairs else None

    base_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for run_id, group in groups.items():
        # Each invocation receives only the shots from this run_id. Its sums,
        # labels, event counts, and plot are created fresh inside that call.
        summary = evaluate_configuration(
            run_id=run_id,
            specs=group,
            files_by_shot=files_by_shot,
            args=args,
            root=root,
            series_name=series_name,
            config_dir=base_dir / run_id,
            expected_pairs_by_shot=expected_pairs_by_shot,
        )
        summaries.append(summary)

    overview = base_dir / "configurations_overview.csv"
    with overview.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "run_id", "parameters", "status", "shots", "recorded_events_used",
            "unique_particle_junction_pairs", "expected_pairs", "missing_pairs",
            "duplicate_pairs", "unexpected_pairs",
        ])
        for summary in summaries:
            writer.writerow([
                summary["run_id"],
                json.dumps(json_safe(summary["grid"]), sort_keys=True),
                summary["status"],
                summary["shots"],
                summary["recorded_events_used"],
                summary["unique_particle_junction_pairs_observed"],
                summary["expected_pairs"],
                summary["missing_pairs"],
                summary["duplicate_pairs"],
                summary["unexpected_pairs"],
            ])

    print(f"\n{len(summaries)} configuration(s) evaluated independently.")
    print(f"Results: {base_dir}")
    print(f"Overview: {overview}")
    for summary in summaries:
        print(f"  {summary['run_id']}: {summary['status']}")

    not_clean = any(summary["status"] in {"FAILED", "INCOMPLETE"} for summary in summaries)
    return 1 if args.strict and not_clean else 0


if __name__ == "__main__":
    sys.exit(main())
