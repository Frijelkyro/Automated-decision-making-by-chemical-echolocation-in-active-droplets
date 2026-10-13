"""Evaluate particle decisions at maze junctions from saved simulation snapshots."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import tomllib
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATTERN = re.compile(r"part_(\d+)\.npz$")
ZONE_SUFFIXES = ("before", "correct", "wrong", "turned_around")
PARTICLE_FIELDS = (
    "series_name",
    "regime_name",
    "config_index",
    "run_id",
    "shot",
    "particle_id",
    "birth_time",
    "junction_id",
    "successes",
    "failures",
    "pending_attempts",
    "no_attempt",
    "success_fraction",
)
JUNCTION_FIELDS = (
    "series_name",
    "regime_name",
    "config_index",
    "run_id",
    "shot",
    "junction_id",
    "particles",
    "successes",
    "failures",
    "pending_attempts",
    "particles_with_no_attempt",
    "success_fraction",
)
AGGREGATE_FIELDS = (
    "series_name",
    "regime_name",
    "config_index",
    "run_id",
    "junction_id",
    "birth_time",
    "bin_start",
    "bin_end",
    "averaging_method",
    "success_fraction",
    "pooled_count_success_fraction",
    "equal_per_shot_success_fraction",
    "success_fraction_std",
    "success_fraction_se",
    "particles",
    "completed_attempt_particles",
    "successes",
    "failures",
    "pending_attempts",
    "particles_with_no_attempt",
    "shots_with_completed_attempts",
)


@dataclass(frozen=True)
class ShotLocation:
    series_name: str
    regime_name: str
    config_index: int
    run_id: str
    shot: int
    data_dir: Path


@dataclass(frozen=True)
class ParticleSnapshot:
    path: Path
    timestep: int
    simulation_time: float
    particle_ids: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    forces: Mapping[str, np.ndarray]


def load_series_config(config_path: str | Path, project_root: str | Path = PROJECT_ROOT) -> dict:
    """Load TOML configuration from either a .toml or equivalent .txt file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(project_root) / path
    with path.open("rb") as config_file:
        config = tomllib.load(config_file)
    if "series" not in config or "regimes" not in config:
        raise ValueError(f"{path} must define [series] and at least one [[regimes]]")
    return config


def discover_shots(
    config: Mapping,
    project_root: str | Path = PROJECT_ROOT,
    regimes: Iterable[str] | None = None,
    configurations: Iterable[int] | None = None,
    shots: Iterable[int] | None = None,
) -> list[ShotLocation]:
    """Expand expected shot paths in the same order as scripts/make_jobs.py."""
    root = Path(project_root).resolve()
    series = config["series"]
    data_root = Path(series["data_root"])
    if not data_root.is_absolute():
        data_root = root / data_root

    regime_filter = set(regimes) if regimes is not None else None
    config_filter = set(configurations) if configurations is not None else None
    shot_filter = set(shots) if shots is not None else None
    found: list[ShotLocation] = []
    missing: list[Path] = []

    for regime in config["regimes"]:
        regime_name = regime["name"]
        if regime_filter is not None and regime_name not in regime_filter:
            continue
        grid = regime["grid"]
        keys = list(grid)
        for config_index, _ in enumerate(itertools.product(*(grid[key] for key in keys))):
            if config_filter is not None and config_index not in config_filter:
                continue
            run_id = f"{regime_name}_c{config_index:04d}"
            expected_shots = range(int(regime["shots"]))
            if shot_filter is not None:
                expected_shots = (shot for shot in expected_shots if shot in shot_filter)
            for shot in expected_shots:
                data_dir = data_root / series["name"] / run_id / f"shot_{shot:03d}"
                if data_dir.is_dir():
                    found.append(
                        ShotLocation(
                            str(series["name"]),
                            str(regime_name),
                            config_index,
                            run_id,
                            shot,
                            data_dir,
                        )
                    )
                else:
                    missing.append(data_dir)

    if missing:
        examples = ", ".join(str(path) for path in missing[:3])
        warnings.warn(
            f"Skipped {len(missing)} expected shot director{'y' if len(missing) == 1 else 'ies'} "
            f"that do not exist; first missing path(s): {examples}",
            RuntimeWarning,
        )
    if not found:
        warnings.warn("No shot directories matched the configuration and filters.", RuntimeWarning)
    return found


def _load_metadata_text(text: str, metadata_path: Path) -> dict:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as original_error:
        lines = text.splitlines()
        filtered = [line for line in lines if not line.lstrip().startswith("neighbors =")]
        if len(filtered) == len(lines):
            raise ValueError(f"Invalid TOML metadata in {metadata_path}: {original_error}") from original_error
        try:
            metadata = tomllib.loads("\n".join(filtered))
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"Invalid TOML metadata in {metadata_path}: {error}") from error
        warnings.warn(
            f"Ignored malformed unused 'neighbors' values in {metadata_path}; "
            "the maze generator serialized NumPy tuple representations instead of TOML.",
            RuntimeWarning,
        )
        return metadata


def load_maze_analysis(analysis_dir: str | Path) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Load maze grid metadata and named decision-zone masks."""
    directory = Path(analysis_dir)
    metadata_path = directory / "maze_metadata.toml"
    mask_path = directory / "zone_masks.npz"
    metadata = _load_metadata_text(metadata_path.read_text(encoding="utf-8"), metadata_path)
    try:
        shape = tuple(int(value) for value in metadata["grid_shape"])
        dx = float(metadata["dx"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{metadata_path} must contain dx and a two-value grid_shape") from error
    if len(shape) != 2 or min(shape) <= 0 or not math.isfinite(dx) or dx <= 0:
        raise ValueError(f"Invalid maze grid dimensions or dx in {metadata_path}")

    zones: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    with np.load(mask_path) as archive:
        for name in archive.files:
            for suffix in ZONE_SUFFIXES:
                ending = f"_{suffix}"
                if name.endswith(ending):
                    junction_id = name[: -len(ending)]
                    mask = np.asarray(archive[name], dtype=bool)
                    if mask.shape != shape:
                        raise ValueError(
                            f"Mask {name} in {mask_path} has shape {mask.shape}; metadata declares {shape}"
                        )
                    zones[junction_id][suffix] = mask
                    break

    if not zones:
        raise ValueError(f"No junction zone masks found in {mask_path}")
    for junction_id, masks in zones.items():
        missing = {"before", "correct", "wrong"} - masks.keys()
        if missing:
            raise ValueError(f"Junction {junction_id} is missing masks: {', '.join(sorted(missing))}")
        overlap = masks["correct"] & masks["wrong"]
        if overlap.any():
            warnings.warn(
                f"Junction {junction_id} has {int(overlap.sum())} cells shared by correct and wrong masks; "
                "visits to those cells will be treated as ambiguous and will not close an attempt.",
                RuntimeWarning,
            )
    metadata["grid_shape"] = shape
    metadata["dx"] = dx
    return metadata, dict(zones)


def map_sim_to_grid(point: tuple[float, float], shape: tuple[int, int], dx: float) -> tuple[int, int]:
    """Apply maze_skeleton_path.sim_to_grid's x-column and flipped-y-row transform."""
    x, y = point
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        raise ValueError(f"Non-finite simulation coordinate: {point}")
    col = int(np.rint(x / dx))
    row = shape[0] - 1 - int(np.rint(y / dx))
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        raise IndexError(f"Point {point} maps outside the maze grid {shape}")
    return row, col


def _particle_ids(archive, particle_count: int, path: Path) -> np.ndarray:
    id_key = next((key for key in ("particle_ids", "particle_id", "ids") if key in archive), None)
    if id_key is None:
        return np.arange(particle_count, dtype=np.int64)
    raw_ids = np.asarray(archive[id_key])
    if raw_ids.ndim != 1 or len(raw_ids) != particle_count:
        raise ValueError(f"Particle IDs in {path} must be a 1D array matching the positions")
    if not np.issubdtype(raw_ids.dtype, np.integer):
        if not np.issubdtype(raw_ids.dtype, np.number) or not np.all(np.isfinite(raw_ids)):
            raise ValueError(f"Particle IDs in {path} must be finite integers")
        if not np.all(raw_ids == np.rint(raw_ids)):
            raise ValueError(f"Particle IDs in {path} must be integers")
    ids = raw_ids.astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"Particle IDs in {path} are not unique")
    return ids


def load_particle_snapshot(path: str | Path, include_forces: bool = True) -> ParticleSnapshot:
    """Load one particle snapshot, including optional force arrays."""
    snapshot_path = Path(path)
    match = SNAPSHOT_PATTERN.fullmatch(snapshot_path.name)
    if match is None:
        raise ValueError(f"Not a part_<timestep>.npz snapshot: {snapshot_path}")
    with np.load(snapshot_path) as archive:
        required = {"x", "y", "vx", "vy", "timestep", "simulation_time"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{snapshot_path} is missing required fields: {', '.join(sorted(missing))}")
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
        vx = np.asarray(archive["vx"], dtype=float)
        vy = np.asarray(archive["vy"], dtype=float)
        if x.ndim != 1 or y.shape != x.shape or vx.shape != x.shape or vy.shape != x.shape:
            raise ValueError(f"Position and velocity arrays in {snapshot_path} must be equal-length 1D arrays")
        timestep = int(np.asarray(archive["timestep"]).item())
        simulation_time = float(np.asarray(archive["simulation_time"]).item())
        if not math.isfinite(simulation_time):
            raise ValueError(f"Non-finite simulation_time in {snapshot_path}")
        particle_ids = _particle_ids(archive, len(x), snapshot_path)
        forces = (
            {key: np.asarray(archive[key]) for key in archive.files if key.startswith("f_")}
            if include_forces
            else {}
        )
    return ParticleSnapshot(
        snapshot_path,
        timestep,
        simulation_time,
        particle_ids,
        np.column_stack((x, y)),
        np.column_stack((vx, vy)),
        forces,
    )


def iter_particle_snapshots(
    data_dir: str | Path, include_forces: bool = False
) -> Iterable[ParticleSnapshot]:
    """Yield snapshots in stored-time order while keeping only one payload in memory."""
    directory = Path(data_dir)
    paths = sorted(
        (path for path in directory.glob("part_*.npz") if SNAPSHOT_PATTERN.fullmatch(path.name)),
        key=lambda path: int(SNAPSHOT_PATTERN.fullmatch(path.name).group(1)),  # type: ignore[union-attr]
    )
    ordered_paths = []
    for path in paths:
        try:
            with np.load(path) as archive:
                required = {"timestep", "simulation_time"}
                missing = required - set(archive.files)
                if missing:
                    raise ValueError(f"missing required fields: {', '.join(sorted(missing))}")
                timestep = int(np.asarray(archive["timestep"]).item())
                simulation_time = float(np.asarray(archive["simulation_time"]).item())
                if not math.isfinite(simulation_time):
                    raise ValueError("non-finite simulation_time")
                ordered_paths.append((simulation_time, timestep, path))
        except (OSError, ValueError, KeyError) as error:
            warnings.warn(f"Skipping incomplete particle snapshot {path}: {error}", RuntimeWarning)
    ordered_paths.sort(key=lambda item: (item[0], item[1]))
    if not ordered_paths:
        raise ValueError(f"No readable part_<timestep>.npz files found in {directory}")
    if len(ordered_paths) == 1:
        warnings.warn(
            f"Only one particle snapshot is available in {directory}; decision transitions may be incomplete.",
            RuntimeWarning,
        )
    for _, _, path in ordered_paths:
        try:
            yield load_particle_snapshot(path, include_forces=include_forces)
        except (OSError, ValueError, KeyError) as error:
            warnings.warn(f"Skipping incomplete particle snapshot {path}: {error}", RuntimeWarning)


def load_particle_snapshots(data_dir: str | Path, include_forces: bool = True) -> list[ParticleSnapshot]:
    """Compatibility helper that materializes snapshots; use the iterator for large shots."""
    return list(iter_particle_snapshots(data_dir, include_forces=include_forces))


def load_birth_times(data_dir: str | Path, particle_ids: Iterable[int]) -> dict[int, float]:
    """Read birth times from the newest simulation state, never from simulation_time."""
    directory = Path(data_dir)
    ids = {int(particle_id) for particle_id in particle_ids}
    state_paths = list(directory.glob("param*_state*.npz"))
    candidates = []
    for path in state_paths:
        try:
            with np.load(path) as state:
                if "birth_times" not in state:
                    continue
                simulation_time = float(np.asarray(state["simulation_time"]).item()) if "simulation_time" in state else -math.inf
                resume_step = int(np.asarray(state["resume_step"]).item()) if "resume_step" in state else -1
                birth_times = np.asarray(state["birth_times"], dtype=float).copy()
                if birth_times.ndim != 1:
                    raise ValueError("birth_times must be a 1D array")
                candidates.append((simulation_time, resume_step, not path.name.endswith("_checkpoint.npz"), path, birth_times))
        except (OSError, ValueError, KeyError) as error:
            warnings.warn(f"Ignoring unusable state file {path}: {error}", RuntimeWarning)
    if not candidates:
        warnings.warn(
            f"No readable param*_state*.npz with birth_times was found in {directory}; "
            "birth-time outputs will be NaN.",
            RuntimeWarning,
        )
        return {particle_id: math.nan for particle_id in ids}

    _, _, _, state_path, birth_times = max(candidates, key=lambda item: item[:3])
    missing = sorted(particle_id for particle_id in ids if particle_id < 0 or particle_id >= len(birth_times))
    if missing:
        warnings.warn(
            f"State {state_path} has birth times for {len(birth_times)} array indices, but "
            f"{len(missing)} snapshot particle ID(s) are outside that range; those birth times will be NaN.",
            RuntimeWarning,
        )
    return {
        particle_id: float(birth_times[particle_id]) if 0 <= particle_id < len(birth_times) else math.nan
        for particle_id in ids
    }


class DecisionAccumulator:
    """Incremental per-particle state for processing one snapshot at a time."""

    def __init__(
        self,
        zones: Mapping[str, Mapping[str, np.ndarray]],
        grid_shape: tuple[int, int],
        dx: float,
        birth_times: Mapping[int, float] | None = None,
        identifiers: Mapping[str, object] | None = None,
    ) -> None:
        self.zones = zones
        self.grid_shape = grid_shape
        self.dx = dx
        self.birth_times = birth_times or {}
        self.identifiers = identifiers or {}
        self.particle_ids: list[int] = []
        self.id_to_slot: dict[int, int] = {}
        self.state = {
            junction_id: {
                "successes": np.zeros(0, dtype=np.int64),
                "failures": np.zeros(0, dtype=np.int64),
                "attempt_open": np.zeros(0, dtype=bool),
                "previous_before": np.zeros(0, dtype=bool),
                "previous_correct": np.zeros(0, dtype=bool),
                "previous_wrong": np.zeros(0, dtype=bool),
            }
            for junction_id in zones
        }

    def _ensure_particle_ids(self, particle_ids: np.ndarray) -> None:
        new_ids = sorted(set(int(value) for value in particle_ids) - self.id_to_slot.keys())
        if not new_ids:
            return
        old_count = len(self.particle_ids)
        self.particle_ids.extend(new_ids)
        self.id_to_slot.update({particle_id: old_count + offset for offset, particle_id in enumerate(new_ids)})
        new_count = len(new_ids)
        for junction_state in self.state.values():
            for name, values in junction_state.items():
                extension = np.zeros(new_count, dtype=values.dtype)
                junction_state[name] = np.concatenate((values, extension))

    def process(self, snapshot: ParticleSnapshot) -> None:
        self._ensure_particle_ids(snapshot.particle_ids)
        particle_slots = np.fromiter(
            (self.id_to_slot[int(particle_id)] for particle_id in snapshot.particle_ids),
            dtype=np.int64,
            count=len(snapshot.particle_ids),
        )
        positions = snapshot.positions
        finite = np.isfinite(positions).all(axis=1)
        rows = np.full(len(positions), -1, dtype=np.int64)
        cols = np.full(len(positions), -1, dtype=np.int64)
        finite_indices = np.flatnonzero(finite)
        if len(finite_indices):
            cols[finite_indices] = np.rint(positions[finite_indices, 0] / self.dx).astype(np.int64)
            rows[finite_indices] = self.grid_shape[0] - 1 - np.rint(
                positions[finite_indices, 1] / self.dx
            ).astype(np.int64)
        valid = (
            finite
            & (rows >= 0)
            & (rows < self.grid_shape[0])
            & (cols >= 0)
            & (cols < self.grid_shape[1])
        )
        valid_slots = particle_slots[valid]

        particle_count = len(self.particle_ids)
        for junction_id, masks in self.zones.items():
            junction_state = self.state[junction_id]
            current_before = np.zeros(particle_count, dtype=bool)
            current_correct = np.zeros(particle_count, dtype=bool)
            current_wrong = np.zeros(particle_count, dtype=bool)
            if valid.any():
                valid_rows = rows[valid]
                valid_cols = cols[valid]
                current_before[valid_slots] = masks["before"][valid_rows, valid_cols]
                current_correct[valid_slots] = masks["correct"][valid_rows, valid_cols]
                current_wrong[valid_slots] = masks["wrong"][valid_rows, valid_cols]

            entered_before = current_before & ~junction_state["previous_before"]
            entered_correct = current_correct & ~junction_state["previous_correct"]
            entered_wrong = current_wrong & ~junction_state["previous_wrong"]
            ambiguous = current_correct & current_wrong
            success_events = (
                junction_state["attempt_open"]
                & entered_correct
                & ~entered_wrong
                & ~ambiguous
            )
            failure_events = (
                junction_state["attempt_open"]
                & entered_wrong
                & ~entered_correct
                & ~ambiguous
            )
            completed = success_events | failure_events
            terminal_entry = entered_correct | entered_wrong
            opened = (
                ~junction_state["attempt_open"]
                & entered_before
                & ~terminal_entry
                & ~ambiguous
            )
            junction_state["successes"] += success_events
            junction_state["failures"] += failure_events
            junction_state["attempt_open"] = (
                (junction_state["attempt_open"] & ~completed) | opened
            )
            junction_state["previous_before"] = current_before
            junction_state["previous_correct"] = current_correct
            junction_state["previous_wrong"] = current_wrong

    def rows(self) -> list[dict]:
        results = []
        for junction_id, junction_state in self.state.items():
            for slot, particle_id in sorted(enumerate(self.particle_ids), key=lambda item: item[1]):
                successes = int(junction_state["successes"][slot])
                failures = int(junction_state["failures"][slot])
                pending = int(junction_state["attempt_open"][slot])
                completed = successes + failures
                results.append(
                    {
                        **self.identifiers,
                        "particle_id": particle_id,
                        "birth_time": float(self.birth_times.get(particle_id, math.nan)),
                        "junction_id": junction_id,
                        "successes": successes,
                        "failures": failures,
                        "pending_attempts": pending,
                        "no_attempt": int(completed == 0 and pending == 0),
                        "success_fraction": successes / completed if completed else math.nan,
                    }
                )
        return results


def evaluate_decisions(
    snapshots: Iterable[ParticleSnapshot],
    zones: Mapping[str, Mapping[str, np.ndarray]],
    grid_shape: tuple[int, int],
    dx: float,
    birth_times: Mapping[int, float] | None = None,
    identifiers: Mapping[str, object] | None = None,
) -> list[dict]:
    """Evaluate edge-triggered junction attempts for every observed particle."""
    accumulator = DecisionAccumulator(zones, grid_shape, dx, birth_times, identifiers)
    for snapshot in snapshots:
        accumulator.process(snapshot)
    return accumulator.rows()


def evaluate_shot_decisions(
    data_dir: str | Path,
    zones: Mapping[str, Mapping[str, np.ndarray]],
    grid_shape: tuple[int, int],
    dx: float,
    identifiers: Mapping[str, object] | None = None,
) -> list[dict]:
    """Stream a shot in filename order when verified, with a sorted fallback."""
    directory = Path(data_dir)
    paths = sorted(
        (path for path in directory.glob("part_*.npz") if SNAPSHOT_PATTERN.fullmatch(path.name)),
        key=lambda path: int(SNAPSHOT_PATTERN.fullmatch(path.name).group(1)),  # type: ignore[union-attr]
    )
    if not paths:
        raise ValueError(f"No part_<timestep>.npz files found in {directory}")

    accumulator = None
    birth_times: dict[int, float] = {}
    previous_key = None
    needs_sorted_pass = False
    for path in paths:
        try:
            snapshot = load_particle_snapshot(path, include_forces=False)
        except (OSError, ValueError, KeyError) as error:
            warnings.warn(f"Skipping incomplete particle snapshot {path}: {error}", RuntimeWarning)
            continue
        sort_key = (snapshot.simulation_time, snapshot.timestep)
        if previous_key is not None and sort_key < previous_key:
            needs_sorted_pass = True
            break
        if accumulator is None:
            birth_times = load_birth_times(directory, snapshot.particle_ids)
            accumulator = DecisionAccumulator(zones, grid_shape, dx, birth_times, identifiers)
        accumulator.process(snapshot)
        previous_key = sort_key

    if accumulator is None:
        raise ValueError(f"No readable part_<timestep>.npz files found in {directory}")
    if needs_sorted_pass:
        warnings.warn(
            f"Stored physical times in {directory} are not ordered by timestep; restarting this shot "
            "with a metadata-sorted streaming pass.",
            RuntimeWarning,
        )
        accumulator = DecisionAccumulator(zones, grid_shape, dx, birth_times, identifiers)
        for snapshot in iter_particle_snapshots(directory, include_forces=False):
            accumulator.process(snapshot)
    return accumulator.rows()


def summarize_junctions(particle_rows: Sequence[Mapping]) -> list[dict]:
    """Roll per-particle decisions up to one row per shot and junction."""
    grouped: dict[tuple, list[Mapping]] = defaultdict(list)
    keys = ("series_name", "regime_name", "config_index", "run_id", "shot", "junction_id")
    for row in particle_rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    summaries = []
    for group_key, rows in grouped.items():
        successes = sum(int(row["successes"]) for row in rows)
        failures = sum(int(row["failures"]) for row in rows)
        pending = sum(int(row["pending_attempts"]) for row in rows)
        summaries.append(
            {
                **dict(zip(keys, group_key)),
                "particles": len(rows),
                "successes": successes,
                "failures": failures,
                "pending_attempts": pending,
                "particles_with_no_attempt": sum(int(row["no_attempt"]) for row in rows),
                "success_fraction": successes / (successes + failures) if successes + failures else math.nan,
            }
        )
    return summaries


def _birth_bins(
    rows: Sequence[Mapping], birth_bin_width: float | None, birth_time_tolerance: float
) -> list[tuple[float, float, float, list[Mapping]]]:
    valid_rows = [row for row in rows if math.isfinite(float(row["birth_time"]))]
    if birth_bin_width is not None:
        if not math.isfinite(birth_bin_width) or birth_bin_width <= 0:
            raise ValueError("birth_bin_width must be a positive finite number")
        bins: dict[int, list[Mapping]] = defaultdict(list)
        for row in valid_rows:
            bins[math.floor(float(row["birth_time"]) / birth_bin_width)].append(row)
        return [
            (index * birth_bin_width + birth_bin_width / 2, index * birth_bin_width,
             (index + 1) * birth_bin_width, members)
            for index, members in sorted(bins.items())
        ]

    if not math.isfinite(birth_time_tolerance) or birth_time_tolerance < 0:
        raise ValueError("birth_time_tolerance must be a non-negative finite number")
    ordered = sorted(valid_rows, key=lambda row: float(row["birth_time"]))
    clusters: list[list[Mapping]] = []
    for row in ordered:
        birth_time = float(row["birth_time"])
        if not clusters or birth_time - float(clusters[-1][0]["birth_time"]) > birth_time_tolerance:
            clusters.append([row])
        else:
            clusters[-1].append(row)
    result = []
    for members in clusters:
        times = [float(row["birth_time"]) for row in members]
        result.append((float(np.mean(times)), min(times), max(times), members))
    return result


def aggregate_birth_times(
    particle_rows: Sequence[Mapping],
    averaging: str = "pooled-count",
    birth_bin_width: float | None = None,
    birth_time_tolerance: float = 0.0,
) -> list[dict]:
    """Average success fraction by birth time, keeping each parameter run separate."""
    if averaging not in {"pooled-count", "equal-per-shot"}:
        raise ValueError("averaging must be 'pooled-count' or 'equal-per-shot'")
    grouped: dict[tuple, list[Mapping]] = defaultdict(list)
    keys = ("series_name", "regime_name", "config_index", "run_id", "junction_id")
    for row in particle_rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)

    summaries = []
    for group_key, rows in grouped.items():
        for birth_time, bin_start, bin_end, members in _birth_bins(rows, birth_bin_width, birth_time_tolerance):
            by_shot: dict[object, list[Mapping]] = defaultdict(list)
            for row in members:
                by_shot[row.get("shot")].append(row)
            shot_fractions = []
            successes = failures = pending = no_attempt = completed_particles = 0
            for shot_rows in by_shot.values():
                shot_successes = sum(int(row["successes"]) for row in shot_rows)
                shot_failures = sum(int(row["failures"]) for row in shot_rows)
                denominator = shot_successes + shot_failures
                successes += shot_successes
                failures += shot_failures
                pending += sum(int(row["pending_attempts"]) for row in shot_rows)
                no_attempt += sum(int(row["no_attempt"]) for row in shot_rows)
                completed_particles += sum(
                    int(int(row["successes"]) + int(row["failures"]) > 0) for row in shot_rows
                )
                if denominator:
                    shot_fractions.append(shot_successes / denominator)

            pooled_fraction = successes / (successes + failures) if successes + failures else math.nan
            equal_fraction = float(np.mean(shot_fractions)) if shot_fractions else math.nan
            if len(shot_fractions) > 1:
                standard_deviation = float(np.std(shot_fractions, ddof=1))
            else:
                standard_deviation = 0.0 if shot_fractions else math.nan
            standard_error = (
                standard_deviation / math.sqrt(len(shot_fractions))
                if shot_fractions
                else math.nan
            )
            chosen_fraction = pooled_fraction if averaging == "pooled-count" else equal_fraction
            summaries.append(
                {
                    **dict(zip(keys, group_key)),
                    "birth_time": birth_time,
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    "averaging_method": averaging,
                    "success_fraction": chosen_fraction,
                    "pooled_count_success_fraction": pooled_fraction,
                    "equal_per_shot_success_fraction": equal_fraction,
                    "success_fraction_std": standard_deviation,
                    "success_fraction_se": standard_error,
                    "particles": len(members),
                    "completed_attempt_particles": completed_particles,
                    "successes": successes,
                    "failures": failures,
                    "pending_attempts": pending,
                    "particles_with_no_attempt": no_attempt,
                    "shots_with_completed_attempts": len(shot_fractions),
                }
            )
    return summaries


class BirthTimeAggregator:
    """Merge one-shot summaries into compact pooled and equal-shot statistics."""

    GROUP_KEYS = ("series_name", "regime_name", "config_index", "run_id", "junction_id")

    def __init__(
        self,
        averaging: str,
        birth_bin_width: float | None,
        birth_time_tolerance: float,
    ) -> None:
        if averaging not in {"pooled-count", "equal-per-shot"}:
            raise ValueError("averaging must be 'pooled-count' or 'equal-per-shot'")
        self.averaging = averaging
        self.birth_bin_width = birth_bin_width
        self.birth_time_tolerance = birth_time_tolerance
        self.groups: dict[tuple, list[dict]] = defaultdict(list)

    def add_shot(self, particle_rows: Sequence[Mapping]) -> None:
        shot_summaries = aggregate_birth_times(
            particle_rows,
            averaging="pooled-count",
            birth_bin_width=self.birth_bin_width,
            birth_time_tolerance=self.birth_time_tolerance,
        )
        for summary in shot_summaries:
            group_key = tuple(summary[key] for key in self.GROUP_KEYS)
            bins = self.groups[group_key]
            bin_id = None
            if self.birth_bin_width is not None:
                bin_id = math.floor(float(summary["birth_time"]) / self.birth_bin_width)
                bucket = next((item for item in bins if item["bin_id"] == bin_id), None)
            else:
                candidates = [
                    item
                    for item in bins
                    if abs(
                        float(summary["birth_time"])
                        - item["birth_time"] / item["birth_time_weight"]
                    )
                    <= self.birth_time_tolerance
                ]
                bucket = (
                    min(
                        candidates,
                        key=lambda item: abs(
                            float(summary["birth_time"])
                            - item["birth_time"] / item["birth_time_weight"]
                        ),
                    )
                    if candidates
                    else None
                )
            if bucket is None:
                bucket = {
                    "bin_id": bin_id if self.birth_bin_width is not None else None,
                    "birth_time": 0.0,
                    "birth_time_weight": 0,
                    "bin_start": float(summary["bin_start"]),
                    "bin_end": float(summary["bin_end"]),
                    "particles": 0,
                    "completed_attempt_particles": 0,
                    "successes": 0,
                    "failures": 0,
                    "pending_attempts": 0,
                    "particles_with_no_attempt": 0,
                    "shots_with_completed_attempts": 0,
                    "shot_fraction_mean": 0.0,
                    "shot_fraction_m2": 0.0,
                }
                bins.append(bucket)

            particle_count = int(summary["particles"])
            bucket["birth_time"] += float(summary["birth_time"]) * particle_count
            bucket["birth_time_weight"] += particle_count
            bucket["bin_start"] = min(bucket["bin_start"], float(summary["bin_start"]))
            bucket["bin_end"] = max(bucket["bin_end"], float(summary["bin_end"]))
            for field in (
                "particles",
                "completed_attempt_particles",
                "successes",
                "failures",
                "pending_attempts",
                "particles_with_no_attempt",
            ):
                bucket[field] += int(summary[field])

            shot_denominator = int(summary["successes"]) + int(summary["failures"])
            if shot_denominator:
                shot_fraction = int(summary["successes"]) / shot_denominator
                count = bucket["shots_with_completed_attempts"] + 1
                delta = shot_fraction - bucket["shot_fraction_mean"]
                bucket["shot_fraction_mean"] += delta / count
                bucket["shot_fraction_m2"] += delta * (shot_fraction - bucket["shot_fraction_mean"])
                bucket["shots_with_completed_attempts"] = count

    def rows(self) -> list[dict]:
        summaries = []
        for group_key, bins in self.groups.items():
            for bucket in bins:
                successes = bucket["successes"]
                failures = bucket["failures"]
                pooled_fraction = successes / (successes + failures) if successes + failures else math.nan
                shot_count = bucket["shots_with_completed_attempts"]
                standard_deviation = (
                    math.sqrt(bucket["shot_fraction_m2"] / (shot_count - 1))
                    if shot_count > 1
                    else 0.0 if shot_count else math.nan
                )
                standard_error = standard_deviation / math.sqrt(shot_count) if shot_count else math.nan
                equal_fraction = bucket["shot_fraction_mean"] if shot_count else math.nan
                summaries.append(
                    {
                        **dict(zip(self.GROUP_KEYS, group_key)),
                        "birth_time": bucket["birth_time"] / bucket["birth_time_weight"],
                        "bin_start": bucket["bin_start"],
                        "bin_end": bucket["bin_end"],
                        "averaging_method": self.averaging,
                        "success_fraction": pooled_fraction if self.averaging == "pooled-count" else equal_fraction,
                        "pooled_count_success_fraction": pooled_fraction,
                        "equal_per_shot_success_fraction": equal_fraction,
                        "success_fraction_std": standard_deviation,
                        "success_fraction_se": standard_error,
                        **{field: bucket[field] for field in (
                            "particles",
                            "completed_attempt_particles",
                            "successes",
                            "failures",
                            "pending_attempts",
                            "particles_with_no_attempt",
                            "shots_with_completed_attempts",
                        )},
                    }
                )
        return sorted(
            summaries,
            key=lambda row: (row["regime_name"], row["config_index"], row["junction_id"], row["birth_time"]),
        )


def write_csv(path: str | Path, rows: Sequence[Mapping], fieldnames: Sequence[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_birth_time_summary(rows: Sequence[Mapping], output_path: str | Path) -> None:
    """Plot one panel per parameter configuration with standard-error bars."""
    import matplotlib.pyplot as plt

    configurations: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in rows:
        configurations[(row["regime_name"], row["run_id"])].append(row)
    if not configurations:
        warnings.warn("No finite birth-time groups are available to plot.", RuntimeWarning)
        return
    panels = list(configurations.items())
    figure, axes = plt.subplots(len(panels), 1, figsize=(9, max(4, 3.5 * len(panels))), squeeze=False)
    for axis, ((regime_name, run_id), config_rows) in zip(axes[:, 0], panels):
        junction_ids = sorted({row["junction_id"] for row in config_rows})
        for junction_id in junction_ids:
            junction_rows = sorted(
                (row for row in config_rows if row["junction_id"] == junction_id),
                key=lambda row: float(row["birth_time"]),
            )
            axis.errorbar(
                [float(row["birth_time"]) for row in junction_rows],
                [float(row["success_fraction"]) for row in junction_rows],
                yerr=[float(row["success_fraction_se"]) for row in junction_rows],
                marker="o",
                linewidth=1,
                markersize=3,
                capsize=2,
                label=junction_id,
            )
        axis.set(title=f"{regime_name}: {run_id}", xlabel="Particle birth time", ylabel="Success fraction")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.25)
        axis.legend(title="Junction", ncol=min(6, len(junction_ids)))
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _default_analysis_dir(project_root: Path) -> Path:
    candidates = sorted((project_root / "different_mazes").rglob("maze_metadata.toml"))
    if len(candidates) != 1:
        raise ValueError(
            "Specify --maze-analysis-dir; expected exactly one different_mazes/**/maze_metadata.toml, "
            f"found {len(candidates)}."
        )
    return candidates[0].parent


def run_analysis(args: argparse.Namespace) -> tuple[Path, int, int]:
    project_root = Path(args.project_root).resolve()
    config = load_series_config(args.config, project_root)
    shots = discover_shots(config, project_root, args.regime, args.configuration, args.shot)
    analysis_dir = Path(args.maze_analysis_dir) if args.maze_analysis_dir else _default_analysis_dir(project_root)
    if not analysis_dir.is_absolute():
        analysis_dir = project_root / analysis_dir
    metadata, zones = load_maze_analysis(analysis_dir)

    series = config["series"]
    output_dir = Path(args.output_dir) if args.output_dir else Path(series["output_root"]) / series["name"] / "decision_analysis"
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    particle_count = junction_count = 0
    missing_birth_time_count = 0
    aggregator = (
        BirthTimeAggregator(args.averaging, args.birth_bin_width, args.birth_time_tolerance)
        if args.aggregate_shots
        else None
    )
    particle_csv_path = output_dir / "particle_decisions.csv"
    junction_csv_path = output_dir / "junction_summary.csv"
    with particle_csv_path.open("w", newline="", encoding="utf-8") as particle_file, junction_csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as junction_file:
        particle_writer = csv.DictWriter(particle_file, fieldnames=PARTICLE_FIELDS, extrasaction="ignore")
        junction_writer = csv.DictWriter(junction_file, fieldnames=JUNCTION_FIELDS, extrasaction="ignore")
        particle_writer.writeheader()
        junction_writer.writeheader()

        for shot in shots:
            try:
                particle_rows = evaluate_shot_decisions(
                    shot.data_dir,
                    zones,
                    metadata["grid_shape"],
                    metadata["dx"],
                    {
                        "series_name": shot.series_name,
                        "regime_name": shot.regime_name,
                        "config_index": shot.config_index,
                        "run_id": shot.run_id,
                        "shot": shot.shot,
                    },
                )
                junction_rows = summarize_junctions(particle_rows)
                particle_writer.writerows(particle_rows)
                junction_writer.writerows(junction_rows)
                particle_file.flush()
                junction_file.flush()
                particle_count += len(particle_rows)
                junction_count += len(junction_rows)
                missing_birth_time_count += sum(
                    not math.isfinite(float(row["birth_time"])) for row in particle_rows
                )
                if aggregator is not None:
                    aggregator.add_shot(particle_rows)
            except (OSError, ValueError, KeyError, StopIteration) as error:
                warnings.warn(f"Skipped incomplete shot {shot.data_dir}: {error}", RuntimeWarning)

    if aggregator is not None:
        if missing_birth_time_count:
            warnings.warn(
                f"{missing_birth_time_count} particle-junction row(s) have no verified birth time "
                "and are excluded from birth-time averaging.",
                RuntimeWarning,
            )
        aggregate_rows = aggregator.rows()
        write_csv(output_dir / "birth_time_summary.csv", aggregate_rows, AGGREGATE_FIELDS)
        if args.plot:
            plot_birth_time_summary(aggregate_rows, output_dir / "success_fraction_vs_birth_time.png")
    elif args.plot:
        warnings.warn("--plot requires --aggregate-shots; no plot was written.", RuntimeWarning)
    return output_dir, particle_count, junction_count


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "experiments" / "simulation_series_parameters.toml")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--maze-analysis-dir", type=Path)
    parser.add_argument("--regime", action="append", help="Regime name to include; may be repeated")
    parser.add_argument("--configuration", type=int, action="append", help="Grid configuration index; may be repeated")
    parser.add_argument("--shot", type=int, action="append", help="Shot index; may be repeated")
    parser.add_argument("--aggregate-shots", action="store_true", help="Also average across selected shots")
    parser.add_argument("--averaging", choices=("pooled-count", "equal-per-shot"), default="pooled-count")
    parser.add_argument("--birth-bin-width", type=float, help="Fixed-width birth-time bins")
    parser.add_argument("--birth-time-tolerance", type=float, default=0.0, help="Cluster times within this tolerance when no fixed bin width is given")
    parser.add_argument("--plot", action="store_true", help="Plot the birth-time summary (requires --aggregate-shots)")
    parser.add_argument("--output-dir", type=Path, help="Override the configured output_root analysis directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        output_dir, particle_count, junction_count = run_analysis(args)
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error
    print(
        f"Wrote {particle_count} particle-junction rows and {junction_count} "
        f"shot-junction rows to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())