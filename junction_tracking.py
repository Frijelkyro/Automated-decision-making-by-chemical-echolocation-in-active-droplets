import os
from pathlib import Path

import numpy as np


class JunctionTracker:
    """Vectorized junction gate tracking for active particles."""

    def __init__(self, zones_file, num_particles, nx, ny, dx=1.0):
        zones_file = Path(zones_file).expanduser()
        if not zones_file.is_absolute():
            raise ValueError(f"zones_file must be an absolute path: {zones_file}")

        with np.load(zones_file, allow_pickle=False) as zones:
            if "junction_ids" not in zones:
                raise ValueError("zones NPZ must contain junction_ids")

            junction_ids_array = zones["junction_ids"]
            if junction_ids_array.ndim != 1:
                raise ValueError("junction_ids must be a one-dimensional array")
            junction_ids = junction_ids_array.tolist()
            self.junction_ids = np.asarray(junction_ids)
            junction_count = len(junction_ids)

            before = np.zeros((junction_count, nx, ny), dtype=bool)
            correct = np.zeros((junction_count, nx, ny), dtype=bool)
            wrong = np.zeros((junction_count, nx, ny), dtype=bool)

            for junction_index, raw_junction_id in enumerate(junction_ids):
                if isinstance(raw_junction_id, bytes):
                    junction_id = raw_junction_id.decode("utf-8")
                else:
                    junction_id = str(raw_junction_id)
                suffixes = (
                    (before, "before"),
                    (correct, "correct"),
                    (wrong, "wrong"),
                )
                for target, suffix in suffixes:
                    key = f"{junction_id}_{suffix}"
                    if key not in zones:
                        raise ValueError(f"zones NPZ is missing {key}")
                    mask = np.asarray(zones[key], dtype=bool)
                    if mask.shape == (nx, ny):
                        target[junction_index] = mask
                    elif mask.shape == (ny, nx):
                        target[junction_index] = mask.T
                    else:
                        raise ValueError(
                            f"{key} has shape {mask.shape}; expected ({nx}, {ny}) "
                            f"or ({ny}, {nx})"
                        )

        self.before = before
        self.correct = correct
        self.wrong = wrong
        self.armed = np.zeros((int(num_particles), junction_count), dtype=bool)
        self.event_buffer = []
        self.dx = float(dx)
        self.flush_count = 0
        self.flushed_upto_step = -1

    @property
    def events(self):
        return self.event_buffer

    def update(self, global_step, sim_time, px, py, active_mask):
        active_particle_ids = np.flatnonzero(np.asarray(active_mask, dtype=bool))
        if active_particle_ids.size == 0:
            return

        px_array = np.asarray(px, dtype=np.int64)
        py_array = np.asarray(py, dtype=np.int64)
        particle_px = px_array[active_particle_ids]
        particle_py = py_array[active_particle_ids]

        nx, ny = self.before.shape[1:]
        valid = (
            (particle_px >= 0)
            & (particle_px < nx)
            & (particle_py >= 0)
            & (particle_py < ny)
        )
        if not np.any(valid):
            return

        valid_particle_ids = active_particle_ids[valid]
        valid_px = particle_px[valid]
        valid_py = particle_py[valid]

        armed = self.armed[valid_particle_ids].copy()
        armed |= self.before[:, valid_px, valid_py].T

        hit_correct = armed & self.correct[:, valid_px, valid_py].T
        hit_wrong = armed & self.wrong[:, valid_px, valid_py].T
        hit_any = hit_correct | hit_wrong

        correct_particles, correct_junctions = np.nonzero(hit_correct)
        wrong_particles, wrong_junctions = np.nonzero(hit_wrong)

        if correct_particles.size:
            for row, junction in zip(correct_particles, correct_junctions):
                particle_id = int(valid_particle_ids[row])
                self.event_buffer.append(
                    (int(global_step), float(sim_time), particle_id,
                     int(junction), 1)
                )

        if wrong_particles.size:
            for row, junction in zip(wrong_particles, wrong_junctions):
                particle_id = int(valid_particle_ids[row])
                self.event_buffer.append(
                    (int(global_step), float(sim_time), particle_id,
                     int(junction), 0)
                )

        armed[hit_any] = False
        self.armed[valid_particle_ids] = armed

    def flush(self, output_dir, name=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if name is None:
            name = f"decisions_{self.flush_count}"
            self.flush_count += 1
        elif isinstance(name, int):
            name = f"decisions_{name}"

        if self.event_buffer:
            event_array = np.asarray(self.event_buffer)
            global_step = event_array[:, 0].astype(np.int64)
            sim_time = event_array[:, 1].astype(np.float64)
            particle_id = event_array[:, 2].astype(np.int64)
            junction_id = event_array[:, 3].astype(np.int64)
            correct = event_array[:, 4].astype(np.int8)
        else:
            global_step = np.empty(0, dtype=np.int64)
            sim_time = np.empty(0, dtype=np.float64)
            particle_id = np.empty(0, dtype=np.int64)
            junction_id = np.empty(0, dtype=np.int64)
            correct = np.empty(0, dtype=np.int8)

        np.savez_compressed(
            output_dir / f"{name}.npz",
            global_step=global_step,
            global_steps=global_step,
            sim_time=sim_time,
            particle_id=particle_id,
            particle_ids=particle_id,
            junction_id=junction_id,
            junction_ids=self.junction_ids[junction_id],
            decision=correct,
            correct=correct,
            shape=np.asarray(self.before.shape[1:], dtype=np.int64),
        )
        if global_step.size:
            self.flushed_upto_step = max(
                int(self.flushed_upto_step),
                int(global_step.max()),
            )
        self.event_buffer.clear()
        return output_dir / f"{name}.npz"

    def get_state(self, checkpoint_step=None):
        return {
            "armed": self.armed.copy(),
            "event_buffer": list(self.event_buffer),
            "flush_count": int(self.flush_count),
            "flushed_upto_step": int(self.flushed_upto_step),
            "checkpoint_step": (
                None if checkpoint_step is None else int(checkpoint_step)
            ),
        }

    def restore_state(self, state):
        if not isinstance(state, dict):
            return
        armed = state.get("armed")
        if armed is not None:
            armed = np.asarray(armed, dtype=bool)
            if (
                armed.ndim == 2
                and armed.shape[1] == self.armed.shape[1]
                and armed.shape[0] <= self.armed.shape[0]
            ):
                self.armed[:armed.shape[0]] = armed
        events = state.get("event_buffer")
        if events is not None:
            self.event_buffer = list(events)
        if "flush_count" in state:
            self.flush_count = int(state["flush_count"])
        if "flushed_upto_step" in state:
            self.flushed_upto_step = int(state["flushed_upto_step"])
        checkpoint_step = state.get("checkpoint_step")
        if checkpoint_step is not None:
            checkpoint_step = int(checkpoint_step)
            self.event_buffer = [
                event for event in self.event_buffer
                if int(event[0]) <= checkpoint_step
            ]
            if self.flushed_upto_step > checkpoint_step:
                self.flushed_upto_step = checkpoint_step


def truncate_decision_log_files(output_dir, checkpoint_step):
    """Remove decisions that were not yet valid at a recovery checkpoint."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return

    checkpoint_step = int(checkpoint_step)
    for decision_file in sorted(output_dir.glob("decisions_*.npz")):
        try:
            with np.load(decision_file, allow_pickle=False) as data:
                if "global_step" in data:
                    global_step = data["global_step"].astype(np.int64)
                elif "global_steps" in data:
                    global_step = data["global_steps"].astype(np.int64)
                else:
                    continue

                if global_step.size == 0:
                    continue
                if int(global_step.min()) > checkpoint_step:
                    decision_file.unlink()
                    continue
                if int(global_step.max()) <= checkpoint_step:
                    continue

                keep = global_step <= checkpoint_step
                saved = {}
                for key in data.files:
                    array = data[key].copy()
                    if array.ndim > 0 and array.shape[0] == global_step.size:
                        array = array[keep]
                    saved[key] = array
        except (OSError, ValueError):
            continue

        temp = decision_file.with_name(
            f".{decision_file.name}.{os.getpid()}.tmp.npz"
        )
        try:
            np.savez_compressed(temp, **saved)
            temp.replace(decision_file)
        finally:
            if temp.exists():
                temp.unlink()
