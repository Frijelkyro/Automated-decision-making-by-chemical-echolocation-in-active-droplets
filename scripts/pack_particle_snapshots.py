"""Pack particle NPZ snapshots into an analysis-focused, streaming HDF5 file."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import tempfile
import warnings
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


SNAPSHOT_PATTERN = re.compile(r"part_(\d+)\.npz$")
HDF5_FILENAME = "particles_analysis.h5"
HDF5_FORMAT = "particle-decision-analysis-v1"


def _read_snapshot(path: Path):
    match = SNAPSHOT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Not a part_<timestep>.npz snapshot: {path}")
    with np.load(path) as archive:
        required = {"x", "y", "simulation_time"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}")
        timestep = int(np.asarray(archive["timestep"]).item()) if "timestep" in archive else int(match.group(1))
        simulation_time = float(np.asarray(archive["simulation_time"]).item())
        x = np.asarray(archive["x"])
        y = np.asarray(archive["y"])
        if not math.isfinite(simulation_time):
            raise ValueError(f"Non-finite simulation_time in {path}")
        if x.ndim != 1 or y.shape != x.shape or x.size == 0:
            raise ValueError(f"x and y in {path} must be equal-length, nonempty 1D arrays")
        if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise ValueError(f"x and y in {path} must be numeric")
        if "particle_ids" in archive:
            particle_ids = np.asarray(archive["particle_ids"])
            if particle_ids.shape != x.shape or not np.issubdtype(particle_ids.dtype, np.integer):
                raise ValueError(f"particle_ids in {path} must be an integer vector matching x and y")
        else:
            particle_ids = np.arange(x.size, dtype=np.int64)
        if len(np.unique(particle_ids)) != len(particle_ids):
            raise ValueError(f"Particle IDs in {path} are not unique")
        return timestep, simulation_time, x, y, particle_ids


def _create_frame_dataset(hdf5_file, name, particle_count, dtype, compression):
    options = {"compression": compression} if compression else {}
    return hdf5_file.create_dataset(
        name,
        shape=(0, particle_count),
        maxshape=(None, particle_count),
        chunks=(1, particle_count),
        dtype=dtype,
        **options,
    )


def _copy_birth_state(data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for state_path in data_dir.glob("param*_state*.npz"):
        target = output_dir / state_path.name
        try:
            if not os.path.samefile(state_path, target):
                shutil.copy2(state_path, target)
        except FileNotFoundError:
            shutil.copy2(state_path, target)


def pack_shot(
    data_dir: str | Path,
    output_file: str | Path | None = None,
    overwrite: bool = False,
    compression: str | None = None,
) -> Path:
    """Stream one shot into HDF5; source NPZ files are never removed or modified."""
    source_dir = Path(data_dir)
    paths = sorted(
        (path for path in source_dir.glob("part_*.npz") if SNAPSHOT_PATTERN.fullmatch(path.name)),
        key=lambda path: int(SNAPSHOT_PATTERN.fullmatch(path.name).group(1)),  # type: ignore[union-attr]
    )
    if not paths:
        raise ValueError(f"No part_<timestep>.npz files found in {source_dir}")

    destination = Path(output_file) if output_file is not None else source_dir / HDF5_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; pass --overwrite to replace it")

    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent, delete=False
    )
    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()
    temporary_path.unlink()

    try:
        first = _read_snapshot(paths[0])
        particle_count = len(first[2])
        stable_particle_ids = first[4].copy()
        with h5py.File(temporary_path, "w") as hdf5_file:
            hdf5_file.attrs["format"] = HDF5_FORMAT
            hdf5_file.attrs["complete"] = False
            hdf5_file.attrs["source_directory"] = str(source_dir.resolve())
            x_dataset = _create_frame_dataset(hdf5_file, "x", particle_count, first[2].dtype, compression)
            y_dataset = _create_frame_dataset(hdf5_file, "y", particle_count, first[3].dtype, compression)
            hdf5_file.create_dataset("particle_ids", data=stable_particle_ids)
            timestep_dataset = hdf5_file.create_dataset(
                "timestep", shape=(0,), maxshape=(None,), chunks=(1024,), dtype=np.int64
            )
            time_dataset = hdf5_file.create_dataset(
                "simulation_time", shape=(0,), maxshape=(None,), chunks=(1024,), dtype=np.float64
            )

            for frame_index, path in enumerate(
                tqdm(paths, desc=f"Packing {source_dir.name}", unit="file", dynamic_ncols=True)
            ):
                snapshot = first if frame_index == 0 else _read_snapshot(path)
                timestep, simulation_time, x, y, particle_ids = snapshot
                if len(x) != particle_count or y.shape != x.shape or particle_ids.shape != x.shape:
                    raise ValueError(f"Particle count changed in {path}; cannot pack a rectangular HDF5 series")
                if x.dtype != x_dataset.dtype or y.dtype != y_dataset.dtype:
                    raise ValueError(f"Position dtype changed in {path}; refusing lossy conversion")

                for dataset in (x_dataset, y_dataset):
                    dataset.resize(frame_index + 1, axis=0)
                timestep_dataset.resize(frame_index + 1, axis=0)
                time_dataset.resize(frame_index + 1, axis=0)
                x_dataset[frame_index] = x
                y_dataset[frame_index] = y
                if not np.array_equal(particle_ids, stable_particle_ids):
                    raise ValueError(f"Particle ID order changed in {path}; refusing an ambiguous packed series")
                timestep_dataset[frame_index] = timestep
                time_dataset[frame_index] = simulation_time

            hdf5_file.attrs["complete"] = True
            hdf5_file.flush()
        if destination.exists() and not overwrite:
            raise FileExistsError(f"{destination} appeared while packing; leaving it untouched")
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination


def _shot_directories(data_root: Path) -> list[Path]:
    if any(SNAPSHOT_PATTERN.fullmatch(path.name) for path in data_root.glob("part_*.npz")):
        return [data_root]
    return sorted(
        path
        for path in data_root.glob("shot_*")
        if path.is_dir() and any(SNAPSHOT_PATTERN.fullmatch(item.name) for item in path.glob("part_*.npz"))
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="A shot directory or run directory")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional separate output root; each shot gets its own folder and copied birth-state NPZs",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing HDF5 outputs")
    parser.add_argument("--compression", choices=("lzf", "gzip"), help="Optional HDF5 compression (default: none)")
    args = parser.parse_args(argv)

    data_root = args.data_dir
    shots = _shot_directories(data_root)
    if not shots:
        parser.error(f"No particle snapshots found under {data_root}")
    output_root = args.output_root
    if output_root is not None and len(shots) > 1:
        output_root.mkdir(parents=True, exist_ok=True)

    tqdm.write(f"[init] Found {len(shots)} shot(s); source NPZ files will not be deleted.")
    for shot_dir in tqdm(shots, desc="Shots", unit="shot", dynamic_ncols=True):
        if output_root is None:
            output_dir = shot_dir
        elif len(shots) == 1:
            output_dir = output_root
        else:
            output_dir = output_root / shot_dir.name
        output_file = output_dir / HDF5_FILENAME
        try:
            result = pack_shot(
                shot_dir,
                output_file,
                overwrite=args.overwrite,
                compression=args.compression,
            )
            if output_dir.resolve() != shot_dir.resolve():
                _copy_birth_state(shot_dir, output_dir)
            tqdm.write(f"[done] {shot_dir.name}: wrote {result}")
        except (OSError, ValueError, KeyError) as error:
            tqdm.write(f"[skip] {shot_dir}: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())