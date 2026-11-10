"""Safely replace raw per-snapshot NPZ files with full-data HDF5 shot archives."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import cast

import h5py
import numpy as np
from tqdm import tqdm


SHOT_PATTERN = re.compile(r"shot_\d+$")
PART_PATTERN = re.compile(r"part_(\d+)\.npz$")
CONC_PATTERN = re.compile(r"conc_(\d+)\.npz$")
ARCHIVE_FORMAT = "full-shot-hdf5-v1"
GIB = 1024**3


def _tree_files(directory: Path) -> list[Path]:
    files = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Refusing to compact a shot containing symlinks: {path}")
        if path.is_file():
            files.append(path)
    return files


def _tree_size(directory: Path) -> int:
    return sum(path.stat().st_size for path in _tree_files(directory))


def _copy_tree_with_progress(source: Path, destination: Path) -> None:
    files = _tree_files(source)
    destination.mkdir(parents=True, exist_ok=False)
    for path in tqdm(files, desc=f"Copying {source.name} to SSD", unit="file", dynamic_ncols=True):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if target.stat().st_size != path.stat().st_size:
            raise IOError(f"Copy size verification failed for {path}")
    for directory in sorted((item for item in source.rglob("*") if item.is_dir()), key=lambda item: len(item.parts)):
        (destination / directory.relative_to(source)).mkdir(parents=True, exist_ok=True)


def _verify_plain_tree_copy(source: Path, destination: Path) -> None:
    source_files = {path.relative_to(source): path.stat().st_size for path in _tree_files(source)}
    destination_files = {path.relative_to(destination): path.stat().st_size for path in _tree_files(destination)}
    if source_files != destination_files:
        raise IOError(f"Copied raw shot does not match its SSD staging copy: {destination}")


def _estimate_npz_payload_bytes(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            total += sum(member.file_size for member in archive.infolist() if not member.is_dir())
    return total


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    for key, array in arrays.items():
        if not key or "/" in key:
            raise ValueError(f"Unsupported HDF5 dataset name {key!r} in {path}")
        if array.dtype.kind not in "biufcS":
            raise ValueError(f"Unsupported nonnumeric/object array {key!r} ({array.dtype}) in {path}")
    return arrays


def _equal_arrays(first: np.ndarray, second: np.ndarray) -> bool:
    if first.dtype.kind in "fc":
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def _snapshot_paths(directory: Path, pattern: re.Pattern[str]) -> list[Path]:
    matches = []
    for path in directory.glob("*.npz"):
        match = pattern.fullmatch(path.name)
        if match:
            matches.append((int(match.group(1)), path))
    return [path for _, path in sorted(matches)]


def _pack_category(
    source_dir: Path,
    output_path: Path,
    pattern: re.Pattern[str],
    category: str,
    compression: str | None,
) -> int:
    paths = _snapshot_paths(source_dir, pattern)
    if not paths:
        raise ValueError(f"No {category} snapshots found in {source_dir}")

    first_arrays = _load_npz_arrays(paths[0])
    if category == "particles":
        for required in ("x", "y", "timestep", "simulation_time"):
            if required not in first_arrays:
                raise ValueError(f"First particle snapshot {paths[0]} is missing {required!r}")
        if "particle_ids" not in first_arrays:
            first_arrays["particle_ids"] = np.arange(len(first_arrays["x"]), dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    options = {"compression": compression} if compression else {}
    with h5py.File(output_path, "w") as hdf5_file:
        hdf5_file.attrs["format"] = ARCHIVE_FORMAT
        hdf5_file.attrs["category"] = category
        hdf5_file.attrs["complete"] = False
        hdf5_file.attrs["source_snapshot_count"] = len(paths)
        hdf5_file.create_dataset(
            "source_filename",
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype=h5py.string_dtype("utf-8"),
        )

        datasets = {}
        for key, array in first_arrays.items():
            frame_shape = array.shape
            frame_chunk = (1, *frame_shape) if frame_shape else (1,)
            datasets[key] = hdf5_file.create_dataset(
                key,
                shape=(0, *frame_shape) if frame_shape else (0,),
                maxshape=(None, *frame_shape) if frame_shape else (None,),
                chunks=frame_chunk,
                dtype=array.dtype,
                **options,
            )

        filenames = cast(h5py.Dataset, hdf5_file["source_filename"])
        for frame_index, path in enumerate(
            tqdm(paths, desc=f"Packing {category}", unit="file", dynamic_ncols=True)
        ):
            arrays = first_arrays if frame_index == 0 else _load_npz_arrays(path)
            if category == "particles" and "particle_ids" not in arrays:
                arrays["particle_ids"] = np.arange(len(arrays["x"]), dtype=np.int64)
            if arrays.keys() != first_arrays.keys():
                raise ValueError(f"NPZ members changed in {path}; refusing a partial/ambiguous archive")

            filenames.resize(frame_index + 1, axis=0)
            filenames[frame_index] = path.name
            for key, array in arrays.items():
                dataset = datasets[key]
                if array.shape != dataset.shape[1:] or array.dtype != dataset.dtype:
                    raise ValueError(
                        f"Array {key!r} changed shape or dtype in {path}: "
                        f"{array.shape}/{array.dtype} != {dataset.shape[1:]}/{dataset.dtype}"
                    )
                dataset.resize(frame_index + 1, axis=0)
                dataset[frame_index] = array
                if not _equal_arrays(array, np.asarray(dataset[frame_index])):
                    raise IOError(f"Read-back verification failed for {path}:{key}")

        hdf5_file.attrs["complete"] = True
        hdf5_file.flush()

    return len(paths)


def _copy_other_files(source_dir: Path, output_dir: Path) -> int:
    copied = 0
    for path in _tree_files(source_dir):
        if path.parent == source_dir and (PART_PATTERN.fullmatch(path.name) or CONC_PATTERN.fullmatch(path.name)):
            continue
        relative = path.relative_to(source_dir)
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    for directory in sorted((item for item in source_dir.rglob("*") if item.is_dir()), key=lambda item: len(item.parts)):
        (output_dir / directory.relative_to(source_dir)).mkdir(parents=True, exist_ok=True)
    return copied


def _verify_hdf5(path: Path, category: str, expected_frames: int) -> None:
    with h5py.File(path, "r") as hdf5_file:
        if hdf5_file.attrs.get("format") != ARCHIVE_FORMAT:
            raise ValueError(f"Unexpected archive format in {path}")
        if hdf5_file.attrs.get("category") != category:
            raise ValueError(f"Wrong category in {path}")
        if not hdf5_file.attrs.get("complete", False):
            raise ValueError(f"Archive was not marked complete: {path}")
        if int(hdf5_file.attrs.get("source_snapshot_count", -1)) != expected_frames:
            raise ValueError(f"Snapshot count mismatch in {path}")
        for name, dataset in hdf5_file.items():
            if not isinstance(dataset, h5py.Dataset) or len(dataset) != expected_frames:
                raise ValueError(f"Invalid frame-leading dataset {name!r} in {path}")


def _verify_tree_copy(source: Path, destination: Path) -> None:
    source_files = {path.relative_to(source): path.stat().st_size for path in _tree_files(source)}
    destination_files = {path.relative_to(destination): path.stat().st_size for path in _tree_files(destination)}
    if source_files != destination_files:
        raise IOError(f"Copied output differs from verified staging package: {destination}")


def _restore_raw_shot(raw_stage: Path, shot_dir: Path) -> None:
    if shot_dir.exists():
        try:
            _verify_plain_tree_copy(raw_stage, shot_dir)
            return
        except (OSError, IOError):
            failed_path = shot_dir.parent / f".{shot_dir.name}.partial-{uuid.uuid4().hex}"
            os.replace(shot_dir, failed_path)
            tqdm.write(f"[recovery] Moved incomplete HDD shot to {failed_path}")

    restore_path = shot_dir.parent / f".{shot_dir.name}.restore-{uuid.uuid4().hex}"
    try:
        shutil.copytree(raw_stage, restore_path)
        _verify_plain_tree_copy(raw_stage, restore_path)
        os.replace(restore_path, shot_dir)
        tqdm.write(f"[recovery] Restored original shot from SSD staging to {shot_dir}")
    except Exception:
        shutil.rmtree(restore_path, ignore_errors=True)
        raise


def _verify_ssd_device(staging_root: Path, expected_device: str | None) -> tuple[str, int]:
    usage = shutil.disk_usage(staging_root)
    if expected_device is None:
        return "device check disabled", usage.free
    result = subprocess.run(
        ["findmnt", "-n", "-o", "SOURCE", "--target", str(staging_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip().split("[", 1)[0]
    if not actual or Path(actual).resolve() != Path(expected_device).resolve():
        raise OSError(
            f"Staging path {staging_root} is mounted from {actual!r}, not expected {expected_device!r}"
        )
    return actual, usage.free


def _copy_back_verified(staged_package: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing packed directory: {destination}")
    partial = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.partial"
    try:
        shutil.copytree(staged_package, partial)
        _verify_tree_copy(staged_package, partial)
        for category, filename in (("particles", "particles.h5"), ("concentration", "concentration.h5")):
            with h5py.File(partial / filename, "r") as archive:
                frame_count = cast(int, archive.attrs["source_snapshot_count"])
            _verify_hdf5(partial / filename, category, frame_count)
        os.replace(partial, destination)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _pack_one_shot(
    shot_dir: Path,
    packed_root: Path,
    staging_root: Path,
    expected_device: str | None,
    replace_source: bool,
    keep_staging: bool,
    compression: str | None,
    staging_reserve_bytes: int,
    hdd_reserve_bytes: int,
) -> Path:
    if not SHOT_PATTERN.fullmatch(shot_dir.name):
        raise ValueError(f"Not a shot_### directory: {shot_dir}")
    if shot_dir.is_symlink():
        raise ValueError(f"Refusing to replace a symlinked shot directory: {shot_dir}")
    if not shot_dir.is_dir():
        raise FileNotFoundError(shot_dir)
    final_dir = packed_root / shot_dir.name
    if final_dir.exists():
        raise FileExistsError(f"Packed output already exists: {final_dir}")

    stage_device, stage_free = _verify_ssd_device(staging_root, expected_device)
    shot_files = _tree_files(shot_dir)
    source_bytes = sum(path.stat().st_size for path in shot_files)
    snapshot_paths = [
        path for path in shot_files
        if path.parent == shot_dir and (PART_PATTERN.fullmatch(path.name) or CONC_PATTERN.fullmatch(path.name))
    ]
    if not any(PART_PATTERN.fullmatch(path.name) for path in snapshot_paths):
        raise ValueError(f"No part_*.npz snapshots in {shot_dir}")
    if not any(CONC_PATTERN.fullmatch(path.name) for path in snapshot_paths):
        raise ValueError(f"No conc_*.npz snapshots in {shot_dir}")

    snapshot_bytes = sum(path.stat().st_size for path in snapshot_paths)
    expanded_snapshot_bytes = _estimate_npz_payload_bytes(snapshot_paths)
    other_bytes = source_bytes - snapshot_bytes
    hdf5_overhead = max(64 * 1024 * 1024, int(expanded_snapshot_bytes * 0.05))
    estimated_required = source_bytes + expanded_snapshot_bytes + hdf5_overhead + other_bytes + staging_reserve_bytes
    tqdm.write(
        f"[space] {shot_dir.name}: staging device {stage_device}, "
        f"free={stage_free / GIB:.1f} GiB, estimated HDF5 payload={expanded_snapshot_bytes / GIB:.1f} GiB, "
        f"estimated peak need={estimated_required / GIB:.1f} GiB"
    )
    if stage_free < estimated_required:
        raise OSError(
            f"Not enough SSD space for {shot_dir.name}: need about {estimated_required / GIB:.1f} GiB, "
            f"have {stage_free / GIB:.1f} GiB"
        )

    stage_dir = staging_root / shot_dir.name
    if stage_dir.exists():
        raise FileExistsError(
            f"Staging directory already exists: {stage_dir}. Inspect/recover it manually; it was not removed."
        )
    raw_stage = stage_dir / "source"
    package_stage = stage_dir / "hdf5-packed" / shot_dir.name
    try:
        _copy_tree_with_progress(shot_dir, raw_stage)
        _verify_plain_tree_copy(shot_dir, raw_stage)

        part_count = _pack_category(
            raw_stage, package_stage / "particles.h5", PART_PATTERN, "particles", compression
        )
        conc_count = _pack_category(
            raw_stage, package_stage / "concentration.h5", CONC_PATTERN, "concentration", compression
        )
        copied_other_count = _copy_other_files(raw_stage, package_stage)
        _verify_hdf5(package_stage / "particles.h5", "particles", part_count)
        _verify_hdf5(package_stage / "concentration.h5", "concentration", conc_count)
        manifest = {
            "format": ARCHIVE_FORMAT,
            "source_shot": str(shot_dir.resolve()),
            "particle_snapshots": part_count,
            "concentration_snapshots": conc_count,
            "other_files_copied": copied_other_count,
            "source_bytes": source_bytes,
            "created_unix_time": time.time(),
        }
        (package_stage / "packing_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        package_bytes = _tree_size(package_stage)
        hdd_usage = shutil.disk_usage(shot_dir.parent)
        available_after_source_removal = hdd_usage.free + source_bytes if replace_source else hdd_usage.free
        if available_after_source_removal < package_bytes + hdd_reserve_bytes:
            raise OSError(
                f"Not enough HDD space to commit {shot_dir.name}: package needs {package_bytes / GIB:.1f} GiB "
                f"plus {hdd_reserve_bytes / GIB:.1f} GiB reserve; available after replacement would be "
                f"{available_after_source_removal / GIB:.1f} GiB"
            )

        if replace_source:
            # The complete source and verified package remain on SSD until HDD copy-back succeeds.
            shutil.rmtree(shot_dir)
        _copy_back_verified(package_stage, final_dir)
        if not keep_staging:
            try:
                shutil.rmtree(stage_dir)
            except OSError as error:
                tqdm.write(f"[warning] Verified archive committed; could not remove SSD staging {stage_dir}: {error}")
        return final_dir
    except Exception:
        if replace_source and raw_stage.exists():
            try:
                _restore_raw_shot(raw_stage, shot_dir)
            except Exception as recovery_error:
                tqdm.write(
                    f"[recovery error] Original HDD shot could not be restored automatically: {recovery_error}. "
                    f"Raw data remains on SSD at {raw_stage}"
                )
        tqdm.write(
            f"[recovery] Preserved staging data at {stage_dir}. "
            "If the HDD source was already removed, the raw copy is under source/ and the verified package under hdf5-packed/."
        )
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing shot_### folders")
    parser.add_argument("--staging-root", type=Path, default=Path("/var/tmp"), help="Writable SSD staging mount/directory")
    parser.add_argument("--expected-ssd-device", default="/dev/nvme0n1p2", help="Expected findmnt source; use empty string to disable")
    parser.add_argument("--replace-source", action="store_true", help="Delete each raw shot only after its SSD package validates")
    parser.add_argument("--keep-staging", action="store_true", help="Keep successful SSD staging copies")
    parser.add_argument("--compression", choices=("lzf", "gzip"), help="Optional HDF5 compression (default: none)")
    parser.add_argument("--shot", action="append", type=int, help="Only process this shot index; may be repeated")
    parser.add_argument("--staging-reserve-gib", type=float, default=2.0)
    parser.add_argument("--hdd-reserve-gib", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true", help="Report shot sizes and SSD free space without copying")
    args = parser.parse_args(argv)
    if args.staging_reserve_gib < 0 or args.hdd_reserve_gib < 0:
        parser.error("reserve values must be non-negative")

    run_dir = args.run_dir.resolve()
    staging_root = args.staging_root.resolve()
    packed_root = run_dir / "hdf5-packed"
    if not run_dir.is_dir():
        parser.error(f"Run directory does not exist: {run_dir}")
    if not staging_root.is_dir():
        parser.error(f"Staging directory does not exist: {staging_root}")
    shots = sorted(path for path in run_dir.glob("shot_*") if path.is_dir() and SHOT_PATTERN.fullmatch(path.name))
    if args.shot is not None:
        selected = set(args.shot)
        shots = [path for path in shots if int(path.name.split("_")[1]) in selected]
    if not shots:
        parser.error(f"No shot_### directories found in {run_dir}")

    expected_device = args.expected_ssd_device.strip() or None
    device, free_bytes = _verify_ssd_device(staging_root, expected_device)
    print(f"[init] Run: {run_dir}")
    print(f"[init] Found {len(shots)} shots; staging on {device} at {staging_root}")
    print(f"[init] SSD free space: {free_bytes / GIB:.1f} GiB")
    print(f"[init] HDD output: {packed_root}")
    if args.replace_source:
        print("[warning] --replace-source is enabled; originals are removed only after verified SSD staging.")
    else:
        print("[safe mode] Original HDD shots will be retained; output requires additional HDD free space.")

    if args.dry_run:
        for shot_dir in shots:
            source_bytes = _tree_size(shot_dir)
            targeted = sum(
                path.stat().st_size
                for path in _tree_files(shot_dir)
                if path.parent == shot_dir and (PART_PATTERN.fullmatch(path.name) or CONC_PATTERN.fullmatch(path.name))
            )
            other_bytes = source_bytes - targeted
            expanded = _estimate_npz_payload_bytes(
                [
                    path for path in _tree_files(shot_dir)
                    if path.parent == shot_dir and (PART_PATTERN.fullmatch(path.name) or CONC_PATTERN.fullmatch(path.name))
                ]
            )
            needed = (
                source_bytes + expanded + max(64 * 1024 * 1024, int(expanded * 0.05))
                + other_bytes + int(args.staging_reserve_gib * GIB)
            )
            print(
                f"[dry-run] {shot_dir.name}: source={source_bytes / GIB:.2f} GiB, "
                f"snapshot NPZ={targeted / GIB:.2f} GiB, uncompressed payload={expanded / GIB:.2f} GiB, "
                f"estimated SSD peak={needed / GIB:.2f} GiB"
            )
        return 0

    packed_root.mkdir(parents=True, exist_ok=True)
    failures = []
    for shot_dir in tqdm(shots, desc="Shots", unit="shot", dynamic_ncols=True):
        try:
            destination = _pack_one_shot(
                shot_dir,
                packed_root,
                staging_root,
                expected_device,
                args.replace_source,
                args.keep_staging,
                args.compression,
                int(args.staging_reserve_gib * GIB),
                int(args.hdd_reserve_gib * GIB),
            )
            tqdm.write(f"[done] {shot_dir.name}: verified archive at {destination}")
        except Exception as error:
            failures.append((shot_dir.name, str(error)))
            tqdm.write(f"[failed] {shot_dir.name}: {error}")
            break

    if failures:
        print("Stopped after first failure; inspect the preserved staging directory before retrying.")
        return 1
    print(f"[complete] Packed {len(shots)} shot(s) into {packed_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())