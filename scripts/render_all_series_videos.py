
#!/usr/bin/env python3
"""Render every available frame for every run in a simulation series."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "experiments/simulation_series_parameters.toml"
VIDEO_SCRIPT = PROJECT_ROOT / "video_maker_testrun.py"
DEFAULT_MAZE = (
    PROJECT_ROOT
    / "different_mazes/Ran_maze_size_prop_to_droplet.tsv"
)


def fmt_number(value: float | int) -> str:
    """Format numbers for filenames, using '-' instead of decimal points."""
    return f"{value:g}".replace(".", "-")


def load_config(config_path: Path) -> dict:
    with config_path.open("rb") as f:
        return tomllib.load(f)


def iter_run_configurations(config: dict):
    """Yield run IDs and effective parameters in the same order as make_jobs."""
    defaults = config.get("defaults", {})
    for regime in config["regimes"]:
        grid = regime.get("grid", {})
        keys = list(grid)
        combinations = itertools.product(*(grid[key] for key in keys))
        for config_index, values in enumerate(combinations):
            run_id = f"{regime['name']}_c{config_index:04d}"
            yield run_id, defaults | dict(zip(keys, values))


def resolve_maze(config: dict, maze_override: Path | None) -> Path:
    """Resolve an explicit maze or the maze selected by the series profile."""
    if maze_override is not None:
        maze = maze_override.expanduser()
        if not maze.is_absolute() and not maze.is_file():
            maze = PROJECT_ROOT / maze
        return maze.resolve()

    profile_name = config.get("defaults", {}).get("maze_config")
    if profile_name:
        startpoints_path = PROJECT_ROOT / "maze_startpoints_configuration.toml"
        with startpoints_path.open("rb") as startpoints_file:
            profiles = tomllib.load(startpoints_file)["configurations"]
        maze = Path(profiles[profile_name]["maze_file"])
        if not maze.is_absolute():
            maze = PROJECT_ROOT / maze
        return maze.resolve()
    return DEFAULT_MAZE.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render MP4 videos for all simulation runs in a series."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to simulation_series_parameters.toml",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override the data root from the TOML configuration.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override the output root from the TOML configuration.",
    )
    parser.add_argument(
        "--maze",
        type=Path,
        help="Override the maze selected by the series' maze_config profile.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render videos even when the destination MP4 already exists.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)

    series = config["series"]
    series_name = series["name"]

    data_root = args.data_root or Path(series.get("data_root", "data"))
    output_root = args.output_root or Path(series.get("output_root", "output"))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    data_series = (data_root / series_name).resolve()
    output_series = (output_root / series_name).resolve()
    videos_dir = output_series / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    maze_path = resolve_maze(config, args.maze)

    if not VIDEO_SCRIPT.is_file():
        parser.error(f"Video renderer not found: {VIDEO_SCRIPT}")
    if not maze_path.is_file():
        parser.error(f"Maze file not found: {maze_path}")
    if not data_series.is_dir():
        parser.error(f"Data series directory not found: {data_series}")

    # Prefer the project's virtual environment for rendering dependencies.
    venv_python = PROJECT_ROOT / ".venv/bin/python"
    python_executable = (
        str(venv_python) if venv_python.is_file() else sys.executable
    )

    run_dirs = [
        (data_series / run_id, parameters)
        for run_id, parameters in iter_run_configurations(config)
        if (data_series / run_id).is_dir()
    ]

    if not run_dirs:
        parser.error(f"No configured run directories found in {data_series}")

    total = 0
    rendered = 0
    skipped = 0
    failed = 0

    log_path = videos_dir / "render.log"

    with log_path.open("a", encoding="utf-8") as log:
        def report(message: str) -> None:
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        report(f"\nSeries: {series_name}")
        report(f"Data:   {data_series}")
        report(f"Videos: {videos_dir}")
        report(f"Configured runs found: {len(run_dirs)}")

        for run_dir, params in run_dirs:
            shot_dirs = sorted(
                p for p in run_dir.glob("shot_*") if p.is_dir()
            )

            if not shot_dirs:
                report(f"[SKIP] {run_dir.name}: no shot directories")
                skipped += 1
                continue

            for shot_dir in shot_dirs:
                total += 1

                # Only render folders containing the expected simulation data.
                particle_files = list(shot_dir.glob("part_*.npz"))
                concentration_files = list(shot_dir.glob("conc_*.npz"))

                required_files = [
                    "grid.txt",
                ]
                if not (
                    (shot_dir / "param.txt").is_file()
                    or (shot_dir / "param_checkpoint.txt").is_file()
                ):
                    required_files.append("param.txt or param_checkpoint.txt")

                missing = [
                    name for name in required_files
                    if name == "grid.txt" and not (shot_dir / name).is_file()
                ]
                has_params = (
                    (shot_dir / "param.txt").is_file()
                    or (shot_dir / "param_checkpoint.txt").is_file()
                )

                if not particle_files or not concentration_files or not has_params or missing:
                    report(
                        f"[SKIP] {run_dir.name}/{shot_dir.name}: "
                        f"missing/incomplete data "
                        f"(particles={len(particle_files)}, "
                        f"concentrations={len(concentration_files)}, "
                        f"grid={'yes' if (shot_dir / 'grid.txt').is_file() else 'no'}, "
                        f"params={'yes' if has_params else 'no'})"
                    )
                    skipped += 1
                    continue

                filename = run_dir.name
                if "grim_reaper_delay" in params:
                    filename += f"_rt{fmt_number(params['grim_reaper_delay'])}"
                if "emission_rate" in params:
                    filename += f"_er{fmt_number(params['emission_rate'])}"
                if "desired_time" in params:
                    filename += f"_{fmt_number(params['desired_time'])}s"

                # Include the shot ID if this run contains multiple shots.
                if len(shot_dirs) > 1:
                    filename += f"_{shot_dir.name}"

                destination = videos_dir / f"{filename}.mp4"

                if destination.is_file() and not args.overwrite:
                    report(f"[EXISTS] {destination.name}")
                    skipped += 1
                    continue

                command = [
                    python_executable,
                    str(VIDEO_SCRIPT),
                    "--data", str(shot_dir.resolve()),
                    "--maze", str(maze_path),
                    "--frames", "0",
                    "--output", str(destination.resolve()),
                ]

                report(
                    f"[RENDER] {run_dir.name}/{shot_dir.name} -> "
                    f"{destination.name}"
                )
                report(
                    f"         parameters={params}"
                )

                try:
                    result = subprocess.run(
                        command,
                        cwd=PROJECT_ROOT,
                        check=False,
                    )
                except OSError as exc:
                    report(f"[FAIL] Could not launch renderer: {exc}")
                    failed += 1
                    continue

                if result.returncode == 0 and destination.is_file():
                    report(f"[OK] {destination}")
                    rendered += 1
                else:
                    report(
                        f"[FAIL] {run_dir.name}/{shot_dir.name}: "
                        f"renderer exit code={result.returncode}, "
                        f"output_exists={destination.is_file()}"
                    )
                    failed += 1

        report("\nSummary")
        report(f"  Shots considered: {total}")
        report(f"  Rendered:         {rendered}")
        report(f"  Skipped:          {skipped}")
        report(f"  Failed:           {failed}")
        report(f"  Log:              {log_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
