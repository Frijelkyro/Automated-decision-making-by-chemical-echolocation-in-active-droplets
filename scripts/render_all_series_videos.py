
#!/usr/bin/env python3
"""Render every available frame for every run in a simulation series."""

from __future__ import annotations

import argparse
import re
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


def get_run_parameters(config: dict, run_index: int) -> dict:
    """Recover grid parameters from the run index.

    Assumes runs are indexed with grim_reaper_delay as the outer
    loop and emission_rate as the inner loop, matching quadratic_c0000...
    """
    regimes = config["regimes"]
    regime = next(
        (r for r in regimes if r["name"] == "quadratic"),
        None,
    )
    if regime is None:
        raise ValueError("No 'quadratic' regime found in the TOML config.")

    grid = regime["grid"]
    delays = grid["grim_reaper_delay"]
    rates = grid["emission_rate"]

    n_rates = len(rates)
    delay_index, rate_index = divmod(run_index, n_rates)

    if delay_index >= len(delays):
        raise ValueError(
            f"Run index {run_index} exceeds the configured parameter grid."
        )

    defaults = config.get("defaults", {})

    return {
        "grim_reaper_delay": delays[delay_index],
        "emission_rate": rates[rate_index],
        "desired_time": defaults.get("desired_time", 200),
    }


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
        default=DEFAULT_MAZE,
        help="Maze TSV passed to video_maker_testrun.py",
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

    data_root = args.data_root or (
        PROJECT_ROOT / series.get("data_root", "data")
    )
    output_root = args.output_root or (
        PROJECT_ROOT / series.get("output_root", "output")
    )

    data_series = (data_root / series_name).resolve()
    output_series = (output_root / series_name).resolve()
    videos_dir = output_series / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    if not VIDEO_SCRIPT.is_file():
        parser.error(f"Video renderer not found: {VIDEO_SCRIPT}")
    if not args.maze.is_file():
        parser.error(f"Maze file not found: {args.maze}")
    if not data_series.is_dir():
        parser.error(f"Data series directory not found: {data_series}")

    # Prefer the project's virtual environment for rendering dependencies.
    venv_python = PROJECT_ROOT / ".venv/bin/python"
    python_executable = (
        str(venv_python) if venv_python.is_file() else sys.executable
    )

    run_dirs = sorted(
        (
            p for p in data_series.glob("quadratic_c*")
            if p.is_dir() and re.fullmatch(r"quadratic_c\d+", p.name)
        ),
        key=lambda p: int(p.name.removeprefix("quadratic_c")),
    )

    if not run_dirs:
        parser.error(f"No quadratic_c* run directories found in {data_series}")

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
        report(f"Runs discovered: {len(run_dirs)}")

        for run_dir in run_dirs:
            match = re.fullmatch(r"quadratic_c(\d+)", run_dir.name)
            assert match is not None
            run_index = int(match.group(1))

            try:
                params = get_run_parameters(config, run_index)
            except (KeyError, ValueError, IndexError) as exc:
                report(f"[SKIP] {run_dir.name}: cannot map parameters: {exc}")
                skipped += 1
                continue

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

                delay = params["grim_reaper_delay"]
                rate = params["emission_rate"]
                desired_time = params["desired_time"]

                filename = (
                    f"{run_dir.name}_"
                    f"rt{fmt_number(delay)}_"
                    f"er{fmt_number(rate)}_"
                    f"{fmt_number(desired_time)}s"
                )

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
                    "--maze", str(args.maze.resolve()),
                    "--frames", "0",
                    "--output", str(destination.resolve()),
                ]

                report(
                    f"[RENDER] {run_dir.name}/{shot_dir.name} -> "
                    f"{destination.name}"
                )
                report(
                    f"         delay={delay}, rate={rate}, "
                    f"desired_time={desired_time}"
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
