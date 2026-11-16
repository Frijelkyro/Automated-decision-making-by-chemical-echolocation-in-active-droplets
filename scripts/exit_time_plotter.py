#!/usr/bin/env python3
"""Plot particle exit duration by birth time for an experiment series.

Reads particle_success_all.csv from particle_success_fraction_analysis.py and
writes per-run and whole-series plots under the configured output directory.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import math
import tomllib
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def paths_from_config(config_path: Path) -> tuple[str, Path, Path]:
    with project_path(config_path).open("rb") as config_file:
        config = tomllib.load(config_file)
    series = config["series"]
    series_name = str(series["name"])
    output_root = project_path(series.get("output_root", "output"))
    series_output = output_root / series_name
    return series_name, series_output / "particle_success_all.csv", series_output / "exit_time"


def load_exit_durations(csv_path: Path, pattern: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    required = {"run_id", "shot", "birth_time", "exit_time", "exited"}
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            run_id = row["run_id"]
            if not fnmatch.fnmatch(run_id, pattern):
                continue
            try:
                birth_time = float(row["birth_time"])
                exit_time = float(row["exit_time"])
                exited = row["exited"].strip().lower() in {"1", "true", "yes"}
            except (TypeError, ValueError):
                continue
            if exited and math.isfinite(birth_time) and math.isfinite(exit_time):
                grouped[run_id].append(
                    {
                        "shot": row["shot"],
                        "birth_time": birth_time,
                        "exit_duration": exit_time - birth_time,
                    }
                )
    return dict(grouped)


def summarize(rows: list[dict], bin_width: float) -> list[dict]:
    bins: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        bins[math.floor(row["birth_time"] / bin_width)].append(row)
    summary = []
    for bin_index, values in sorted(bins.items()):
        durations = np.asarray([row["exit_duration"] for row in values], dtype=float)
        summary.append(
            {
                "birth_time_bin": bin_index,
                "birth_time": (bin_index + 0.5) * bin_width,
                "mean_exit_duration": float(np.mean(durations)),
                "standard_deviation": float(np.std(durations, ddof=1)) if len(durations) > 1 else 0.0,
                "particle_count": len(values),
                "shot_count": len({row["shot"] for row in values}),
            }
        )
    return summary


def save_plot(rows: list[dict], output_path: Path, title: str, bin_width: float, dpi: int, show: bool) -> list[dict]:
    summary = summarize(rows, bin_width)
    if not summary:
        return []
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        [row["birth_time"] for row in summary],
        [row["mean_exit_duration"] for row in summary],
        yerr=[row["standard_deviation"] for row in summary],
        fmt="o-",
        capsize=3,
    )
    ax.set_xlabel("Birth time [s]")
    ax.set_ylabel("Exit time - birth time [s]")
    ax.set_title(f"{title} (birth-time bins: {bin_width:g} s)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return summary


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scope", "run_id", "birth_time_bin", "birth_time", "mean_exit_duration",
        "standard_deviation", "particle_count", "shot_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Experiment series TOML.")
    parser.add_argument("--csv", type=Path, help="Override particle_success_all.csv path.")
    parser.add_argument("--out-dir", type=Path, help="Override the series exit_time output directory.")
    parser.add_argument("--pattern", default="*", help="Run ID glob filter, e.g. 'qj01_rates_c000*'.")
    parser.add_argument("--bin-width", type=float, default=2.0, help="Birth-time bin width in seconds.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true", help="Display plots interactively as well as saving them.")
    args = parser.parse_args(argv)
    if not math.isfinite(args.bin_width) or args.bin_width <= 0:
        parser.error("--bin-width must be positive and finite")

    try:
        series_name, configured_csv, configured_output = paths_from_config(args.config)
    except (KeyError, OSError, tomllib.TOMLDecodeError, ValueError) as error:
        parser.error(str(error))
    csv_path = project_path(args.csv) if args.csv else configured_csv
    output_dir = project_path(args.out_dir) if args.out_dir else configured_output
    if not csv_path.is_file():
        parser.error(
            f"Exit-time data not found: {csv_path}. Run particle_success_fraction_analysis.py "
            "with this experiment config first to create particle_success_all.csv."
        )

    grouped = load_exit_durations(csv_path, args.pattern)
    if not grouped:
        parser.error(f"No finite exited particles found in {csv_path} for run pattern {args.pattern!r}.")

    summary_rows = []
    all_rows = []
    for run_id, rows in sorted(grouped.items()):
        all_rows.extend(rows)
        summary = save_plot(
            rows,
            output_dir / f"{run_id}_exit_time_vs_birth_time.png",
            run_id,
            args.bin_width,
            args.dpi,
            args.show,
        )
        summary_rows.extend({"scope": "run", "run_id": run_id, **row} for row in summary)
        print(f"{run_id}: {len(rows)} exited particles")

    overall = save_plot(
        all_rows,
        output_dir / "all_runs_exit_time_vs_birth_time.png",
        series_name,
        args.bin_width,
        args.dpi,
        args.show,
    )
    summary_rows.extend({"scope": "all", "run_id": "all", **row} for row in overall)
    write_summary(output_dir / "exit_time_by_birth_time.csv", summary_rows)
    print(f"Saved exit-time plots and summary under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
