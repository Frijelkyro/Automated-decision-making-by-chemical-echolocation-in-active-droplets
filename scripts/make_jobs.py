import argparse
import itertools
import shlex
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "maze_cluster_script.py"
PARAMETER_FLAGS = {
    "beta": "--beta",
    "dt": "--dt",
    "emission_rate": "--emission-rate",
    "grim_reaper_delay": "--grim-reaper-delay",
    "desired_time": "--desired-time",
    "write_every": "--write-every",
    "n_steps": "--n-steps",
    "num_particles": "--num-particles",
    "test_run": "--test-run",
    "resume_simulation": "--resume",
}


def _value_text(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


def generate_jobs(config_path):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    series = config["series"]
    defaults = config.get("defaults", {})
    data_root = Path(series["data_root"])
    output_root = Path(series["output_root"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    jobs = []
    for regime in config["regimes"]:
        grid = regime["grid"]
        keys = list(grid)
        for config_index, values in enumerate(itertools.product(*(grid[key] for key in keys))):
            parameters = defaults | dict(zip(keys, values))
            unknown = parameters.keys() - PARAMETER_FLAGS.keys()
            if unknown:
                raise ValueError(f"Unsupported simulation parameter(s): {', '.join(sorted(unknown))}")

            run_id = f"{regime['name']}_c{config_index:04d}"
            for shot in range(regime["shots"]):
                shot_name = f"shot_{shot:03d}"
                data_dir = data_root / series["name"] / run_id / shot_name
                output_dir = output_root / series["name"] / run_id / shot_name
                args = [sys.executable, str(WRAPPER)]
                for key, value in parameters.items():
                    args.extend([PARAMETER_FLAGS[key], _value_text(value)])
                args.extend([
                    "--run-id", run_id,
                    "--shot", str(shot),
                    "--data-dir", str(data_dir),
                    "--output-dir", str(output_dir),
                ])
                jobs.append(shlex.join(args))

    jobfile = output_root / series["name"] / "jobs.txt"
    jobfile.parent.mkdir(parents=True, exist_ok=True)
    jobfile.write_text("\n".join(jobs) + "\n")
    return jobs, jobfile


def main():
    parser = argparse.ArgumentParser(description="Expand a TOML experiment into run commands.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "simulation_series_parameters.toml",
    )
    args = parser.parse_args()

    jobs, jobfile = generate_jobs(args.config)
    print(f"Wrote {len(jobs)} jobs to {jobfile}")


if __name__ == "__main__":
    main()
