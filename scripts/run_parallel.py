import argparse
import shutil
import subprocess
import time
import tomllib
from pathlib import Path

if __package__:
    from .make_jobs import generate_jobs
else:
    from make_jobs import generate_jobs


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Run experiment jobs with GNU Parallel.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "simulation_series_parameters.toml",
    )
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--progress", "--bar", action="store_true", help="Show GNU Parallel's compact progress bar.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume every shot from its checkpoint instead of skipping completed job-log entries.",
    )
    args = parser.parse_args()

    parallel = shutil.which("parallel")
    if parallel is None:
        parser.error("GNU Parallel is not installed or is not on PATH")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    resume_in_config = config.get("defaults", {}).get("resume_simulation", False) or any(
        True in regime.get("grid", {}).get("resume_simulation", [])
        for regime in config.get("regimes", [])
    )
    resume_checkpoints = args.resume or resume_in_config
    jobs, jobfile = generate_jobs(
        config_path,
        resume_override=True if args.resume else None,
    )
    joblog = jobfile.with_name(
        time.strftime("joblog-resume-%Y%m%d-%H%M%S")
        if resume_checkpoints
        else "joblog"
    )
    command = [parallel, "-j", str(args.jobs), "--joblog", str(joblog)]
    if args.progress:
        command.append("--bar")
    command.append("--halt-on-error=never")  # set to 2 to kill all running jobs and exit as soon as any job fails

    with jobfile.open() as job_input:
        result = subprocess.run(command, stdin=job_input, cwd=ROOT)
    if result.returncode:
        print(f"{result.returncode} job(s) failed; see {joblog}")
    
if __name__ == "__main__":
    main()
