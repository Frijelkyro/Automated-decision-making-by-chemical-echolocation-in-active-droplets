import argparse
import hashlib
import shutil
import subprocess
import tomllib
from pathlib import Path

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
    parser.add_argument("--progress", action="store_true", help="Show GNU Parallel queue progress.")
    parser.add_argument("--resume", action="store_true", help="Resume using the existing GNU Parallel job log.")
    args = parser.parse_args()

    parallel = shutil.which("parallel")
    if parallel is None:
        parser.error("GNU Parallel is not installed or is not on PATH")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    jobs, jobfile = generate_jobs(config_path)
    joblog = jobfile.with_name("joblog")
    signature_file = jobfile.with_suffix(".sha256")
    signature = hashlib.sha256(("\n".join(jobs) + "\n").encode()).hexdigest()
    command = [parallel, "-j", str(args.jobs), "--joblog", str(joblog)]
    if args.progress:
        command.append("--progress")
    resume_simulation = config.get("defaults", {}).get("resume_simulation", False)
    if joblog.exists():
        if not signature_file.exists() or signature_file.read_text().strip() != signature:
            parser.error(
                "Existing series job log does not match this job list; "
                "keep the configuration unchanged or use a new series name"
            )
        if resume_simulation:
            command.append("--resume-failed")
        elif args.resume:
            command.append("--resume")
    else:
        signature_file.write_text(signature + "\n")

    with jobfile.open() as job_input:
        subprocess.run(command, stdin=job_input, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
