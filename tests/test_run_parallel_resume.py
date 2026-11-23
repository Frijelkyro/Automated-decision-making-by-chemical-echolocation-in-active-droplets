import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_parallel

ROOT = Path(__file__).resolve().parents[1]


class RunParallelResumeTests(unittest.TestCase):
    def _write_config(self, root, resume_simulation):
        config_path = root / "series.toml"
        config_path.write_text(
            f"""
[series]
name = "resume-test"
data_root = "{root / 'data'}"
output_root = "{root / 'output'}"

[defaults]
desired_time = 2
resume_simulation = {str(resume_simulation).lower()}

[[regimes]]
name = "run"
shots = 1

[regimes.grid]
emission_rate = [1.0]
""",
            encoding="utf-8",
        )
        return config_path

    def _invoke(self, config_path, *extra_args):
        captured = {}

        def capture_run(command, **kwargs):
            captured["command"] = command
            captured["job_input"] = kwargs["stdin"].read()

        with (
            patch.object(
                sys,
                "argv",
                ["run_parallel.py", "--config", str(config_path), *extra_args],
            ),
            patch.object(run_parallel.shutil, "which", return_value="/usr/bin/parallel"),
            patch.object(run_parallel.subprocess, "run", side_effect=capture_run),
        ):
            run_parallel.main()
        return captured

    def test_resume_flag_reruns_jobs_with_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self._write_config(root, resume_simulation=False)

            captured = self._invoke(config_path, "--resume")

            command = captured["command"]
            joblog = Path(command[command.index("--joblog") + 1])
            self.assertTrue(joblog.name.startswith("joblog-checkpoint-resume-"))
            self.assertIn("--resume true", captured["job_input"])
            self.assertNotIn("--resume-failed", command)

    def test_resume_config_does_not_skip_successful_joblog_entries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self._write_config(root, resume_simulation=True)

            captured = self._invoke(config_path)

            command = captured["command"]
            joblog = Path(command[command.index("--joblog") + 1])
            self.assertTrue(joblog.name.startswith("joblog-checkpoint-resume-"))
            self.assertIn("--resume true", captured["job_input"])
            self.assertNotIn("--resume-failed", command)


if __name__ == "__main__":
    unittest.main()
