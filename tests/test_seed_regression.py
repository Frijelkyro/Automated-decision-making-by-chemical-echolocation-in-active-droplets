import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).resolve().parent / "baseline"


class SeedRegressionTests(unittest.TestCase):
    def _run_baseline(self, directory):
        data_dir = directory / "data"
        output_dir = directory / "output"
        command = [
            sys.executable,
            str(ROOT / "maze_cluster_script.py"),
            "--seed",
            "12345",
            "--test-run",
            "true",
            "--desired-time",
            "20",
            "--dt",
            "0.0005",
            "--n-steps",
            "100",
            "--num-particles",
            "10",
            "--emission-rate",
            "0.5",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
        ]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        return data_dir, output_dir

    def test_fixed_seed_shot_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as first_temp, tempfile.TemporaryDirectory() as second_temp:
            first_data, first_output = self._run_baseline(Path(first_temp))
            second_data, second_output = self._run_baseline(Path(second_temp))

            first_exit = (first_output / "exit_times.txt").read_bytes()
            second_exit = (second_output / "exit_times.txt").read_bytes()
            self.assertEqual(first_exit, second_exit)
            self.assertEqual(first_exit, (BASELINE / "exit_times.txt").read_bytes())

            first_parts = {path.name: path for path in first_data.glob("part_*.npz")}
            second_parts = {path.name: path for path in second_data.glob("part_*.npz")}
            baseline_parts = {path.name: path for path in BASELINE.glob("part_*.npz")}
            self.assertEqual(set(first_parts), set(second_parts))
            self.assertEqual(set(first_parts), set(baseline_parts))

            for name in first_parts:
                first_bytes = first_parts[name].read_bytes()
                self.assertEqual(first_bytes, second_parts[name].read_bytes())
                self.assertEqual(first_bytes, (BASELINE / name).read_bytes())
