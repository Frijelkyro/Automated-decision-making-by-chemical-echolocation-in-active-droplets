import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from junction_tracking import truncate_decision_log_files


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = ROOT / ".venv" / "bin" / "python"
PYTHON = PROJECT_PYTHON if PROJECT_PYTHON.exists() else sys.executable


class JunctionTrackerResumeTest(unittest.TestCase):
    @staticmethod
    def _read_decision_log(directory):
        rows = []
        for filename in sorted(directory.glob("decisions_*.npz")):
            with np.load(filename, allow_pickle=False) as data:
                for key in ("global_step", "sim_time", "particle_id", "junction_id", "decision"):
                    if key not in data:
                        raise AssertionError(f"{filename.name} is missing {key}")
                count = data["global_step"].size
                for index in range(count):
                    rows.append(
                        (
                            int(data["global_step"][index]),
                            float(data["sim_time"][index]),
                            int(data["particle_id"][index]),
                            int(data["junction_id"][index]),
                            int(data["decision"][index]),
                        )
                    )
        return rows

    def test_decision_log_cleanup_truncates_straddles_and_deletes_future_files(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            np.savez(
                output_dir / "decisions_straddle.npz",
                global_step=np.asarray([8, 9, 10, 11]),
                global_steps=np.asarray([8, 9, 10, 11]),
                sim_time=np.asarray([0.8, 0.9, 1.0, 1.1]),
                particle_id=np.asarray([1, 1, 2, 2]),
                junction_id=np.asarray([0, 0, 1, 1]),
                decision=np.asarray([1, 0, 1, 0]),
            )
            np.savez(
                output_dir / "decisions_future.npz",
                global_step=np.asarray([12, 13]),
                sim_time=np.asarray([1.2, 1.3]),
                particle_id=np.asarray([1, 1]),
                junction_id=np.asarray([0, 0]),
                decision=np.asarray([1, 1]),
            )

            truncate_decision_log_files(output_dir, 10)

            self.assertFalse((output_dir / "decisions_future.npz").exists())
            with np.load(output_dir / "decisions_straddle.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(
                    data["global_step"],
                    np.asarray([8, 9, 10]),
                )
                np.testing.assert_array_equal(
                    data["decision"],
                    np.asarray([1, 0, 1]),
                )

    def test_ten_loops_resume_from_second_last_checkpoint_matches_straight_run(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            zones_file = temp_path / "zones.npz"
            np.savez(
                zones_file,
                junction_ids=np.asarray(["J0"]),
                J0_before=np.zeros((99, 110), dtype=bool),
                J0_correct=np.zeros((99, 110), dtype=bool),
                J0_wrong=np.zeros((99, 110), dtype=bool),
            )

            straight_dir = temp_path / "straight"
            straight_dir.mkdir()
            resume_dir = temp_path / "resume"
            resume_dir.mkdir()

            common = [
                str(PYTHON),
                str(ROOT / "maze_cluster_script.py"),
                "--desired-time",
                "1",
                "--n-steps",
                "100",
                "--dt",
                "0.001",
                "--emission-rate",
                "10",
                "--num-particles",
                "10",
                "--seed",
                "12345",
                "--maze-config",
                "Ran_maze_size_prop_to_droplet",
                "--track-decisions",
                "--zones-file",
                str(zones_file),
                "--decision-flush-loops",
                "100",
            ]

            straight = subprocess.run(
                common
                + ["--data-dir", str(straight_dir), "--output-dir", str(straight_dir)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(straight.returncode, 0, straight.stderr)

            second_last = straight_dir / "ckpt_previous.npz"
            self.assertTrue(second_last.exists())
            shutil.copy2(second_last, resume_dir / "ckpt_previous.npz")

            resumed = subprocess.run(
                common
                + [
                    "--resume",
                    "true",
                    "--data-dir",
                    str(resume_dir),
                    "--output-dir",
                    str(resume_dir),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)

            straight_log = self._read_decision_log(straight_dir)
            resumed_log = self._read_decision_log(resume_dir)
            self.assertEqual(resumed_log, straight_log)


if __name__ == "__main__":
    unittest.main()
