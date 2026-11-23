import tempfile
import unittest
from pathlib import Path

from scripts.render_all_series_videos import has_parameter_file


class RenderAllSeriesVideosTests(unittest.TestCase):
    def test_accepts_numbered_parameter_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            shot_dir = Path(temporary_directory)
            (shot_dir / "param00.txt").touch()

            self.assertTrue(has_parameter_file(shot_dir))

    def test_accepts_legacy_parameter_file_names(self):
        for filename in ("param.txt", "param_checkpoint.txt"):
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    shot_dir = Path(temporary_directory)
                    (shot_dir / filename).touch()

                    self.assertTrue(has_parameter_file(shot_dir))

    def test_rejects_nonmatching_parameter_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            shot_dir = Path(temporary_directory)
            (shot_dir / "param_checkpoint.npz").touch()
            (shot_dir / "parameters.txt").touch()

            self.assertFalse(has_parameter_file(shot_dir))


if __name__ == "__main__":
    unittest.main()
