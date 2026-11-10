import math
import csv
import sys
import tempfile
import tomllib
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
from unittest.mock import patch

import numpy as np

from scripts import maze_skeleton_path as maze_analysis
from scripts import compact_shot_directories as compactor
from scripts import particle_decision_analysis as analysis
from scripts import pack_particle_snapshots as packer


def _write_snapshot(directory, timestep, simulation_time, positions):
    positions = np.asarray(positions, dtype=float)
    np.savez(
        directory / f"part_{timestep}.npz",
        timestep=timestep,
        simulation_time=simulation_time,
        x=positions[:, 0],
        y=positions[:, 1],
        vx=np.zeros(len(positions)),
        vy=np.zeros(len(positions)),
    )


def _maze_artifacts(directory):
    directory.mkdir(parents=True, exist_ok=True)
    before = np.zeros((5, 5), dtype=np.uint8)
    correct = np.zeros_like(before)
    wrong = np.zeros_like(before)
    before[3, 1] = 1
    correct[3, 2] = 1
    wrong[3, 3] = 1
    np.savez_compressed(directory / "zone_masks.npz", J01_before=before, J01_correct=correct, J01_wrong=wrong)
    (directory / "maze_metadata.toml").write_text("dx = 1.0\ngrid_shape = [5, 5]\n", encoding="utf-8")


class ParticleDecisionAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_zone_masks_expand_across_corridor_and_to_dead_end(self):
        shape = (7, 15)
        free = np.zeros(shape, dtype=bool)
        free[2:5, 1:14] = True
        free[0:7, 6:9] = True
        path = [(3, col) for col in range(1, 14)]
        skeleton = np.zeros(shape, dtype=bool)
        branch = [(row, 7) for row in range(2, -1, -1)]
        for cell in path + branch:
            skeleton[cell] = True
        graph = {cell: [] for cell in path + branch}
        for first, second in zip(path, path[1:]):
            graph[first].append(second)
            graph[second].append(first)
        branch_chain = [(3, 7)] + branch
        for first, second in zip(branch_chain, branch_chain[1:]):
            graph[first].append(second)
            graph[second].append(first)
        junctions = [
            {"id": "J01", "center": (3, 7), "cluster": {(3, 7)}},
            {"id": "J02", "center": (3, 11), "cluster": {(3, 11)}},
        ]

        zones = maze_analysis.build_zones(junctions, graph, skeleton, free, path)

        self.assertTrue(all((row, 2) in zones["J01"]["before"] for row in (2, 3, 4)))
        self.assertTrue(all((row, 9) in zones["J01"]["correct"] for row in (2, 3, 4)))
        self.assertIn((0, 7), zones["J01"]["wrong"])
        self.assertIn((1, 6), zones["J01"]["wrong"])

    def test_metadata_neighbors_are_valid_toml_integer_pairs(self):
        path = [(1, 1), (1, 2)]
        junctions = [
            {
                "id": "J01",
                "center": (1, 1),
                "path_index": 0,
                "branch_count": 1,
                "kind": "Y",
                "neighbors": [(np.int64(2), np.int64(3)), (np.int64(4), np.int64(5))],
            }
        ]
        metadata_path = self.root / "maze_metadata.toml"
        maze_analysis.save_metadata(
            metadata_path,
            Path("maze.tsv"),
            {"wall_value": 1, "free_mask": np.zeros((6, 6), dtype=bool)},
            path,
            junctions,
            {},
            (1, 1),
            (1, 2),
            Path("plot.png"),
        )

        parsed = tomllib.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(parsed["junctions"][0]["neighbors"], [[2, 3], [4, 5]])
        self.assertEqual(len(parsed["junctions"]), 1)

    def test_birth_before_gate_and_nan_retirement(self):
        data_dir = self.root / "shot_000"
        data_dir.mkdir()
        maze_dir = self.root / "maze"
        _maze_artifacts(maze_dir)
        np.savez(
            data_dir / "param_state.npz",
            birth_times=np.array([1.0, 0.0, 0.0]),
            simulation_time=2.0,
            resume_step=2,
        )
        _write_snapshot(data_dir, 0, 0.0, [(1, 1), (2, 1), (1, 1)])
        _write_snapshot(data_dir, 1, 1.0, [(1, 1), (2, 1), (np.nan, np.nan)])
        _write_snapshot(data_dir, 2, 2.0, [(2, 1), (2, 1), (3, 1)])

        events, summary = analysis.analyze(data_dir, maze_dir, shot_id="shot_000")
        by_id = {row["particle_id"]: row for row in summary}

        self.assertEqual([(row["particle_id"], row["decision"]) for row in events], [(0, "correct")])
        self.assertEqual(by_id[0]["correct_junctions"], 1)
        self.assertEqual(by_id[0]["success_fraction"], 1.0)
        self.assertEqual(by_id[1]["decided_junctions"], 0)
        self.assertEqual(by_id[2]["decided_junctions"], 0)

    def test_nonmonotonic_snapshot_times_are_processed_chronologically(self):
        data_dir = self.root / "shot_000"
        data_dir.mkdir()
        maze_dir = self.root / "maze"
        _maze_artifacts(maze_dir)
        np.savez(data_dir / "param_state.npz", birth_times=np.array([0.0]), simulation_time=2.0)
        _write_snapshot(data_dir, 0, 0.0, [(1, 1)])
        _write_snapshot(data_dir, 50, 2.0, [(3, 1)])
        _write_snapshot(data_dir, 100, 1.0, [(2, 1)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            events, summary = analysis.analyze(data_dir, maze_dir)

        self.assertEqual([row["decision"] for row in events], ["correct"])
        self.assertEqual(summary[0]["success_fraction"], 1.0)

    def test_packed_hdf5_is_analyzable_without_raw_npz_files(self):
        data_dir = self.root / "shot_000"
        data_dir.mkdir()
        maze_dir = self.root / "maze"
        _maze_artifacts(maze_dir)
        np.savez(data_dir / "param_state.npz", birth_times=np.array([0.0]), simulation_time=2.0)
        _write_snapshot(data_dir, 0, 0.0, [(1, 1)])
        _write_snapshot(data_dir, 50, 2.0, [(3, 1)])
        _write_snapshot(data_dir, 100, 1.0, [(2, 1)])

        hdf5_path = packer.pack_shot(data_dir)
        self.assertTrue((data_dir / "part_0.npz").is_file())
        for snapshot in data_dir.glob("part_*.npz"):
            snapshot.unlink()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            events, summary = analysis.analyze(data_dir, maze_dir)

        self.assertTrue(hdf5_path.is_file())
        self.assertEqual([row["decision"] for row in events], ["correct"])
        self.assertEqual(summary[0]["success_fraction"], 1.0)

    def test_full_shot_compaction_preserves_all_fields_and_replaces_safely(self):
        run_dir = self.root / "run_c0000"
        shot_dir = run_dir / "shot_000"
        shot_dir.mkdir(parents=True)
        staging_root = self.root / "ssd"
        staging_root.mkdir()
        maze_dir = self.root / "maze"
        _maze_artifacts(maze_dir)
        np.savez(
            shot_dir / "param_state.npz",
            birth_times=np.array([0.0]),
            simulation_time=1.0,
            resume_step=1,
        )
        (shot_dir / "grid.txt").write_text("grid metadata", encoding="utf-8")
        _write_snapshot(shot_dir, 0, 0.0, [(1, 1)])
        _write_snapshot(shot_dir, 1, 1.0, [(2, 1)])
        for timestep, sim_time, value in ((0, 0.0, 1.0), (1, 1.0, 2.0)):
            np.savez(
                shot_dir / f"conc_{timestep}.npz",
                timestep=timestep,
                simulation_time=sim_time,
                concentration=np.full((3, 4), value, dtype=np.float32),
            )
        for timestep in (0, 1):
            with np.load(shot_dir / f"part_{timestep}.npz") as snapshot:
                arrays = {key: snapshot[key].copy() for key in snapshot.files}
            arrays["custom_diagnostic"] = np.array([timestep, timestep + 10], dtype=np.int16)
            arrays["f_wall"] = np.full((1, 2), timestep, dtype=np.float32)
            np.savez(shot_dir / f"part_{timestep}.npz", **arrays)

        final_dir = compactor._pack_one_shot(
            shot_dir,
            run_dir / "hdf5-packed",
            staging_root,
            expected_device=None,
            replace_source=True,
            keep_staging=False,
            compression=None,
            staging_reserve_bytes=0,
            hdd_reserve_bytes=0,
        )

        self.assertFalse(shot_dir.exists())
        self.assertTrue((final_dir / "grid.txt").is_file())
        self.assertTrue((final_dir / "param_state.npz").is_file())
        self.assertFalse((staging_root / "shot_000").exists())
        with __import__("h5py").File(final_dir / "particles.h5", "r") as archive:
            self.assertEqual(archive.attrs["complete"], True)
            self.assertEqual(archive["custom_diagnostic"][1].tolist(), [1, 11])
            self.assertEqual(archive["f_wall"].shape[0], 2)
        with __import__("h5py").File(final_dir / "concentration.h5", "r") as archive:
            self.assertEqual(archive["concentration"][1].shape, (3, 4))

        events, summary = analysis.analyze(final_dir, maze_dir)
        self.assertEqual([row["decision"] for row in events], ["correct"])
        self.assertEqual(summary[0]["success_fraction"], 1.0)

    def test_insufficient_ssd_space_leaves_hdd_source_untouched(self):
        run_dir = self.root / "run_c0000"
        shot_dir = run_dir / "shot_000"
        shot_dir.mkdir(parents=True)
        staging_root = self.root / "ssd"
        staging_root.mkdir()
        _write_snapshot(shot_dir, 0, 0.0, [(1, 1)])
        np.savez(shot_dir / "conc_0.npz", timestep=0, simulation_time=0.0, concentration=np.zeros((2, 2)))

        with self.assertRaises(OSError):
            compactor._pack_one_shot(
                shot_dir,
                run_dir / "hdf5-packed",
                staging_root,
                expected_device=None,
                replace_source=True,
                keep_staging=False,
                compression=None,
                staging_reserve_bytes=10**15,
                hdd_reserve_bytes=0,
            )

        self.assertTrue(shot_dir.is_dir())
        self.assertFalse((run_dir / "hdf5-packed" / "shot_000").exists())

    def test_copyback_failure_restores_original_hdd_shot(self):
        run_dir = self.root / "run_c0000"
        shot_dir = run_dir / "shot_000"
        shot_dir.mkdir(parents=True)
        staging_root = self.root / "ssd"
        staging_root.mkdir()
        _write_snapshot(shot_dir, 0, 0.0, [(1, 1)])
        np.savez(shot_dir / "conc_0.npz", timestep=0, simulation_time=0.0, concentration=np.zeros((2, 2)))

        with patch.object(compactor, "_copy_back_verified", side_effect=OSError("simulated HDD failure")):
            with self.assertRaisesRegex(OSError, "simulated HDD failure"):
                compactor._pack_one_shot(
                    shot_dir,
                    run_dir / "hdf5-packed",
                    staging_root,
                    expected_device=None,
                    replace_source=True,
                    keep_staging=True,
                    compression=None,
                    staging_reserve_bytes=0,
                    hdd_reserve_bytes=0,
                )

        self.assertTrue((shot_dir / "part_0.npz").is_file())
        self.assertTrue((shot_dir / "conc_0.npz").is_file())
        self.assertTrue((staging_root / "shot_000" / "source" / "part_0.npz").is_file())

    def test_birth_time_bins_average_particle_fractions_across_shots(self):
        bins = analysis.BirthTimeBins(0.01)
        bins.add(
            [
                {"shot": "shot_000", "birth_time": 6.0, "success_fraction": 0.9},
                {"shot": "shot_001", "birth_time": 6.0, "success_fraction": 0.8},
            ]
        )
        bins.add(
            [
                {"shot": "shot_002", "birth_time": 6.004, "success_fraction": 0.6},
                {"shot": "shot_002", "birth_time": 6.02, "success_fraction": 0.25},
            ]
        )

        summaries = bins.rows()
        first_bin = summaries[0]

        self.assertEqual(first_bin["particle_count"], 3)
        self.assertEqual(first_bin["shot_count"], 3)
        self.assertAlmostEqual(first_bin["mean_success_fraction"], (0.9 + 0.8 + 0.6) / 3)
        self.assertEqual(summaries[1]["particle_count"], 1)

    def test_run_directory_cli_averages_all_shot_particle_fractions(self):
        run_dir = self.root / "series" / "run_c0000"
        maze_dir = self.root / "maze"
        output_dir = self.root / "results"
        _maze_artifacts(maze_dir)
        for shot, terminal_col in ((0, 2), (1, 3)):
            shot_dir = run_dir / f"shot_{shot:03d}"
            shot_dir.mkdir(parents=True)
            np.savez(
                shot_dir / "param_state.npz",
                birth_times=np.array([0.0]),
                simulation_time=1.0,
                resume_step=1,
            )
            _write_snapshot(shot_dir, 0, 0.0, [(1, 1)])
            _write_snapshot(shot_dir, 1, 1.0, [(terminal_col, 1)])

        with patch(
            "sys.argv",
            [
                "particle_decision_analysis.py",
                "--data-dir", str(run_dir),
                "--maze-analysis-dir", str(maze_dir),
                "--output-dir", str(output_dir),
                "--birth-bin-width", "0.01",
            ],
        ):
            analysis.main()

        with (output_dir / "birth_time_summary.csv").open(encoding="utf-8") as result_file:
            summary = next(csv.DictReader(result_file))
        with (output_dir / "particle_success.csv").open(encoding="utf-8") as particle_file:
            particles = list(csv.DictReader(particle_file))

        self.assertEqual(len(particles), 2)
        self.assertEqual(summary["particle_count"], "2")
        self.assertEqual(summary["shot_count"], "2")
        self.assertEqual(float(summary["mean_success_fraction"]), 0.5)
        self.assertTrue((output_dir / "success_vs_birth_time.png").is_file())


if __name__ == "__main__":
    unittest.main()