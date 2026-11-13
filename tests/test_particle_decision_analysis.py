import math
import csv
import sys
import tempfile
import tomllib
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import maze_skeleton_path as maze_analysis
from scripts import particle_decision_analysis as analysis
from scripts import plot_path_concentration as concentration_plot


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

    def test_compact_zone_masks_fill_local_corridor_within_five_by_five(self):
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

        zones = maze_analysis.build_zones(junctions, graph, skeleton, free, path, zone_extent="compact")

        for cells in zones["J01"].values():
            self.assertTrue(all(1 <= row <= 5 and 5 <= col <= 9 for row, col in cells))
        self.assertIn((2, 5), zones["J01"]["before"])
        self.assertIn((2, 9), zones["J01"]["correct"])
        self.assertIn((1, 7), zones["J01"]["wrong"])

    def test_start_positions_load_from_maze_config_and_allow_overrides(self):
        config_path = self.root / "startpoints.toml"
        config_path.write_text(
            '[configurations.QJ01_config0]\nemitter_position = [5.0, 11.05]\n'
            'static_source_position = [100.2, 10.2]\n',
            encoding="utf-8",
        )

        configured = maze_analysis.load_start_positions(
            config_path, Path("QJ01.tsv"), config_name="QJ01_config0"
        )
        overridden = maze_analysis.load_start_positions(
            config_path,
            Path("QJ01.tsv"),
            emitter_override=[10, 4],
            goal_override=[110, 40],
            config_name="QJ01_config0",
        )

        self.assertEqual(configured, ((5.0, 11.05), (100.2, 10.2)))
        self.assertEqual(overridden, ((10.0, 4.0), (110.0, 40.0)))

    def test_shared_startpoint_profiles_match_simulation_initial_conditions(self):
        config_path = Path(__file__).resolve().parents[1] / "maze_startpoints_configuration.toml"
        profiles = tomllib.loads(config_path.read_text(encoding="utf-8"))["configurations"]

        self.assertEqual(profiles["QJ01_config0"]["emitter_position"], [5.0, 11.05])
        self.assertEqual(profiles["QJ01_config0"]["static_source_position"], [100.2, 10.2])
        self.assertEqual(profiles["QJ01_config0"]["exit_radius"], 12.0)
        self.assertEqual(
            profiles["Ran_maze_size_prop_to_droplet"]["emitter_position"], [2.1, 84.1]
        )
        self.assertEqual(profiles["Testrun"]["maze_file"], "different_mazes/Ran_maze_size_prop_to_droplet.tsv")
        self.assertEqual(profiles["Testrun"]["emitter_position"], [2.1, 14.8])
        self.assertEqual(profiles["Testrun"]["static_source_position"], [42.5, 10.5])

    def test_concentration_plot_paths_resolve_from_experiment_config(self):
        config_path = self.root / "series.toml"
        startpoints_path = self.root / "startpoints.toml"
        maze_path = self.root / "mazes" / "QJ01.tsv"
        config_path.write_text(
            f'[series]\nname = "demo"\ndata_root = "{self.root / "data"}"\n'
            f'output_root = "{self.root / "output"}"\n\n'
            '[defaults]\nmaze_config = "QJ01_config0"\n',
            encoding="utf-8",
        )
        startpoints_path.parent.mkdir(parents=True, exist_ok=True)
        startpoints_path.write_text(
            f'[configurations.QJ01_config0]\nmaze_file = "{maze_path}"\n',
            encoding="utf-8",
        )

        series_dir, maze_meta, output_dir, jobs, resolved_maze = concentration_plot.resolve_config_paths(
            config_path, startpoints_path
        )

        self.assertEqual(series_dir, self.root / "data" / "demo")
        self.assertEqual(maze_meta, self.root / "mazes" / "QJ01_analysis" / "maze_metadata.toml")
        self.assertEqual(output_dir, self.root / "output" / "demo" / "path_concentration")
        self.assertEqual(jobs, self.root / "output" / "demo" / "jobs.txt")
        self.assertEqual(resolved_maze, self.root / "mazes" / "QJ01.tsv")

    def test_concentration_plot_cli_runs_from_experiment_config(self):
        maze_path = self.root / "mazes" / "QJ01.tsv"
        maze_path.parent.mkdir(parents=True)
        maze = np.ones((5, 5), dtype=int)
        maze[2, 1:4] = 0
        np.savetxt(maze_path, maze, delimiter="\t", fmt="%d")
        shot_dir = self.root / "data" / "demo" / "custom_c0000" / "shot_000"
        shot_dir.mkdir(parents=True)
        np.savez(shot_dir / "conc_0.npz", concentration=np.ones((5, 5)), timestep=0)

        config_path = self.root / "series.toml"
        config_path.write_text(
            f'[series]\nname = "demo"\ndata_root = "{self.root / "data"}"\n'
            f'output_root = "{self.root / "output"}"\n\n'
            '[defaults]\nmaze_config = "QJ01_config0"\n',
            encoding="utf-8",
        )
        startpoints_path = self.root / "startpoints.toml"
        startpoints_path.write_text(
            f'[configurations.QJ01_config0]\nmaze_file = "{maze_path}"\n',
            encoding="utf-8",
        )
        metadata_path = self.root / "mazes" / "QJ01_analysis" / "maze_metadata.toml"
        metadata_path.parent.mkdir()
        metadata_path.write_text(
            'maze_file = "/old-host/old-project/different_mazes/QJ01.tsv"\n'
            'wall_value = 1\nemitter_grid = [2, 1]\n'
            'goal_grid = [2, 3]\npath_length = 3\n',
            encoding="utf-8",
        )

        with patch(
            "sys.argv",
            [
                "plot_path_concentration.py",
                "--config", str(config_path),
                "--startpoints-config", str(startpoints_path),
            ],
        ):
            concentration_plot.main()

        output = self.root / "output" / "demo" / "path_concentration" / "custom_c0000_shot000.png"
        self.assertTrue(output.is_file())

    def test_maze_cli_accepts_positions_and_zone_extent(self):
        args = maze_analysis.build_argument_parser().parse_args(
            [
                "--maze", "different_mazes/QJ01.tsv",
                "--maze-config", "QJ01_config0",
                "--emitter-position", "5.0", "11.05",
                "--static-source-position", "100.2", "10.2",
                "--zone-extent", "compact",
            ]
        )

        self.assertEqual(args.maze, Path("different_mazes/QJ01.tsv"))
        self.assertEqual(args.maze_config, "QJ01_config0")
        self.assertEqual(args.emitter_position, [5.0, 11.05])
        self.assertEqual(args.static_source_position, [100.2, 10.2])
        self.assertEqual(args.zone_extent, "compact")

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

    def test_experiment_config_cli_discovers_series_shots(self):
        series_dir = self.root / "data" / "demo"
        shot_dir = series_dir / "custom_c0000" / "shot_000"
        shot_dir.mkdir(parents=True)
        maze_dir = self.root / "different_mazes" / "QJ01_analysis"
        _maze_artifacts(maze_dir)
        np.savez(shot_dir / "param_state.npz", birth_times=np.array([0.0]), simulation_time=1.0)
        _write_snapshot(shot_dir, 0, 0.0, [(1, 1)])
        _write_snapshot(shot_dir, 1, 1.0, [(2, 1)])

        config_path = self.root / "series.toml"
        config_path.write_text(
            f'[series]\nname = "demo"\ndata_root = "{self.root / "data"}"\n'
            f'output_root = "{self.root / "output"}"\n\n'
            '[defaults]\nmaze_config = "QJ01_config0"\n',
            encoding="utf-8",
        )
        startpoints_path = self.root / "startpoints.toml"
        startpoints_path.write_text(
            f'[configurations.QJ01_config0]\nmaze_file = "{self.root / "different_mazes" / "QJ01.tsv"}"\n',
            encoding="utf-8",
        )

        with patch(
            "sys.argv",
            [
                "particle_decision_analysis.py",
                "--config", str(config_path),
                "--startpoints-config", str(startpoints_path),
            ],
        ):
            analysis.main()

        result_dir = self.root / "output" / "demo" / "decision_analysis"
        with (result_dir / "particle_success.csv").open(encoding="utf-8") as particle_file:
            particles = list(csv.DictReader(particle_file))
        self.assertEqual(len(particles), 1)
        self.assertEqual(particles[0]["run_id"], "custom_c0000")
        self.assertTrue((result_dir / "success_vs_birth_time.png").is_file())


if __name__ == "__main__":
    unittest.main()