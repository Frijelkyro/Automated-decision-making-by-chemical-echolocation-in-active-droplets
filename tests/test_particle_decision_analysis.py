import math
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from scripts import particle_decision_analysis as analysis


def _snapshot(tmp_path, filename, timestep, simulation_time, positions):
    positions = np.asarray(positions, dtype=float)
    path = tmp_path / filename
    np.savez(
        path,
        timestep=timestep,
        simulation_time=simulation_time,
        x=positions[:, 0],
        y=positions[:, 1],
        vx=np.zeros(len(positions)),
        vy=np.zeros(len(positions)),
    )
    return analysis.load_particle_snapshot(path)


def _zones():
    before = np.zeros((5, 5), dtype=bool)
    correct = np.zeros_like(before)
    wrong = np.zeros_like(before)
    before[3, 1] = True
    correct[3, 2] = True
    wrong[3, 3] = True
    return {"J01": {"before": before, "correct": correct, "wrong": wrong}}


class ParticleDecisionAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_discover_shots_follows_grid_order_and_filters(self):
        config = {
            "series": {"name": "demo", "data_root": "data"},
            "regimes": [
                {
                    "name": "r1",
                    "shots": 2,
                    "grid": {"first": [10, 20], "second": [1, 2]},
                }
            ],
        }
        for config_index in range(4):
            for shot in range(2):
                path = self.tmp_path / "data" / "demo" / f"r1_c{config_index:04d}" / f"shot_{shot:03d}"
                path.mkdir(parents=True)

        all_shots = analysis.discover_shots(config, self.tmp_path)
        found = analysis.discover_shots(config, self.tmp_path, configurations=[2], shots=[1])

        self.assertEqual([shot.config_index for shot in all_shots], [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual([(shot.config_index, shot.shot) for shot in found], [(2, 1)])
        self.assertEqual(found[0].data_dir, self.tmp_path / "data" / "demo" / "r1_c0002" / "shot_001")

    def test_snapshots_sort_by_physical_time_then_timestep(self):
        _snapshot(self.tmp_path, "part_100.npz", 100, 2.0, [(0, 0)])
        _snapshot(self.tmp_path, "part_300.npz", 300, 1.0, [(0, 0)])
        _snapshot(self.tmp_path, "part_200.npz", 200, 1.0, [(0, 0)])

        snapshots = analysis.load_particle_snapshots(self.tmp_path, include_forces=False)

        self.assertEqual([(item.simulation_time, item.timestep) for item in snapshots], [(1.0, 200), (1.0, 300), (2.0, 100)])

    def test_shot_stream_restarts_in_stored_physical_time_order(self):
        _snapshot(self.tmp_path, "part_0.npz", 0, 0.0, [(1, 1)])
        _snapshot(self.tmp_path, "part_50.npz", 50, 2.0, [(3, 1)])
        _snapshot(self.tmp_path, "part_100.npz", 100, 1.0, [(2, 1)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = analysis.evaluate_shot_decisions(
                self.tmp_path,
                _zones(),
                (5, 5),
                1.0,
            )[0]

        self.assertEqual(row["successes"], 1)
        self.assertEqual(row["failures"], 0)


    def test_coordinate_mapping_flips_y_to_top_origin_row(self):
        self.assertEqual(analysis.map_sim_to_grid((2.0, 1.0), (5, 5), 1.0), (3, 2))
        self.assertEqual(analysis.map_sim_to_grid((0.0, 0.0), (5, 5), 1.0), (4, 0))

    def test_state_transitions_count_success_and_failure_once(self):
        snapshots = [
            _snapshot(self.tmp_path, "part_0.npz", 0, 0.0, [(1, 1), (1, 1)]),
            _snapshot(self.tmp_path, "part_1.npz", 1, 1.0, [(2, 1), (1, 1)]),
            _snapshot(self.tmp_path, "part_2.npz", 2, 2.0, [(2, 2), (3, 1)]),
            _snapshot(self.tmp_path, "part_3.npz", 3, 3.0, [(2, 2), (3, 1)]),
            _snapshot(self.tmp_path, "part_4.npz", 4, 4.0, [(2, 2), (3, 1)]),
        ]

        rows = analysis.evaluate_decisions(snapshots, _zones(), (5, 5), 1.0)
        by_id = {row["particle_id"]: row for row in rows}

        self.assertEqual(by_id[0]["successes"], 1)
        self.assertEqual(by_id[0]["failures"], 0)
        self.assertEqual(by_id[0]["success_fraction"], 1.0)
        self.assertEqual(by_id[1]["successes"], 0)
        self.assertEqual(by_id[1]["failures"], 1)
        self.assertEqual(by_id[1]["success_fraction"], 0.0)

    def test_pending_attempt_and_no_attempt_are_separate(self):
        snapshots = [
            _snapshot(self.tmp_path, "part_0.npz", 0, 0.0, [(1, 1), (4, 1)]),
            _snapshot(self.tmp_path, "part_1.npz", 1, 1.0, [(1, 1), (4, 1)]),
        ]

        rows = analysis.evaluate_decisions(snapshots, _zones(), (5, 5), 1.0)
        by_id = {row["particle_id"]: row for row in rows}

        self.assertEqual(by_id[0]["pending_attempts"], 1)
        self.assertEqual(by_id[0]["no_attempt"], 0)
        self.assertTrue(math.isnan(by_id[0]["success_fraction"]))
        self.assertEqual(by_id[1]["pending_attempts"], 0)
        self.assertEqual(by_id[1]["no_attempt"], 1)
        self.assertTrue(math.isnan(by_id[1]["success_fraction"]))

    def test_correct_wrong_overlap_is_ambiguous_and_does_not_close(self):
        zones = _zones()
        zones["J01"]["wrong"][3, 2] = True
        snapshots = [
            _snapshot(self.tmp_path, "part_0.npz", 0, 0.0, [(1, 1)]),
            _snapshot(self.tmp_path, "part_1.npz", 1, 1.0, [(2, 1)]),
            _snapshot(self.tmp_path, "part_2.npz", 2, 2.0, [(2, 2)]),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = analysis.evaluate_decisions(snapshots, zones, (5, 5), 1.0)[0]

        self.assertEqual(row["pending_attempts"], 1)
        self.assertEqual(row["successes"], row["failures"], 0)

    def test_malformed_unused_neighbors_are_ignored_for_metadata(self):
        text = "dx = 1.0\ngrid_shape = [5, 5]\n[[junctions]]\nneighbors = [(np.int64(1), np.int64(2))]"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metadata = analysis._load_metadata_text(text, self.tmp_path / "maze_metadata.toml")

        self.assertEqual(metadata["grid_shape"], [5, 5])
        self.assertTrue(caught)

    def test_aggregation_distinguishes_pooled_and_equal_shot(self):
        rows = []
        for shot, successes, failures in ((0, 9, 1), (1, 0, 1)):
            for index in range(successes + failures):
                rows.append(
                    {
                        "series_name": "s",
                        "regime_name": "r",
                        "config_index": 0,
                        "run_id": "r_c0000",
                        "shot": shot,
                        "junction_id": "J01",
                        "birth_time": 2.0,
                        "successes": int(index < successes),
                        "failures": int(index >= successes),
                        "pending_attempts": 0,
                        "no_attempt": 0,
                    }
                )

        pooled = analysis.aggregate_birth_times(rows, averaging="pooled-count")[0]
        equal = analysis.aggregate_birth_times(rows, averaging="equal-per-shot")[0]
        pooled_stream = analysis.BirthTimeAggregator("pooled-count", None, 0.0)
        equal_stream = analysis.BirthTimeAggregator("equal-per-shot", None, 0.0)
        for shot in (0, 1):
            shot_rows = [row for row in rows if row["shot"] == shot]
            pooled_stream.add_shot(shot_rows)
            equal_stream.add_shot(shot_rows)
        pooled_incremental = pooled_stream.rows()[0]
        equal_incremental = equal_stream.rows()[0]

        self.assertAlmostEqual(pooled["success_fraction"], 9 / 11)
        self.assertEqual(equal["success_fraction"], 0.45)
        self.assertAlmostEqual(equal["pooled_count_success_fraction"], 9 / 11)
        self.assertEqual(equal["shots_with_completed_attempts"], 2)
        self.assertAlmostEqual(pooled_incremental["success_fraction"], pooled["success_fraction"])
        self.assertAlmostEqual(equal_incremental["success_fraction"], equal["success_fraction"])

    def test_zero_completed_attempts_produces_nan(self):
        rows = [
            {
                "series_name": "s",
                "regime_name": "r",
                "config_index": 0,
                "run_id": "r_c0000",
                "shot": 0,
                "junction_id": "J01",
                "birth_time": 2.0,
                "successes": 0,
                "failures": 0,
                "pending_attempts": 1,
                "no_attempt": 0,
            }
        ]

        self.assertTrue(math.isnan(analysis.summarize_junctions(rows)[0]["success_fraction"]))
        self.assertTrue(math.isnan(analysis.aggregate_birth_times(rows)[0]["success_fraction"]))


if __name__ == "__main__":
    unittest.main()