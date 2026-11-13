import csv
import math
import numpy as np
import pytest

from scripts.particle_success_fraction_analysis import (
    _configuration_plot_name,
    _maze_coordinate_to_sim,
    _parameters_from_grid,
    _read_jobs,
    _summarize_birth_times,
    load_maze_oriented,
    centerline_path,
    index_shot,
    main,
    aggregate,
    plot_configuration,
    score_shot,
)


@pytest.fixture
def simple_maze_file(tmp_path):
    raw = np.zeros((100, 100), dtype=int)
    raw[0, :] = 1
    raw[-1, :] = 1
    raw[:, 0] = 1
    raw[:, -1] = 1
    file_path = tmp_path / "maze.tsv"
    np.savetxt(file_path, raw, delimiter="\t", fmt="%d")
    return file_path


def test_load_maze_oriented_and_path(simple_maze_file):
    start = (2, 2)
    end = (75, 75)
    maze = load_maze_oriented(simple_maze_file, start=start, end=end)
    assert maze.shape == (100, 100)
    assert maze[start] == 1
    assert maze[end] == 1
    path = centerline_path(maze, start, end, wall_weight=30.0, wall_power=2.0)
    assert len(path) > 0


def test_index_shot_uses_simulation_time(tmp_path):
    shot = tmp_path / "shot_000"
    shot.mkdir()
    np.savez(shot / "part_000.npz", simulation_time=1.0, timestep=100, x=np.array([2.0]), y=np.array([81.0]))
    np.savez(shot / "part_010.npz", simulation_time=3.0, timestep=110, x=np.array([2.0]), y=np.array([81.0]))
    idx = index_shot(shot)
    assert [entry["simulation_time"] for entry in idx] == [1.0, 3.0]
    assert [entry["timestep"] for entry in idx] == [100, 110]


def test_score_shot_handles_nonuniform_and_exit_inside_interval(tmp_path):
    shot = tmp_path / "shot_000"
    shot.mkdir()
    start = (2, 2)
    end = (8, 8)
    raw = np.zeros((20, 20), dtype=int)
    raw[0, :] = 1
    raw[-1, :] = 1
    raw[:, 0] = 1
    raw[:, -1] = 1
    path = np.array([start, (3, 2), (5, 4), end], dtype=float)
    file_path = tmp_path / "maze.tsv"
    np.savetxt(file_path, raw, delimiter="\t", fmt="%d")

    np.savez(shot / "part_000.npz", simulation_time=1.0, timestep=1, x=np.array([2.0]), y=np.array([2.0]))
    np.savez(shot / "part_001.npz", simulation_time=3.0, timestep=2, x=np.array([18.0]), y=np.array([18.0]))
    np.savez(shot / "part_002.npz", simulation_time=5.0, timestep=3, x=np.array([18.0]), y=np.array([18.0]))

    np.savez(shot / "param_state.npz", birth_times=np.array([0.0]), exit_trigger_time=np.array([4.0]))

    result = score_shot(
        shot,
        maze=load_maze_oriented(file_path, start=start, end=end),
        path=path,
        tolerance=1.0,
        logger=None,
    )
    assert result["rows"][0]["success_fraction"] > 0.0
    assert result["rows"][0]["exited"] is True


def test_score_shot_never_exited_and_birth_after_final_snapshot(tmp_path):
    shot = tmp_path / "shot_000"
    shot.mkdir()
    start = (2, 2)
    end = (8, 8)
    raw = np.zeros((20, 20), dtype=int)
    raw[0, :] = 1
    raw[-1, :] = 1
    raw[:, 0] = 1
    raw[:, -1] = 1
    path = np.array([start, (3, 2), end], dtype=float)
    file_path = tmp_path / "maze.tsv"
    np.savetxt(file_path, raw, delimiter="\t", fmt="%d")

    np.savez(shot / "part_000.npz", simulation_time=1.0, timestep=1, x=np.array([2.0]), y=np.array([2.0]))
    np.savez(shot / "part_001.npz", simulation_time=3.0, timestep=2, x=np.array([2.0]), y=np.array([2.0]))

    np.savez(shot / "param_state.npz", birth_times=np.array([5.0]), exit_trigger_time=np.array([np.inf]))
    result = score_shot(
        shot,
        maze=load_maze_oriented(file_path, start=start, end=end),
        path=path,
        tolerance=1.0,
        logger=None,
    )
    row = result["rows"][0]
    assert math.isnan(row["success_fraction"])


def test_score_shot_case3_counts_and_nan_endpoints(tmp_path):
    shot = tmp_path / "shot_000"
    shot.mkdir()
    start = (2, 2)
    end = (8, 8)
    raw = np.zeros((20, 20), dtype=int)
    raw[0, :] = 1
    raw[-1, :] = 1
    raw[:, 0] = 1
    raw[:, -1] = 1
    path = np.array([start, (3, 2), (4, 2)], dtype=float)
    file_path = tmp_path / "maze.tsv"
    np.savetxt(file_path, raw, delimiter="\t", fmt="%d")
    np.savez(shot / "part_000.npz", simulation_time=1.0, timestep=1, x=np.array([np.nan]), y=np.array([np.nan]))
    np.savez(shot / "part_001.npz", simulation_time=3.0, timestep=2, x=np.array([2.0]), y=np.array([2.0]))
    np.savez(shot / "param_state.npz", birth_times=np.array([0.0]), exit_trigger_time=np.array([5.0]))
    result = score_shot(
        shot,
        maze=load_maze_oriented(file_path, start=start, end=end),
        path=path,
        tolerance=1.0,
        logger=None,
    )
    assert result["diagnostics"]["case3_count"] >= 0


def test_score_shot_detects_particle_length_mismatch(tmp_path):
    shot = tmp_path / "shot_000"
    shot.mkdir()
    np.savez(shot / "part_000.npz", simulation_time=1.0, timestep=1, x=np.array([1.0, 2.0]), y=np.array([1.0]))
    np.savez(shot / "param_state.npz", birth_times=np.array([0.0]), exit_trigger_time=np.array([np.inf]))
    with pytest.raises(ValueError):
        score_shot(shot, maze=np.ones((5, 5)), path=np.array([[0.0, 0.0]]), tolerance=1.0)


def test_aggregate_and_plot_average_success_by_birth_time(tmp_path):
    series = tmp_path / "data" / "batch"
    shot_rows = {}
    for shot_index, fractions in enumerate(((0.6, 0.8), (0.8, 0.9))):
        shot = series / "run_c0000" / f"shot_{shot_index:03d}"
        shot.mkdir(parents=True)
        rows = [
            {"particle_id": 0, "birth_time": 0.0, "success_fraction": fractions[0], "traversal_time": 1.0, "n_samples": 2, "exit_time": 2.0, "exited": True},
            {"particle_id": 1, "birth_time": 200.0, "success_fraction": fractions[1], "traversal_time": 1.0, "n_samples": 2, "exit_time": 2.0, "exited": True},
        ]
        with (shot / "particle_success.txt").open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(
                    f"{row['particle_id']} {row['success_fraction']:.6f} 1.000000 2 "
                    f"{row['birth_time']:.6f} 2.000000 1\n"
                )
        shot_rows[shot_index] = rows

    output = tmp_path / "results"
    output.mkdir()
    aggregate(series, output)
    with (output / "particle_success_by_birth_time.csv").open(encoding="utf-8") as file:
        summary = list(csv.DictReader(file))

    assert [float(row["birth_time"]) for row in summary] == [0.5, 200.5]
    assert [float(row["mean_success_fraction"]) for row in summary] == pytest.approx([0.7, 0.85])
    assert [float(row["standard_deviation_success_fraction"]) for row in summary] == pytest.approx([np.std([0.6, 0.8], ddof=1), np.std([0.8, 0.9], ddof=1)])
    assert [int(row["shot_count"]) for row in summary] == [2, 2]

    plot_path = output / "success_fraction.png"
    plot_configuration("run_c0000", shot_rows, plot_path, show_individual_points=True)
    assert plot_path.is_file()


def test_birth_time_bins_use_width_and_report_standard_deviation():
    summary = _summarize_birth_times(
        [
            {"birth_time": 0.1, "success_fraction": 0.4, "shot": "shot_0"},
            {"birth_time": 0.8, "success_fraction": 0.6, "shot": "shot_1"},
            {"birth_time": 1.2, "success_fraction": 0.9, "shot": "shot_1"},
        ],
        bin_width=1.0,
    )

    assert [row["birth_time_bin"] for row in summary] == [0, 1]
    assert [row["birth_time"] for row in summary] == [0.5, 1.5]
    assert summary[0]["mean_success_fraction"] == pytest.approx(0.5)
    assert summary[0]["standard_deviation_success_fraction"] == pytest.approx(np.std([0.4, 0.6], ddof=1))
    assert math.isnan(summary[1]["standard_deviation_success_fraction"])
    with pytest.raises(ValueError):
        _summarize_birth_times([], bin_width=0.0)


def test_parameter_metadata_controls_plot_name_and_maze_coordinate(tmp_path):
    jobs = tmp_path / "jobs.txt"
    jobs.write_text(
        "python run.py --grim-reaper-delay 6.0 --emission-rate 1.0 "
        "--desired-time 400 --run-id emission_1_c0000 --shot 0\n",
        encoding="utf-8",
    )
    parameters = _read_jobs(jobs)["emission_1_c0000"]
    maze_path = tmp_path / "example_maze.tsv"

    assert _configuration_plot_name("emission_1_c0000", parameters, maze_path) == (
        "rt6-0_er1-0_400s_example_maze.png"
    )
    assert _maze_coordinate_to_sim((27, 3), (110, 99)) == (3, 82)

    config = {
        "defaults": {"grim_reaper_delay": 6.0, "desired_time": 400},
        "regimes": [{"name": "emission_1", "grid": {"emission_rate": [1.0]}}],
    }
    assert _parameters_from_grid(config)["emission_1_c0000"]["emission_rate"] == "1.0"


def test_main_selects_maze_metadata_and_saves_parameter_named_pngs(tmp_path, monkeypatch):
    maze_dir = tmp_path / "mazes"
    maze_dir.mkdir()
    raw_maze = np.ones((7, 7), dtype=int)
    raw_maze[1:6, 1:6] = 0
    maze_file = maze_dir / "maze.tsv"
    np.savetxt(maze_file, raw_maze, delimiter="\t", fmt="%d")
    metadata_file = maze_dir / "maze_metadata.toml"
    metadata_file.write_text(
        'maze_file = "maze.tsv"\nwall_value = 1\nemitter_grid = [1, 1]\ngoal_grid = [5, 5]\n',
        encoding="utf-8",
    )

    shot = tmp_path / "data" / "series" / "run_c0000" / "shot_000"
    shot.mkdir(parents=True)
    np.savez(shot / "param_state.npz", birth_times=np.array([0.0]), exit_trigger_time=np.array([1.0]))
    for timestep, simulation_time, position in ((0, 0.0, (1.0, 5.0)), (1, 1.0, (5.0, 1.0))):
        np.savez(
            shot / f"part_{timestep:03d}.npz",
            timestep=timestep,
            simulation_time=simulation_time,
            x=np.array([position[0]]),
            y=np.array([position[1]]),
        )

    jobs = tmp_path / "output" / "series" / "jobs.txt"
    jobs.parent.mkdir(parents=True)
    jobs.write_text(
        "python run.py --grim-reaper-delay 6.0 --emission-rate 1.0 "
        "--desired-time 10 --run-id run_c0000 --shot 0\n",
        encoding="utf-8",
    )
    png_dir = tmp_path / "all_pngs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "particle_success_fraction_analysis.py",
            "--series-dir", str(shot.parents[1]),
            "--maze-meta", str(metadata_file),
            "--out-dir", str(png_dir),
            "--birth-bin-width", "2.0",
            "--show-individual-points",
        ],
    )

    assert main() == 0
    assert (png_dir / "rt6-0_er1-0_10s_maze.png").is_file()
    assert (png_dir / "maze_maze_path_tolerance.png").is_file()
    with (tmp_path / "data" / "output" / "series" / "particle_success_by_birth_time.csv").open(encoding="utf-8") as file:
        summary = list(csv.DictReader(file))
    assert len(summary) == 1
    assert float(summary[0]["birth_time"]) == 1.0
    assert float(summary[0]["mean_success_fraction"]) == 1.0


def test_main_config_selects_maze_profile_without_metadata(tmp_path, monkeypatch):
    maze_file = tmp_path / "mazes" / "QJ01.tsv"
    maze_file.parent.mkdir()
    raw_maze = np.ones((7, 7), dtype=int)
    raw_maze[1:6, 1:6] = 0
    np.savetxt(maze_file, raw_maze, delimiter="\t", fmt="%d")
    series_root = tmp_path / "data" / "qj01"
    series_root.mkdir(parents=True)

    config = tmp_path / "series.toml"
    config.write_text(
        f'[series]\nname = "qj01"\ndata_root = "{tmp_path / "data"}"\n'
        f'output_root = "{tmp_path / "output"}"\n\n'
        '[defaults]\nmaze_config = "QJ01_config0"\n',
        encoding="utf-8",
    )
    startpoints = tmp_path / "startpoints.toml"
    startpoints.write_text(
        f'[configurations.QJ01_config0]\nmaze_file = "{maze_file}"\n'
        'emitter_position = [1, 1]\nstatic_source_position = [5, 5]\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "particle_success_fraction_analysis.py",
            "--config", str(config),
            "--startpoints-config", str(startpoints),
            "--path-only",
        ],
    )

    assert main() == 0
    assert (tmp_path / "output" / "qj01" / "maze_QJ01_path_tolerance.png").is_file()
