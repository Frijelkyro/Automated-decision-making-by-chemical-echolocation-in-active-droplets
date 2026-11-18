import numpy as np

from list_of_functions import (
    build_free_cell_index_map,
    expand_concentration,
    free_cell_linear_index,
    write_concentration,
)
from resume_simulation import resume_simulation_from_file


def test_concentration_snapshot_stores_and_restores_free_cells(tmp_path):
    maze = np.array([[1, 0, 1], [0, 1, 1]])
    free_cell_index_map = build_free_cell_index_map(maze)
    concentration = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    prefix = str(tmp_path / "conc")

    write_concentration(
        prefix,
        concentration,
        0,
        0.0,
        1,
        free_cell_index_map=free_cell_index_map,
    )

    with np.load(f"{prefix}_0.npz") as data:
        stored = data["concentration"]
        grid_shape = tuple(data["grid_shape"])

    assert stored.shape == (4,)
    assert grid_shape == maze.shape
    assert free_cell_linear_index(1, 1, free_cell_index_map) == 2
    restored = expand_concentration(stored, free_cell_index_map)
    assert np.array_equal(restored[maze != 0], concentration[0][maze != 0])
    assert np.all(restored[maze == 0] == 0)
    legacy_grid = np.ones(maze.shape, dtype=np.float32)
    assert np.array_equal(
        expand_concentration(legacy_grid, free_cell_index_map), legacy_grid
    )


def test_write_concentration_prunes_snapshots_outside_timestep_window(tmp_path):
    prefix = str(tmp_path / "conc")
    concentration = np.ones((1, 2, 2), dtype=np.float32)

    for timestep in (0, 100, 200):
        write_concentration(prefix, concentration, timestep, timestep, 1)

    write_concentration(
        prefix,
        concentration,
        300,
        300,
        1,
        retention_steps=200,
    )

    assert sorted(path.name for path in tmp_path.glob("conc_*.npz")) == [
        "conc_100.npz",
        "conc_200.npz",
        "conc_300.npz",
    ]


def test_resume_uses_checkpoint_concentration_without_heatmap_file(tmp_path):
    decision_file = tmp_path / "junction_decisions.csv"
    decision_file.write_text(
        "timestep,simulation_time,particle_id,junction_id,correct\n"
        "8,0.8,0,J01,1\n"
        "10,1.0,0,J01,0\n"
        "12,1.2,0,J01,1\n",
        encoding="utf-8",
    )
    (tmp_path / "param_checkpoint.txt").write_text(
        "num_particles: 1\n"
        "file_prefix_conc: conc\n"
        "file_prefix_part: part\n"
        f"junction_decisions_filename: {decision_file}\n"
        "dt: 0.1\n"
    )
    np.savez_compressed(
        tmp_path / "param_state_checkpoint.npz",
        birth_times=np.array([0.0]),
        active_mask=np.array([True]),
        dead_tracker=np.array([False]),
        exit_trigger_time=np.array([np.inf]),
        resume_step=10,
        simulation_time=1.0,
        concentration=np.full((2, 2), 7.0, dtype=np.float32),
        junction_ids=np.array(["J01"]),
        junction_pending=np.array([[True]]),
        junction_decided=np.array([[False]]),
        junction_decision_events=np.array(
            [[9, 0.9, 0, 0, 1], [10, 1.0, 0, 0, 0], [11, 1.1, 0, 0, 0]],
            dtype=float,
        ),
    )
    np.savez(
        tmp_path / "part_10.npz",
        x=np.array([1.0]),
        y=np.array([2.0]),
        vx=np.array([0.0]),
        vy=np.array([0.0]),
        theta=np.array([0.0]),
        omega=np.array([0.0]),
    )

    restored = resume_simulation_from_file(
        str(tmp_path),
        str(tmp_path / "param.txt"),
        np.ones((2, 2), dtype=np.int8),
        3,
    )

    assert np.array_equal(restored[0][0], np.full((2, 2), 7.0))
    assert restored[9] == 10
    assert restored[12]["junction_pending"].tolist() == [[True]]
    assert restored[12]["junction_ids"].tolist() == ["J01"]
    assert restored[12]["junction_decision_events"] == [(9, 0.9, 0, 0, 1)]
    assert decision_file.read_text(encoding="utf-8").count("\n") == 3