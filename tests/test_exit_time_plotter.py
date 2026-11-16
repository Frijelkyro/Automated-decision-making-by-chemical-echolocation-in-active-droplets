import csv

from scripts.exit_time_plotter import main


def test_config_plots_per_run_and_series_exit_durations(tmp_path):
    series_dir = tmp_path / "output" / "demo"
    series_dir.mkdir(parents=True)
    config = tmp_path / "series.toml"
    config.write_text(
        f'[series]\nname = "demo"\ndata_root = "{tmp_path / "data"}"\n'
        f'output_root = "{tmp_path / "output"}"\n',
        encoding="utf-8",
    )
    aggregate = series_dir / "particle_success_all.csv"
    with aggregate.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["run_id", "shot", "birth_time", "exit_time", "exited"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"run_id": "run_c0000", "shot": "shot_000", "birth_time": 0.5, "exit_time": 4.5, "exited": "True"},
                {"run_id": "run_c0000", "shot": "shot_001", "birth_time": 1.0, "exit_time": 7.0, "exited": "1"},
                {"run_id": "run_c0000", "shot": "shot_001", "birth_time": 2.0, "exit_time": "inf", "exited": "False"},
                {"run_id": "other_c0000", "shot": "shot_000", "birth_time": 0.5, "exit_time": 2.5, "exited": "True"},
            ]
        )

    assert main(["--config", str(config), "--pattern", "run_*"]) == 0

    output_dir = series_dir / "exit_time"
    assert (output_dir / "run_c0000_exit_time_vs_birth_time.png").is_file()
    assert (output_dir / "all_runs_exit_time_vs_birth_time.png").is_file()
    with (output_dir / "exit_time_by_birth_time.csv").open(encoding="utf-8") as csv_file:
        summary = list(csv.DictReader(csv_file))
    assert len(summary) == 2
    assert summary[0]["scope"] == "run"
    assert summary[0]["particle_count"] == "2"
    assert float(summary[0]["mean_exit_duration"]) == 5.0
    assert summary[1]["scope"] == "all"
