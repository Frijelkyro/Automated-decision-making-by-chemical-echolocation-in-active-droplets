Copyright (C) 2025 Aritra K. Mukhopadhyay.

# README.md

* Project: Automated decision-making by chemical echolocation in active droplets
* Authors: Aritra K. Mukhopadhyay, Ran Niu, Linhui Fu, Kai Feng, Christopher Fujta, Qiang Zhao, Jinping Qu, and Benno Liebchen.
* Description: This project simulates a scalar chemical field coupled to particle motion in maze-shaped domains. Active droplets use chemical sensing, self-propulsion, interactions, and wall avoidance to navigate autonomously.
* Contact: aritra.mukhopadhyay@pkm.tu-darmstadt.de, benno.liebchen@pkm.tu-darmstadt.de

## What this project does

- Runs time-dependent simulations of a scalar chemical field coupled to particle motion.
- Supports moving and static point sources, optional advection, chemotaxis, particle-particle interactions, wall avoidance, and self-propulsion.
- Writes concentration and particle `.npz` snapshots, checkpoint bundles, parameter/grid metadata, optional junction decision logs, and rendered MP4 videos.

## File descriptions

- `maze_cluster_script.py` - main simulation driver and CLI.
- `list_of_functions.py` - numerical solver, force models, snapshot writers, and checkpoint handling.
- `maze_functions.py` - maze generators and TSV loaders.
- `resume_simulation.py` - checkpoint restoration and snapshot cleanup.
- `storage.py` - compact maze-aware concentration storage.
- `junction_tracking.py` - junction gate tracking and decision-log flushing.
- `video_maker_testrun.py` - renders one shot directory to MP4.
- `scripts/render_all_series_videos.py` - renders every available shot in a simulation series.
- `scripts/view_decisions.py` - prints junction decision-log rows.

## Requirements

Python 3.11+ is required because the scripts use the standard-library `tomllib` TOML parser. Install the scientific dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The key packages are NumPy, Matplotlib, SciPy, scikit-image, pandas, and tqdm. `ffmpeg` must be available on `PATH` to save MP4 videos.

## Running a simulation

Run a test shot:

```bash
python maze_cluster_script.py --seed 12345 --test-run true \
  --desired-time 20 --dt 0.0005 --n-steps 100 \
  --num-particles 10 --emission-rate 0.5 \
  --data-dir data/test-shot --output-dir output/test-shot
```

The same wrapper accepts experiment and directory overrides:

```bash
python maze_cluster_script.py --beta -8 --emission-rate 0.5 --dt 0.0005 \
  --grim-reaper-delay 6 --desired-time 800 --part-every 100 --conc-every 200 \
  --n-steps 100 --test-run false --resume false --run-id beta_m8_e05 --shot 0 \
  --seed 12345 --maze-config Ran_maze_size_prop_to_droplet \
  --data-dir data/beta_m8_e05/shot_000 \
  --output-dir output/beta_m8_e05/shot_000
```

### CLI flags

- `--seed <int>` - initialize NumPy and Python RNG state before the simulation; default `0`.
- `--part-every <steps>` - write particle snapshots at this global-step frequency.
- `--conc-every <steps>` - write concentration snapshots at this frequency; it must be a positive multiple of `--part-every`.
- `--write-every <steps>` - hidden compatibility alias for snapshot frequency.
- `--resume true|false` - resume from `ckpt_previous.npz`.
- `--maze-config <name>` - select a profile in `maze_startpoints_configuration.toml`.
- `--startpoints-config <path>` - override the startpoint configuration file.
- `--track-decisions` - enable junction tracking.
- `--zones-file <path>` - override the maze profile's `zones_file`.
- `--decision-flush-loops <n>` - flush decision-log buffers every N solver loops.
- `--test-run true` - force the `Testrun` maze/startpoint profile.

Experiment defaults and parameter grids live in `experiments/simulation_series_parameters.toml`. Generate job commands with:

```bash
python scripts/make_jobs.py
```

Launch generated jobs with GNU Parallel:

```bash
python scripts/run_parallel.py --jobs 3
```

To extend or continue an experiment, set `resume_simulation = true` in its TOML and run `scripts/run_parallel.py` again, or pass `--resume` to force checkpoint resume for every configured shot. Checkpoint-resume dispatch runs every shot command again under a fresh GNU Parallel job log; GNU Parallel must not skip successful entries because the simulations themselves decide what work remains. Each shot resumes from `ckpt_previous.npz`, and completed shots will do no further solver loops if their checkpoint already meets the requested duration. Increasing `desired_time` is supported, including the additional particles implied by a longer duration. Keep the other physics and maze settings unchanged while resuming; use a new series name and output directories for different settings.

## Maze and zone configuration

- `maze_startpoints_configuration.toml` contains named profiles under `[configurations]`.
- A profile has `maze_file`, `emitter_position`, optional `emitter_jitter`, `static_source_position`, `exit_radius`, and optional `zones_file`.
- `junction_zones.toml` in a maze analysis directory defines active decision zones. `[defaults]` provides `before_distance`, `correct_distance`, `wrong_distance`, and per-zone radius settings; matching `[junctions.Jxx]` entries override defaults.
- `zone_masks.npz` is generated by `scripts/maze_skeleton_path.py`. It stores `<Jxx>_before`, `<Jxx>_correct`, and `<Jxx>_wrong` masks plus `junction_ids`, `maze_hash`, and simulation `shape`.

## Output format

A shot directory typically contains:

- `param.txt` - recorded parameters.
- `param_checkpoint.txt` - parameter snapshot associated with the recovery checkpoint.
- `grid.txt` - grid dimensions and maze-grid data.
- `part_<global-step>.npz` - particle positions and forces.
- `conc_<global-step>.npz` - concentration snapshots.
- `free_cells.npz` - the free-cell index used to unpack compact concentration snapshots.
- `ckpt_current.npz` and `ckpt_previous.npz` - newest and recovery checkpoint bundles.

Particle snapshots contain `timestep`, `simulation_time`, `x`, `y`, `vx`, `vy`, `theta`, `omega`, and the force arrays `f_self_propulsion`, `f_chemotaxis`, `f_interaction`, and `f_wall` when force saving is enabled.

Concentration snapshots use `format="free_v1"` and store C-order values for free maze cells in `concentration`, together with `grid_shape` and `maze_hash`. `free_cells.npz` stores `free_index`, `maze_hash`, and `format="free_index_v1"`. Legacy full-grid concentration NPZ files remain readable.

Checkpoint bundles contain concentration and particle state, `masks`, `active_mask`, `dead_tracker`, `birth_times`, `exit_times`/`exit_trigger_time`, `step`/`resume_step`, `time`/`simulation_time`, `extra_state`, `dt`, `num_particles`, snapshot prefixes, maze/profile fields, and source/exit parameters. `extra_state` contains NumPy/Python RNG state and optional junction-tracker state.

`exit_times.txt` in the output directory has the header `ExitTime Beta JobID ParticleID`. When tracking is enabled, the output directory receives `decisions_<n>.npz` files containing `global_step`, `sim_time`, `particle_id`, `junction_id`, and `decision`/`correct`, plus `junction_ids` and `shape`. View them with:

```bash
python scripts/view_decisions.py <shot-output-directory> --head 20
```

## Video rendering

Render one shot:

```bash
python video_maker_testrun.py --data <shot-data-directory> \
  --maze different_mazes/Ran_maze_size_prop_to_droplet.tsv \
  --frames 0 --output output/video.mp4
```

`--frames 0` renders all available snapshots. Render all discovered shots in a series:

```bash
python scripts/render_all_series_videos.py --overwrite
```
