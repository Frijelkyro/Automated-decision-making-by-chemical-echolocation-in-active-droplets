## T0 Seed and regression baseline
- Files changed:
  - `maze_cluster_script.py`: added `--seed`, NumPy/Python seeding, and per-loop `perf_counter()` runtime output.
  - `scripts/make_jobs.py`: added seed forwarding support for experiment TOMLs.
  - `tests/test_seed_regression.py`: added fixed-seed byte-identity regression coverage.
  - `tests/baseline/`: stored the fixed-seed `exit_times.txt` and `part_*.npz` baseline.
- New/changed interfaces (signatures, file formats, CLI flags):
  - Added `--seed <int>` to `maze_cluster_script.py`; default is `0`.
  - Added `seed` support to generated experiment jobs.
- Behavior changes vs. before (esp. physics/RNG):
  - NumPy and Python RNGs are initialized before maze/config and particle initialization.
  - Simulation behavior is reproducible for the same seed and configuration.
  - Added console output in the form `Loop <index>: <runtime> s`; no physics equations changed.
- How to test (exact command):
  - `.venv/bin/python -m unittest tests.test_seed_regression`
  - Baseline command: `.venv/bin/python maze_cluster_script.py --seed 12345 --test-run true --desired-time 20 --dt 0.0005 --n-steps 100 --num-particles 10 --emission-rate 0.5 --data-dir <data-dir> --output-dir <output-dir>`
- Known issues / TODO for later tasks:
  - Runtime output is wall-clock-dependent and is not written to files.

## T1 Housekeeping
- Files changed:
  - `maze_cluster_script.py`
  - `list_of_functions.py`
  - `maze_functions.py`
  - `scripts/run_parallel.py`
  - `master_run.sh` (deleted)
  - `HANDOFF.md`
- New/changed interfaces (signatures, file formats, CLI flags):
  - None.
- Behavior changes vs. before (esp. physics/RNG):
  - Housekeeping only. `drops_added_incremental` always used its enabled behavior, and the unused alternate branch was removed.
  - Advection logic and the `update_flow_field` call path were retained because advection remains configurable.
  - No RNG or physics equations changed.
- How to test (exact command):
  - `.venv/bin/python -m unittest tests.test_seed_regression`
  - `python -m compileall -q maze_cluster_script.py list_of_functions.py maze_functions.py scripts/run_parallel.py`
- Known issues / TODO for later tasks:
  - Removed functions/code: `compute_birth_times`, `load_c_from_file`, `initialize_c_from_file`, `noflux`, `full_traj`, `column_names`, and the `drops_added_incremental = False` branch.

## T2 Refactor the solver interface
- Files changed:
  - `list_of_functions.py`
  - `maze_cluster_script.py`
  - `HANDOFF.md`
- New/changed interfaces (signatures, file formats, CLI flags):
  - `SimParams` fields: `dx`, `dc`, `dp`, `bp`, `moving_source_production_strength`, `static_source_production_strength`, `moving_source_decay_rate`, `static_source_decay_rate`, `sp_decay_rate`, `global_decay_strength`, `self_propulsion_speed`, `self_propulsion_frequency`, `dr`, `particle_mass`, `particle_moment_inertia`, `lx`, `ly`, `n_xbins`, `n_ybins`, `n_steps`, `dt`, `gamma`, `time_loop`, `write_every`, `num_particles`, `param_filename`, `grid_filename`, `file_prefix_conc`, `file_prefix_part`, `advection`, `massive_particle`, `moving_source_decay`, `static_source_decay`, `epsilon_lj`, `static_source_position`, `exit_radius`, `birth_times`, `active_mask`, `dead_tracker`, `death_zone_map`, `exit_zone_map`, `birth_zone_map`, `grim_reaper_delay`, `exit_trigger_time`, `emitter_position`, `permeability`, `exit_wall_mask`, `maze_file`, `maze_config`, `emitter_jitter`, `run_id`, `shot`, `seed`.
  - `SolverResult` fields: `concentration`, `position`, `theta`, `velocity`, `ang_velocity`, `forces_self_propulsion`, `forces_chemotaxis`, `forces_interaction`, `forces_wall`, `finished`, `exit_timestep`, `exit_trigger_time`.
  - `save_snapshot(file_prefix_conc, file_prefix_part, concentration, particles, theta, velocity, ang_velocity, forces_self_propulsion, forces_chemotaxis, forces_interaction, forces_wall, timestep, simulation_time, n_steps, save_forces=True)`.
- Behavior changes vs. before (esp. physics/RNG):
  - Solver parameters are passed through `SimParams`; the return value is `SolverResult` and the completion flag is named `finished`.
  - Concentration and particle NPZ writes share `save_snapshot`.
  - Python and NumPy inputs, equations, RNG draws, and serialized parameter names remain unchanged.
  - Argument defaults for the direct simulation script now match the `qj01_rate_reaper_sweep_200s.toml` defaults for `desired_time`, `dt`, `beta`, `write_every`, `n_steps`, and `test_run`.
  - Simulation time remains unitless in code; for plotting, multiply it by 10 seconds (`simulation_time * 10 sec`) from the 1e-3 normal-second time unit.
- How to test (exact command):
  - `.venv/bin/python -m unittest tests.test_seed_regression`
  - `python -m compileall -q maze_cluster_script.py list_of_functions.py scripts/run_parallel.py`
- Known issues / TODO for later tasks:
  - `SimParams.to_parameter_dict()` retains legacy parameter-file key names for byte-compatible checkpoint output.

## T3 Bug fixes
- Files changed:
  - `list_of_functions.py`
  - `scripts/run_parallel.py`
  - `tests/baseline/`
  - `HANDOFF.md`
- New/changed interfaces (signatures, file formats, CLI flags):
  - GNU Parallel jobs now use `--halt-on-error=2`; existing job logs remain the failure record and `--resume-failed` remains the retry path.
  - No Python function signatures or file formats changed.
- Behavior changes vs. before (esp. physics/RNG):
  - Particle spawn bin checks now recompute `px_bins`/`py_bins` from current positions at each solver entry.
  - Snapshot writes use the global `timestep` instead of `local_step`, correcting cross-solver-call write cadence.
  - An OOB particle is saved in the snapshot but is no longer marked as an exit or dead particle; the solver raises `RuntimeError`, producing a nonzero shot process exit.
  - No force calculation, RNG, or concentration equations changed.
- How to test (exact command):
  - `.venv/bin/python -m unittest tests.test_seed_regression`
  - `python -m compileall -q maze_cluster_script.py list_of_functions.py scripts/run_parallel.py scripts/make_jobs.py`
- Known issues / TODO for later tasks:
  - The fixed-seed regression baseline was re-recorded after the T3 `write_every` fix using seed `12345`, test-run 20 seconds, `dt=0.0005`, 10 particles, and emission rate `0.5`.

## T4 Decouple snapshot frequencies and checkpoints
- Files changed:
  - `list_of_functions.py`
  - `maze_cluster_script.py`
  - `resume_simulation.py`
  - `video_maker_testrun.py`
  - `scripts/make_jobs.py`
  - `HANDOFF.md`
- New/changed interfaces (signatures, file formats, CLI flags):
  - Added `--part-every` and `--conc-every`; concentration cadence must be a positive multiple of particle cadence.
  - Checkpoints are stored in `ckpt_current.npz` and `ckpt_previous.npz`.
  - Bundle keys include `conc`/`concentration`, `particles`, `theta`, `velocity`, `ang_velocity`, `masks`, `active_mask`, `dead_tracker`, `birth_times`, `exit_times`/`exit_trigger_time`, `step`/`resume_step`, `time`/`simulation_time`, `extra_state`, `dt`, `num_particles`, `file_prefix_conc`, `file_prefix_part`, `maze_config`, `static_source_position`, and `exit_radius`.
  - Resume loads `ckpt_previous.npz`; the old `_checkpoint.txt` and `_state_checkpoint.npz` names are no longer used.
- Behavior changes vs. before (esp. physics/RNG):
  - Particle and concentration NPZ files now use separate global-step frequencies.
  - Checkpoints are written at the end of every solver chunk and atomically promoted with `os.replace`.
  - Chunk step/time are checkpointed at the end of the chunk, and NumPy/Python RNG states are stored in `extra_state` for resumed runs.
  - Test runs always resolve to the `Testrun` startpoint profile, even if `--maze-config` also specifies another profile.
  - Video rendering discovers numbered `param*.txt` snapshots or falls back to `ckpt_previous.npz` for startpoint coordinates.
- How to test (exact command):
  - `.venv/bin/python -m unittest tests.test_seed_regression`
  - `python -m compileall -q maze_cluster_script.py list_of_functions.py resume_simulation.py video_maker_testrun.py scripts/make_jobs.py`
- Known issues / TODO for later tasks:
  - `extra_state` is the extension point for future tracker state. Preserve it as an object/dict, or add named keys to `write_param_snapshot` and restore them in `resume_simulation_from_file`.

## T5 Compact concentration storage
- Files changed:
  - `storage.py`
  - `list_of_functions.py`
  - `resume_simulation.py`
  - `video_maker_testrun.py`
  - `scripts/plot_path_concentration.py`
  - `tests/test_compact_concentration.py`
  - `HANDOFF.md`
- New/changed interfaces (signatures, file formats, CLI flags):
  - Added `storage.build_free_index(maze)`, `storage.pack_conc(concentration, free_index)`, `storage.unpack_conc(packed, free_index, grid_shape=None)`, and `storage.cell_to_lin(row, col, free_index)`.
  - Added `storage.write_concentration(...)` and `storage.read_concentration(...)`.
  - Each shot writes `free_cells.npz` once with `free_index`, `maze_hash`, and `format="free_index_v1"`.
  - Concentration snapshots use `format="free_v1"`, store only C-order free-cell values in `concentration`, and include `grid_shape` and the maze hash.
  - Legacy full-grid concentration NPZ files remain readable by `_read_concentration`, video rendering, and path plotting.
- Behavior changes vs. before (esp. physics/RNG):
  - Concentration snapshots are packed without changing the live solver array or physics/RNG behavior.
  - NPZ output is selected after timing plain and compressed writes and comparing file sizes.
  - Mapping/hash validation prevents an incompatible maze from unpacking concentration snapshots.
- How to test (exact command):
  - `.venv/bin/python -m unittest tests.test_seed_regression`
  - `.venv/bin/python -m unittest tests.test_compact_concentration`
  - `.venv/bin/python -m compileall -q storage.py list_of_functions.py resume_simulation.py video_maker_testrun.py scripts/plot_path_concentration.py`
  - Render a fresh shot: `.venv/bin/python video_maker_testrun.py --data <shot-data-dir> --frames 5 --output <output.mp4>`
- Known issues / TODO for later tasks:
  - Checkpoint bundles still contain full concentration state; future tasks may compact embedded checkpoint concentrations too.

## T6 Skeleton script: disjoint, configurable zones
- Files changed:
  - `scripts/maze_skeleton_path.py`
  - `different_mazes/Ran_maze_size_prop_to_droplet_analysis/junction_zones.toml`
  - `different_mazes/Ran_maze_size_prop_to_droplet_analysis/zone_masks.npz`
  - `different_mazes/Ran_maze_size_prop_to_droplet_analysis/maze_metadata.toml`
  - `different_mazes/Ran_maze_size_prop_to_droplet_analysis/maze_decision_zones.png`
  - `HANDOFF.md`
- New/changed interfaces (signatures, file formats, CLI flags):
  - `zone_masks.npz` keys are `<Jxx>_before`, `<Jxx>_correct`, `<Jxx>_wrong`, plus `junction_ids`, `maze_hash`, and `shape`.
  - Mask shape is the simulation orientation `(ny, nx)` produced by `np.rot90(mask, -1)`; `shape` is the stored simulation shape.
  - Added `--before-length` and `--zone-length`; defaults are `4` and `7`.
  - `junction_zones.toml` uses `[defaults]` keys `before_length`, `zone_length`, `maze_hash`, and `junction_ids`, with `[junctions.Jxx]` overrides and a raw-grid `center` array.
- Behavior changes vs. before (esp. physics/RNG):
  - All labels use one nearest-skeleton-cell expansion pass.
  - `before` covers at least the configured number of route cells toward the junction and can cross earlier junctions.
  - `correct`/`wrong` are depth-limited BFS branches from the junction cluster.
  - Correct and wrong masks are asserted disjoint; junctions inside the configured exit radius are skipped.
- How to test (exact command):
  - `.venv/bin/python scripts/maze_skeleton_path.py --maze different_mazes/Ran_maze_size_prop_to_droplet.tsv`
  - The run verifies each saved mask against `grid_to_sim` and asserts every junction's correct/wrong intersection is empty.
- Known issues / TODO for later tasks:
  - `zone_extent=compact` is retained for compatibility, but gate construction remains driven by the `before_length` and `zone_length` BFS depths.

## T6 Follow-up: Per-junction circular gates
- Files changed:
  - `scripts/maze_skeleton_path.py`
  - `scripts/particle_decision_analysis.py`
  - `tests/test_particle_decision_analysis.py`
  - `different_mazes/Ran_maze_size_prop_to_droplet_analysis/zone_config.toml`
  - `different_mazes/Ran_maze_size_prop_to_droplet_analysis/maze_decision_zones.png`
- New/changed interfaces:
  - The earlier `zone_config.toml` settings are now legacy and ignored; `junction_zones.toml` is the active source.
  - Active per-junction raw-grid keys are `center`, `before_center`, `correct_center`, `wrong_center`, and optional radius keys.
  - Masks are now local circular/free-corridor gates rather than nearest-skeleton-corridor expansions.
- Behavior changes:
  - Plot overlays no longer cover the whole maze corridor.
  - Magenta crosses mark skeleton dead ends.
  - Correct and wrong masks remain asserted disjoint.
  - The particle decision loader accepts the NPZ simulation `shape` key.

### T6 radius override
- Files changed:
  - `scripts/maze_skeleton_path.py`
  - `different_mazes/Ran_maze_size_prop_to_droplet_analysis/junction_zones.toml`
- Interface:
  - `[defaults]` accepts `radius = 3.5`.
  - `[junctions.Jxx]` accepts `radius`; it applies to before, correct, and wrong gates.
<<<<<<< HEAD
### T6 configuration precedence clarification
- Files changed:
  - `scripts/maze_skeleton_path.py`
  - `different_mazes/QJ01_analysis/junction_zones.toml`
  - `different_mazes/QJ01_analysis/zone_config.toml`
  - `different_mazes/QJ01_analysis/maze_metadata.toml`
  - `different_mazes/Ran_maze_size_prop_to_droplet_analysis/junction_zones.toml`
  - `different_mazes/Ran_maze_size_prop_to_droplet_analysis/zone_config.toml`
- Interfaces:
  - `junction_zones.toml` is the active zone configuration; matching `[junctions.Jxx]` entries override `[defaults]`.
  - Added `--zone-radius R` for a run-only default-radius override; explicit junction radii still take precedence.
  - `zone_config.toml` is legacy and ignored; `maze_metadata.toml` is generated output, not a zone input.

## T6 independent zone settings and junction drops
- Files changed:
  - `scripts/maze_skeleton_path.py`
  - `tests/test_particle_decision_analysis.py`
  - `different_mazes/QJ01_analysis/junction_zones.toml`
  - `different_mazes/Ran_maze_size_prop_to_droplet_analysis/junction_zones.toml`
- Interfaces:
  - `[defaults]` and `[junctions.Jxx]` use independent `before_distance`, `correct_distance`, `wrong_distance` and per-zone `before_radius`, `correct_radius`, `wrong_radius`.
  - `drop = true` skips that junction's zones without renumbering later junction IDs; config entries retain `drop = false` by default.
  - Added `--before-distance`, `--correct-distance`, and `--wrong-distance`; legacy `--zone-length` remains as a hidden compatibility alias setting both correct and wrong distances.
=======
>>>>>>> 76bf27b (first proper implementation of junction decision tracking in the simulation loop (+bugs?))

## T7 JunctionTracker and solver integration
- Files changed:
  - `junction_tracking.py`
  - `list_of_functions.py`
  - `maze_cluster_script.py`
  - `scripts/make_jobs.py`
  - `maze_startpoints_configuration.toml`
  - `HANDOFF.md`
- New/changed interfaces (signatures, file formats, CLI flags):
  - Added `JunctionTracker(zones_file, num_particles, nx, ny, dx=1.0)` and `JunctionTracker.update(global_step, sim_time, px, py, active_mask)`.
  - State attributes for T8: `before`, `correct`, `wrong` have shape `(J, nx, ny)`, `armed` has shape `(N, J)`, and `event_buffer`/`events` hold `(global_step, sim_time, particle_id, junction_id, correct)` tuples.
  - Added `--track-decisions`, `--zones-file`, and `--decision-flush-loops`; maze profiles accept `zones_file`.
  - Each flush writes `decisions_<n>.npz` in the shot output directory. NPZ keys are `global_step`, `sim_time`, `particle_id`, `junction_id`, `correct`, plus `junction_ids`, `shape`, and compatibility aliases `global_steps`, `particle_ids`, `decision`.
  - `junction_tracker` state is checkpointed under `extra_state["junction_tracker"]`.
- Behavior changes vs. before (esp. physics/RNG):
  - Tracking is inert and introduces no RNG or physics changes unless `--track-decisions` is enabled.
  - Armed particles persist until one of their junction correct/wrong gates fires; no timeout is added.
  - The tracker runs after the solver's OOB check, in batched NumPy operations, and logs numeric junction indices and `correct=1`/`wrong=0`.
  - Buffers flush every configured solver-loop count, at normal shot end, and from the OOB catch path.
- How to test (exact command):
  - `.venv/bin/python -m compileall -q junction_tracking.py maze_cluster_script.py list_of_functions.py scripts/make_jobs.py`
  - `tmpdir=$(mktemp -d); .venv/bin/python maze_cluster_script.py --desired-time 1 --n-steps 100 --seed 12345 --maze-config Ran_maze_size_prop_to_droplet --track-decisions --decision-flush-loops 5 --data-dir "$tmpdir" --output-dir "$tmpdir" && find "$tmpdir" -name 'decisions_*.npz' -print`
  - Run paired identical seeds with tracking omitted and compare `exit_times.txt` and snapshot outputs.
- Known issues / TODO for later tasks:
  - The existing fixed-seed suite currently differs from its stored `exit_times.txt` baseline on particle exit time; paired track-off runs should be used to isolate baseline equality until that baseline is reconciled.

## T7 decision log viewer
- Files changed:
  - `scripts/view_decisions.py`
  - `HANDOFF.md`
- New/changed interfaces (signatures, file formats, CLI flags):
  - Added `python scripts/view_decisions.py <shot-directory> [--head N]`.
- Behavior changes vs. before (esp. physics/RNG):
  - Read-only; no simulation or physics changes.
- How to test (exact command):
  - `.venv/bin/python scripts/view_decisions.py <shot-output-directory> --head 20`
- Known issues / TODO for later tasks:
  - The viewer prints the first N rows from each `decisions_*.npz`; use `--head 0` to print only filenames.

## T8 Resume compatibility for the tracker
- Files changed:
  - `junction_tracking.py`
  - `maze_cluster_script.py`
  - `tests/test_junction_tracker_resume.py`
  - `HANDOFF.md`
- New/changed interfaces (signatures, file formats, CLI flags):
  - `JunctionTracker.get_state(checkpoint_step=None)` adds `flushed_upto_step` and `checkpoint_step` to the junction-tracker dictionary stored under `extra_state["junction_tracker"]`.
  - Restored state includes `armed`, `event_buffer`, `flush_count`, `flushed_upto_step`, and the checkpoint step.
  - Added `truncate_decision_log_files(output_dir, checkpoint_step)` for post-resume decision-log cleanup.
- Behavior changes vs. before (esp. physics/RNG):
  - On resume, tracker state and pending events are restored and pending events after the checkpoint step are discarded.
  - Flushed decision NPZ files beginning after the checkpoint step are deleted; straddling files are rewritten without events after the checkpoint step.
  - No RNG or physics behavior changes.
- How to test (exact command):
  - `.venv/bin/python -m unittest tests.test_junction_tracker_resume`
- Known issues / TODO for later tasks:
  - None.
