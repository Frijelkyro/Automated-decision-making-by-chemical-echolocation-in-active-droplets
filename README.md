
Copyright (C) 2025 Aritra K. Mukhopadhyay.

# README.md

* Project: Automated decision-making by chemical
echolocation in active droplets

* Authors: Aritra K. Mukhopadhyay, Ran Niu, Linhui Fu, Kai Feng, Christopher Fujta, Qiang Zhao, Jinping Qu, and
Benno Liebchen.

* Description: Motile microorganisms, like bacteria and algae, unify abilities like self-propulsion, autonomous navigation, and decision-making on the micron scale. While recent breakthroughs have led to the creation of synthetic microswimmers and nanoagents that can also self-propel, they still lack the functionality and sophistication of their biological counterparts. This study pioneers a mechanism enabling synthetic agents to autonomously navigate and make decisions, allowing them to solve mazes and transport cargo through complex environments without requiring external cues or guidance. The mechanism exploits chemo-hydrodynamic signals, produced by agents like active droplets or colloids, to remotely sense and respond to their environment - similar to echolocation. Our research paves the way for endowing autonomous, motile synthetic agents with functionalities that have been so far exclusive to biological organisms.
               
* Contact: aritra.mukhopadhyay@pkm.tu-darmstadt.de, benno.liebchen@pkm.tu-darmstadt.de               




## What this project does

- Runs time-dependent simulations of a scalar chemical field coupled to particle motion inside maze-shaped domains.
- Supports moving and static point sources, optional advection, chemotactic response, particle–particle interactions (Lennard-Jones style), wall avoidance, and self-propulsion.
- Writes particle snapshots at the `--write-every` timestep interval and concentration snapshots at a configurable multiple of that interval. Only concentration snapshots within the latest `2 * n_steps` timesteps are retained; particle snapshots are kept for video rendering.


## File descriptions

- `maze_cluster_script.py` — example driver that sets simulation parameters, generates/loads a maze, runs the solver loop and saves outputs to `data/`.
- `list_of_functions.py` — numerical routines and I/O helpers (solvers, force models, write functions).
- `maze_functions.py` — maze generators and loaders (random, box, custom, load from TSV).
- `video_maker.py` — reads `data/` files and renders an MP4 animation of concentration + particle trajectories.
- `master_run.sh` — simple helper to run the cluster script followed by the video maker.

## How to run the simulations

Prerequisites

- Python 3.8+ (code was tested on Python 3.10+). Ensure you have a working Python environment with the scientific stack.
- Recommended packages: `numpy`, `matplotlib`, `seaborn` (for the video script), and `ffmpeg` (for saving MP4s).

Install minimal dependencies (example using pip):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib seaborn
# ffmpeg should be available on PATH for `video_maker.py` to save MP4s
```

Run an example simulation

```bash
# From the repository root
./master_run.sh

# Or run directly
python maze_cluster_script.py
python video_maker.py
```

By default the example driver writes outputs into `data/` (see `maze_cluster_script.py` for parameter settings). Outputs include:

- `data/param.txt` — recorded simulation parameters
- `data/grid.txt` — grid description
- `data/conc_*.npz` — concentration snapshots, retained within the latest `2 * n_steps` timesteps
- `data/part_*.npz` — particle snapshots
- `data/particle_trajectory.mp4` — rendered video (created by `video_maker.py`)

### Experiment series

The single-run wrapper accepts parameter and directory overrides, so it can also be run without GNU Parallel:

```bash
python maze_cluster_script.py --beta -8 --emission-rate 0.5 --dt 0.0005 \
	--grim-reaper-delay 6 --desired-time 800 --write-every 100 --n-steps 100 \
	--test-run false --resume false --run-id beta_m8_e05 --shot 0 \
	--data-dir data/beta_m8_e05/shot_000 \
	--output-dir output/beta_m8_e05/shot_000
```

Experiment defaults and parameter grids live in `experiments/simulation_series_parameters.toml`. Generate its command list with:

```bash
python scripts/make_jobs.py
```

If GNU Parallel is installed, launch the generated list with three concurrent jobs:

```bash
python scripts/run_parallel.py --jobs 3
```

When `resume_simulation = true`, rerunning a series with the same name resumes its GNU Parallel job log and each unfinished shot from its latest checkpoint. Keep its configuration unchanged when resuming; use a new series name for changed parameters. Shots without a checkpoint start fresh. Each series has isolated data and output directories. The configured `emission_1_reaper_6_400s` series runs 200 shots.



### Junction decision tracking

Generate maze junction zones before a run with `python scripts/maze_skeleton_path.py`. This creates `zone_masks.npz` and an editable `zone_config.toml` in that maze's `*_analysis/` directory. Distances are measured in skeleton cells: the before zone is never closer than 4 cells to its junction, and correct/wrong zones default to 7 cells. Correct and wrong masks are kept disjoint. Edit each `[JunctionID]` table in `zone_config.toml` to change those distances.

When zone masks are present beside the configured maze, the solver records each particle's first correct (`1`) or wrong (`0`) decision per junction in `junction_decisions.csv` in the data directory. Rows contain timestep, simulation time, particle ID, junction ID, and the decision value. Records are buffered and written every 10 solver loops by default; change this with `--decision-write-every-loops`. Pass `--junction-zone-masks` to use a different mask file. The decision state is checkpointed with the simulation, and resuming truncates decision rows and buffered events beyond the restored checkpoint timestep.

## Configuration and parameters

- `maze_cluster_script.py` contains the main parameter block (diffusion coefficients, time step `dt`, `n_steps`, `dx`, particle counts, source strengths, etc.). Edit that file to experiment with different regimes.
- `maze_functions.py` provides multiple maze sources: random generator, `box_maze`, `custom_maze`, and `maze_from_file(file.tsv)` to load maze TSVs in `different_mazes/`.

Example: change grid size and time steps in `maze_cluster_script.py`:

```python
# change domain size and resolution
dx = 1.0
Lx = 200.0
Ly = 200.0
n_xbins = int(Lx/dx)
n_ybins = int(Ly/dx)
n_steps = 4000
```

## Output format

- Concentration files (`data/conc_<t>.npz`) contain `timestep`, `simulation_time`, `grid_shape`, and a one-dimensional `concentration` array with values only for free maze cells. Readers reconstruct the 2D grid from the recorded maze geometry; older full-grid snapshots remain readable.
- Particle files (`data/part_<t>.npz`) contain `timestep`, `simulation_time`, and per-particle position, velocity, orientation, and angular velocity arrays.

`--concentration-every-particle-snapshots` controls the heatmap cadence: `2` writes one concentration snapshot for every two particle snapshots. The video holds each heatmap while displaying the intervening particle snapshots; it renders only the particle history covered by the retained concentration snapshots. The default is `2`.
