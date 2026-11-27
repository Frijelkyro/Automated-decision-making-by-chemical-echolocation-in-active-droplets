import atexit
import shutil
import signal
import sys
import zlib
import argparse
import os
import random
import tomllib
from pathlib import Path
import numpy as np
from maze_functions import get_exit_wall_mask, maze_from_file
from resume_simulation import resume_simulation_from_file
from list_of_functions import (
    chemical_solver,
    initialize_c,
    SimParams,
    write_param_snapshot,
)
from junction_tracking import JunctionTracker, truncate_decision_log_files

from time import perf_counter
from tqdm import tqdm


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


parser = argparse.ArgumentParser(description="Run one maze simulation shot.")
parser.add_argument("--dt", type=float, default=5e-4)
parser.add_argument("--beta", type=float, default=-8)
parser.add_argument("--emission-rate", type=float, default=0.5)
parser.add_argument("--grim-reaper-delay", type=float, default=6.0)
parser.add_argument("--desired-time", type=float, default=200)
parser.add_argument("--part-every", type=int)
parser.add_argument("--conc-every", type=int)
parser.add_argument("--write-every", type=int, help=argparse.SUPPRESS)
parser.add_argument("--n-steps", type=int, default=100)
parser.add_argument("--num-particles", type=int, default=60)
parser.add_argument("--test-run", type=parse_bool, default=False)
parser.add_argument("--maze-config", help="Named maze/startpoint profile from the configuration TOML.")
parser.add_argument(
    "--startpoints-config",
    type=Path,
    default=Path(__file__).with_name("maze_startpoints_configuration.toml"),
)
parser.add_argument("--resume", type=parse_bool, default=False)
parser.add_argument("--run-id", default="default")
parser.add_argument("--shot", type=int, default=0)
parser.add_argument("--data-dir", default="data")
parser.add_argument("--output-dir")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--track-decisions", action="store_true", default=False)
parser.add_argument("--zones-file", type=Path)
parser.add_argument("--decision-flush-loops", type=int, default=100)
parser.add_argument("--stage-dir", help="tmpfs directory (e.g. /dev/shm/maze_stage) to buffer conc/part files.")
parser.add_argument("--stage-budget-gb", type=float, default=1.0, help="Flush staged files to --data-dir once this size is reached.")
parser.add_argument("--resume-from", default="ckpt_previous.npz",
                    help="Checkpoint file name or step number, e.g. 291599 or ckpt_291599.npz")
parser.add_argument("--reseed-on-resume", action="store_true",
                    help="Draw fresh noise after restoring the RNG state.")
args = parser.parse_args()

# choose set-seed + shot number and config name for the seed
effective_seed = int(np.random.SeedSequence(
    [args.seed, args.shot, zlib.crc32(args.run_id.encode())]
).generate_state(1)[0])
np.random.seed(effective_seed)
random.seed(effective_seed)

resume_ckpt = args.resume_from
if resume_ckpt.isdigit():
    resume_ckpt = f"ckpt_{resume_ckpt}.npz"

config_path = args.startpoints_config.expanduser()
if not config_path.is_absolute():
    config_path = Path(__file__).resolve().parent / config_path
try:
    with config_path.open("rb") as config_file:
        configurations = tomllib.load(config_file)["configurations"]
    # Test runs always use the dedicated startpoint profile, regardless of
    # experiment/job defaults that also happen to set --maze-config.
    maze_config_name = "Testrun" if args.test_run else (
        args.maze_config or "Ran_maze_size_prop_to_droplet"
    )
    maze_config = configurations[maze_config_name]
    maze_file = Path(maze_config["maze_file"])
    if not maze_file.is_absolute():
        maze_file = Path(__file__).resolve().parent / maze_file
    emitter_position = np.asarray(maze_config["emitter_position"], dtype=np.float32)
    emitter_jitter = np.asarray(maze_config.get("emitter_jitter", [0.0, 0.0]), dtype=np.float32)
    static_source_position = tuple(map(float, maze_config["static_source_position"]))
    configured_exit_radius = float(maze_config["exit_radius"])
    configured_zones_file = maze_config.get("zones_file")
    if emitter_position.shape != (2,) or emitter_jitter.shape != (2,):
        raise ValueError("emitter_position and emitter_jitter must each contain [x, y].")
except (KeyError, OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as error:
    parser.error(f"Invalid maze/startpoint configuration in {config_path}: {error}")

# This script is used to run the chemical solver on a maze with one particle.
# It initializes the parameters, generates a maze, and runs the simulation.


init_t0 = perf_counter()  # time tracking
beta = args.beta

dc = 2e2  # diffusion coefficient of the chemical (100)
dp = 1e0  # noise strength for the particle (0.1)
bp = beta * 1e4  # chemotactic sensitivity (CR: -16000, CA:16000)
moving_source_production_strength = 1.0  # strength of the source
moving_source_decay_rate = 1 / 70  # characteristic decay rate of the source
global_decay_strength = 0.0  # evaporation of chemical from environment
self_propulsion_speed = 0.0 * 1e1  # self-propulsion speed of the particle (1.0)
sp_decay_rate = 0.0  # characteristic decay rate of the self-propulsion
self_propulsion_frequency = 0.0  # self-propulsion angular velocity of the particle
dr = 3.0  # rotational diffusion coefficient of the particle (1.0)
particle_mass = 1e-1  # mass of the particle (0.001, 0.2)
particle_moment_inertia = 4e-2  # moment of inertia of the particle (0.001, 0.06)

epsilon_lj = 0.10  # Lennard-Jones potential parameter for interaction between particles
static_source_production_strength = 0.0  # Strength of the static source
static_source_decay_rate = 0.0  # characteristic decay rate of the source

advection = False  # whether to include advection term in the chemical equation
massive_particle = True  # whether to include mass in the particle equation
moving_source_decay = False
static_source_decay = False

# Simulation parameters
dx = 1.0  # grid spacing
lx = 1e2  # domain size
ly = 1e2  # domain size
n_xbins = int(lx / dx)  # number of bins in x direction
n_ybins = int(ly / dx)  # number of bins in y direction
n_steps = args.n_steps
dt = args.dt  # time step size
# A dt time increment is a unit of simulation time; code logic uses unitless time.
# The physical time unit is 1e-3 normal seconds, so plotting normal time
# from simulation_time uses simulation_time * 10 sec.
desired_time = args.desired_time
time_loop = int(np.ceil(desired_time / n_steps / dt)) if desired_time else 2250
gamma = (dc * dt) / (dx**2)  # gamma parameter

# local_step = 0       # integer, position within the current solver call
simulation_time = 0.0  # physical time to start with TODO in seconds?
# start_step = 0

part_every = args.part_every or args.write_every or 100
conc_every = args.conc_every or args.write_every or 100
if part_every <= 0:
    parser.error("--part-every must be positive")
if conc_every <= 0:
    parser.error("--conc-every must be positive")
if conc_every % part_every != 0:
    parser.error("--conc-every must be a multiple of --part-every")
if args.decision_flush_loops < 1:
    parser.error("--decision-flush-loops must be positive")

num_particles = args.num_particles  # Number of particles
emission_rate = args.emission_rate  # droplets per second
if desired_time:
    num_particles = int(np.ceil(emission_rate * desired_time))
test_run = args.test_run

# Data directory
data = args.data_dir
output_dir = args.output_dir or data
os.makedirs(data, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
if not os.path.exists(data):
    os.makedirs(data)
param_filename = os.path.join(data, "param.txt")
grid_filename = os.path.join(data, "grid.txt")
file_prefix_conc = os.path.join(data, "conc")
file_prefix_part = os.path.join(data, "part")
stage_dir = None
stage_budget = int(args.stage_budget_gb * 1024**3)
if args.stage_dir:
    stage_dir = Path(args.stage_dir) / (
        f"{zlib.crc32(os.path.abspath(data).encode()):08x}_{args.run_id}_{args.shot}"
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    file_prefix_conc = os.path.join(stage_dir, "conc")
    file_prefix_part = os.path.join(stage_dir, "part")
    param_filename = os.path.join(stage_dir, "param.txt")

def stage_bytes():
    return sum(f.stat().st_size for f in stage_dir.iterdir()
               if f.is_file() and not f.name.startswith((".", "ckpt_")))

def flush_stage():
    """Move staged files to the real data dir (copy to temp name, then atomic rename)."""
    if stage_dir is None or not stage_dir.exists():
        return
    files = [f for f in stage_dir.iterdir()
             if f.is_file() and not f.name.startswith(".")]
    files.sort(key=lambda f: f.name.startswith("ckpt_"))  # ckpt_* last
    for f in files:
        tmp = Path(data) / f".{f.name}.tmp"
        shutil.copy2(f, tmp)
        os.replace(tmp, Path(data) / f.name)
        if not f.name.startswith("ckpt_"):
            f.unlink()

if stage_dir is not None:
    atexit.register(flush_stage)  # normal end, finished, uncaught exceptions

stop_requested = False
def _on_signal(signum, _frame):
    print(f"INTERRUPTED by SIGTERM at step {globals().get('global_step')}, "
          f"t={globals().get('simulation_time')}", flush=True)
    global stop_requested
    stop_requested = True   # just set a flag; the loop handles it

signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)

maze = maze_from_file(str(maze_file))
wall = np.transpose(np.where(maze == 0))

exit_radius = configured_exit_radius
grim_reaper_delay = args.grim_reaper_delay
exit_wall_radius = 20.0  # radius for the leaky exit wall (this also removes particles when they get <2 pixels close)
permeability = 0.0  # permeability of the exit wall (0 = no-flux, >0 = leaky)

if test_run:
    n_steps = int(n_steps * 0.1 // 1)
elif desired_time:
    if n_steps < 2:
        parser.error("--n-steps must be at least 2 for a full simulation")
    time_loop = int(np.ceil(desired_time / ((n_steps - 1) * dt)))

# death zone and reaper timer
death_zone_map = np.zeros_like(maze, dtype=bool)
X, Y = np.indices(maze.shape)
cx, cy = np.rint(
    np.array(static_source_position) / dx
)  # static source position hold exit position
exit_zone_map = ((X - cx) ** 2 + (Y - cy) ** 2) <= (exit_radius / dx) ** 2
death_zone_map = ((X - cx) ** 2 + (Y - cy) ** 2) <= (exit_radius * 0.9 / dx) ** 2

birth_cx, birth_cy = np.rint(emitter_position / dx).astype(int)
min_clearance = 3 * dx
birth_zone_map = ((X - birth_cx) ** 2 + (Y - birth_cy) ** 2) <= (
    min_clearance / dx
) ** 2
# open walls (if leaky)
exit_wall_mask = get_exit_wall_mask(maze, static_source_position, dx, exit_wall_radius)

# Initial condition everywhere inside the grid
c_initial = 0.0
conc = initialize_c(c_initial, n_steps, maze)

# Calculate arrays safely using the master num_particles variable
p = np.full((num_particles, n_steps, 2), 0.0, dtype=np.float32)
v = np.full((num_particles, n_steps, 2), 0.0, dtype=np.float32)
theta = np.full((num_particles, n_steps), 0.0, dtype=np.float32)
omega = np.full((num_particles, n_steps), 0.0, dtype=np.float32)

active_mask = np.zeros(num_particles, dtype=bool)
dead_tracker = np.zeros(num_particles, dtype=bool)
exit_trigger_time = np.full(num_particles, np.inf)


# Default: All particles start at the same emitter location and activate at delayed birth times.
p[:, 0, :] = emitter_position + np.random.uniform(-emitter_jitter, emitter_jitter, size=(num_particles, 2))
v[:, 0, :] = 0.0
theta[:, 0] = np.random.uniform(0, 2.0 * np.pi, size=num_particles)
omega[:, 0] = 0.0

birth_times = np.array(
    [i / emission_rate for i in range(num_particles)],
    dtype=np.float64,
)
initial_particles = p
initial_theta = theta
initial_velocity = v
initial_ang_velocity = omega
initial_active_mask = active_mask
initial_dead_tracker = dead_tracker
initial_exit_trigger_time = exit_trigger_time
initial_birth_times = birth_times

# Resume settings
resume_simulation = args.resume

resume_step = 0  # this will be read from the last sim
resume_old_dt = np.inf
resume_new_dt = dt
resumed_from_checkpoint = False

if stage_dir is not None:
    if resume_simulation:
        flush_stage()          # rescue what a killed run left in tmpfs
    else:
        for f in stage_dir.iterdir():   # fresh start: don't inherit stale files
            if f.is_file():
                f.unlink()

if resume_simulation:
    checkpoint_previous = os.path.join(data, resume_ckpt)
    checkpoint_exists = os.path.exists(checkpoint_previous)

    if checkpoint_exists:
        (
            conc,
            p,
            theta,
            v,
            omega,
            active_mask,
            dead_tracker,
            exit_trigger_time,
            birth_times,
            resume_step,
            simulation_time,
            resume_old_dt,
        ) = resume_simulation_from_file(
            data,
            param_filename,
            maze,
            n_steps,
            resume_ckpt
        )
        checkpoint_num_particles = len(active_mask)
        if checkpoint_num_particles < num_particles:
            p = np.concatenate(
                (p, initial_particles[checkpoint_num_particles:]), axis=0
            )
            theta = np.concatenate(
                (theta, initial_theta[checkpoint_num_particles:]), axis=0
            )
            v = np.concatenate(
                (v, initial_velocity[checkpoint_num_particles:]), axis=0
            )
            omega = np.concatenate(
                (omega, initial_ang_velocity[checkpoint_num_particles:]), axis=0
            )
            active_mask = np.concatenate(
                (active_mask, initial_active_mask[checkpoint_num_particles:])
            )
            dead_tracker = np.concatenate(
                (dead_tracker, initial_dead_tracker[checkpoint_num_particles:])
            )
            exit_trigger_time = np.concatenate(
                (
                    exit_trigger_time,
                    initial_exit_trigger_time[checkpoint_num_particles:],
                )
            )
            birth_times = np.concatenate(
                (birth_times, initial_birth_times[checkpoint_num_particles:])
            )
        else:
            num_particles = checkpoint_num_particles
        resumed_from_checkpoint = True

        print(
            f"Resuming from timestep {resume_step} "
            f"(t: {simulation_time:.6f} s, old dt: {resume_old_dt}, new dt: {dt})"
        )
        if desired_time:
            chunk_duration = (
                n_steps * dt
                if test_run
                else (n_steps - 1) * dt
            )
            time_loop = int(
                np.ceil((desired_time - simulation_time) / chunk_duration)
            )
    else:
        print(f"No checkpoint found in {data}; starting this shot from scratch.")

junction_tracker = None
if args.track_decisions:
    zones_file = args.zones_file or configured_zones_file
    if zones_file is None:
        parser.error("--track-decisions requires --zones-file or a zones_file in the maze profile")
    zones_file = Path(zones_file).expanduser()
    if not zones_file.is_absolute():
        zones_file = Path(__file__).resolve().parent / zones_file
    junction_tracker = JunctionTracker(
        zones_file,
        num_particles,
        maze.shape[0],
        maze.shape[1],
        dx=dx,
    )
    if resumed_from_checkpoint:
        try:
            with np.load(os.path.join(data, "ckpt_previous.npz"), allow_pickle=True) as state:
                if "extra_state" in state:
                    extra_state = state["extra_state"].item()
                    if isinstance(extra_state, dict):
                        junction_tracker.restore_state(extra_state.get("junction_tracker"))
        except (OSError, ValueError):
            parser.error("Unable to restore junction tracker state from ckpt_previous.npz")
        truncate_decision_log_files(output_dir, resume_step)

# Build one parameter source and serialize its legacy parameter-file keys from it.
sim_params = SimParams(
    dx=dx,
    dc=dc,
    dp=dp,
    bp=bp,
    moving_source_production_strength=moving_source_production_strength,
    static_source_production_strength=static_source_production_strength,
    moving_source_decay_rate=moving_source_decay_rate,
    static_source_decay_rate=static_source_decay_rate,
    sp_decay_rate=sp_decay_rate,
    global_decay_strength=global_decay_strength,
    self_propulsion_speed=self_propulsion_speed,
    self_propulsion_frequency=self_propulsion_frequency,
    dr=dr,
    particle_mass=particle_mass,
    particle_moment_inertia=particle_moment_inertia,
    lx=lx,
    ly=ly,
    n_xbins=n_xbins,
    n_ybins=n_ybins,
    n_steps=n_steps,
    dt=dt,
    gamma=gamma,
    time_loop=time_loop,
    part_every=part_every,
    conc_every=conc_every,
    num_particles=num_particles,
    param_filename=param_filename,
    grid_filename=grid_filename,
    file_prefix_conc=file_prefix_conc,
    file_prefix_part=file_prefix_part,
    advection=advection,
    massive_particle=massive_particle,
    moving_source_decay=moving_source_decay,
    static_source_decay=static_source_decay,
    epsilon_lj=epsilon_lj,
    static_source_position=static_source_position,
    exit_radius=exit_radius,
    birth_times=birth_times,
    active_mask=active_mask,
    dead_tracker=dead_tracker,
    death_zone_map=death_zone_map,
    exit_zone_map=exit_zone_map,
    birth_zone_map=birth_zone_map,
    grim_reaper_delay=grim_reaper_delay,
    exit_trigger_time=exit_trigger_time,
    emitter_position=emitter_position,
    permeability=permeability,
    exit_wall_mask=exit_wall_mask,
    maze_file=str(maze_file),
    maze_config=maze_config_name,
    emitter_jitter=tuple(emitter_jitter),
    run_id=args.run_id,
    shot=args.shot,
    seed=effective_seed,
    junction_tracker=junction_tracker,
    decision_output_dir=output_dir,
    decision_flush_loops=args.decision_flush_loops,
)
parameter_dict = sim_params.to_parameter_dict()

# --------------- time tracking ---------------------
init_duration_perf_metric = perf_counter() - init_t0  # time tracking
print(
    f"Initialization time: {init_duration_perf_metric:.3f} s | particles: {num_particles} | n_steps/loop: {n_steps} | maze.shape: {maze.shape} | concentration shape: {conc.shape} | emission_rate: {emission_rate} | dt: {dt}",
    flush=True,
)
simulation_t0 = perf_counter()  # time tracking
n_active = active_mask.sum()
#pbar = tqdm(range(time_loop), desc="Simulation", unit="loop")
pbar = tqdm(
    range(time_loop),
    desc=f"Shot {args.shot}",
    unit="loop",
    disable=True,
)

global_step = resume_step + 1 if resumed_from_checkpoint else 0
finished = False
print(f"START resumed={resumed_from_checkpoint} step={global_step} "
      f"t={simulation_time:.2f}/{desired_time} loops_planned={time_loop}"
      + ("  -> NOTHING TO DO" if time_loop <= 0 else ""), flush=True)

# ---------------- Simulation loop ------------------
for i in pbar:
    loop_t0 = perf_counter()  # time tracking

    try:
        solver_result = chemical_solver(
            conc,
            p,
            theta,
            v,
            omega,
            maze,
            sim_params,
            start_step=global_step,
            start_time=simulation_time,
        )
    except RuntimeError:
        if junction_tracker is not None:
            junction_tracker.flush(output_dir)
        raise
    conc = solver_result.concentration
    p = solver_result.position
    theta = solver_result.theta
    v = solver_result.velocity
    omega = solver_result.ang_velocity
    exit_trigger_time = solver_result.exit_trigger_time
    exit_timestep = solver_result.exit_timestep
    finished = solver_result.finished

    end_step = global_step + n_steps - 1
    end_time = simulation_time + (n_steps - 1) * dt
    parameter_dict["active_mask"] = active_mask
    parameter_dict["dead_tracker"] = dead_tracker
    parameter_dict["exit_trigger_time"] = exit_trigger_time
    parameter_dict["birth_times"] = birth_times
    extra_state = {
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if junction_tracker is not None:
        extra_state["junction_tracker"] = junction_tracker.get_state(end_step)
    write_param_snapshot(
        parameter_dict,
        end_time,
        end_step,
        concentration=conc[end_step % n_steps],
        particles=p[:, end_step % n_steps, :],
        theta=theta[:, end_step % n_steps],
        velocity=v[:, end_step % n_steps, :],
        ang_velocity=omega[:, end_step % n_steps],
        extra_state=extra_state,
    )

    global_step += n_steps - 1
    global_step += 1
    simulation_time = end_time
    if stage_dir is not None and stage_bytes() >= stage_budget:
        flush_stage()

    if (
        junction_tracker is not None
        and (i + 1) % args.decision_flush_loops == 0
    ):
        junction_tracker.flush(output_dir)

    if finished:
        conc[-1, :, :] = conc[exit_timestep % n_steps, :, :]
        loop_duration = perf_counter() - loop_t0
        print(f"FINISHED: Loop {i}: step {end_step} t={simulation_time:.2f}/{desired_time} "
              f"{loop_duration:.3f}s", flush=True)
        break
    if stop_requested:
        print(f"PAUSED at step {end_step}, t={simulation_time:.2f}", flush=True)
        if junction_tracker is not None:
            junction_tracker.flush(output_dir)
        flush_stage()
        sys.exit(143)
    # set the first particles parameters for the next loop to the most recent from the current loop
    conc[0, :, :] = conc[-1, :, :]
    p[:, 0, :] = p[:, -1, :]
    theta[:, 0] = theta[:, -1]
    omega[:, 0] = omega[:, -1]
    v[:, 0, :] = v[:, -1, :]

    # time tracking
    n_active = active_mask.sum()
    pbar.set_postfix(
        timestep=end_step,
        active=n_active,
        simulation_time=f"{simulation_time:.0f}s",
    )
    loop_duration = perf_counter() - loop_t0
    if i % 100 == 0:
        print(f"Loop {i}: step {end_step} t={simulation_time:.2f}/{desired_time} "
              f"{loop_duration:.3f}s", flush=True)

if junction_tracker is not None:
    junction_tracker.flush(output_dir)

job_id = args.shot

filename2 = os.path.join(output_dir, "exit_times.txt")

with open(filename2, "w") as f:
    f.write("ExitTime Beta JobID ParticleID\n")
    for particle_id in range(num_particles):
        if not np.isfinite(exit_trigger_time[particle_id]):
            f.write(f"{-1} {beta} {job_id} {particle_id}\n")
        else:
            f.write(
                f"{max((exit_trigger_time [particle_id]-birth_times[particle_id]), -1)} {beta} {job_id} {particle_id}\n"
            )


sim_duration_perf_metric = perf_counter() - simulation_t0

print(
    f"\nThis simulation duration (performance metric): {sim_duration_perf_metric:.3f} s"
)
