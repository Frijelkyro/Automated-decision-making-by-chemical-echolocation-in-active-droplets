import ast
import csv
import os
import numpy as np
from list_of_functions import build_free_cell_index_map, expand_concentration


def _truncate_junction_decisions(filename, resume_step):
    if not os.path.exists(filename):
        return set()

    with open(filename, newline="", encoding="utf-8") as event_file:
        reader = csv.DictReader(event_file)
        if reader.fieldnames is None or "timestep" not in reader.fieldnames:
            raise ValueError(f"Invalid junction-decision file header: {filename}")
        fieldnames = reader.fieldnames
        rows = [row for row in reader if int(row["timestep"]) <= resume_step]

    written_events = {
        (
            int(row["timestep"]),
            int(row["particle_id"]),
            row["junction_id"],
            int(row["correct"]),
        )
        for row in rows
    }
    temporary_filename = filename + ".tmp"
    with open(temporary_filename, "w", newline="", encoding="utf-8") as event_file:
        writer = csv.DictWriter(event_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_filename, filename)
    return written_events


def _read_parameter_file(filename):
    parameters = {}

    with open(filename) as f:
        for line in f:
            if ":" not in line:
                continue

            key, value = map(str.strip, line.split(":", 1))

            try:
                parameters[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parameters[key] = value

    return parameters


def _read_concentration(filename, n_steps, free_cell_index_map):
    with np.load(filename) as data:
        concentration = expand_concentration(
            data["concentration"], free_cell_index_map
        )

    c = np.zeros((n_steps, *concentration.shape), dtype=concentration.dtype)
    c[0] = concentration
    return c


def _read_particles(filename, num_particles, n_steps):
    with np.load(filename) as data:
        p = np.zeros((num_particles, n_steps, 2), dtype=data["x"].dtype)
        v = np.zeros((num_particles, n_steps, 2), dtype=data["vx"].dtype)
        theta = np.zeros((num_particles, n_steps), dtype=data["theta"].dtype)
        omega = np.zeros((num_particles, n_steps), dtype=data["omega"].dtype)

        p[:, 0, 0] = data["x"]
        p[:, 0, 1] = data["y"]
        v[:, 0, 0] = data["vx"]
        v[:, 0, 1] = data["vy"]
        theta[:, 0] = data["theta"]
        omega[:, 0] = data["omega"]

    return p, theta, v, omega


def resume_simulation_from_file(data, param_filename, maze, n_steps):
    state_filename = param_filename.removesuffix(".txt") + "_state_checkpoint.npz"
    param_filename = param_filename.removesuffix(".txt") + "_checkpoint.txt"

    if not os.path.exists(param_filename):
        raise FileNotFoundError(f"Checkpoint parameters not found: {param_filename}")
    if not os.path.exists(state_filename):
        raise FileNotFoundError(f"Checkpoint state not found: {state_filename}")
    print(f"Using recovery parameter file: {param_filename}")

    old = _read_parameter_file(param_filename)

    with np.load(state_filename) as state:
        birth_times = state["birth_times"]
        active_mask = state["active_mask"]
        dead_tracker = state["dead_tracker"]
        exit_trigger_time = state["exit_trigger_time"]
        resume_step = int(state["resume_step"])
        resume_time = float(state["simulation_time"])
        checkpoint_concentration = (
            state["concentration"].copy()
            if "concentration" in state.files
            else None
        )
        decision_state = {}
        if "junction_pending" in state.files:
            decision_state["junction_ids"] = state["junction_ids"].astype(str)
            decision_state["junction_pending"] = state["junction_pending"].copy()
            decision_state["junction_decided"] = state["junction_decided"].copy()
            buffered_events = state["junction_decision_events"].copy()
            decision_state["junction_decision_events"] = [
                (
                    int(event[0]),
                    float(event[1]),
                    int(event[2]),
                    int(event[3]),
                    int(event[4]),
                )
                for event in buffered_events
                if int(event[0]) <= resume_step
            ]

    num_particles = int(old["num_particles"])
    conc_prefix = os.path.join(data, os.path.basename(old["file_prefix_conc"]))
    part_prefix = os.path.join(data, os.path.basename(old["file_prefix_part"]))
    decision_filename = old.get(
        "junction_decisions_filename",
        os.path.join(data, "junction_decisions.csv"),
    )
    written_events = _truncate_junction_decisions(decision_filename, resume_step)
    if decision_state:
        junction_ids = decision_state["junction_ids"].tolist()
        decision_state["junction_decision_events"] = [
            event
            for event in decision_state["junction_decision_events"]
            if (
                event[0],
                event[2],
                junction_ids[event[3]],
                event[4],
            )
            not in written_events
        ]

    for prefix in (conc_prefix, part_prefix):
        prefix = os.path.basename(prefix)
        for filename in os.listdir(data):
            if filename.startswith(prefix + "_") and filename.endswith(".npz"):
                step = int(filename.rsplit("_", 1)[1][:-4])
                if step > resume_step:
                    os.remove(os.path.join(data, filename))

    if checkpoint_concentration is None:
        conc = _read_concentration(
            f"{conc_prefix}_{resume_step}.npz",
            n_steps,
            build_free_cell_index_map(maze),
        )
    else:
        conc = np.zeros(
            (n_steps, *checkpoint_concentration.shape),
            dtype=checkpoint_concentration.dtype,
        )
        conc[0] = checkpoint_concentration

    p, theta, v, omega = _read_particles(
        f"{part_prefix}_{resume_step}.npz",
        num_particles,
        n_steps,
    )
    print(f"using particle file: {part_prefix}_{resume_step}.npz")
    if checkpoint_concentration is None:
        print(f"using conc file: {conc_prefix}_{resume_step}.npz")
    else:
        print("using concentration from checkpoint state")

    return (
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
        resume_time,
        float(old["dt"]),
        decision_state,
    )
