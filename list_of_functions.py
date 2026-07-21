import numpy as np
from maze_functions import get_exit_wall_mask, leaky_exit_wall


# Define the initial conditions
def initialize_c(c_initial, n_steps, maze):
    # Initialize solution: the grid of u(k, i, j)
    nx = maze.shape[0]
    ny = maze.shape[1]
    c = np.full((n_steps, nx, ny), c_initial, dtype=np.float32)

    return c


def initialize_c_from_file(c_initial, n_steps, maze, filename):
    nx = maze.shape[0]
    ny = maze.shape[1]
    c = np.full((n_steps, nx, ny), c_initial, dtype=np.float32)
    c[0] = np.loadtxt(filename, skiprows=3).reshape((nx, ny))
    return c


def compute_birth_steps(num_particles, emission_rate, dt):
    emission_interval = 1.0 / emission_rate
    return np.array(
        [int(round(i * emission_interval / dt)) for i in range(num_particles)],
        dtype=int,
    )


def active_particle_mask(timestep, birth_steps):
    return timestep >= birth_steps


# define the source function for a moving point source
def moving_point_source(
    s,
    production_strength,
    timestep,
    particle_positions,
    dx,
    dt,
    moving_source_decay_rate,
    n_steps,
):
    t = timestep % n_steps
    for particle_id in range(particle_positions.shape[0]):
        x_bin = int(np.rint(particle_positions[particle_id, 0] / dx))
        y_bin = int(np.rint(particle_positions[particle_id, 1] / dx))
        s[t, x_bin, y_bin] += (production_strength / (dx**2)) * np.exp(
            -timestep * dt * moving_source_decay_rate
        )
        # s[t, x_bin, y_bin] += (production_strength / (dx ** 2))
    return s


# define the source function for a static point source
def static_point_source(
    s,
    production_strength,
    timestep,
    source_position,
    dx,
    dt,
    static_source_decay_rate,
    n_steps,
):
    t = timestep % n_steps
    x_bin = int(np.rint(source_position[0] / dx))
    y_bin = int(np.rint(source_position[1] / dx))
    s[t, x_bin, y_bin] += (production_strength / (dx**2)) * np.exp(
        -timestep * dt * static_source_decay_rate
    )
    # s[t, x_bin, y_bin] += (production_strength / (dx ** 2))
    return s


# update the flow field based on the distance from the particle
def update_flow_field(
    ux, uy, timestep, particle_positions, dx, dt, moving_source_decay_rate, n_steps
):
    t = timestep % n_steps
    # Calculate the shape of the flow velocity arrays
    nx, ny = ux[t, :, :].shape

    # Create mesh grids of indices
    x_indices, y_indices = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")

    for particle_id in range(particle_positions.shape[0]):
        # Calculate the distances from the particle coordinates to grid points
        dist_x = x_indices * dx - particle_positions[particle_id, 0]
        dist_y = y_indices * dx - particle_positions[particle_id, 1]

        # Calculate the distances using the 1/r^n dependency, avoiding division by zero
        r = np.sqrt(dist_x**2 + dist_y**2)
        # r[r < 1e-6] = 1e-6  # Avoid division by zero by adding a small value

        # Calculate the flow field based on the 1/r^n dependency
        # n=1
        # ux_update = 10**(-1) * dist_x / r**(n+1)
        # uy_update = 10**(-1) * dist_y / r**(n+1)

        # Calculate the flow field based on the re^(-r) dependency
        ux_update = (
            2.3
            * 10 ** (4)
            * dist_x
            * np.exp(-3.5 * r)
            * np.exp(-timestep * dt * moving_source_decay_rate)
        )
        uy_update = (
            2.3
            * 10 ** (4)
            * dist_y
            * np.exp(-3.5 * r)
            * np.exp(-timestep * dt * moving_source_decay_rate)
        )

        # Update the flow field
        ux[t, :, :] += ux_update
        uy[t, :, :] += uy_update

    return ux, uy


# define noflux boundary condition
def noflux(
    conc, t, x_diff_indices, y_diff_indices, x_diff_neg_indices, y_diff_neg_indices
):
    conc[t, x_diff_indices[:, 0], x_diff_indices[:, 1]] = conc[
        t, x_diff_indices[:, 0], x_diff_indices[:, 1] + 1
    ]
    conc[t, x_diff_neg_indices[:, 0], x_diff_neg_indices[:, 1] + 1] = conc[
        t, x_diff_neg_indices[:, 0], x_diff_neg_indices[:, 1]
    ]

    conc[t, y_diff_indices[:, 0], y_diff_indices[:, 1]] = conc[
        t, y_diff_indices[:, 0] + 1, y_diff_indices[:, 1]
    ]
    conc[t, y_diff_neg_indices[:, 0] + 1, y_diff_neg_indices[:, 1]] = conc[
        t, y_diff_neg_indices[:, 0], y_diff_neg_indices[:, 1]
    ]

    return conc


# define noflux boundary condition for a maze
def noflux_maze(
    conc,
    t,
    x_diff_indices,
    y_diff_indices,
    x_diff_neg_indices,
    y_diff_neg_indices,
    maze,
    regular_wall_mask=None,
):
    conc[t, x_diff_indices[:, 0], x_diff_indices[:, 1] + 1] = conc[
        t, x_diff_indices[:, 0], x_diff_indices[:, 1] + 2
    ]
    conc[t, x_diff_neg_indices[:, 0], x_diff_neg_indices[:, 1]] = conc[
        t, x_diff_neg_indices[:, 0], x_diff_neg_indices[:, 1] - 1
    ]

    conc[t, y_diff_indices[:, 0] + 1, y_diff_indices[:, 1]] = conc[
        t, y_diff_indices[:, 0] + 2, y_diff_indices[:, 1]
    ]
    conc[t, y_diff_neg_indices[:, 0], y_diff_neg_indices[:, 1]] = conc[
        t, y_diff_neg_indices[:, 0] - 1, y_diff_neg_indices[:, 1]
    ]

    if regular_wall_mask is not None:
        conc[t, regular_wall_mask] = 0.0
    else:
        conc[t, maze == 0] = 0.0

    return conc


def chemotaxis_force(position, t, c, maze, dx, Bp, active_mask=None):
    num_particles = position.shape[0]
    all_forces = np.zeros((num_particles, 2))
    if active_mask is None:
        active_mask = np.ones(num_particles, dtype=bool)

    for particle_id in np.where(active_mask)[0]:
        x_bin = int(np.rint(position[particle_id, t, 0] / dx))
        y_bin = int(np.rint(position[particle_id, t, 1] / dx))

        particle_force = np.zeros(2)

        if maze[x_bin, y_bin] != 0:
            particle_force[0] = (
                Bp
                * (c[t + 1, x_bin + 1, y_bin] - c[t + 1, x_bin - 1, y_bin])
                / (2.0 * dx)
            )
            particle_force[1] = (
                Bp
                * (c[t + 1, x_bin, y_bin + 1] - c[t + 1, x_bin, y_bin - 1])
                / (2.0 * dx)
            )

        if maze[x_bin + 1, y_bin] == 0 or maze[x_bin - 1, y_bin] == 0:
            particle_force[0] = 0.0
            particle_force[1] = (
                Bp
                * (c[t + 1, x_bin, y_bin + 1] - c[t + 1, x_bin, y_bin - 1])
                / (2.0 * dx)
            )

        if maze[x_bin, y_bin + 1] == 0 or maze[x_bin, y_bin - 1] == 0:
            particle_force[0] = (
                Bp
                * (c[t + 1, x_bin + 1, y_bin] - c[t + 1, x_bin - 1, y_bin])
                / (2.0 * dx)
            )
            particle_force[1] = 0.0

        if (maze[x_bin + 1, y_bin] == 0 or maze[x_bin - 1, y_bin] == 0) and (
            maze[x_bin, y_bin + 1] == 0 or maze[x_bin, y_bin - 1] == 0
        ):
            particle_force[0] = 0.0
            particle_force[1] = 0.0

        all_forces[particle_id] = particle_force

    return all_forces


def interaction_force(position, t, dx, epsilon_LJ, active_mask=None):
    num_particles = position.shape[0]
    interaction_forces = np.zeros((num_particles, num_particles, 2))
    if active_mask is None:
        active_mask = np.ones(num_particles, dtype=bool)

    active_indices = np.where(active_mask)[0]
    sigma = dx

    for i_index in range(len(active_indices)):
        i = active_indices[i_index]
        for j_index in range(i_index + 1, len(active_indices)):
            j = active_indices[j_index]
            disp = position[j, t, :] - position[i, t, :]
            dist = np.linalg.norm(disp)

            if dist != 0:
                lj_force = (
                    epsilon_LJ
                    * (12.0 * (sigma / dist) ** 12 - 6.0 * (sigma / dist) ** 6)
                    * disp
                    / dist**2
                )
                interaction_forces[i, j] = -lj_force
                interaction_forces[j, i] = lj_force

    return interaction_forces


def wall_force(position, t, wall_coords, dx, active_mask=None):
    num_particles = position.shape[0]
    if active_mask is None:
        active_mask = np.ones(num_particles, dtype=bool)

    r_cut = 2.0 * dx
    sigma = r_cut / 2.0 ** (1.0 / 6.0)
    forces = np.zeros((num_particles, 2))

    for i in np.where(active_mask)[0]:
        disp_from_wall = position[i, t, :] - wall_coords
        dist = np.linalg.norm(disp_from_wall, axis=1)
        if min(dist) < r_cut:
            index = np.argmin(dist)

            # WCA potential
            epsilon_wca = 10.0
            force_magnitude = (
                4.0
                * epsilon_wca
                * (
                    12.0 * (sigma / dist[index]) ** 12
                    - 6.0 * (sigma / dist[index]) ** 6
                )
            )
            forces[i] = force_magnitude * disp_from_wall[index] / (dist[index] ** 2)

            # Soft potential
            # epsilon_soft = 10000.0
            # force_magnitude = epsilon_soft * (np.pi / r_cut) * np.sin(np.pi * dist[index] / r_cut)
            # forces[i] = force_magnitude * disp_from_wall[index] / dist[index]

    return forces


def self_propulsion_force(
    position,
    timestep,
    theta,
    dt,
    sp_decay_rate,
    n_steps,
    self_propulsion_speed,
    active_mask=None,
):
    t = timestep % n_steps
    num_particles = position.shape[0]
    forces = np.zeros((num_particles, 2))
    if active_mask is None:
        active_mask = np.ones(num_particles, dtype=bool)

    for i in np.where(active_mask)[0]:
        forces[i, 0] = (
            self_propulsion_speed
            * np.cos(theta[i, t])
            * np.exp(-timestep * dt * sp_decay_rate)
        )
        forces[i, 1] = (
            self_propulsion_speed
            * np.sin(theta[i, t])
            * np.exp(-timestep * dt * sp_decay_rate)
        )
        # forces[i, 1] = self_propulsion_speed * np.sin(theta[i, t])

    return forces


def write_parameters(**parameters):
    param_filename = parameters.get("param_filename", "parameters.txt")
    with open(param_filename, "w") as param_file:
        for key, value in parameters.items():
            # param_file.write(f"{key}: {value:.2e}\n")
            param_file.write(f"{key}: {value}\n")


def write_grid(grid_filename, nx, ny):
    with open(grid_filename, "w") as f:
        f.write("BOX:\n")
        f.write(f"0 {nx}\n")
        f.write(f"0 {ny}\n")
        f.write("SHAPE:\n")
        f.write(f"{nx} {ny}\n")
        f.write("COORDINATES: X Y\n")

        for i in range(nx):
            for j in range(ny):
                f.write(f"{i} {j}\n")


def write_concentration(file_prefix, c, time_steps, n_steps):
    for t in time_steps:
        filename = f"{file_prefix}_{t}.txt"
        with open(filename, "w") as f:
            f.write("TIMESTEP:\n")
            f.write(f"{t}\n")
            f.write("DATA:c\n")
            np.savetxt(f, c[t % n_steps].ravel(), fmt="%.16f")


def write_particles(
    file_prefix,
    particles,
    theta,
    velocity,
    ang_velocity,
    forces_self_propulsion,
    forces_chemotaxis,
    forces_interaction,
    forces_wall,
    time_steps,
    num_particles,
    n_steps,
):
    for t in time_steps:
        filename = f"{file_prefix}_{t}.txt"
        with open(filename, "w") as f:
            f.write("TIMESTEP:\n")
            f.write(f"{t}\n")
            f.write(
                "DATA: particle_id x y theta vx vy omega f_spx f_spy f_chemx f_chemy f_intx f_inty f_wallx f_wally\n"
            )
            for particle_id in range(num_particles):
                f.write(
                    f"{particle_id} {particles[particle_id, t % n_steps, 0]} {particles[particle_id, t % n_steps, 1]} {theta[particle_id, t % n_steps]} "
                    f"{velocity[particle_id, t % n_steps, 0]} {velocity[particle_id, t % n_steps, 1]} {ang_velocity[particle_id, t % n_steps]} "
                    f"{forces_self_propulsion[particle_id, 0]} {forces_self_propulsion[particle_id, 1]} "
                    f"{forces_chemotaxis[particle_id, 0]} {forces_chemotaxis[particle_id, 1]} "
                    f"{forces_interaction[particle_id, :, 0].sum()} {forces_interaction[particle_id, :, 1].sum()} "
                    f"{forces_wall[particle_id, 0]} {forces_wall[particle_id, 1]}\n"
                )


def exit_condition(timestep, exit_times, px_bins, py_bins, exit_zone_map, dead_tracker):
    # A particle exits if it lands on a True cell in the exit map,
    # hasn't exited yet
    new_exits = exit_zone_map[px_bins, py_bins] & (exit_times == 0.0)

    if np.any(new_exits):
        print(f"{np.sum(new_exits)} particle(s) reached the exit region!")
        exit_times[new_exits] = timestep

    # Check if all particles have successfully exited or died
    has_exited_or_died = (exit_times > 0.0) | dead_tracker
    return np.all(has_exited_or_died)


def chemical_solver(
    c, position, theta, velocity, ang_velocity, maze, exit_times, start_step=0, **kwargs
):

    # unpacking the kwargs
    dx = kwargs.get("dx", 1)
    Dc = kwargs.get("Dc", 1)
    Dp = kwargs.get("Dp", 1)
    Bp = kwargs.get("Bp", 1)
    moving_source_production_strength = kwargs.get(
        "moving_source_production_strength", 1
    )
    static_source_production_strength = kwargs.get(
        "static_source_production_strength", 1
    )
    moving_source_decay_rate = kwargs.get("moving_source_decay_rate", 0)
    static_source_decay_rate = kwargs.get("static_source_decay_rate", 0)
    sp_decay_rate = kwargs.get("sp_decay_rate", 0)
    global_decay_strength = kwargs.get("global_decay_strength", 1)
    self_propulsion_speed = kwargs.get("self_propulsion_speed", 1)
    self_propulsion_frequency = kwargs.get("self_propulsion_frequency", 1)
    Dr = kwargs.get("Dr", 1)
    M = kwargs.get("M", 1)
    J = kwargs.get("J", 1)
    Lx = kwargs.get("Lx", 1)
    Ly = kwargs.get("Ly", 1)
    n_xbins = kwargs.get("n_xbins", 1)
    n_ybins = kwargs.get("n_ybins", 1)
    n_steps = kwargs.get("n_steps", 1)
    dt = kwargs.get("dt", 1)
    gamma = kwargs.get("gamma", 1)
    total_time = kwargs.get("total_time", 1)
    time_loop = kwargs.get("time_loop", 1)
    write_every = kwargs.get("write_every", 1)
    num_particles = kwargs.get("num_particles", 1)
    param_filename = kwargs.get("param_filename", 1)
    grid_filename = kwargs.get("grid_filename", 1)
    file_prefix_conc = kwargs.get("file_prefix_conc", 1)
    file_prefix_part = kwargs.get("file_prefix_part", 1)
    advection = kwargs.get("advection", False)
    massive_particle = kwargs.get("massive_particle", False)
    moving_source_decay = kwargs.get("moving_source_decay", False)
    static_source_decay = kwargs.get("static_source_decay", False)
    epsilon_LJ = kwargs.get("epsilon_LJ", 1)
    static_source_position = kwargs.get("static_source_position", (1, 1))
    exit_radius = kwargs.get("exit_radius", 20)
    birth_steps = kwargs.get("birth_steps", np.zeros(num_particles, dtype=int))
    active_mask = kwargs.get("active_mask", np.zeros(num_particles, dtype=bool))
    dead_tracker = kwargs.get("dead_tracker", np.zeros(num_particles, dtype=bool))
    death_zone_map = kwargs.get("death_zone_map", np.zeros_like(maze, dtype=bool))
    exit_zone_map = kwargs.get("exit_zone_map", np.zeros_like(maze, dtype=bool))
    grim_reaper_delay = kwargs.get("grim_reaper_delay", 0)
    exit_trigger_time = kwargs.get("exit_trigger_time", np.full(num_particles, np.inf))
    emitter_position = np.array(
        kwargs.get("emitter_position", (0.0, 0.0)), dtype=np.float32
    )
    permeability = kwargs.get("permeability", 0.0)
    drops_added_incremental = kwargs.get("drops_added_incremental", True)
    exit_wall_mask = kwargs.get("exit_wall_mask", None)
    grim_reaper_delay_timestep = grim_reaper_delay / dt

    nt, nx, ny = c.shape
    wall = maze == 0
    x_diff = np.diff(maze, axis=1)
    y_diff = np.diff(maze, axis=0)

    # Precalculate x_diff and y_diff indices
    x_diff_indices = np.transpose(np.where(x_diff == 1))
    y_diff_indices = np.transpose(np.where(y_diff == 1))
    x_diff_neg_indices = np.transpose(np.where(x_diff == -1))
    y_diff_neg_indices = np.transpose(np.where(y_diff == -1))

    # Filter boundary-transition indices and wall coordinates for leaky walls
    if permeability > 0.0 and exit_wall_mask is not None:
        in_exit_wall_x = exit_wall_mask[x_diff_indices[:, 0], x_diff_indices[:, 1]]
        x_diff_indices_reg = x_diff_indices[~in_exit_wall_x]

        in_exit_wall_x_neg = exit_wall_mask[
            x_diff_neg_indices[:, 0], x_diff_neg_indices[:, 1] + 1
        ]
        x_diff_neg_indices_reg = x_diff_neg_indices[~in_exit_wall_x_neg]

        in_exit_wall_y = exit_wall_mask[y_diff_indices[:, 0], y_diff_indices[:, 1]]
        y_diff_indices_reg = y_diff_indices[~in_exit_wall_y]

        in_exit_wall_y_neg = exit_wall_mask[
            y_diff_neg_indices[:, 0] + 1, y_diff_neg_indices[:, 1]
        ]
        y_diff_neg_indices_reg = y_diff_neg_indices[~in_exit_wall_y_neg]

        regular_wall_mask = (maze == 0) & ~exit_wall_mask
        wall_coords = np.transpose(np.where((maze == 0) & ~exit_wall_mask)) * dx
    else:
        x_diff_indices_reg = x_diff_indices
        x_diff_neg_indices_reg = x_diff_neg_indices
        y_diff_indices_reg = y_diff_indices
        y_diff_neg_indices_reg = y_diff_neg_indices
        regular_wall_mask = maze == 0
        wall_coords = np.transpose(np.where(maze == 0)) * dx

    source = np.full((nt, nx, ny), 0.0, dtype=np.float32)
    ux = np.full((nt, nx, ny), 0.0, dtype=np.float32)
    uy = np.full((nt, nx, ny), 0.0, dtype=np.float32)

    f_sp = np.full((num_particles, n_steps, 2), 0.0, dtype=np.float32)
    f_chem = np.full((num_particles, n_steps, 2), 0.0, dtype=np.float32)
    f_int = np.full((num_particles, n_steps, 2), 0.0, dtype=np.float32)
    f_wall = np.full((num_particles, n_steps, 2), 0.0, dtype=np.float32)

    exit = False
    exit_timestep = np.inf

    px_bins = np.zeros(num_particles, dtype=int)
    py_bins = np.zeros(num_particles, dtype=int)

    for timestep in range(start_step, start_step + nt - 1):
        t = timestep % nt

        if drops_added_incremental:
            new_births = birth_steps == timestep
            min_clearance = 1.5 * dx

            spawning_indices = np.where(new_births)[
                0
            ]  # indices of particles trying to spawn right now
            if len(spawning_indices) > 0:
                is_clear = True

                if np.any(active_mask):
                    # Pre-extract active positions for speed
                    active_positions = position[active_mask, t, :]

                    # Vectorized distance calculation to the single emitter point
                    dist_to_emitter = np.linalg.norm(
                        active_positions - emitter_position, axis=1
                    )

                    if np.min(dist_to_emitter) < min_clearance:
                        is_clear = False

                if is_clear:
                    first_particle_idx = spawning_indices[0]
                    active_mask[first_particle_idx] = (
                        True  # Spawn the first particle in the queue
                    )

                    # Delay the rest (if any) because they can't occupy the exact same spot
                    if len(spawning_indices) > 1:
                        birth_steps[spawning_indices[1:]] += 1
                else:
                    # The emitter is blocked; delay EVERYONE in the queue
                    birth_steps[spawning_indices] += 1
        else:
            if timestep == 0:
                active_mask[:] = True

        inactive_mask = ~active_mask
        position[inactive_mask, t + 1, :] = position[inactive_mask, t, :]
        velocity[inactive_mask, t + 1, :] = 0.0
        theta[inactive_mask, t + 1] = theta[inactive_mask, t]
        ang_velocity[inactive_mask, t + 1] = ang_velocity[inactive_mask, t]

        A = c[t, 2:, 1:-1]  # c[k, i+1, j]
        B = c[t, :-2, 1:-1]  # c[k, i-1, j]
        C = c[t, 1:-1, 2:]  # c[k, i, j+1]
        D = c[t, 1:-1, :-2]  # c[k, i, j-1]
        E = c[t, 1:-1, 1:-1]  # c[k, i, j]

        particle_positions = position[:, t, :]
        source = moving_point_source(
            source,
            moving_source_production_strength,
            timestep,
            particle_positions[active_mask],
            dx,
            dt,
            moving_source_decay_rate,
            n_steps,
        )
        source = static_point_source(
            source,
            static_source_production_strength,
            timestep,
            static_source_position,
            dx,
            dt,
            static_source_decay_rate,
            n_steps,
        )
        S = source[t, 1:-1, 1:-1]
        result = (
            gamma * (A + B + C + D - 4 * E)
            + E
            + S * dt
            - global_decay_strength * dt * E
        )

        if advection == True:
            # Calculate fluid velocity
            ux, uy = update_flow_field(
                ux,
                uy,
                timestep,
                particle_positions[active_mask],
                dx,
                dt,
                moving_source_decay_rate,
                n_steps,
            )

            # Calculate advection term
            adv = -(dt / (2.0 * dx)) * (
                np.multiply(ux[t, 1:-1, 1:-1], (A - B))
                + np.multiply(uy[t, 1:-1, 1:-1], (C - D))
            )

            # Add advection term to the result
            result += adv

        # Set the newly computed heatmap at time k+1
        c[t + 1, 1:-1, 1:-1] = result

        # Noflux boundary condition for maze
        noflux_maze(
            c,
            t + 1,
            x_diff_indices_reg,
            y_diff_indices_reg,
            x_diff_neg_indices_reg,
            y_diff_neg_indices_reg,
            maze,
            regular_wall_mask,
        )

        # Leaky boundary condition for exit walls
        if permeability > 0.0 and exit_wall_mask is not None:
            c = leaky_exit_wall(c, t + 1, exit_wall_mask, permeability, dt)

        # Forces
        forces_interaction = interaction_force(
            position, t, dx, epsilon_LJ, active_mask=active_mask
        )
        forces_chemotaxis = chemotaxis_force(
            position, t, c, maze, dx, Bp, active_mask=active_mask
        )
        forces_wall = wall_force(position, t, wall_coords, dx, active_mask=active_mask)
        forces_self_propulsion = self_propulsion_force(
            position,
            timestep,
            theta,
            dt,
            sp_decay_rate,
            n_steps,
            self_propulsion_speed,
            active_mask=active_mask,
        )

        forces_interaction_sum = forces_interaction.sum(axis=1)

        f_int[:, t, :] = forces_interaction_sum
        f_chem[:, t, :] = forces_chemotaxis
        f_wall[:, t, :] = forces_wall
        f_sp[:, t, :] = forces_self_propulsion

        dt_noise = np.sqrt(2.0 * Dp * dt)
        dr_noise = np.sqrt(2.0 * Dr * dt)
        # Generate random noise for ALL particles at once (faster than generating it inside a loop)
        noise_x = dt_noise * np.random.normal(0, 1, size=num_particles)
        noise_y = dt_noise * np.random.normal(0, 1, size=num_particles)
        noise_theta = dr_noise * np.random.normal(0, 1, size=num_particles)

        if massive_particle:
            # Update Active particles simultaneously using the mask ---
            # Position
            position[active_mask, t + 1, 0] = (
                position[active_mask, t, 0] + dt * velocity[active_mask, t, 0]
            )
            position[active_mask, t + 1, 1] = (
                position[active_mask, t, 1] + dt * velocity[active_mask, t, 1]
            )

            # Velocity
            velocity[active_mask, t + 1, 0] = velocity[active_mask, t, 0] + (1 / M) * (
                -dt * velocity[active_mask, t, 0]
                + dt * forces_self_propulsion[active_mask, 0]
                + dt * forces_chemotaxis[active_mask, 0]
                + dt * forces_interaction_sum[active_mask, 0]
                + dt * forces_wall[active_mask, 0]
                + noise_x[active_mask]
            )
            velocity[active_mask, t + 1, 1] = velocity[active_mask, t, 1] + (1 / M) * (
                -dt * velocity[active_mask, t, 1]
                + dt * forces_self_propulsion[active_mask, 1]
                + dt * forces_chemotaxis[active_mask, 1]
                + dt * forces_interaction_sum[active_mask, 1]
                + dt * forces_wall[active_mask, 1]
                + noise_y[active_mask]
            )

            # Angular
            theta[active_mask, t + 1] = (
                theta[active_mask, t] + dt * ang_velocity[active_mask, t]
            )
            ang_velocity[active_mask, t + 1] = ang_velocity[active_mask, t] + (
                1 / J
            ) * (
                -dt * ang_velocity[active_mask, t]
                + dt * self_propulsion_frequency
                + noise_theta[active_mask]
            )
        else:
            # --- Update Remaining Active Particles ---
            position[active_mask, t + 1, 0] = position[active_mask, t, 0] + (
                dt * forces_self_propulsion[active_mask, 0]
                + dt * forces_chemotaxis[active_mask, 0]
                + dt * forces_interaction_sum[active_mask, 0]
                + dt * forces_wall[active_mask, 0]
                + noise_x[active_mask]
            )
            position[active_mask, t + 1, 1] = position[active_mask, t, 1] + (
                dt * forces_self_propulsion[active_mask, 1]
                + dt * forces_chemotaxis[active_mask, 1]
                + dt * forces_interaction_sum[active_mask, 1]
                + dt * forces_wall[active_mask, 1]
                + noise_y[active_mask]
            )

            theta[active_mask, t + 1] = theta[active_mask, t] + (
                dt * self_propulsion_frequency + noise_theta[active_mask]
            )

            # Calculate velocities based on position change
            velocity[active_mask, t + 1, 0] = (
                position[active_mask, t + 1, 0] - position[active_mask, t, 0]
            ) / dt
            velocity[active_mask, t + 1, 1] = (
                position[active_mask, t + 1, 1] - position[active_mask, t, 1]
            ) / dt
            ang_velocity[active_mask, t + 1] = (
                theta[active_mask, t + 1] - theta[active_mask, t]
            ) / dt

        if np.any(active_mask):
            px_bins[active_mask] = np.rint(position[active_mask, t, 0] / dx).astype(int)
            py_bins[active_mask] = np.rint(position[active_mask, t, 1] / dx).astype(int)

            # Clip indices safely within maze boundaries
            px_bins[active_mask] = np.clip(px_bins[active_mask], 0, maze.shape[0] - 1)
            py_bins[active_mask] = np.clip(py_bins[active_mask], 0, maze.shape[1] - 1)

        # Write concentration and particle data at specified intervals
        if timestep == 0:
            write_parameters(**kwargs)
            write_grid(grid_filename, nx, ny)

        if timestep % write_every == 0:
            write_concentration(file_prefix_conc, c, [timestep], n_steps)
            write_particles(
                file_prefix_part,
                position,
                theta,
                velocity,
                ang_velocity,
                forces_self_propulsion,
                forces_chemotaxis,
                forces_interaction,
                forces_wall,
                [timestep],
                num_particles,
                n_steps,
            )

        # exit condition

        hit_exit_zone = (
            active_mask & death_zone_map[px_bins, py_bins] & np.isinf(exit_trigger_time)
        )

        if np.any(hit_exit_zone):
            exit_trigger_time[hit_exit_zone] = timestep
            print("particle(s) reached the exit zone at timestep: " + str(start_step + timestep) + " ;Simulation Time: "+ str((start_step + timestep)*dt))

        ready_to_reap = (
            ~dead_tracker
            & np.isfinite(exit_trigger_time)
            & ((timestep - exit_trigger_time) >= grim_reaper_delay_timestep)
        )

        if np.any(ready_to_reap):
            dead_tracker[ready_to_reap] = True
            position[ready_to_reap, t:, :] = np.nan
            velocity[ready_to_reap, t:, :] = np.nan
            exit_times[ready_to_reap] = exit_trigger_time[ready_to_reap]  # use timestep to find time of removal alternatively

        active_mask[dead_tracker] = False

        # Check if all particles have successfully exited or died
        has_exited_or_died = (exit_times > 0.0) | dead_tracker

        # if exit_condition(
        #     timestep, exit_times, px_bins, py_bins, exit_zone_map, dead_tracker
        # ):
        if np.all(has_exited_or_died):
            write_concentration(file_prefix_conc, c, [timestep], n_steps)
            write_particles(
                file_prefix_part,
                position,
                theta,
                velocity,
                ang_velocity,
                forces_self_propulsion,
                forces_chemotaxis,
                forces_interaction,
                forces_wall,
                [timestep],
                num_particles,
                n_steps,
            )
            exit = True
            exit_timestep = timestep
            break

    return (
        c,
        position,
        theta,
        velocity,
        ang_velocity,
        f_sp,
        f_chem,
        f_int,
        f_wall,
        exit,
        exit_timestep,
        exit_trigger_time,
    )
