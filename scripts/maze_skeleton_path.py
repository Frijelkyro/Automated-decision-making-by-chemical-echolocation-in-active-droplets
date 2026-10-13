#!/usr/bin/env python3
"""Maze skeleton, optimal path, junctions, decision zones, and plotting.

This script validates the TSV wall/free convention, loads the maze, builds a
skeleton, finds the shortest wall-avoiding route from an emitter to a goal, and
identifies local Y-junction decision zones.
"""

from __future__ import annotations

import heapq
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAZE_FILE = Path("different_mazes/Ran_maze_size_prop_to_droplet.tsv")
EMITTER_POSITION = (2.1, 84.1)
STATIC_SOURCE_POSITION = (90.2, 10.5)  # project convention from maze_cluster_script.py
GOAL_TOLERANCE = 2.0
DX = 1.0

# Preferred wall convention for TSVs. The project traditionally inverts raw TSVs,
# but the actual file convention is 1 = wall and 0 = free space.
WALL_VALUE = 1

COST_LAMBDA = 60.0
COST_EPS = 0.5
COST_P = 2.5
SNAP_RADIUS = 12.0

ZONE_DISTANCES = {
    "J01": {"before": 8, "correct": 10, "wrong": 8},
    "J02": {"before": 6, "correct": 12, "wrong": 7},
}
DEFAULT_ZONE_DISTANCE = {"before": 8, "correct": 10, "wrong": 8}


def analysis_dir_for_maze(maze_path: str | Path) -> Path:
    """Return a per-maze directory for persisted analysis artifacts."""
    base_path = Path(maze_path)
    return base_path.parent / f"{base_path.stem}_analysis"


def normalize_zone_cfg(raw_cfg: object) -> dict[str, int]:
    """Normalize a user-editable zone config dictionary onto the allowed keys."""
    cfg = dict(DEFAULT_ZONE_DISTANCE)
    if not isinstance(raw_cfg, dict):
        return cfg
    for key in ("before", "correct", "wrong"):
        value = raw_cfg.get(key, DEFAULT_ZONE_DISTANCE[key])
        try:
            cfg[key] = int(value)
        except (TypeError, ValueError):
            cfg[key] = int(DEFAULT_ZONE_DISTANCE[key])
    return cfg


def load_zone_config(config_path: Path, junction_ids: list[str]) -> dict[str, dict[str, int]]:
    """Load per-junction distances, preserving existing values and filling defaults."""
    config: dict[str, dict[str, int]] = {}
    if config_path.exists():
        try:
            loaded = tomllib.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config = {jid: normalize_zone_cfg(loaded.get(jid)) for jid in loaded if isinstance(jid, str)}
        except (tomllib.TOMLDecodeError, OSError):
            config = {}

    final: dict[str, dict[str, int]] = {}
    for junction_id in junction_ids:
        final[junction_id] = normalize_zone_cfg(config.get(junction_id, DEFAULT_ZONE_DISTANCE))
    for extra_key, value in sorted(config.items()):
        if extra_key not in final:
            final[extra_key] = normalize_zone_cfg(value)
    return final


def write_zone_config(config_path: Path, zone_cfg: dict[str, dict[str, int]]) -> None:
    """Persist user-editable decision-zone distances as TOML."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# User-editable decision-zone distances for each junction.",
        "# Adjust values and rerun the script to regenerate masks and plots.",
        "",
    ]
    for junction_id in sorted(zone_cfg):
        lines.append(f"[{junction_id}]")
        for key in ("before", "correct", "wrong"):
            lines.append(f"{key} = {int(zone_cfg[junction_id].get(key, DEFAULT_ZONE_DISTANCE[key]))}")
        lines.append("")
    config_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def save_zone_masks(mask_path: Path, zones: dict[str, dict[str, set[tuple[int, int]]]], shape: tuple[int, int]) -> None:
    """Save each zone mask as a compressed uint8 array for downstream evaluation."""
    arrays: dict[str, np.ndarray] = {}
    for junction_id, masks in zones.items():
        for label in ("before", "correct", "wrong"):
            arr = np.zeros(shape, dtype=np.uint8)
            for rr, cc in masks.get(label, set()):
                arr[rr, cc] = 1
            arrays[f"{junction_id}_{label}"] = arr
    if arrays:
        np.savez_compressed(mask_path, **arrays)
    else:
        np.savez_compressed(mask_path, junction_ids=np.array([], dtype=str))


def save_metadata(metadata_path: Path, maze_path: Path, maze_data: dict, path_cells: list[tuple[int, int]],
                 junctions: list[dict], zones: dict[str, dict[str, set[tuple[int, int]]]],
                 emitter_grid: tuple[int, int], goal_grid: tuple[int, int], output_plot: Path) -> None:
    """Save maze metadata and junction summaries for future reuse."""
    lines = [
        f'maze_file = "{str(maze_path)}"',
        f'wall_value = {int(maze_data["wall_value"])}',
        f'dx = {float(DX)}',
        f'goal_tolerance = {float(GOAL_TOLERANCE)}',
        f'grid_shape = [{maze_data["free_mask"].shape[0]}, {maze_data["free_mask"].shape[1]}]',
        f'path_length = {len(path_cells)}',
        f'emitter_grid = [{emitter_grid[0]}, {emitter_grid[1]}]',
        f'goal_grid = [{goal_grid[0]}, {goal_grid[1]}]',
        f'plot_path = "{str(output_plot)}"',
        "",
        "[[junctions]]",
    ]

    for junction in junctions:
        center = junction["center"]
        lines.extend([
            "",
            "[[junctions]]",
            f'id = "{junction["id"]}"',
            f'center = [{center[0]}, {center[1]}]',
            f'path_index = {int(junction.get("path_index", 0))}',
            f'branch_count = {int(junction.get("branch_count", 0))}',
            f'kind = "{junction.get("kind", "Y")}"',
            f'neighbors = {list(junction.get("neighbors", []))}',  # TODO dont let neigbors break the TOML syntax
        ])

    metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


OUTPUT_PLOT = Path("output/maze_decision_zones.png")
OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def detect_wall_value(grid: np.ndarray, preferred: int = WALL_VALUE) -> int:
    """Infer the wall/free encoding from the data, with a manual override.

    We assume a valid maze has walls on the border, so the value with the larger
    border occupancy is treated as the wall label. If the input is ambiguous, the
    preferred value is kept.
    """
    values = np.unique(grid)
    if preferred in values:
        border_score = {
            1: float(np.mean(grid[0, :] == 1) + np.mean(grid[-1, :] == 1) + np.mean(grid[:, 0] == 1) + np.mean(grid[:, -1] == 1)),
            0: float(np.mean(grid[0, :] == 0) + np.mean(grid[-1, :] == 0) + np.mean(grid[:, 0] == 0) + np.mean(grid[:, -1] == 0)),
        }
        if border_score.get(1, -1.0) >= border_score.get(0, -1.0):
            return 1
        return 0

    if 0 in values and 1 in values:
        score0 = float(np.mean(grid[0, :] == 0) + np.mean(grid[-1, :] == 0) + np.mean(grid[:, 0] == 0) + np.mean(grid[:, -1] == 0))
        score1 = float(np.mean(grid[0, :] == 1) + np.mean(grid[-1, :] == 1) + np.mean(grid[:, 0] == 1) + np.mean(grid[:, -1] == 1))
        return 1 if score1 >= score0 else 0
    return int(preferred)


def load_maze(path: str | Path, wall_value: int | None = None) -> dict:
    """Load the TSV maze, validate encoding, and produce free-space mask.

    The project's own helper flips the maze on load, but the original TSV is
    checked here so the script can handle both conventions cleanly.
    """
    grid = np.loadtxt(path, delimiter="\t", dtype=float)
    if grid.ndim != 2:
        raise ValueError(f"Maze must be 2D, got shape {grid.shape} from {path}.")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"Maze contains NaN/Inf values in {path}.")
    if not np.isin(grid, [0.0, 1.0]).all():
        raise ValueError(f"Maze must contain only 0 and 1 values; found {np.unique(grid)[:10]}.")

    effective_wall_value = detect_wall_value(grid, preferred=wall_value if wall_value is not None else WALL_VALUE)
    free_mask = (grid != effective_wall_value)
    if not free_mask.any():
        raise ValueError("No free-space cells were found in the maze.")

    distance = ndi.distance_transform_edt(~free_mask)
    skeleton = skeletonize(free_mask)
    return {
        "grid": grid,
        "wall_value": effective_wall_value,
        "free_mask": free_mask,
        "distance": distance,
        "skeleton": skeleton,
    }


def sim_to_grid(point: tuple[float, float], shape: tuple[int, int], dx: float = DX) -> tuple[int, int]:
    """Map a simulation coordinate (x, y) to a grid index.

    The raw TSV is treated as an array with row 0 at the top. `y` is therefore
    converted as if the grid origin is the lower-left corner in simulation space,
    while the array itself still keeps its native row orientation.
    """
    x, y = point
    col = int(np.rint(x / dx))
    row = int(np.rint(y / dx))
    row = shape[0] - 1 - row
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        raise IndexError(f"Point {point} maps outside the maze grid {shape}.")
    return row, col


def grid_to_sim(cell: tuple[int, int], shape: tuple[int, int], dx: float = DX) -> tuple[float, float]:
    row, col = cell
    y = (shape[0] - 1 - row) * dx
    x = col * dx
    return x, y


def snap_to_free(point: tuple[float, float], free_mask: np.ndarray, dx: float = DX, tolerance: float = GOAL_TOLERANCE,
                clearance: np.ndarray | None = None, radius: float | None = None):
    """Snap a point to the most suitable free cell near it.

    We prefer cells that stay close to the target but also lie in the center of the
    local corridor, rather than choosing the absolute nearest wall-adjacent cell.
    """
    rows, cols = free_mask.shape
    if not free_mask.any():
        raise ValueError("Free mask is empty.")
    if radius is None:
        radius = tolerance

    coords = np.argwhere(free_mask)
    if len(coords) == 0:
        raise ValueError("No free cells available to snap the position to.")

    x, y = point
    cell_x = coords[:, 1] * dx
    cell_y = (rows - 1 - coords[:, 0]) * dx
    d = np.hypot(cell_x - x, cell_y - y)
    mask = d <= max(radius, tolerance)
    if not np.any(mask):
        raise ValueError(f"Could not snap {point} within tolerance {tolerance}. Best distance was {d.min():.2f}.")

    candidate_idx = np.where(mask)[0]
    if clearance is None:
        clearance = ndi.distance_transform_edt(~free_mask)
    scores = clearance[coords[candidate_idx, 0], coords[candidate_idx, 1]] - 0.5 * d[mask]
    best_idx = candidate_idx[int(np.argmax(scores))]
    best = tuple(coords[best_idx])
    if d[best_idx] > tolerance:
        raise ValueError(f"Could not snap {point} within tolerance {tolerance}. Best distance was {d[best_idx]:.2f}.")
    return tuple(best)


def build_graph(skeleton: np.ndarray):
    """Create a sparse 8-connected graph from the skeleton pixels."""
    graph = {}
    h, w = skeleton.shape
    for r, c in np.argwhere(skeleton):
        nbrs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and skeleton[rr, cc]:
                    nbrs.append((rr, cc))
        if nbrs:
            graph[(int(r), int(c))] = nbrs
    return graph


def shortest_path(graph, start, goal):
    if start == goal:
        return [start]
    q = [(start, [start])]
    seen = {start}
    while q:
        current, path = q.pop(0)
        for nxt in graph.get(current, []):
            if nxt == goal:
                return path + [goal]
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, path + [nxt]))
    return []


def junction_clusters(graph):
    """Group connected skeleton pixels with degree at least 3 into one junction."""
    candidates = {p for p, nbrs in graph.items() if len(nbrs) >= 3}
    seen = set()
    clusters = []
    for p in sorted(candidates):
        if p in seen:
            continue
        comp = set()
        stack = [p]
        seen.add(p)
        while stack:
            u = stack.pop()
            comp.add(u)
            for v in graph.get(u, []):
                if v in candidates and v not in seen:
                    seen.add(v)
                    stack.append(v)
        clusters.append(comp)
    return sorted(clusters, key=lambda c: (min(r for r, _ in c), min(c for _, c in c)))


def find_junctions(skeleton: np.ndarray, graph) -> list[dict]:
    """Produce a list of valid Y-junctions and mark ambiguous ones."""
    junctions = []
    for cluster in junction_clusters(graph):
        if not cluster:
            continue
        center = np.mean(np.array(list(cluster), dtype=float), axis=0)
        center = tuple(np.rint(center).astype(int))
        neighbors = sorted({n for p in cluster for n in graph.get(p, []) if n not in cluster})
        if len(neighbors) != 3:
            continue
        junctions.append(
            {
                "id": "Jxx",
                "center": center,
                "cluster": cluster,
                "neighbors": neighbors,
                "branch_count": len(neighbors),
                "kind": "Y",
            }
        )
    return junctions


def order_junctions_by_path(junctions: list[dict], path: list[tuple[int, int]]) -> list[dict]:
    """Re-order junction IDs by the path distance from the start, not by cluster index."""
    if not junctions:
        return []
    ordered = []
    for junction in junctions:
        center = junction["center"]
        nearest = min(path, key=lambda p: abs(p[0] - center[0]) + abs(p[1] - center[1]))
        junction = dict(junction)
        junction["path_index"] = path.index(nearest)
        ordered.append(junction)
    ordered.sort(key=lambda j: j["path_index"])
    for idx, junction in enumerate(ordered, 1):
        junction["id"] = f"J{idx:02d}"
    return ordered


def branch_mask(center, target, graph, skeleton, max_steps):
    """Return a small local branch neighborhood up to max_steps cells from center."""
    if center == target:
        return {center}
    path = shortest_path(graph, center, target)
    if not path:
        return set()
    keep = set(path[: min(len(path), max_steps + 1)])
    expanded = set()
    h, w = skeleton.shape
    for cell in keep:
        r, c = cell
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if (rr, cc) != center:
                    expanded.add((rr, cc))
    return expanded



ZONE_SIDE_CELLS = 5  # Odd number of grid cells per side


def build_zones(
    junctions: list[dict],
    graph,
    skeleton: np.ndarray,
    free_mask: np.ndarray,
    path: list[tuple[int, int]],
    zone_cfg: dict,
    dx: float = DX,
) -> dict:
    """Create square before/correct/wrong zones centered along each branch.

    before: center lies `before` cells back along the incoming optimal path.
    correct: center lies `correct` cells forward along the optimal path.
    wrong: center lies `wrong` cells along the alternative branch.

    Each square is clipped to free space. Different zones may overlap.
    """
    if not path:
        return {}

    path_set = set(path)
    path_indices = {cell: i for i, cell in enumerate(path)}
    half = max(0, int(ZONE_SIDE_CELLS) // 2)
    result = {}

    def square_mask(center, distance_cells):
        """Return a square centered on a grid cell, clipped to free space."""
        r, c = center
        r0, r1 = max(0, r - half), min(free_mask.shape[0], r + half + 1)
        c0, c1 = max(0, c - half), min(free_mask.shape[1], c + half + 1)

        mask = set()
        for rr in range(r0, r1):
            for cc in range(c0, c1):
                if free_mask[rr, cc]:
                    mask.add((rr, cc))
        return mask

    def trace_alternative(cluster, steps):
        """Trace a non-optimal branch away from the junction."""
        # Find branch edges leaving the junction cluster that are not
        # part of the optimal path.
        edges = sorted(
            (u, v)
            for u in cluster
            for v in graph.get(u, [])
            if v not in cluster and v not in path_set
        )
        if not edges:
            return None

        # Try each alternative edge. Follow a shortest branch path while
        # preventing a return through the junction or onto the optimal route.
        best_path = []
        for _, start in edges:
            queue = [(start, [start])]
            seen = {start}
            while queue:
                node, branch_path = queue.pop(0)
                if len(branch_path) > len(best_path):
                    best_path = branch_path
                if len(branch_path) >= steps:
                    continue

                for nxt in sorted(graph.get(node, [])):
                    if nxt in seen or nxt in cluster or nxt in path_set:
                        continue
                    seen.add(nxt)
                    queue.append((nxt, branch_path + [nxt]))

        return best_path[min(steps - 1, len(best_path) - 1)] if best_path else None

    for junction in junctions:
        jid = junction["id"]
        cluster = set(junction.get("cluster", []))
        if not cluster:
            cluster = {junction["center"]}

        # Anchor the junction to the closest point on the optimal path.
        path_index = min(
            range(len(path)),
            key=lambda i: min(
                abs(path[i][0] - r) + abs(path[i][1] - c)
                for r, c in cluster
            ),
        )

        cfg = zone_cfg.get(jid, DEFAULT_ZONE_DISTANCE)
        distances = {
            name: max(1, int(round(float(cfg.get(name, default)) / dx)))
            for name, default in DEFAULT_ZONE_DISTANCE.items()
        }

        # Incoming and outgoing centers are measured along the ordered
        # optimal path, not from a rounded junction-center pixel.
        before_index = max(0, path_index - distances["before"])
        correct_index = min(len(path) - 1, path_index + distances["correct"])

        before_center = path[before_index]
        correct_center = path[correct_index]

        wrong_center = trace_alternative(cluster, distances["wrong"])

        zones_for_junction = {
            "before": square_mask(before_center, distances["before"]),
            "correct": square_mask(correct_center, distances["correct"]),
            "wrong": (
                square_mask(wrong_center, distances["wrong"])
                if wrong_center is not None
                else set()
            ),
        }

        result[jid] = zones_for_junction

        print(
            f"{jid}: before center={before_center}, "
            f"correct center={correct_center}, "
            f"wrong center={wrong_center}; "
            f"cells={ {k: len(v) for k, v in zones_for_junction.items()} }"
        )

    return result


def build_path(free_mask: np.ndarray, skeleton: np.ndarray, start: tuple[float, float], goal: tuple[float, float],
               dx: float = DX, lam: float = COST_LAMBDA, eps: float = COST_EPS, power: float = COST_P):
    """Find the shortest wall-avoiding path using Dijkstra on the skeleton graph."""
    clearance = ndi.distance_transform_edt(~free_mask)
    graph = build_graph(skeleton)
    if not graph:
        raise ValueError("Skeleton graph is empty; cannot compute a path.")

    start_row, start_col = sim_to_grid(start, free_mask.shape, dx)
    goal_row, goal_col = sim_to_grid(goal, free_mask.shape, dx)

    start_candidates = [p for p in graph if abs(p[0] - start_row) + abs(p[1] - start_col) <= 15]
    goal_candidates = [p for p in graph if abs(p[0] - goal_row) + abs(p[1] - goal_col) <= 15]
    if not start_candidates or not goal_candidates:
        raise ValueError(f"No skeleton endpoints found near the emitter or goal: start={start}, goal={goal}.")

    start_cell = min(start_candidates, key=lambda p: ((p[0] - start_row) ** 2 + (p[1] - start_col) ** 2, -clearance[p]))
    goal_cell = min(goal_candidates, key=lambda p: ((p[0] - goal_row) ** 2 + (p[1] - goal_col) ** 2, -clearance[p]))

    pq = [(0.0, start_cell)]
    dist = {start_cell: 0.0}
    prev = {start_cell: None}

    while pq:
        cost, cell = heapq.heappop(pq)
        if cell == goal_cell:
            break
        if cost > dist.get(cell, np.inf):
            continue
        for nxt in graph.get(cell, []):
            step = np.hypot(abs(nxt[0] - cell[0]), abs(nxt[1] - cell[1]))
            local_clearance = min(clearance[cell], clearance[nxt])
            penalty = 1.0 + lam / (local_clearance + eps) ** power
            new_cost = cost + step * penalty
            if new_cost < dist.get(nxt, np.inf):
                dist[nxt] = new_cost
                prev[nxt] = cell
                heapq.heappush(pq, (new_cost, nxt))

    if goal_cell not in prev:
        raise ValueError(f"No valid path exists from {start} to {goal} on the skeleton graph.")

    path = []
    cur = goal_cell
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, start_cell, goal_cell


def plot_maze(grid: np.ndarray, free_mask: np.ndarray, skeleton: np.ndarray, path: list[tuple[int, int]],
              junctions: list[dict], zones: dict, emitter_cell: tuple[int, int], goal_cell: tuple[int, int],
              output_path: str | Path):
    """Plot the free space, skeleton, path, junctions, and decision zones."""
    fig, ax = plt.subplots(figsize=(10, 8))
    background = np.where(free_mask, 0.0, 1.0)
    ax.imshow(background, cmap="Greys", origin="upper", vmin=0, vmax=1)
    ax.imshow(skeleton.astype(float), cmap="cividis", alpha=0.5, origin="upper")

    if path:
        path_r = [p[0] for p in path]
        path_c = [p[1] for p in path]
        ax.plot(path_c, path_r, color="black", linewidth=2.0, alpha=0.9)

    for junction in junctions:
        r, c = junction["center"]
        ax.plot(c, r, "o", color="gold", markersize=7)
        ax.text(c + 1.0, r + 1.0, junction["id"], color="gold", fontsize=9, weight="bold")

    # for jid, masks in zones.items():
    #     for zone_name, color in (("before", "tab:blue"), ("correct", "tab:green"), ("wrong", "tab:red")):
    #         mask = masks.get(zone_name, set())
    #         if not mask:
    #             continue
    #         arr = np.zeros_like(background, dtype=float)
    #         for rr, cc in mask:
    #             arr[rr, cc] = 1.0
    #         arr = ndi.binary_dilation(arr.astype(bool), iterations=1).astype(float)
    #         ax.imshow(arr, origin="upper", cmap=matplotlib.colors.ListedColormap([color]), alpha=0.7, vmin=0, vmax=1)

    for jid, masks in zones.items():
        for zone_name, color in (
            ("before", "tab:blue"),
            ("correct", "tab:green"),
            ("wrong", "tab:red"),
        ):
            mask = masks.get(zone_name, set())
            if not mask:
                print(f"{jid} {zone_name}: EMPTY")
                continue

            arr = np.zeros_like(background, dtype=bool)
            for rr, cc in mask:
                arr[rr, cc] = True

            arr = ndi.binary_dilation(arr, iterations=1)

            # RGBA overlay: zone color where True, fully transparent elsewhere.
            rgba = np.zeros((*arr.shape, 4), dtype=float)
            rgba[arr] = matplotlib.colors.to_rgba(color, alpha=0.7)

            ax.imshow(
                rgba,
                origin="upper",
                interpolation="nearest",
            )

            print(f"{jid} {zone_name}: {len(mask)} cells")

    # markers for emitter and goal
    ax.plot(emitter_cell[1], emitter_cell[0], "s", color="dodgerblue", markersize=8)
    ax.plot(goal_cell[1], goal_cell[0], "*", color="crimson", markersize=10)
    ax.set_title("Maze, skeleton, optimal path, junctions, and decision zones")
    ax.set_xlabel("x (grid columns)")
    ax.set_ylabel("y (grid rows)")
    ax.legend(
        [
            plt.Line2D([0], [0], color="black", lw=2),
            plt.Line2D([0], [0], color="tab:blue", lw=4),
            plt.Line2D([0], [0], color="tab:green", lw=4),
            plt.Line2D([0], [0], color="tab:red", lw=4),
        ],
        ["optimal path", "before", "correct", "wrong"],
        loc="upper right",
    )
    ax.set_aspect("equal")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    path = Path(MAZE_FILE)
    analysis_dir = analysis_dir_for_maze(path)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    maze_data = load_maze(path)
    free_mask = maze_data["free_mask"]
    skeleton = maze_data["skeleton"]
    dist = maze_data["distance"]

    path_cells, start_cell, goal_cell = build_path(free_mask, skeleton, EMITTER_POSITION, STATIC_SOURCE_POSITION, dx=DX)
    emitter_grid = start_cell
    goal_grid = goal_cell

    graph = build_graph(skeleton)
    # drop the last junction, as this is most likely already at the goal
    junctions = order_junctions_by_path(find_junctions(skeleton, graph), path_cells)[:-1]
    junction_ids = [junction["id"] for junction in junctions]
    config_path = analysis_dir / "zone_config.toml"
    zone_cfg = load_zone_config(config_path, junction_ids)
    zones = build_zones(junctions, graph, skeleton, free_mask, path_cells, zone_cfg, dx=DX)

    for jid, masks in zones.items():
        print(
            f"{jid}: "
            f"before={len(masks['before'])}, "
            f"correct={len(masks['correct'])}, "
            f"wrong={len(masks['wrong'])}"
        )

    # Save a user-editable TOML config; changes are preserved across reruns.
    write_zone_config(config_path, zone_cfg)

    output_plot = analysis_dir / "maze_decision_zones.png"
    save_zone_masks(analysis_dir / "zone_masks.npz", zones, free_mask.shape)
    save_metadata(analysis_dir / "maze_metadata.toml", path, maze_data, path_cells, junctions, zones, emitter_grid, goal_grid, output_plot)

    print(f"Maze file: {path}")
    print(f"Encoded wall value: {maze_data['wall_value']}")
    print(f"Emitter on route: {emitter_grid} -> sim {grid_to_sim(emitter_grid, free_mask.shape, DX)}")
    print(f"Goal on route: {goal_grid} -> sim {grid_to_sim(goal_grid, free_mask.shape, DX)}")
    print(f"Path length: {len(path_cells)} cells")
    print(f"Detected junctions: {', '.join(j['id'] for j in junctions) if junctions else 'none'}")
    print(f"Zone config: {config_path}")
    print(f"Zone summary: {list(zones.keys())}")
    print(f"Saved analysis artifacts under {analysis_dir}")

    plot_maze(maze_data["grid"], free_mask, skeleton, path_cells, junctions, zones, emitter_grid, goal_grid, output_plot)

    result = {
        "maze_file": str(path),
        "wall_value": maze_data["wall_value"],
        "free_mask": free_mask,
        "distance_transform": dist,
        "skeleton": skeleton,
        "path": path_cells,
        "path_sim": [grid_to_sim(cell, free_mask.shape, DX) for cell in path_cells],
        "junctions": junctions,
        "zones": zones,
        "analysis_dir": str(analysis_dir),
        "zone_config": zone_cfg,
        "zone_config_path": str(config_path),
        "mask_path": str(analysis_dir / "zone_masks.npz"),
        "metadata_path": str(analysis_dir / "maze_metadata.toml"),
        "plot_path": str(output_plot),
    }
    return result


if __name__ == "__main__":
    main()
