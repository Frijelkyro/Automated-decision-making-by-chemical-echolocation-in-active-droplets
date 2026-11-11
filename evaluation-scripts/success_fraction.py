from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maze_functions import maze_from_file

DATA = ROOT / "data/emission_1_reaper_6_400s/emission_1_c0000"
MAZE = ROOT / "different_mazes/Ran_maze_size_prop_to_droplet.tsv"
TOLERANCE = 1.0  # Set this to your desired distance in maze units.
START = (0.0, 0.0)  # Set to the actual particle starting position.
OUT = ROOT / "evaluation-scripts/success_fractions"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    maze = maze_from_file(MAZE)

    # Build a centerline from the maze's shortest path.
    # This assumes maze_functions provides a 2D grid-like maze.
    from scipy.sparse.csgraph import dijkstra
    from scipy.sparse import csr_matrix

    grid = np.asarray(maze)
    walkable = grid.astype(bool)
    points = np.argwhere(walkable)
    index = {tuple(p): i for i, p in enumerate(points)}
    rows, cols = [], []
    for p, i in index.items():
        for step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (p[0] + step[0], p[1] + step[1])
            if q in index:
                rows.append(i)
                cols.append(index[q])

    graph = csr_matrix((np.ones(len(rows)), (rows, cols)),
                       shape=(len(points), len(points)))
    start, end = index[tuple(points[0])], index[tuple(points[-1])]
    _, predecessors = dijkstra(graph, directed=False, indices=start,
                               return_predecessors=True)
    path = [end]
    while path[-1] != start:
        path.append(predecessors[path[-1]])
        if path[-1] < 0:
            raise ValueError("No path through maze")
    centerline = points[path[::-1]]
    tree = cKDTree(centerline)

    for shot in sorted(DATA.glob("shot_*")):
        files = sorted(shot.glob("part_*.npz"))
        if not files:
            continue

        good = np.zeros(400, dtype=int)
        total = np.zeros(400, dtype=int)

        for file in files:
            with np.load(file) as d:
                x, y = d["x"], d["y"]
            valid = np.isfinite(x) & np.isfinite(y)
            valid &= ~((x == START[0]) & (y == START[1]))
            xy = np.column_stack((x[valid], y[valid]))
            dist = np.full(400, np.inf)
            dist[valid] = tree.query(xy)[0]
            total += valid
            good += valid & (dist <= TOLERANCE)

        fraction = np.divide(good, total, out=np.full(400, np.nan),
                             where=total > 0)
        np.savetxt(OUT / f"{shot.name}_success_fraction.csv",
                   np.column_stack((np.arange(400), fraction)),
                   delimiter=",", header="particle_id,success_fraction",
                   comments="")

if __name__ == "__main__":
    main()
