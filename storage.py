import hashlib
import os
import tempfile
import time

import numpy as np


FREE_FORMAT = "free_v1"


def maze_hash(maze):
    data = np.asarray(maze)
    hasher = hashlib.sha256()
    hasher.update(str(data.shape).encode("ascii"))
    hasher.update(data.dtype.str.encode("ascii"))
    hasher.update(data.tobytes(order="C"))
    return hasher.hexdigest()


def build_free_index(maze):
    """Build a -1-padded C-order linear index for maze free cells."""
    maze = np.asarray(maze)
    free_index = np.full(maze.shape, -1, dtype=np.int64)
    free_positions = np.argwhere(maze != 0)
    for linear, (row, col) in enumerate(free_positions):
        free_index[row, col] = linear
    return free_index


def cell_to_lin(row, col=None, free_index=None):
    """Return the free-cell linear index for ``(row, col, free_index)``.

    ``cell_to_lin(free_index, row, col)`` is also accepted.
    """
    if isinstance(row, np.ndarray):
        if free_index is not None:
            free_index, row, col = row, int(col), int(free_index)
        elif isinstance(col, (tuple, list, np.ndarray)):
            free_index = row
            row, col = int(col[0]), int(col[1])
        else:
            raise ValueError("cell coordinates are required")
    if free_index is None:
        raise ValueError("free_index is required")
    return int(np.asarray(free_index)[row, col])


def pack_conc(conc, free_index):
    conc = np.asarray(conc)
    free_index = np.asarray(free_index)
    if conc.shape[-2:] != free_index.shape:
        raise ValueError(
            f"concentration shape {conc.shape[-2:]} does not match maze shape {free_index.shape}"
        )
    flat_conc = np.asarray(conc).reshape((-1, free_index.size))
    free_positions = free_index.ravel() >= 0
    return np.ascontiguousarray(flat_conc[:, free_positions])


def unpack_conc(packed, free_index, grid_shape=None):
    packed = np.asarray(packed)
    free_index = np.asarray(free_index)
    if packed.ndim == 1:
        packed = packed.reshape(1, -1)
    if grid_shape is None:
        grid_shape = free_index.shape
    if not np.asarray(packed).shape[-1] == np.count_nonzero(free_index >= 0):
        raise ValueError("packed concentration has the wrong free-cell count")
    result = np.zeros((*packed.shape[:-1], *grid_shape), dtype=packed.dtype)
    flat_result = result.reshape((-1, free_index.size))
    flat_index = free_index.ravel()
    free_positions = flat_index >= 0
    flat_result[:, free_positions] = packed.reshape((-1, int(free_positions.sum())))
    return result[0] if packed.shape[0] == 1 and packed.ndim == 2 else result


def save_free_index(directory, maze):
    directory = os.fspath(directory)
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, "free_cells.npz")
    if os.path.exists(filename):
        with np.load(filename) as existing:
            if str(existing["maze_hash"].item()) == maze_hash(maze):
                return filename
    free_index = build_free_index(maze)
    np.savez(
        filename,
        free_index=free_index,
        maze_hash=np.asarray(maze_hash(maze)),
        format=np.asarray("free_index_v1"),
    )
    return filename


def load_free_index(directory, maze=None):
    filename = os.path.join(os.fspath(directory), "free_cells.npz")
    with np.load(filename) as data:
        if maze is not None:
            actual = maze_hash(maze)
            stored = str(data["maze_hash"].item())
            if stored != actual:
                raise ValueError(
                    f"free_cells.npz maze hash mismatch: stored {stored}, expected {actual}"
                )
        return data["free_index"].copy()


def read_concentration(filename, directory=None, maze=None):
    filename = os.fspath(filename)
    with np.load(filename) as data:
        if "format" in data and str(data["format"].item()) == FREE_FORMAT:
            packed = data["concentration"].copy()
            if directory is None:
                directory = os.path.dirname(filename)
            free_index = load_free_index(directory, maze)
            grid_shape = tuple(int(v) for v in data["grid_shape"]) if "grid_shape" in data else free_index.shape
            return unpack_conc(packed, free_index, grid_shape)
        return data["concentration"].copy()


def _save_npz_benchmark(path, data):
    """Benchmark temporary writes and choose compression when size savings are useful."""
    plain_fd, plain_path = tempfile.mkstemp(suffix=".npz")
    os.close(plain_fd)
    compressed_fd, compressed_path = tempfile.mkstemp(suffix=".npz")
    os.close(compressed_fd)
    try:
        start = time.perf_counter()
        np.savez(plain_path, **data)
        plain_time = time.perf_counter() - start
        plain_size = os.path.getsize(plain_path)

        start = time.perf_counter()
        np.savez_compressed(compressed_path, **data)
        compressed_time = time.perf_counter() - start
        compressed_size = os.path.getsize(compressed_path)

        use_compressed = (
            compressed_size < plain_size * 0.85
            or compressed_time < plain_time
        )
        chosen = compressed_path if use_compressed else plain_path
        with open(chosen, "rb") as source:
            with open(path, "wb") as destination:
                destination.write(source.read())
    finally:
        for temporary in (plain_path, compressed_path):
            if os.path.exists(temporary):
                os.unlink(temporary)


def write_concentration(
    filename,
    conc,
    free_index,
    maze_hash_value=None,
    timestep=None,
    simulation_time=None,
):
    filename = os.fspath(filename)
    packed = pack_conc(conc, free_index)
    data = {
        "format": np.asarray(FREE_FORMAT),
        "concentration": packed,
        "grid_shape": np.asarray(free_index.shape, dtype=np.int64),
    }
    if maze_hash_value is not None:
        data["maze_hash"] = np.asarray(maze_hash_value)
    if timestep is not None:
        data["timestep"] = np.asarray(timestep)
    if simulation_time is not None:
        data["simulation_time"] = np.asarray(simulation_time)
    _save_npz_benchmark(filename, data)
