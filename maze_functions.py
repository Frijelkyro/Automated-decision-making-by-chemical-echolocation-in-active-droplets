# Define the maze functions

# Import the necessary libraries
import numpy as np
import random

def random_maze(width, height):
    maze = np.ones((height, width), dtype=int)

    def is_valid(x, y):
        return 1 <= x < width - 1 and 1 <= y < height - 1

    def is_wall(x, y):
        return maze[y][x] == 1

    def can_carve(x, y):
        return is_valid(x, y) and is_wall(x, y)

    def carve(x, y):
        maze[y][x] = 0

    def carve_path(x1, y1, x2, y2):
        for y in range(min(y1, y2), max(y1, y2) + 1):
            for x in range(min(x1, x2), max(x1, x2) + 1):
                carve(x, y)

    stack = [(2, 2)]
    maze[2][2] = 0

    while stack:
        x, y = stack[-1]
        neighbors = []

        for dx, dy in [(4, 0), (-4, 0), (0, 4), (0, -4)]:
            nx, ny = x + dx, y + dy
            if can_carve(nx, ny):
                neighbors.append((nx, ny))

        if neighbors:
            nx, ny = random.choice(neighbors)
            carve_path(x, y, nx, ny)
            stack.append((nx, ny))
        else:
            stack.pop()

    # Fill first and last rows with 0s
    maze[0, :] = 0
    maze[height - 1, :] = 0

    # Fill first and last columns with 0s
    maze[:, 0] = 0
    maze[:, width - 1] = 0

    return maze


#define a maze within a single box
def box_maze(nx, ny):
    # Initialize solution: the grid of u(k, i, j)
    m = np.full((nx, ny), 1)

    # Set the boundary walls
    m[0, :] = 0
    m[:, 0] = 0
    m[-1, :] = 0
    m[:, -1] = 0

    return m

#define a custom maze
def custom_maze(nx, ny):
    # Initialize solution: the grid of u(k, i, j)
    m = np.full((nx, ny), 1)

    # Set the boundary walls
    m[0, :] = 0
    m[:, 0] = 0
    m[-1, :] = 0
    m[:, -1] = 0
    
    #add extra walls
    m[100:, 100] = 0
    m[100, 100:] = 0    
    

    return m

#maze from a file
def maze_from_file(file_path):
    # Read the TSV file into a 2D NumPy array
    data = np.loadtxt(file_path, delimiter="\t", dtype=int)
    
    # Invert the values (0s to 1s and 1s to 0s)
    inverted_data = 1 - data
    
    # Rotate the matrix by 90 degrees clockwise
    rotated_data = np.rot90(inverted_data, k=-1)
    
    return rotated_data


#function to load concentrations from a file
def load_c_from_file(maze, filename):
    nx=maze.shape[0]
    ny=maze.shape[1]
    c = np.loadtxt(filename, skiprows=3).reshape((nx, ny))
    return c


#function to load trajectory from a file
def load_traj_from_file(filename):
    traj = np.loadtxt(filename, skiprows=1)
    return traj