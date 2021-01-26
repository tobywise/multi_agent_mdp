import numpy as np
from scipy.spatial.distance import cdist, euclidean
from typing import Union
from numba import njit

def grid_coords(state_grid:np.ndarray) -> Union[list, np.ndarray]:
    """
    Gets state coordinate (X, Y) indices and state IDs from an NxN grid of states

    Args:
        state_grid (np.ndarray): Grid of states, shape (x, y)

    Returns:
        list: List of state IDs
        np.ndarray: Array of state coordinate indices, shape (n_states, 2)
    """

    if not isinstance(state_grid, np.ndarray):
        raise TypeError("Grid must be a numpy array")

    if state_grid.ndim != 2:
        raise AttributeError("Grid must be 2 dimensional")

    grid_idx = np.array(np.where(~np.isnan(state_grid))).T
    grid_state_id = list(range(grid_idx.shape[0]))

    return grid_state_id, grid_idx

@njit
def offset_distance(x1:int, y1:int, x2:int, y2:int) -> int:
    """
    Gets distance between two hex cells using offset coordinates (in "even-q" format)

    Args:
        x1 (int): X position of cell 1
        y1 (int): Y position of cell 1
        x2 (int): X position of cell 2
        y2 (int): Y position of cell 2

    Returns:
        int: Distance between cells
    """

    ac = offset_to_cube(x1,y1)
    bc = offset_to_cube(x2,y2)
    f = cube_distance(ac, bc)

    return f

@njit
def cube_distance(a:np.ndarray, b:np.ndarray) -> int:
    """
    Calculates the distance between a pair of hexagons (in cube coordinate space). Each coordinate
    is represented by an array of 3 values.

    Args:
        a (np.ndarray): X, Y, Z coordinate of the first hexagon
        b (np.ndarray): X, Y, Z coordinate of the first hexagon

    Returns:
        int: distance between the two hexagons
    """
    assert len(a) == 3 & len(b) == 3, 'Length of both coordinates must be 3 (x, y, z)'

    return (np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]) + np.abs(a[2] - b[2])) / 2

@njit
def hex_dist(XA:list, XB:list) -> np.ndarray:
    """
    Gets the distance between two vectors of hex coordinates (in offset coordinate space)

    Args:
        XA (list): List of A coordinates (each represented as a tuple)
        XB (list): List of B coordinates (each represented as a tuple)

    Returns:
        np.ndarray: [description]
    """

    dist = np.zeros((len(XA), len(XB)))

    for n, i in enumerate(XA):
        for nn, j in enumerate(XB):
            dist[n, nn] = offset_distance(i[0], i[1], j[0], j[1])

    return dist

@njit
def offset_to_cube(col:int, row:int) -> np.ndarray:
    """
    Takes offset hex coordinates (a column and a row) and converts to cube coordinates.

    Args:
        col (int): Column coordinate
        row (int): Row coordinate
    Returns:
        np.ndarray: (x, y, z) coordinates in cube coordinate space
    """
    col = int(col)
    row = int(row)
    x = col
    z = row - (col + np.bitwise_and(col, 1)) / 2
    y = -x-z
    f = np.array([x,y,z])
    return f


def get_action_from_states_hex(A:int, B:int, T:np.ndarray, coords:np.ndarray) -> int:
    """
    Takes a pair of adjacent states on a hex grid and gives the action needed to move from A to B.

    Only works for a deterministic MDP where each state leads to another through a single action.

    Args:
        A (int): State A ID
        B (int): State B ID
        T (np.ndarray): Adjacency matrix
        coords (np.ndarray): Grid coordinates of all hexagons

    Returns:
        int: Action that moves from A to B
    """

    if not T[A, B]:
        raise ValueError("A and B are not adjacent")

    x1, y1 = coords[A, :]
    x2, y2 = coords[B, :]

    dx, dy = x2-x1, y2-y1

    if dx == 0:
        if dy > 0:
            action = 0
        if dy < 0:
            action = 3
    else:
        if x1 % 2:
            if dx > 0 and dy == 0:
                action = 1
            elif dx < 0 and dy == 0:
                action = 5
            elif dx > 0 and dy < 0:
                action = 2
            elif dx < 0 and dy < 0:
                action = 4
        else:
            if dx > 0 and dy > 0:
                action = 1
            elif dx < 0 and dy > 0:
                action = 5
            elif dx > 0 and dy == 0:
                action = 2
            elif dx < 0 and dy == 0:
                action = 4
    
    return action

def get_action_from_states_square(A:int, B:int, T:np.ndarray, coords:np.ndarray) -> int:
    """
    Takes a pair of adjacent states on square grid and gives the action needed to move from A to B. Assumes
    that the agent can move up, down, left, right - no diagonal moves.

    Only works for a deterministic MDP where each state leads to another through a single action.

    Args:
        A (int): State A ID
        B (int): State B ID
        T (np.ndarray): Adjacency matrix
        coords (np.ndarray): Grid coordinates of all hexagons

    Returns:
        int: Action that moves from A to B
    """

    if not T[A, B]:
        raise ValueError("A and B are not adjacent")

    x1, y1 = coords[A, :]
    x2, y2 = coords[B, :]

    dx, dy = x2-x1, y2-y1

    if dx == 0:
        if dy > 0:
            action = 0
        if dy < 0:
            action = 2
    elif dx > 0:
        action = 1
    elif dx < 0:
        action = 3

    return action

def hex_adjacency(grid_idx:np.ndarray) -> np.ndarray:
    """
    Generates an adjacency matrix from an array of state coordinates. Grid must represent a hexagonal grid using
    "even-q" coordinates

    Args:
        grid_idx (np.ndarray): State coordinate indices, shape (n_states, 2)

    Returns:
        np.ndarray: Adjacency matrix, shape (n_states, n_states)
    """

    adjacency = (hex_dist(grid_idx,grid_idx)==1).astype(int)
    
    return adjacency

def square_adjacency(grid_coords:np.ndarray) -> np.ndarray:
    """
    Generates an adjacency matrix from an array of state coordinates. Grid must be sqaure

    Args:
        grid_coords (np.ndarray): State coordinate indices, shape (n_states, 2)

    Returns:
        np.ndarray: Adjacency matrix, shape (n_states, n_states)
    """
    return (cdist(grid_coords, grid_coords) == 1).astype(int)
