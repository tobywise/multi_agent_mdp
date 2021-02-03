from maMDP.grid_utils import *
import pytest
import numpy as np

@pytest.fixture()
def empty_grid_coords():
    """Generates grid coords and indexes for an empty grid"""
    grid_state_id, grid_idx = grid_coords(np.zeros((15, 10)))
    out = (grid_state_id, grid_idx)
    return out

@pytest.fixture()
def empty_hex_grid_distance():
    _, grid_idx = grid_coords(np.zeros((15, 10)))
    return hex_dist(grid_idx, grid_idx)

@pytest.fixture()
def hex_adjacency_matrix():
    _, grid_idx = grid_coords(np.zeros((15, 10)))
    return hex_adjacency(grid_idx)

@pytest.fixture()
def square_adjacency_matrix():
    _, grid_idx = grid_coords(np.zeros((15, 10)))
    return square_adjacency(grid_idx)


def test_grid_coords_number_of_outputs(empty_grid_coords):
    assert len(empty_grid_coords) == 2

def test_grid_state_ids_equal_number_states(empty_grid_coords):
    grid_state_id, grid_idx = empty_grid_coords
    assert len(grid_state_id) == 150

def test_grid_state_ids_unique(empty_grid_coords):
    grid_state_id, grid_idx = empty_grid_coords
    assert np.all(np.unique(grid_state_id, return_counts=True)[1] == 1)

def test_grid_coords_max_x(empty_grid_coords):
    grid_state_id, grid_idx = empty_grid_coords
    assert grid_idx[:, 0].max() == 14

def test_grid_coords_max_y(empty_grid_coords):
    grid_state_id, grid_idx = empty_grid_coords
    assert grid_idx[:, 1].max() == 9


def test_offest_distance_equals_0():
    assert offset_distance(3, 3, 3, 3) == 0

@pytest.mark.parametrize("x2,y2", [
    (3, 2),
    (4, 2),
    (4, 3),
    (3, 4),
    (2, 3),
    (2, 2)
])
def test_offset_distance_equals_1(x2, y2):
    assert offset_distance(3, 3, x2, y2) == 1

@pytest.mark.parametrize("x2,y2", [
    (2, 6),
    (0, 0),
    (2, 5),
    (1, 6),
    (5, 4),
    (0, 4)
])
def test_offset_distance_not_equal_1(x2, y2):
    assert offset_distance(3, 3, x2, y2) > 1


def test_cube_distance_equals_0():
    a_coord = np.array([0, 0, 0])
    b_coord = np.array([0, 0, 0])
    assert cube_distance(a_coord, b_coord) == 0

@pytest.mark.parametrize("b_coord", [
    np.array([1, 0, -1]),
    np.array([1, -1, 0]),
    np.array([0, -1, 1]),
    np.array([-1, 0, 1]),
    np.array([-1, 1, 0]),
    np.array([0, 1, -1]),
])
def test_cube_distance_equals_1(b_coord):
    a_coord = np.array([0, 0, 0])
    assert cube_distance(a_coord, b_coord) == 1

@pytest.mark.parametrize("b_coord", [
    np.array([1, 2, -1]),
    np.array([1, 7, 0]),
    np.array([3, -1, 1]),
    np.array([0, 7, 0]),
    np.array([3, 3, 3]),
    np.array([0, 2, -11]),
])
def test_cube_distance_not_equal_1(b_coord):
    a_coord = np.array([0, 0, 0])
    assert cube_distance(a_coord, b_coord) > 1


def test_hex_dist_diagonal_zero(empty_hex_grid_distance):
    assert np.all(np.diag(empty_hex_grid_distance) == 0) 

def test_hex_dist_symmetric(empty_hex_grid_distance):
    assert np.isclose(empty_hex_grid_distance, empty_hex_grid_distance.T).all()

def test_hex_dist_positive(empty_hex_grid_distance):
    assert np.all(empty_hex_grid_distance >= 0)

@pytest.mark.parametrize("offset_coords,cube_coords", [
    ([0, 0], np.array([0, 0, 0])),
    ([0, -1], np.array([0, 1, -1])),
    ([1, 0], np.array([1, 0, -1])),
    ([1, 1], np.array([1, -1, 0])),
    ([0, 1], np.array([0, -1, 1])),
    ([-1, 1], np.array([-1, 0, 1])),
    ([-1, 0], np.array([-1, 1, 0])),
])
def test_offset_to_cube(offset_coords, cube_coords):
    assert np.all(offset_to_cube(*offset_coords) == cube_coords)


@pytest.mark.parametrize("A,B,expected", [
    (33, 34, 0),
    (33, 43, 1),
    (33, 42, 2),
    (33, 32, 3),
    (33, 22, 4),
    (33, 23, 5),
])
def test_get_action_from_states_hex(A, B, expected):
    _, grid_idx = grid_coords(np.zeros((15, 10)))
    adjacency = hex_adjacency(grid_idx)
    action = get_action_from_states_hex(A, B, adjacency, grid_idx)
    assert action == expected


@pytest.mark.parametrize("A,B,expected", [
    (33, 34, 0),
    (33, 43, 1),
    (33, 32, 2),
    (33, 23, 3),
])
def test_get_action_from_states_square(A, B, expected):
    _, grid_idx = grid_coords(np.zeros((15, 10)))
    adjacency = square_adjacency(grid_idx)
    action = get_action_from_states_square(A, B, adjacency, grid_idx)
    assert action == expected


def test_hex_adjacency_diagonal(hex_adjacency_matrix):
    assert np.all(np.diag(hex_adjacency_matrix) == 0)

def test_hex_adjacency_symmetric(hex_adjacency_matrix):
    assert np.isclose(hex_adjacency_matrix, hex_adjacency_matrix.T).all()

def test_hex_adjacency_positive(hex_adjacency_matrix):
    assert np.all(hex_adjacency_matrix >= 0)


def test_square_adjacency_diagonal(square_adjacency_matrix):
    assert np.all(np.diag(square_adjacency_matrix) == 0)

def test_square_adjacency_symmetric(square_adjacency_matrix):
    assert np.isclose(square_adjacency_matrix, square_adjacency_matrix.T).all()

def test_square_adjacency_positive(square_adjacency_matrix):
    assert np.all(square_adjacency_matrix >= 0)