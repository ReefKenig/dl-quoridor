"""
Board State Tensor Specification
==================================
This file defines the EXACT tensor layout and provides a reference
implementation for computing each channel. Reef must produce identical
output from his QuoridorEnv.state_to_tensor().

Tensor shape: (board_size, board_size, 10)
All values normalized to [0, 1].
Indexed as tensor[row, col, channel].

Channel Breakdown
-----------------
Ch 0: Player 0 position    — binary, 1.0 at pawn location
Ch 1: Player 1 position    — binary, 1.0 at pawn location
Ch 2: Player 0 horizontal walls — binary, 1.0 where P0 placed horizontal walls
Ch 3: Player 0 vertical walls   — binary, 1.0 where P0 placed vertical walls
Ch 4: Player 1 horizontal walls — binary, 1.0 where P1 placed horizontal walls
Ch 5: Player 1 vertical walls   — binary, 1.0 where P1 placed vertical walls
Ch 6: Player 0 walls remaining  — uniform plane, value = remaining / max_walls
Ch 7: Player 1 walls remaining  — uniform plane, value = remaining / max_walls
Ch 8: Player 0 distance map     — BFS distance from each cell to P0's goal row, normalized by max possible distance
Ch 9: Player 1 distance map     — BFS distance from each cell to P1's goal row, normalized by max possible distance

Design Rationale
-----------------
- Channels 0-1: Spatial pawn positions, directly usable by convolutions.
- Channels 2-5: Wall ownership separated per player. The network can learn which walls belong to whom (important for strategy).
- Channels 6-7: Scalar resource info broadcast to every cell. This tells the network how many walls each player can still place.
- Channels 8-9: Pathfinding heuristic. Distance maps encode BFS shortest path information considering current wall layout. This is the most computationally expensive channel to compute but gives the network crucial strategic signal about board connectivity.

Wall Encoding Detail
--------------------
Walls span 2 cells. A horizontal wall at intersection (r, c) blocks movement between:
    (r, c)↔(r+1, c)  and  (r, c+1)↔(r+1, c+1)

A vertical wall at intersection (r, c) blocks movement between:
    (r, c)↔(r, c+1)  and  (r+1, c)↔(r+1, c+1)

In the tensor, we mark BOTH cells covered by each wall:
    Horizontal wall at (r, c): tensor[r, c, ch] = 1 AND tensor[r, c+1, ch] = 1
    Vertical wall at (r, c):   tensor[r, c, ch] = 1 AND tensor[r+1, c, ch] = 1

This keeps the spatial structure — convolutions can "see" wall extent.
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Set


# Wall grid size = board_size - 1
# Walls are indexed by their top-left intersection coordinate (r, c)
# where r ∈ [0, board_size-2] and c ∈ [0, board_size-2]


def compute_pawn_channel(board_size: int, pawn_row: int, pawn_col: int) -> np.ndarray:
    """
    Channel 0 or 1: binary pawn position.

    Returns:
        (board_size, board_size) array with 1.0 at pawn location, 0.0 elsewhere.
    """
    channel = np.zeros((board_size, board_size), dtype=np.float32)
    channel[pawn_row, pawn_col] = 1.0
    return channel


def compute_wall_channel(
    board_size: int,
    walls: List[Tuple[int, int]],
    orientation: str,
) -> np.ndarray:
    """
    Channels 2-5: wall placement map.

    Args:
        board_size: board dimension
        walls: list of (row, col) intersection coordinates where walls are placed
        orientation: "horizontal" or "vertical"

    Returns:
        (board_size, board_size) array with 1.0 at cells covered by walls.
    """
    channel = np.zeros((board_size, board_size), dtype=np.float32)
    wall_grid = board_size - 1

    for r, c in walls:
        if not (0 <= r < wall_grid and 0 <= c < wall_grid):
            continue

        if orientation == "horizontal":
            # Horizontal wall at (r, c) spans columns c and c+1
            channel[r, c] = 1.0
            if c + 1 < board_size:
                channel[r, c + 1] = 1.0
        elif orientation == "vertical":
            # Vertical wall at (r, c) spans rows r and r+1
            channel[r, c] = 1.0
            if r + 1 < board_size:
                channel[r + 1, c] = 1.0

    return channel


def compute_walls_remaining_channel(
    board_size: int,
    walls_remaining: int,
    max_walls: int,
) -> np.ndarray:
    """
    Channel 6 or 7: walls remaining as uniform plane.

    Value = walls_remaining / max_walls, broadcast to every cell.
    When all walls are placed, entire plane is 0.0.
    When no walls are placed yet, entire plane is 1.0.
    """
    if max_walls == 0:
        value = 0.0
    else:
        value = walls_remaining / max_walls

    return np.full((board_size, board_size), value, dtype=np.float32)


def compute_distance_map(
    board_size: int,
    goal_row: int,
    h_walls: Set[Tuple[int, int]],
    v_walls: Set[Tuple[int, int]],
) -> np.ndarray:
    """
    Channel 8 or 9: BFS distance from each cell to the goal row,
    considering current wall placements.

    Uses multi-source BFS starting from all cells in the goal row.
    Walls block edges between adjacent cells.

    Args:
        board_size: board dimension
        goal_row: target row (0 for P0, board_size-1 for P1)
        h_walls: set of (r, c) where horizontal walls are placed (any player)
        v_walls: set of (r, c) where vertical walls are placed (any player)

    Returns:
        (board_size, board_size) array, normalized to [0, 1].
        Cells unreachable from goal get value 1.0 (maximum distance).
    """
    max_dist = board_size * board_size  # upper bound
    dist = np.full((board_size, board_size), max_dist, dtype=np.float32)

    # Multi-source BFS from all cells in the goal row
    queue = deque()
    for col in range(board_size):
        dist[goal_row, col] = 0
        queue.append((goal_row, col))

    while queue:
        r, c = queue.popleft()
        current_dist = dist[r, c]

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < board_size and 0 <= nc < board_size):
                continue

            if dist[nr, nc] <= current_dist + 1:
                continue  # already visited with shorter/equal path

            # Check if wall blocks this edge
            if _is_blocked(r, c, nr, nc, h_walls, v_walls, board_size):
                continue

            dist[nr, nc] = current_dist + 1
            queue.append((nr, nc))

    # Normalize by board_size².  This is large enough to prevent
    # clipping even on heavily-walled boards (where BFS distances can
    # far exceed 2*board_size).  Unreachable cells (dist == max_dist)
    # map to exactly 1.0.
    norm_factor = board_size * board_size
    normalized = np.clip(dist / norm_factor, 0.0, 1.0)

    return normalized


def _is_blocked(
    r: int,
    c: int,
    nr: int,
    nc: int,
    h_walls: Set[Tuple[int, int]],
    v_walls: Set[Tuple[int, int]],
    board_size: int,
) -> bool:
    """
    Check if movement from (r, c) to (nr, nc) is blocked by any wall.

    Movement direction determines which walls to check:

    Moving UP (r->r-1, same col):
        Blocked by horizontal wall at (r-1, c) or (r-1, c-1)

    Moving DOWN (r->r+1, same col):
        Blocked by horizontal wall at (r, c) or (r, c-1)

    Moving LEFT (same row, c->c-1):
        Blocked by vertical wall at (r, c-1) or (r-1, c-1)

    Moving RIGHT (same row, c->c+1):
        Blocked by vertical wall at (r, c) or (r-1, c)
    """
    dr = nr - r
    dc = nc - c
    wall_grid = board_size - 1

    if dr == -1 and dc == 0:  # UP
        wr = r - 1
        if 0 <= wr < wall_grid:
            if (wr, c) in h_walls and c < wall_grid:
                return True
            if c - 1 >= 0 and (wr, c - 1) in h_walls:
                return True

    elif dr == 1 and dc == 0:  # DOWN
        wr = r
        if 0 <= wr < wall_grid:
            if (wr, c) in h_walls and c < wall_grid:
                return True
            if c - 1 >= 0 and (wr, c - 1) in h_walls:
                return True

    elif dr == 0 and dc == -1:  # LEFT
        wc = c - 1
        if 0 <= wc < wall_grid:
            if (r, wc) in v_walls and r < wall_grid:
                return True
            if r - 1 >= 0 and (r - 1, wc) in v_walls:
                return True

    elif dr == 0 and dc == 1:  # RIGHT
        wc = c
        if 0 <= wc < wall_grid:
            if (r, wc) in v_walls and r < wall_grid:
                return True
            if r - 1 >= 0 and (r - 1, wc) in v_walls:
                return True

    return False


def build_tensor(
    board_size: int,
    p0_pos: Tuple[int, int],
    p1_pos: Tuple[int, int],
    p0_h_walls: List[Tuple[int, int]],
    p0_v_walls: List[Tuple[int, int]],
    p1_h_walls: List[Tuple[int, int]],
    p1_v_walls: List[Tuple[int, int]],
    p0_walls_remaining: int,
    p1_walls_remaining: int,
    max_walls: int,
) -> np.ndarray:
    """
    Build the complete 10-channel tensor from raw game state.

    This is the REFERENCE IMPLEMENTATION. Reef's state_to_tensor() must
    produce identical output given the same inputs.

    Args:
        board_size: 5 or 9
        p0_pos: (row, col) of player 0's pawn
        p1_pos: (row, col) of player 1's pawn
        p0_h_walls: list of (r, c) horizontal walls placed by P0
        p0_v_walls: list of (r, c) vertical walls placed by P0
        p1_h_walls: list of (r, c) horizontal walls placed by P1
        p1_v_walls: list of (r, c) vertical walls placed by P1
        p0_walls_remaining: walls P0 can still place
        p1_walls_remaining: walls P1 can still place
        max_walls: max walls per player (5 for 5x5, 10 for 9x9)

    Returns:
        np.ndarray of shape (board_size, board_size, 10), dtype float32
    """
    tensor = np.zeros((board_size, board_size, 10), dtype=np.float32)

    # Ch 0-1: Pawn positions
    tensor[:, :, 0] = compute_pawn_channel(board_size, *p0_pos)
    tensor[:, :, 1] = compute_pawn_channel(board_size, *p1_pos)

    # Ch 2-5: Wall placements per player per orientation
    tensor[:, :, 2] = compute_wall_channel(board_size, p0_h_walls, "horizontal")
    tensor[:, :, 3] = compute_wall_channel(board_size, p0_v_walls, "vertical")
    tensor[:, :, 4] = compute_wall_channel(board_size, p1_h_walls, "horizontal")
    tensor[:, :, 5] = compute_wall_channel(board_size, p1_v_walls, "vertical")

    # Ch 6-7: Walls remaining (normalized, broadcast)
    tensor[:, :, 6] = compute_walls_remaining_channel(
        board_size, p0_walls_remaining, max_walls
    )
    tensor[:, :, 7] = compute_walls_remaining_channel(
        board_size, p1_walls_remaining, max_walls
    )

    # Ch 8-9: Distance maps (BFS considering ALL walls from both players)
    all_h_walls = set(p0_h_walls) | set(p1_h_walls)
    all_v_walls = set(p0_v_walls) | set(p1_v_walls)

    # P0 goal = row 0, P1 goal = row (board_size - 1)
    tensor[:, :, 8] = compute_distance_map(
        board_size, goal_row=0, h_walls=all_h_walls, v_walls=all_v_walls
    )
    tensor[:, :, 9] = compute_distance_map(
        board_size,
        goal_row=board_size - 1,
        h_walls=all_h_walls,
        v_walls=all_v_walls,
    )

    return tensor


# =========================================================================
#  Validation
# =========================================================================


def validate_tensor_spec():
    """
    Test the reference implementation on known board states.
    If this passes, the spec is self-consistent.
    """
    # --- Test 1: Initial state (no walls) ---
    tensor = build_tensor(
        board_size=5,
        p0_pos=(4, 2),
        p1_pos=(0, 2),
        p0_h_walls=[],
        p0_v_walls=[],
        p1_h_walls=[],
        p1_v_walls=[],
        p0_walls_remaining=5,
        p1_walls_remaining=5,
        max_walls=5,
    )

    assert tensor.shape == (5, 5, 10), f"Bad shape: {tensor.shape}"
    assert tensor.dtype == np.float32

    # P0 at (4, 2)
    assert tensor[4, 2, 0] == 1.0
    assert tensor.sum(axis=(0, 1))[0] == 1.0  # exactly one cell set

    # P1 at (0, 2)
    assert tensor[0, 2, 1] == 1.0
    assert tensor.sum(axis=(0, 1))[1] == 1.0

    # No walls
    assert tensor[:, :, 2].sum() == 0.0
    assert tensor[:, :, 3].sum() == 0.0
    assert tensor[:, :, 4].sum() == 0.0
    assert tensor[:, :, 5].sum() == 0.0

    # Full walls remaining
    assert np.allclose(tensor[:, :, 6], 1.0)  # 5/5 = 1.0
    assert np.allclose(tensor[:, :, 7], 1.0)

    # Distance maps: no walls, so BFS = Manhattan distance to goal row
    assert tensor[0, 0, 8] == 0.0  # goal row
    assert np.isclose(tensor[4, 0, 8], 4.0 / 25)  # 4 steps, norm=5²
    assert tensor[4, 0, 9] == 0.0  # goal row
    assert np.isclose(tensor[0, 0, 9], 4.0 / 25)

    print("  Test 1 (initial state): PASS")

    # --- Test 2: State with walls ---
    tensor = build_tensor(
        board_size=5,
        p0_pos=(4, 2),
        p1_pos=(0, 2),
        p0_h_walls=[(2, 1)],
        p0_v_walls=[],
        p1_h_walls=[],
        p1_v_walls=[],
        p0_walls_remaining=4,
        p1_walls_remaining=5,
        max_walls=5,
    )

    # Wall channel 2 should have marks at (2, 1) and (2, 2)
    assert tensor[2, 1, 2] == 1.0
    assert tensor[2, 2, 2] == 1.0
    assert tensor[2, 0, 2] == 0.0  # not covered by this wall

    # P0 walls remaining: 4/5 = 0.8
    assert np.allclose(tensor[:, :, 6], 0.8)

    # Distance map should reflect the wall
    dist_no_wall = 3.0 / 25  # without wall: 3 steps up, norm=5²
    dist_with_wall = tensor[3, 1, 8]
    assert (
        dist_with_wall > dist_no_wall
    ), f"Wall should increase distance: {dist_with_wall} <= {dist_no_wall}"

    print("  Test 2 (with walls): PASS")

    # --- Test 3: All values in [0, 1] ---
    assert tensor.min() >= 0.0, f"Negative value: {tensor.min()}"
    assert tensor.max() <= 1.0, f"Value > 1: {tensor.max()}"

    print("  Test 3 (value range): PASS")

    # --- Test 4: Symmetry check ---
    tensor_sym = build_tensor(
        board_size=5,
        p0_pos=(4, 2),
        p1_pos=(0, 2),
        p0_h_walls=[],
        p0_v_walls=[],
        p1_h_walls=[],
        p1_v_walls=[],
        p0_walls_remaining=5,
        p1_walls_remaining=5,
        max_walls=5,
    )
    # P0 distance map should be vertical mirror of P1 distance map
    p0_dist = tensor_sym[:, :, 8]
    p1_dist = tensor_sym[:, :, 9]
    assert np.allclose(p0_dist, p1_dist[::-1, :]), "Distance maps not symmetric"

    print("  Test 4 (symmetry): PASS")

    print("\nAll tensor spec tests passed.")


if __name__ == "__main__":
    print("--- Tensor Spec Validation ---")
    validate_tensor_spec()
