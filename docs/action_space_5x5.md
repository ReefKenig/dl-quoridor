# Quoridor 5×5 — Action Space Specification

## Overview

This document defines the **exact** action space for the 5×5 Quoridor board.
Every number here must match the environment implementation (`quoridor_env.py`)
and the neural network output layer (`network.py`). A mismatch means silent
training corruption.

## Board Layout

```
     0   1   2   3   4       (columns)
   +---+---+---+---+---+
0  |   |   | P1|   |   |     P1 starts at (0, 2)
   +---+---+---+---+---+
1  |   |   |   |   |   |
   +---+---+---+---+---+
2  |   |   |   |   |   |
   +---+---+---+---+---+
3  |   |   |   |   |   |
   +---+---+---+---+---+
4  |   |   | P0|   |   |     P0 starts at (4, 2)
   +---+---+---+---+---+

P0 goal: reach row 0
P1 goal: reach row 4
```

## 1. Pawn Moves

### Basic moves (4 directions)

A pawn can move one cell in any cardinal direction: UP, DOWN, LEFT, RIGHT.
This gives a **maximum** of 4 basic moves. Edge/wall constraints reduce
this at runtime, but the action space must allocate slots for all 4.

### Jump moves

When a pawn is adjacent to the opponent, it can jump:

- **Straight jump**: Jump over the opponent to the cell behind them (same
  direction). This is possible only if the cell behind is empty and not
  blocked by a wall.
- **Diagonal jumps**: If the straight jump is blocked (wall or board edge),
  the pawn can jump diagonally to either side of the opponent. This gives
  up to 2 diagonal options per blocked direction.

Maximum possible jump destinations from any position:

- Straight jumps: 4 (one per direction)
- Diagonal jumps: 4 (UL, UR, DL, DR — only possible when straight is blocked)

Total unique pawn destinations (cells reachable in one move):
**Up to 8 positions**, but in practice never all 8 simultaneously.

### Encoding pawn moves

We encode pawn moves as **relative direction**, not absolute destination.
This keeps the action index stable regardless of board position.

```
Index | Move
------|--------------
  0   | UP            (row - 1)
  1   | DOWN          (row + 1)
  2   | LEFT          (col - 1)
  3   | RIGHT         (col + 1)
  4   | JUMP_UP       (row - 2)
  5   | JUMP_DOWN     (row + 2)
  6   | JUMP_LEFT     (col - 2)
  7   | JUMP_RIGHT    (col + 2)
  8   | JUMP_UP_LEFT  (row - 1, col - 1)
  9   | JUMP_UP_RIGHT (row - 1, col + 1)
 10   | JUMP_DOWN_LEFT  (row + 1, col - 1)
 11   | JUMP_DOWN_RIGHT (row + 1, col + 1)
```

**Pawn action slots: 12**

Most of these are invalid at any given board state. The environment's
`get_valid_actions()` returns only the legal subset. The policy network
outputs probabilities over all 12 + wall slots; invalid ones get masked to 0.

## 2. Wall Placements

Walls in Quoridor span **2 cells** and are placed between cell intersections.

### Wall positions on 5×5

A wall is identified by its **top-left intersection** coordinate.
On a 5×5 board, intersections form a 4×4 grid (between the 5 rows/cols).

```
Intersections (where walls anchor):
  (0,0) (0,1) (0,2) (0,3)
  (1,0) (1,1) (1,2) (1,3)
  (2,0) (2,1) (2,2) (2,3)
  (3,0) (3,1) (3,2) (3,3)
```

Grid size for wall anchors: **(board_size - 1) × (board_size - 1) = 4 × 4 = 16**

### Wall orientations

Each position supports two orientations:

- **Horizontal**: blocks vertical movement between two pairs of cells
- **Vertical**: blocks horizontal movement between two pairs of cells

### Wall action count

```
Horizontal walls: 4 × 4 = 16
Vertical walls:   4 × 4 = 16
Total wall slots:           32
```

### Encoding wall placements

```
Index range | Meaning
------------|----------------------------
12 - 27     | Horizontal wall at (r, c)
            |   index = 12 + r * 4 + c
            |   r ∈ [0, 3], c ∈ [0, 3]
28 - 43     | Vertical wall at (r, c)
            |   index = 28 + r * 4 + c
            |   r ∈ [0, 3], c ∈ [0, 3]
```

## 3. Total Action Space

```
Component           | Count
--------------------|------
Pawn moves (basic)  |    4
Pawn jumps          |    8
Horizontal walls    |   16
Vertical walls      |   16
--------------------|------
TOTAL               |   44
```

**`action_space_size = 44`** for the 5×5 board.

### Comparison with 9×9

For reference, the 9×9 board has:

```
Pawn moves:       12   (same relative encoding)
Horizontal walls: 64   (8 × 8)
Vertical walls:   64   (8 × 8)
TOTAL:           140
```

Note: the design document says 132 (4 pawn + 128 walls). This is incorrect —
it doesn't account for jump moves. The correct number for 9×9 is **140**.

## 4. Action Index Mapping (5×5)

Complete lookup table:

```python
# Pawn moves
MOVE_UP             = 0
MOVE_DOWN           = 1
MOVE_LEFT           = 2
MOVE_RIGHT          = 3
JUMP_UP             = 4
JUMP_DOWN           = 5
JUMP_LEFT           = 6
JUMP_RIGHT          = 7
JUMP_UP_LEFT        = 8
JUMP_UP_RIGHT       = 9
JUMP_DOWN_LEFT      = 10
JUMP_DOWN_RIGHT     = 11

# Wall placements
WALL_H_OFFSET       = 12   # horizontal walls: indices 12-27
WALL_V_OFFSET       = 28   # vertical walls:   indices 28-43
WALL_GRID_SIZE      = 4    # (board_size - 1)

def wall_h_index(row, col):
    """Action index for horizontal wall at intersection (row, col)."""
    return WALL_H_OFFSET + row * WALL_GRID_SIZE + col

def wall_v_index(row, col):
    """Action index for vertical wall at intersection (row, col)."""
    return WALL_V_OFFSET + row * WALL_GRID_SIZE + col

ACTION_SPACE_SIZE_5x5 = 44
```

## 5. Validation Requirement

Reef's `QuoridorEnv.action_space_size` MUST return **44** for 5×5.
The policy network's output layer MUST have **44** units for 5×5.
The MCTS policy vector MUST be length **44**.

Any mismatch between these three values = broken training.

## 6. Wall Count

Each player starts with a limited number of walls:

| Board Size | Walls per Player |
|-----------|------------------|
| 5×5       | 5                |
| 9×9       | 10               |

Once a player has placed all their walls, wall placement actions are
removed from `get_valid_actions()`.

## 7. Legality Constraints (enforced by `get_valid_actions`)

A wall placement is **illegal** if:
1. The position is already occupied by another wall
2. The wall overlaps/crosses an existing wall
3. The wall completely blocks a player's path to their goal row
   (connectivity check via BFS — required by Quoridor rules)
4. The player has no walls remaining

A pawn move is **illegal** if:
1. A wall blocks the movement in that direction
2. The destination is off the board
3. Jump rules are not satisfied (opponent not adjacent, etc.)
