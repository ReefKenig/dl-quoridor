"""Wall-aware pathing shared by both tensor specs — one BFS, one blocking rule.

Previously duplicated in tensor_spec.py (2-player) and tensor_spec_mp.py
(N-player) with identical semantics but different signatures.
"""
from collections import deque

import numpy as np

# Distance planes divide by DIST_NORM_MULTIPLE * board_size. A straight run is
# board_size-1 steps, so this leaves headroom for wall detours while keeping one
# step of progress at ~1/(2*bs) — 0.056 at 9x9, against binary planes at 1.0.
# The old bs**2 divisor never clipped but put a step at 1/81, so the racing
# signal arrived an order of magnitude below every other channel.
# Trade-off: detours longer than 2*bs now clip to 1.0, the same value as
# unreachable. Walls can never fully block (the env enforces _paths_survive),
# so that only conflates "very far" with "no path", which is the intent.
DIST_NORM_MULTIPLE = 2

# Tensor spec version. The distance-plane divisor changed between the two, which
# rescales channels 8/9 by board_size/2 — 2.5x at 5x5, 4.5x at 9x9. A model only
# ever sees the spec it trained under, so every checkpoint predating the change
# must keep asking for V1; runs/MODELS.json records which one each was trained on.
SPEC_V1_DIST_SQ = 1      # divisor = board_size ** 2
SPEC_V2_DIST_2BS = 2     # divisor = DIST_NORM_MULTIPLE * board_size
CURRENT_SPEC = SPEC_V2_DIST_2BS


def dist_norm(board_size, spec_version=CURRENT_SPEC):
    """Divisor the distance planes are normalized by, per tensor spec version."""
    if spec_version == SPEC_V1_DIST_SQ:
        return board_size * board_size
    if spec_version == SPEC_V2_DIST_2BS:
        return DIST_NORM_MULTIPLE * board_size
    raise ValueError(f"unknown tensor spec version: {spec_version!r}")


def wall_blocks(r, c, nr, nc, h_walls, v_walls, board_size):
    """Is the step (r, c) -> (nr, nc) blocked by a wall? Orthogonal steps only.

    A horizontal wall at (wr, wc) blocks the vertical edge below cells
    (wr, wc) and (wr, wc+1); a vertical wall blocks the horizontal edge right
    of cells (wr, wc) and (wr+1, wc).
    """
    dr, dc = nr - r, nc - c
    W = board_size - 1

    if dr == -1 and dc == 0:        # up
        wr = r - 1
        if 0 <= wr < W:
            if (wr, c) in h_walls and c < W:
                return True
            if c - 1 >= 0 and (wr, c - 1) in h_walls:
                return True
    elif dr == 1 and dc == 0:       # down
        wr = r
        if 0 <= wr < W:
            if (wr, c) in h_walls and c < W:
                return True
            if c - 1 >= 0 and (wr, c - 1) in h_walls:
                return True
    elif dr == 0 and dc == -1:      # left
        wc = c - 1
        if 0 <= wc < W:
            if (r, wc) in v_walls and r < W:
                return True
            if r - 1 >= 0 and (r - 1, wc) in v_walls:
                return True
    elif dr == 0 and dc == 1:       # right
        wc = c
        if 0 <= wc < W:
            if (r, wc) in v_walls and r < W:
                return True
            if r - 1 >= 0 and (r - 1, wc) in v_walls:
                return True

    return False


def distance_map(board_size, goal, h_walls, v_walls, spec_version=CURRENT_SPEC):
    """Normalized BFS distance from every cell to a goal edge.

    goal is ('row', k) or ('col', k). Returns (board_size, board_size) float32
    in [0, 1]; unreachable cells and detours past the norm both read 1.0.
    spec_version picks the divisor — pass a checkpoint's own spec, not the
    current one, or the model sees planes on a scale it never trained on.
    """
    bs = board_size
    unreachable = bs * bs
    dist = np.full((bs, bs), unreachable, dtype=np.float32)

    kind, k = goal
    cells = [(k, c) for c in range(bs)] if kind == "row" else [(r, k) for r in range(bs)]
    queue = deque()
    for cell in cells:
        dist[cell] = 0
        queue.append(cell)

    while queue:
        r, c = queue.popleft()
        d = dist[r, c]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < bs and 0 <= nc < bs):
                continue
            if dist[nr, nc] <= d + 1:
                continue
            if wall_blocks(r, c, nr, nc, h_walls, v_walls, bs):
                continue
            dist[nr, nc] = d + 1
            queue.append((nr, nc))

    return np.clip(dist / dist_norm(bs, spec_version), 0.0, 1.0)
