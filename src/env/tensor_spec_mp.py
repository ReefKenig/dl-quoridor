"""
N-player tensor (shared walls). Channels = 3N + 3:
  N   pawn planes (one per seat, absolute)
  2   shared wall maps (all-H, all-V)
  N   walls-remaining planes (per seat, broadcast)
  N   distance-to-own-goal BFS maps (per seat)
  1   current-player plane (normalized to [0,1])
All values in [0,1]. Indexed tensor[row, col, channel].
"""
import numpy as np
from collections import deque


def _pawn_plane(bs, pos):
    ch = np.zeros((bs, bs), np.float32)
    ch[pos] = 1.0
    return ch


def _wall_plane(bs, walls, orient):
    ch = np.zeros((bs, bs), np.float32)
    W = bs - 1
    for r, c in walls:
        if not (0 <= r < W and 0 <= c < W):
            continue
        ch[r, c] = 1.0
        if orient == "h" and c + 1 < bs:
            ch[r, c + 1] = 1.0
        elif orient == "v" and r + 1 < bs:
            ch[r + 1, c] = 1.0
    return ch


def _remaining_plane(bs, rem, mx):
    return np.full((bs, bs), 0.0 if mx == 0 else rem / mx, np.float32)


def _blocked(r, c, nr, nc, h, v, bs):
    dr, dc = nr - r, nc - c
    W = bs - 1
    if dr == -1 and dc == 0:
        wr = r - 1
        if 0 <= wr < W and ((wr, c) in h and c < W or (c - 1 >= 0 and (wr, c - 1) in h)):
            return True
    elif dr == 1 and dc == 0:
        wr = r
        if 0 <= wr < W and ((wr, c) in h and c < W or (c - 1 >= 0 and (wr, c - 1) in h)):
            return True
    elif dr == 0 and dc == -1:
        wc = c - 1
        if 0 <= wc < W and ((r, wc) in v and r < W or (r - 1 >= 0 and (r - 1, wc) in v)):
            return True
    elif dr == 0 and dc == 1:
        wc = c
        if 0 <= wc < W and ((r, wc) in v and r < W or (r - 1 >= 0 and (r - 1, wc) in v)):
            return True
    return False


def _distance_map(bs, goal, h, v):
    """goal = ('row', k) or ('col', k). Multi-source BFS from the goal edge."""
    INF = bs * bs
    dist = np.full((bs, bs), INF, np.float32)
    q = deque()
    kind, k = goal
    cells = [(k, c) for c in range(bs)] if kind == "row" else [(r, k)
                                                               for r in range(bs)]
    for cell in cells:
        dist[cell] = 0
        q.append(cell)
    while q:
        r, c = q.popleft()
        d = dist[r, c]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < bs and 0 <= nc < bs):
                continue
            if dist[nr, nc] <= d + 1:
                continue
            if _blocked(r, c, nr, nc, h, v, bs):
                continue
            dist[nr, nc] = d + 1
            q.append((nr, nc))
    return np.clip(dist / (bs * bs), 0.0, 1.0)


def build_tensor_mp(board_size, positions, h_walls, v_walls, remaining,
                    max_walls, goals, current_player):
    N = len(positions)
    bs = board_size
    planes = []
    for i in range(N):
        planes.append(_pawn_plane(bs, positions[i]))
    planes.append(_wall_plane(bs, h_walls, "h"))
    planes.append(_wall_plane(bs, v_walls, "v"))
    for i in range(N):
        planes.append(_remaining_plane(bs, remaining[i], max_walls))
    for i in range(N):
        planes.append(_distance_map(bs, goals[i], set(h_walls), set(v_walls)))
    turn_val = current_player / (N - 1) if N > 1 else 0.0
    planes.append(np.full((bs, bs), turn_val, np.float32))
    return np.stack(planes, axis=-1)   # (bs, bs, 3N+3)
