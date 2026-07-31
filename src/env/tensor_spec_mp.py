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

from src.env.pathing import distance_map


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
        planes.append(distance_map(bs, goals[i], set(h_walls), set(v_walls)))
    turn_val = current_player / (N - 1) if N > 1 else 0.0
    planes.append(np.full((bs, bs), turn_val, np.float32))
    return np.stack(planes, axis=-1)   # (bs, bs, 3N+3)
