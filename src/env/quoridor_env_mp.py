"""
N-player Quoridor engine (2..4). Decisions baked in:
  - Jumps: official rules. Jump one adjacent pawn straight if the landing is
    in-board, not wall-separated, and empty; else step diagonally beside it
    (no wall, must be empty). No double-jumps. Action space stays 44.
  - Walls: SHARED (two global sets) for pathing/tensor; per-seat remaining counts.
  - Seats: universal map, first N used. 3-player == 4-player minus seat 3,
    so seat 2 (left->right) is the unpaired/advantaged player.
"""
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

from src.env.env_interface import QuoridorEnvInterface
from src.env.pathing import CURRENT_SPEC
from src.env.tensor_spec_mp import build_tensor_mp


MOVE_MAP = {
    (-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3,
    (-2, 0): 4, (2, 0): 5, (0, -2): 6, (0, 2): 7,
    (-1, -1): 8, (-1, 1): 9, (1, -1): 10, (1, 1): 11,
}
ACTION_TO_MOVE = {v: k for k, v in MOVE_MAP.items()}

NUM_MOVE_ACTIONS = 12


def compute_action_space_size(board_size: int) -> int:
    """Pawn moves plus the two wall orientations. Derived from NUM_MOVE_ACTIONS
    so the layout and the size can never drift apart."""
    return NUM_MOVE_ACTIONS + 2 * (board_size - 1) ** 2


def wall_offsets(board_size: int):
    """(horizontal_offset, vertical_offset) into the action space."""
    return NUM_MOVE_ACTIONS, NUM_MOVE_ACTIONS + (board_size - 1) ** 2


def wall_action(is_h: bool, r: int, c: int, board_size: int) -> int:
    """Action index for the wall slot at (r, c)."""
    h_off, v_off = wall_offsets(board_size)
    return (h_off if is_h else v_off) + r * (board_size - 1) + c


def decode_wall_action(action: int, board_size: int):
    """(is_h, r, c) for a wall action, or None for a pawn move."""
    h_off, v_off = wall_offsets(board_size)
    if action < h_off:
        return None
    W = board_size - 1
    is_h = action < v_off
    w = action - (h_off if is_h else v_off)
    return is_h, w // W, w % W


def seat_specs(board_size: int):
    """Return list of (start_pos, goal) for seats 0..3. goal=('row'|'col', k)."""
    bs = board_size
    mid = bs // 2
    return [
        ((bs - 1, mid), ("row", 0)),        # seat 0: bottom -> top
        ((0, mid),      ("row", bs - 1)),   # seat 1: top -> bottom
        ((mid, 0),      ("col", bs - 1)),   # seat 2: left -> right
        ((mid, bs - 1), ("col", 0)),        # seat 3: right -> left
    ]


@dataclass
class QuoridorStateMP:
    board_size: int
    num_players: int
    positions: List[Tuple[int, int]]
    h_walls: Set[Tuple[int, int]]          # shared
    v_walls: Set[Tuple[int, int]]          # shared
    walls_remaining: List[int]
    max_walls: int
    goals: List[Tuple[str, int]]
    current_player: int
    turn_count: int
    game_over: bool
    winner: Optional[int]


class QuoridorEnvMP(QuoridorEnvInterface):
    def __init__(self, board_size=5, num_players=4, max_turns=300,
                 debug=False, max_walls_per_player=None, wall_budget=None,
                 spec_version=CURRENT_SPEC):
        assert 2 <= num_players <= 4
        self.board_size = board_size
        self.num_players = num_players
        # Walls each player STARTS with; None = max_walls_per_player. The
        # early-training curriculum ramps this up from 0 (a pure race), because
        # at 9x9 walls are 128 of 140 actions and an untrained policy walls
        # ~91% of the time. max_walls_per_player stays fixed either way so the
        # walls-remaining plane keeps one scale.
        self.wall_budget = wall_budget
        # Must match the spec the consuming model trained under; see pathing.py.
        self.spec_version = spec_version
        if max_walls_per_player is not None:
            self.max_walls_per_player = max_walls_per_player
        else:
            # NOTE: official 4-player Quoridor gives 5 walls each on a 9x9
            # board. For the 5x5 POC these are deliberately scaled down
            # (N=2 → 3, N≥3 → 2) to keep games short. With only 2 walls
            # per seat at N=4, wall strategy is nearly vestigial — the POC
            # mostly learns pathfinding/racing, not blocking. To demonstrate
            # coalition-emergence / leader-blocking, raise this to ≥4.
            self.max_walls_per_player = (3 if num_players == 2 else 2) if board_size == 5 else (
                10 if num_players == 2 else 5)
        self.max_turns = max_turns
        self.debug = debug
        self._specs = seat_specs(board_size)[:num_players]

    @property
    def action_space_size(self) -> int:
        return compute_action_space_size(self.board_size)

    @property
    def starting_walls(self):
        """Walls each player begins with — the curriculum budget, or the max."""
        if self.wall_budget is None:
            return self.max_walls_per_player
        return max(0, min(self.wall_budget, self.max_walls_per_player))

    def reset(self) -> QuoridorStateMP:
        starts = [s for s, _ in self._specs]
        goals = [g for _, g in self._specs]
        return QuoridorStateMP(
            board_size=self.board_size, num_players=self.num_players,
            positions=list(starts), h_walls=set(), v_walls=set(),
            walls_remaining=[self.starting_walls] * self.num_players,
            max_walls=self.max_walls_per_player, goals=goals,
            current_player=0, turn_count=0, game_over=False, winner=None,
        )

    def get_current_player(self, state) -> int:
        return state.current_player

    def clone_state(self, state) -> QuoridorStateMP:
        return QuoridorStateMP(
            board_size=state.board_size, num_players=state.num_players,
            positions=list(state.positions), h_walls=state.h_walls.copy(),
            v_walls=state.v_walls.copy(), walls_remaining=list(state.walls_remaining),
            max_walls=state.max_walls, goals=list(state.goals),
            current_player=state.current_player, turn_count=state.turn_count,
            game_over=state.game_over, winner=state.winner,
        )

    def step(self, state, action):
        ns = self.clone_state(state)
        cp = ns.current_player
        slot = decode_wall_action(action, self.board_size)
        if slot is None:
            dr, dc = ACTION_TO_MOVE[action]
            r, c = ns.positions[cp]
            ns.positions[cp] = (r + dr, c + dc)
        else:
            is_h, r, c = slot
            (ns.h_walls if is_h else ns.v_walls).add((r, c))
            ns.walls_remaining[cp] -= 1
        reward, done = self._check_terminal(ns)
        ns.current_player = (cp + 1) % ns.num_players
        ns.turn_count += 1
        if not done:
            self._skip_stuck_players(ns)
        return ns, reward, done, {"winner": ns.winner if done else None}

    def get_valid_actions(self, state):
        if state.game_over:
            return np.array([], dtype=np.int64)
        cp = state.current_player
        pos = state.positions[cp]
        others = set(state.positions[i]
                     for i in range(state.num_players) if i != cp)
        h, v = state.h_walls, state.v_walls
        valid = []
        for tgt in self._pawn_moves(pos, others, h, v):
            valid.append(MOVE_MAP[(tgt[0] - pos[0], tgt[1] - pos[1])])
        if state.walls_remaining[cp] > 0:
            W = self.board_size - 1
            blockers = self._player_blockers(state, h, v)
            for r in range(W):
                for c in range(W):
                    if self._valid_h(r, c, h, v):
                        if self._paths_survive(state, blockers, True, r, c,
                                               h | {(r, c)}, v):
                            valid.append(wall_action(True, r, c, self.board_size))
                    if self._valid_v(r, c, h, v):
                        if self._paths_survive(state, blockers, False, r, c,
                                               h, v | {(r, c)}):
                            valid.append(wall_action(False, r, c, self.board_size))
        return np.array(valid, dtype=np.int64)

    def state_to_tensor(self, state):
        return build_tensor_mp(
            board_size=state.board_size, positions=state.positions,
            h_walls=state.h_walls, v_walls=state.v_walls,
            remaining=state.walls_remaining, max_walls=state.max_walls,
            goals=state.goals, current_player=state.current_player,
            spec_version=self.spec_version,
        )

    # ---------- helpers ----------
    def _at_goal(self, pos, goal):
        kind, k = goal
        return pos[0] == k if kind == "row" else pos[1] == k

    def _check_terminal(self, state):
        cp = state.current_player
        if self._at_goal(state.positions[cp], state.goals[cp]):
            state.game_over = True
            state.winner = cp
            return 1.0, True
        if state.turn_count >= self.max_turns:
            state.game_over = True
            state.winner = None
            return 0.0, True
        return 0.0, False

    def _skip_stuck_players(self, state):
        """After advancing current_player, skip any player who has zero
        legal moves (locally boxed in by pawns + walls). Mutates state
        in-place. Safe: the path-check rule guarantees a goal-path always
        exists, so the stuck player can be freed by others' moves — this
        just skips their turn until they're unstuck."""
        for _ in range(state.num_players):
            if self._player_has_moves(state):
                return
            state.current_player = (state.current_player + 1) % state.num_players
            state.turn_count += 1
            if state.turn_count >= self.max_turns:
                state.game_over = True
                state.winner = None
                return

    def _player_has_moves(self, state):
        """Quick check: does current_player have at least one legal action?"""
        cp = state.current_player
        pos = state.positions[cp]
        others = set(state.positions[i]
                     for i in range(state.num_players) if i != cp)
        if self._pawn_moves(pos, others, state.h_walls, state.v_walls):
            return True
        # No pawn moves; check if any wall placement is legal
        if state.walls_remaining[cp] > 0:
            W = self.board_size - 1
            for r in range(W):
                for c in range(W):
                    if self._valid_h(r, c, state.h_walls, state.v_walls):
                        if self._all_paths(state, state.h_walls | {(r, c)}, state.v_walls):
                            return True
                    if self._valid_v(r, c, state.h_walls, state.v_walls):
                        if self._all_paths(state, state.h_walls, state.v_walls | {(r, c)}):
                            return True
        return False

    def _can_move(self, p1, p2, h, v):
        r1, c1 = p1
        r2, c2 = p2
        if c1 == c2:
            rm = min(r1, r2)
            if (rm, c1) in h or (rm, c1 - 1) in h:
                return False
        elif r1 == r2:
            cm = min(c1, c2)
            if (r1, cm) in v or (r1 - 1, cm) in v:
                return False
        return True

    def _pawn_moves(self, pos, others, h, v):
        N = self.board_size
        moves = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < N and 0 <= nc < N):
                continue
            if not self._can_move(pos, (nr, nc), h, v):
                continue
            if (nr, nc) not in others:
                moves.append((nr, nc))                      # normal step
            else:
                lr, lc = nr + dr, nc + dc                    # straight landing
                straight = (0 <= lr < N and 0 <= lc < N and
                            self._can_move((nr, nc), (lr, lc), h, v) and
                            (lr, lc) not in others)
                if straight:
                    moves.append((lr, lc))                  # jump one pawn
                else:
                    for ddr, ddc in [(-dc, dr), (dc, -dr)]:  # diagonals beside it
                        sr, sc = nr + ddr, nc + ddc
                        if (0 <= sr < N and 0 <= sc < N and
                                self._can_move((nr, nc), (sr, sc), h, v) and
                                (sr, sc) not in others):
                            moves.append((sr, sc))
        return moves

    def _valid_h(self, r, c, h, v):
        if (r, c) in v:
            return False
        if (r, c) in h or (r, c - 1) in h or (r, c + 1) in h:
            return False
        return True

    def _valid_v(self, r, c, h, v):
        if (r, c) in h:
            return False
        if (r, c) in v or (r - 1, c) in v or (r + 1, c) in v:
            return False
        return True

    def _shortest_path(self, start, goal, h, v):
        """BFS route from `start` to its goal as a list of cells, or None."""
        N = self.board_size
        if self._at_goal(start, goal):
            return [start]
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cur[0] + dr, cur[1] + dc
                nx = (nr, nc)
                if not (0 <= nr < N and 0 <= nc < N) or nx in prev:
                    continue
                if not self._can_move(cur, nx, h, v):
                    continue
                prev[nx] = cur
                if self._at_goal(nx, goal):
                    path = [nx]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return path
                q.append(nx)
        return None

    def _path_blockers(self, path):
        """Wall slots that would cut `path`, as (h_slots, v_slots).

        Inverse of _can_move: a vertical step at column c is cut by an h-wall at
        (r, c) or (r, c-1); a horizontal step at row r by a v-wall at (r, c) or
        (r-1, c).
        """
        hb, vb = set(), set()
        for a, b in zip(path, path[1:]):
            if a[1] == b[1]:
                rm, c = min(a[0], b[0]), a[1]
                hb.add((rm, c))
                hb.add((rm, c - 1))
            else:
                r, cm = a[0], min(a[1], b[1])
                vb.add((r, cm))
                vb.add((r - 1, cm))
        return hb, vb

    def distance_to_goal(self, state, player):
        """Moves for `player` to reach their goal from `state`, or None if walled in."""
        path = self._shortest_path(state.positions[player], state.goals[player],
                                   state.h_walls, state.v_walls)
        return None if path is None else len(path) - 1

    def _player_blockers(self, state, h, v):
        """Per-player path-cutting wall slots, or None if any player is already
        walled in (callers then fall back to the exhaustive check)."""
        out = []
        for i in range(state.num_players):
            path = self._shortest_path(state.positions[i], state.goals[i], h, v)
            if path is None:
                return None
            out.append(self._path_blockers(path))
        return out

    def _paths_survive(self, state, blockers, is_h, r, c, h, v):
        """Would every player still reach their goal with this wall placed?

        A wall that misses a player's stored path leaves that path intact, so
        only the players it actually cuts need a fresh BFS. That is what makes
        wall enumeration cheap in the opening, where nothing is walled in yet.
        """
        if blockers is None:
            return self._all_paths(state, h, v)
        for i, (hb, vb) in enumerate(blockers):
            if (r, c) in (hb if is_h else vb):
                if not self._has_path(state.positions[i], state.goals[i], h, v):
                    return False
        return True

    def _all_paths(self, state, h, v):
        for i in range(state.num_players):
            if not self._has_path(state.positions[i], state.goals[i], h, v):
                return False
        return True

    def _has_path(self, start, goal, h, v):
        N = self.board_size
        q = deque([start])
        seen = {start}
        while q:
            cur = q.popleft()
            if self._at_goal(cur, goal):
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cur[0] + dr, cur[1] + dc
                nx = (nr, nc)
                if 0 <= nr < N and 0 <= nc < N and nx not in seen and self._can_move(cur, nx, h, v):
                    seen.add(nx)
                    q.append(nx)
        return False
