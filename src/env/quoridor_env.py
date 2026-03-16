import numpy as np
import copy
from collections import deque
from typing import Set, Tuple, List, Optional
from dataclasses import dataclass

from src.env.env_interface import QuoridorEnvInterface
from src.env.tensor_spec import build_tensor


def compute_action_space_size(board_size: int) -> int:
    return 12 + 2 * (board_size - 1) ** 2


# Directional encoding
MOVE_MAP = {
    (-1, 0): 0,
    (1, 0): 1,
    (0, -1): 2,
    (0, 1): 3,  # Basic
    (-2, 0): 4,
    (2, 0): 5,
    (0, -2): 6,
    (0, 2): 7,  # Straight jumps
    (-1, -1): 8,
    (-1, 1): 9,
    (1, -1): 10,
    (1, 1): 11,  # Diagonal jumps
}
ACTION_TO_MOVE = {v: k for k, v in MOVE_MAP.items()}


@dataclass
class QuoridorState:
    board_size: int
    p0_pos: Tuple[int, int]
    p1_pos: Tuple[int, int]
    p0_h_walls: Set[Tuple[int, int]]
    p0_v_walls: Set[Tuple[int, int]]
    p1_h_walls: Set[Tuple[int, int]]
    p1_v_walls: Set[Tuple[int, int]]
    p0_walls: int
    p1_walls: int
    max_walls: int
    current_player: int
    turn_count: int
    game_over: bool
    winner: Optional[int]


"""
Quoridor Game Environment
==========================
Owner: Reef
"""


class QuoridorEnv(QuoridorEnvInterface):
    def __init__(self, is_poc: bool = True, max_turns: int = 150, debug: bool = False):
        self.is_poc = is_poc
        self.board_size = 5 if is_poc else 9
        self.max_walls_per_player = 5 if is_poc else 10
        self.max_turns = max_turns
        self.debug = debug

    @property
    def action_space_size(self) -> int:
        return compute_action_space_size(self.board_size)

    def reset(self) -> QuoridorState:
        mid_col = self.board_size // 2

        return QuoridorState(
            board_size=self.board_size,
            p0_pos=(self.board_size - 1, mid_col),
            p1_pos=(0, mid_col),
            p0_h_walls=set(),
            p0_v_walls=set(),
            p1_h_walls=set(),
            p1_v_walls=set(),
            p0_walls=self.max_walls_per_player,
            p1_walls=self.max_walls_per_player,
            max_walls=self.max_walls_per_player,
            current_player=0,
            turn_count=0,
            game_over=False,
            winner=None,
        )

    def get_current_player(self, state: QuoridorState) -> int:
        return state.current_player

    def clone_state(self, state: QuoridorState) -> QuoridorState:
        return QuoridorState(
            board_size=state.board_size,
            p0_pos=state.p0_pos,
            p1_pos=state.p1_pos,
            p0_h_walls=state.p0_h_walls.copy(),
            p0_v_walls=state.p0_v_walls.copy(),
            p1_h_walls=state.p1_h_walls.copy(),
            p1_v_walls=state.p1_v_walls.copy(),
            p0_walls=state.p0_walls,
            p1_walls=state.p1_walls,
            max_walls=state.max_walls,
            current_player=state.current_player,
            turn_count=state.turn_count,
            game_over=state.game_over,
            winner=state.winner,
        )

    def step(
        self, state: QuoridorState, action: int
    ) -> Tuple[QuoridorState, float, bool, dict]:
        if self.debug:
            valid = self.get_valid_actions(state)
            assert action in valid, f"Invalid action {action}. Valid: {valid}"

        new_state = self.clone_state(state)

        W = self.board_size - 1
        h_offset = 12
        v_offset = 12 + W**2

        # 1. Pawn moves
        if action < 12:
            dr, dc = ACTION_TO_MOVE[action]
            if new_state.current_player == 0:
                new_state.p0_pos = (
                    new_state.p0_pos[0] + dr, new_state.p0_pos[1] + dc)
            else:
                new_state.p1_pos = (
                    new_state.p1_pos[0] + dr, new_state.p1_pos[1] + dc)

        # 2. Horizontal walls
        elif action < v_offset:
            w = action - h_offset
            r, c = w // W, w % W
            if new_state.current_player == 0:
                new_state.p0_h_walls.add((r, c))
                new_state.p0_walls -= 1
            else:
                new_state.p1_h_walls.add((r, c))
                new_state.p1_walls -= 1

        # 3. Vertical walls
        else:
            w = action - v_offset
            r, c = w // W, w % W
            if new_state.current_player == 0:
                new_state.p0_v_walls.add((r, c))
                new_state.p0_walls -= 1
            else:
                new_state.p1_v_walls.add((r, c))
                new_state.p1_walls -= 1

        reward, done = self._check_terminal_state(new_state)
        new_state.current_player = 1 - new_state.current_player
        new_state.turn_count += 1

        info = {"winner": new_state.winner if done else None}
        return new_state, reward, done, info

    def get_valid_actions(self, state: QuoridorState) -> np.ndarray:
        if state.game_over:
            return np.array([], dtype=np.int64)

        valid_actions = []
        all_h = state.p0_h_walls | state.p1_h_walls
        all_v = state.p0_v_walls | state.p1_v_walls

        pos = state.p0_pos if state.current_player == 0 else state.p1_pos
        opp = state.p1_pos if state.current_player == 0 else state.p0_pos

        # 1. Pawn actions
        moves = self._get_pawn_moves(pos, opp, all_h, all_v, self.board_size)
        for move in moves:
            dr, dc = move[0] - pos[0], move[1] - pos[1]
            valid_actions.append(MOVE_MAP[(dr, dc)])

        # 2. Wall actions
        walls_left = state.p0_walls if state.current_player == 0 else state.p1_walls
        if walls_left > 0:
            W = self.board_size - 1
            h_offset = 12
            v_offset = 12 + W**2

            for r in range(W):
                for c in range(W):
                    if self._is_valid_h_wall(r, c, all_h, all_v):
                        temp_h = all_h | {(r, c)}
                        if self._has_path(
                            state.p0_pos, 0, temp_h, all_v, self.board_size
                        ) and self._has_path(
                            state.p1_pos,
                            self.board_size - 1,
                            temp_h,
                            all_v,
                            self.board_size,
                        ):
                            valid_actions.append(h_offset + r * W + c)

                    if self._is_valid_v_wall(r, c, all_h, all_v):
                        temp_v = all_v | {(r, c)}
                        if self._has_path(
                            state.p0_pos, 0, all_h, temp_v, self.board_size
                        ) and self._has_path(
                            state.p1_pos,
                            self.board_size - 1,
                            all_h,
                            temp_v,
                            self.board_size,
                        ):
                            valid_actions.append(v_offset + r * W + c)

        return np.array(valid_actions, dtype=np.int64)

    def state_to_tensor(self, state: QuoridorState) -> np.ndarray:
        return build_tensor(
            board_size=state.board_size,
            p0_pos=state.p0_pos,
            p1_pos=state.p1_pos,
            p0_h_walls=list(state.p0_h_walls),
            p0_v_walls=list(state.p0_v_walls),
            p1_h_walls=list(state.p1_h_walls),
            p1_v_walls=list(state.p1_v_walls),
            p0_walls_remaining=state.p0_walls,
            p1_walls_remaining=state.p1_walls,
            max_walls=state.max_walls,
        )

    # ==============================================================
    # Helper Methods
    # ==============================================================

    def _check_terminal_state(self, state: QuoridorState) -> Tuple[float, bool]:
        p0_row, _ = state.p0_pos
        p1_row, _ = state.p1_pos

        if p0_row == 0:
            state.game_over = True
            state.winner = 0
            return 1.0, True

        if p1_row == state.board_size - 1:
            state.game_over = True
            state.winner = 1
            return 1.0, True

        if state.turn_count >= self.max_turns:
            state.game_over = True
            state.winner = None
            return 0.0, True

        return 0.0, False

    def _can_move(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int],
        h_walls: Set[Tuple[int, int]],
        v_walls: Set[Tuple[int, int]],
    ) -> bool:
        r1, c1 = pos1
        r2, c2 = pos2
        if c1 == c2:
            r_min = min(r1, r2)
            if (r_min, c1) in h_walls or (r_min, c1 - 1) in h_walls:
                return False
        elif r1 == r2:
            c_min = min(c1, c2)
            if (r1, c_min) in v_walls or (r1 - 1, c_min) in v_walls:
                return False
        return True

    def _get_pawn_moves(
        self,
        pos: Tuple[int, int],
        opp: Tuple[int, int],
        h_walls: Set[Tuple[int, int]],
        v_walls: Set[Tuple[int, int]],
        N: int,
    ) -> List[Tuple[int, int]]:
        moves = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if (
                0 <= nr < N
                and 0 <= nc < N
                and self._can_move(pos, (nr, nc), h_walls, v_walls)
            ):
                if (nr, nc) == opp:
                    jr, jc = opp[0] + dr, opp[1] + dc
                    if (
                        0 <= jr < N
                        and 0 <= jc < N
                        and self._can_move(opp, (jr, jc), h_walls, v_walls)
                    ):
                        moves.append((jr, jc))
                    else:
                        for ddr, ddc in [(-dc, dr), (dc, -dr)]:
                            djr, djc = opp[0] + ddr, opp[1] + ddc
                            if (
                                0 <= djr < N
                                and 0 <= djc < N
                                and self._can_move(opp, (djr, djc), h_walls, v_walls)
                            ):
                                moves.append((djr, djc))
                else:
                    moves.append((nr, nc))
        return moves

    def _is_valid_h_wall(
        self,
        r: int,
        c: int,
        h_walls: Set[Tuple[int, int]],
        v_walls: Set[Tuple[int, int]],
    ) -> bool:
        if (r, c) in v_walls:
            return False
        if (r, c) in h_walls or (r, c - 1) in h_walls or (r, c + 1) in h_walls:
            return False
        return True

    def _is_valid_v_wall(
        self,
        r: int,
        c: int,
        h_walls: Set[Tuple[int, int]],
        v_walls: Set[Tuple[int, int]],
    ) -> bool:
        if (r, c) in h_walls:
            return False
        if (r, c) in v_walls or (r - 1, c) in v_walls or (r + 1, c) in v_walls:
            return False
        return True

    def _has_path(
        self,
        start_pos: Tuple[int, int],
        goal_row: int,
        h_walls: Set[Tuple[int, int]],
        v_walls: Set[Tuple[int, int]],
        N: int,
    ) -> bool:
        queue = deque([start_pos])
        visited = {start_pos}

        while queue:
            curr = queue.popleft()
            if curr[0] == goal_row:
                return True

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                nxt = (nr, nc)
                if (
                    0 <= nr < N
                    and 0 <= nc < N
                    and nxt not in visited
                    and self._can_move(curr, nxt, h_walls, v_walls)
                ):
                    visited.add(nxt)
                    queue.append(nxt)
        return False
