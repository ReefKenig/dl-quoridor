import numpy as np
import copy
from collections import deque
from typing import Set, Tuple, List
from dataclasses import dataclass

from src.env.env_interface import QuoridorEnvInterface

@dataclass
class QuoridorState:
    board_size: int
    p0_pos: Tuple[int, int]
    p1_pos: Tuple[int, int]
    h_walls: Set[Tuple[int, int]]
    v_walls: Set[Tuple[int, int]]
    p0_walls: int
    p1_walls: int
    current_player: int
    turn_count: int
    game_over: bool
    winner: int

"""
Quoridor Game Environment
==========================
Owner: Reef

Implement QuoridorEnvInterface for the 5x5 board.
MCTS and the training loop depend on this contract.

When done, run: python -m tests.test_mcts
If MCTS beats random at >80% win rate with your env, integration works.
"""

class QuoridorEnv(QuoridorEnvInterface):
    def __init__(self, board_size: int = 9, max_walls: int = 4, max_turns: int = 150):
        self.board_size = board_size
        self.max_walls = max_walls
        self.max_turns = max_turns
        
    @property
    def action_space_size(self) -> int:
        return self.board_size ** 2 + 2 * (self.board_size - 1) ** 2  # pawn moves + wall placements
        
    def reset(self) -> QuoridorState:
        # Center column is N // 2
        mid_col = self.board_size // 2
        
        return QuoridorState(
            board_size=self.board_size,
            p0_pos=(0, mid_col),
            p1_pos=(self.board_size - 1, mid_col),
            h_walls=set(),
            v_walls=set(),
            p0_walls=self.max_walls,
            p1_walls=self.max_walls,
            current_player=0,
            turn_count=0,
            game_over=False,
            winner=-1
        )
    
    def get_current_player(self, state: QuoridorState) -> int:
        return state.current_player
    
    def clone_state(self, state: QuoridorState) -> QuoridorState:
        return copy.deepcopy(state)
    
    def step(self, state: QuoridorState, action: int) -> Tuple[QuoridorState, float, bool, dict]:
        new_state = copy.deepcopy(state)
        
        N = self.board_size
        W = N - 1
        pawn_moves = N ** 2
        h_walls_offset = pawn_moves + W ** 2
        
        # Determine action type and execute
        if action < pawn_moves:
            self._execute_pawn_move(new_state, action, N)
        elif action < h_walls_offset:
            self._execute_h_wall(new_state, action, W, pawn_moves)
        else:
            self._execute_v_wall(new_state, action, W, h_walls_offset)
            
        # Check terminal state
        reward, done = self._check_terminal_state(new_state)
        
        new_state.current_player = 1 - new_state.current_player
        new_state.turn_count += 1
        
        return new_state, reward, done, {}
    
    def get_valid_actions(self, state: QuoridorState) -> np.ndarray:
        if state.game_over:
            return np.array([], dtype=np.int64)
    
        valid_actions = []
        valid_actions.extend(self._get_valid_pawn_actions(state))
        valid_actions.extend(self._get_valid_h_wall_actions(state))
        valid_actions.extend(self._get_valid_v_wall_actions(state))
        
        return np.array(valid_actions, dtype=np.int64)
    
    def state_to_tensor(self, state: QuoridorState) -> np.ndarray:
        tensor = np.zeros((self.board_size, self.board_size, 10), dtype=np.float32)
        
        tensor[state.p0_pos[0], state.p0_pos[1], 0] = 1.0
        tensor[state.p1_pos[0], state.p1_pos[1], 1] = 1.0
        
        for r, c in state.h_walls: tensor[r, c, 2] = 1.0
        for r, c in state.v_walls: tensor[r, c, 3] = 1.0
        
        tensor[:, :, 4] = state.p0_walls / self.max_walls
        tensor[:, :, 5] = state.p1_walls / self.max_walls
        
        tensor[:, :, 6] = 1.0 if state.current_player == 0 else 0.0
        
        tensor[self.board_size - 1, :, 7] = 1.0 # P0 goal
        tensor[0, :, 8] = 1.0 # P1 goal
        
        tensor[:, :, 9] = state.turn_count / self.max_turns
        
        return tensor
    
    # === Helper Methods ===
    
    def _can_move(self, pos1: Tuple[int, int], pos2: Tuple[int, int], h_walls: Set[Tuple[int, int]], v_walls: Set[Tuple[int, int]]) -> bool:
        r1, c1 = pos1
        r2, c2 = pos2
        if c1 == c2:
            r_min = min(r1, r2)
            if (r_min, c1) in h_walls or (r_min, c1 - 1) in h_walls: return False
        elif r1 == r2:
            c_min = min(c1, c2)
            if (r1, c_min) in v_walls or (r1 - 1, c_min) in v_walls: return False
        return True
    
    def _get_pawn_moves(self, pos: Tuple[int, int], opp: Tuple[int, int], h_walls: Set[Tuple[int, int]], v_walls: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        moves = []
        N = self.board_size
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not self._is_within_bounds(nr, nc, N):
                continue
            
            if self._can_move(pos, (nr, nc), h_walls, v_walls):
                if (nr, nc) == opp:
                    jump_moves = self._get_jump_moves((nr, nc), dr, dc, h_walls, v_walls, N)
                    moves.extend(jump_moves)
                else:
                    moves.append((nr, nc))
        
        return moves
    
    def _is_valid_h_wall(self, r: int, c: int, h_walls: Set[Tuple[int, int]], v_walls: Set[Tuple[int, int]]) -> bool:
        if [(r, c), (r, c - 1), (r, c + 1)] in h_walls or (r, c) in v_walls: return False
        return True
    
    def _is_valid_v_wall(self, r: int, c: int, h_walls: Set[Tuple[int, int]], v_walls: Set[Tuple[int, int]]) -> bool:
        if [(r, c), (r - 1, c), (r + 1, c)] in v_walls or (r, c) in h_walls: return False
        return True
    
    def _has_path(self, start_pos: Tuple[int, int], goal_row: int, h_walls: Set[Tuple[int, int]], v_walls: Set[Tuple[int, int]], board_size: int) -> bool:
        queue = deque([start_pos])
        visited = {start_pos}
        N = board_size
        
        while queue:
            curr = queue.popleft()
            if curr[0] == goal_row:
                return True
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                nxt = (nr, nc)
                if 0 <= nr < N and 0 <= nc < N and nxt not in visited and self._can_move(curr, nxt, h_walls, v_walls):
                    visited.add(nxt)
                    queue.append(nxt)
                    
        return False
    
    def _get_valid_pawn_actions(self, state: QuoridorState) -> List[int]:
        """Get valid pawn move actions for the current player."""
        player = state.current_player
        pos = state.p0_pos if player == 0 else state.p1_pos
        opp_pos = state.p1_pos if player == 0 else state.p0_pos
        moves = self._get_pawn_moves(pos, opp_pos, state.h_walls, state.v_walls)
        N = self.board_size
        actions = []
        for move in moves:
            row, col = move
            action = row * N + col
            actions.append(action)
        return actions
    
    def _get_valid_h_wall_actions(self, state: QuoridorState) -> List[int]:
        """Get valid horizontal wall placement actions."""
        player = state.current_player
        walls_left = state.p0_walls if player == 0 else state.p1_walls
        if walls_left <= 0:
            return []
        
        N = self.board_size
        W = N - 1
        pawn_moves = N ** 2
        actions = []
        for r in range(W):
            for c in range(W):
                if self._is_valid_h_wall(r, c, state.h_walls, state.v_walls):
                    action = pawn_moves + r * W + c
                    actions.append(action)
        return actions
    
    def _get_valid_v_wall_actions(self, state: QuoridorState) -> List[int]:
        """Get valid vertical wall placement actions."""
        player = state.current_player
        walls_left = state.p0_walls if player == 0 else state.p1_walls
        if walls_left <= 0:
            return []
        
        N = self.board_size
        W = N - 1
        pawn_moves = N ** 2
        h_walls_offset = pawn_moves + W ** 2
        actions = []
        for r in range(W):
            for c in range(W):
                if self._is_valid_v_wall(r, c, state.h_walls, state.v_walls):
                    action = h_walls_offset + r * W + c
                    actions.append(action)
        return actions
    
    def _execute_pawn_move(self, state: QuoridorState, action: int, N: int) -> None:
        """Execute a pawn move action."""
        row = action // N
        col = action % N
        
        if state.current_player == 0:
            state.p0_pos = (row, col)
        else:
            state.p1_pos = (row, col)
            
    def _execute_h_wall(self, state: QuoridorState, action: int, W: int, pawn_moves) -> None:
        """Execute a horizontal wall placement action."""
        wall_action = action - pawn_moves
        r = wall_action // W
        c = wall_action % W
        
        state.h_walls.add((r, c))
        
        if state.current_player == 0:
            state.p0_walls -= 1
        else:
            state.p1_walls -= 1
            
    def _execute_v_wall(self, state: QuoridorState, action: int, W: int, h_walls_offset) -> None:
        """Execute a vertical wall placement action."""
        wall_action = action - h_walls_offset
        r = wall_action // W
        c = wall_action % W
        
        state.v_walls.add((r, c))
        
        if state.current_player == 0:
            state.p0_walls -= 1
        else:
            state.p1_walls -= 1
            
    def _check_terminal_state(self, state: QuoridorState) -> Tuple[float, bool]:
        """Check if game is over and compute reward."""
        p0_row, _ = state.p0_pos
        p1_row, _ = state.p1_pos
        
        # Check win conditions
        if p0_row == 0:
            state.game_over = True
            state.winner = 0
            return 1.0, True
        
        if p1_row == self.board_size - 1:
            state.game_over = True
            state.winner = 1
            return -1.0, True
        
        # Check turn limit
        if state.turn_count >= self.max_turns:
            state.game_over = True
            state.winner = -1
            return 0.0, True
        
        return 0.0, False
    
    def _is_within_bounds(self, r: int, c: int, N: int) -> bool:
        """Check if a position is within the board boundaries."""
        return 0 <= r < N and 0 <= c < N

    def _get_jump_moves(self, opp_pos: Tuple[int, int], dr: int, dc: int, h_walls: Set[Tuple[int, int]], v_walls: Set[Tuple[int, int]], N: int) -> List[Tuple[int, int]]:
        """Get possible jump moves over the opponent."""
        moves = []
        jr, jc = opp_pos[0] + dr, opp_pos[1] + dc
        
        if self._is_within_bounds(jr, jc, N) and self._can_move(opp_pos, (jr, jc), h_walls, v_walls):
            moves.append((jr, jc))
        else:
            # Diagonal jumps if straight jump is blocked
            for ddr, ddc in [(-dc, dr), (dc, -dr)]:  # Perpendicular directions
                djr, djc = opp_pos[0] + ddr, opp_pos[1] + ddc
                if self._is_within_bounds(djr, djc, N) and self._can_move(opp_pos, (djr, djc), h_walls, v_walls):
                    moves.append((djr, djc))
        
        return moves