import numpy as np
import copy
from collections import deque
from typing import Set, Tuple, List, Optional
from dataclasses import dataclass

from src.env.env_interface import QuoridorEnvInterface
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
        
        self.action_space_size = self.board_size ** 2 + 2 * (self.board_size - 1) ** 2  # pawn moves + wall placements
        self.reset()
        
    def reset(self):
        # Center column is N // 2
        mid_col = self.board_size // 2
        
        self.p0_pos = (0, mid_col)
        self.p1_pos = (self.board_size - 1, mid_col)
        
        self.h_walls: Set[Tuple[int, int]] = set()
        self.v_walls: Set[Tuple[int, int]] = set()
        
        self.p0_walls = self.max_walls
        self.p1_walls = self.max_walls
        
        self.current_player = 0
        self.turn_count = 0
        self.game_over = False
        self.winner = -1
        
        return self
    
    def get_current_player(self) -> int:
        return self.current_player
    
    def clone_state(self) -> 'QuoridorEnv':
        return copy.deepcopy(self)
    
    def step(self, action: int) -> Tuple[bool, int]:
        if self.game_over:
            return self.game_over, self.winner
        
        N = self.board_size
        W = N - 1
        pawn_moves = N ** 2
        h_walls_offset = pawn_moves + W ** 2
        
        # Pawn moves
        if action < pawn_moves:
            row, col = action // N, action % N
            if self.current_player == 0:
                self.p0_pos = (row, col)
            else:
                self.p1_pos = (row, col)
                
        # Horizontal walls
        elif action < h_walls_offset:
            w = action - pawn_moves
            row, col = w // W, w % W
            self.h_walls.add((row, col))
            if self.current_player == 0:
                self.p0_walls -= 1
            else:
                self.p1_walls -= 1
                
        # Vertical walls
        elif action < self.action_space_size:
            w = action - h_walls_offset
            row, col = w // W, w % W
            self.v_walls.add((row, col))
            if self.current_player == 0:
                self.p0_walls -= 1
            else:
                self.p1_walls -= 1
                
        self.turn_count += 1
        
        # Check win condition
        if self.p0_pos[0] == self.board_size - 1:
            self.game_over = True
            self.winner = 0
        elif self.p1_pos[0] == 0:
            self.game_over = True
            self.winner = 1
        elif self.turn_count >= self.max_turns:
            self.game_over = True
            self.winner = -1  # draw
            
        self.current_player = 1 - self.current_player
        return self.game_over, self.winner
    
    def get_valid_actions(self) -> List[int]:
        if self.game_over:
            return []
        
        valid_actions = []
        N = self.board_size
        W = N - 1
        pawn_moves = N ** 2
        h_walls_offset = pawn_moves + W ** 2
        
        # Pawn moves
        moves = self._get_pawn_moves(self.current_player)
        for row, col in moves:
            valid_actions.append(row * N + col)
            
        # Wall placements
        walls_left = self.p0_walls if self.current_player == 0 else self.p1_walls
        if walls_left > 0:
            for r in range(W):
                for c in range(W):
                    # Check horizontal walls
                    if self._is_valid_h_wall(r, c):
                        self.h_walls.add((r, c))
                        if self._has_path(self.p0_pos, N - 1) and self._has_path(self.p1_pos, 0):
                            valid_actions.append(pawn_moves + r * W + c)
                        self.h_walls.remove((r, c))
                        
                    # Check vertical walls
                    if self._is_valid_v_wall(r, c):
                        self.v_walls.add((r, c))
                        if self._has_path(self.p0_pos, N - 1) and self._has_path(self.p1_pos, 0):
                            valid_actions.append(h_walls_offset + r * W + c)
                        self.v_walls.remove((r, c))
                        
        return valid_actions
    
    def state_to_tensor(self) -> np.ndarray:
        tensor = np.zeros((self.board_size, self.board_size, 10), dtype=np.float32)
        
        tensor[self.p0_pos[0], self.p0_pos[1], 0] = 1.0
        tensor[self.p1_pos[0], self.p1_pos[1], 1] = 1.0
        
        for r, c in self.h_walls: tensor[r, c, 2] = 1.0
        for r, c in self.v_walls: tensor[r, c, 3] = 1.0
        
        tensor[:, :, 4] = self.p0_walls / self.max_walls
        tensor[:, :, 5] = self.p1_walls / self.max_walls
        
        tensor[:, :, 6] = 1.0 if self.current_player == 0 else 0.0
        
        tensor[self.board_size - 1, :, 7] = 1.0 # P0 goal
        tensor[0, :, 8] = 1.0 # P1 goal
        
        tensor[:, :, 9] = self.turn_count / self.max_turns
        
        return tensor
    
    # === Helper Methods ===
    
    def _can_move(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        r1, c1 = pos1
        r2, c2 = pos2
        if c1 == c2:
            r_min = min(r1, r2)
            if (r_min, c1) in self.h_walls or (r_min, c1 - 1) in self.h_walls: return False
        elif r1 == r2:
            c_min = min(c1, c2)
            if (r1, c_min) in self.v_walls or (r1 - 1, c_min) in self.v_walls: return False
        return True
    
    def _get_pawn_moves(self, player: int) -> List[Tuple[int, int]]:
        pos = self.p0_pos if player == 0 else self.p1_pos
        opp = self.p1_pos if player == 0 else self.p0_pos
        moves = []
        N = self.board_size
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < N and 0 <= nc < N and self._can_move(pos, (nr, nc)):
                if (nr, nc) == opp:
                    jr, jc = nr + dr, nc + dc
                    if 0 <= jr < N and 0 <= jc < N and self._can_move((nr, nc), (jr, jc)):
                        moves.append((jr, jc))
                    else:
                        if dr != 0:
                            for ddc in [-1, 1]:
                                dr_diag, dc_diag = nr, nc + ddc
                                if 0 <= dr_diag < N and 0 <= dc_diag < N and self._can_move((nr, nc), (dr_diag, dc_diag)):
                                    moves.append((dr_diag, dc_diag))
                        else:
                            for ddr in [-1, 1]:
                                dr_diag, dc_diag = nr + ddr, nc
                                if 0 <= dr_diag < N and 0 <= dc_diag < N and self._can_move((nr, nc), (dr_diag, dc_diag)):
                                    moves.append((dr_diag, dc_diag))
                else:
                    moves.append((nr, nc))
        return moves
    
    def _is_valid_h_wall(self, r: int, c: int) -> bool:
        if [(r, c), (r, c - 1), (r, c + 1)] in self.h_walls or (r, c) in self.v_walls: return False
        return True
    
    def _is_valid_v_wall(self, r: int, c: int) -> bool:
        if [(r, c), (r - 1, c), (r + 1, c)] in self.v_walls or (r, c) in self.h_walls: return False
        return True
    
    def _has_path(self, start_pos: Tuple[int, int], goal_row: int) -> bool:
        queue = deque([start_pos])
        visited = {start_pos}
        N = self.board_size
        
        while queue:
            curr = queue.poplef()
            if curr[0] == goal_row:
                return True
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                nxt = (nr, nc)
                if 0 <= nr < N and 0 <= nc < N and nxt not in visited and self._can_move(curr, nxt):
                    visited.add(nxt)
                    queue.append(nxt)
                    
        return False
