"""
MP engine tests: jump edge cases and tensor parity (N=2 MP vs 2p).
Run: pytest tests/test_env_mp.py -v
"""
import numpy as np
import pytest

from src.env.quoridor_env_mp import QuoridorEnvMP
from src.env.tensor_spec import build_tensor
from src.env.tensor_spec_mp import build_tensor_mp


# ── Item 1: jump edge case — straight blocked + one diagonal pawn + other diagonal wall ──

class TestJumpEdgeCases:
    """Cover multi-pawn adjacency scenarios on 5×5 with 4 pawns."""

    def setup_method(self):
        self.env = QuoridorEnvMP(board_size=5, num_players=4)

    def _targets(self, pos, others, h=frozenset(), v=frozenset()):
        return sorted(self.env._pawn_moves(pos, set(others), set(h), set(v)))

    def test_straight_blocked_one_diag_pawn_other_diag_wall(self):
        """Straight landing occupied, one diagonal occupied by pawn,
        other diagonal blocked by wall → only zero moves in that direction."""
        # Mover at (2,2), adjacent pawn at (1,2).
        # Straight landing (0,2) occupied by another pawn → diagonals considered.
        # Diagonal (1,1) occupied by a pawn → unavailable.
        # Diagonal (1,3) blocked by a vertical wall at (0,2) → unavailable.
        # Result: no jump at all upward.
        targets = self._targets(
            (2, 2),
            # pawns blocking straight + one diag
            [(1, 2), (0, 2), (1, 1)],
            v=frozenset([(0, 2)]),               # wall blocking the other diag
        )
        # (1,2), (0,2), (1,1) are all occupied or wall-blocked from (1,2)
        assert (0, 2) not in targets
        assert (1, 2) not in targets
        assert (1, 1) not in targets
        assert (1, 3) not in targets  # wall-blocked diagonal

    def test_straight_blocked_one_diag_pawn_other_diag_open(self):
        """Straight blocked by pawn, one diagonal occupied, other open → one diagonal."""
        # Mover (2,2), adjacent (1,2), straight (0,2) has a pawn.
        # Diagonal (1,1) has a pawn → blocked.
        # Diagonal (1,3) is open → allowed.
        targets = self._targets((2, 2), [(1, 2), (0, 2), (1, 1)])
        assert (1, 3) in targets
        assert (1, 1) not in targets
        assert (0, 2) not in targets

    def test_straight_blocked_both_diags_pawns(self):
        """Straight blocked, both diagonals occupied → no upward jump at all."""
        targets = self._targets((2, 2), [(1, 2), (0, 2), (1, 1), (1, 3)])
        up_targets = [t for t in targets if t[0] < 2]
        assert up_targets == []  # can't go up at all


# ── Item 5: tensor parity — MP at N=2 must reproduce 2p tensor channels ──

class TestTensorParity:
    """Assert build_tensor_mp at N=2 reproduces build_tensor's channels
    for identical game states."""

    def _compare(self, board_size, p0, p1, h_walls, v_walls,
                 p0_rem, p1_rem, max_w):
        """Build both tensors and compare corresponding channels."""
        # 2p spec: per-player wall ownership. For parity we attribute
        # all walls to P0 (simplest; the distance maps use the union anyway).
        t2p = build_tensor(
            board_size=board_size,
            p0_pos=p0, p1_pos=p1,
            p0_h_walls=list(h_walls), p0_v_walls=list(v_walls),
            p1_h_walls=[], p1_v_walls=[],
            p0_walls_remaining=p0_rem, p1_walls_remaining=p1_rem,
            max_walls=max_w,
        )
        tmp = build_tensor_mp(
            board_size=board_size,
            positions=[p0, p1],
            h_walls=set(h_walls), v_walls=set(v_walls),
            remaining=[p0_rem, p1_rem],
            max_walls=max_w,
            goals=[("row", 0), ("row", board_size - 1)],
            current_player=0,
        )

        # Channel mapping:
        # 2p Ch0 (P0 pawn)          ↔ MP Ch0
        # 2p Ch1 (P1 pawn)          ↔ MP Ch1
        # 2p Ch2 (P0 h-walls)       ↔ MP Ch2 (shared h-walls, all attributed to P0)
        # 2p Ch3 (P0 v-walls)       ↔ MP Ch3 (shared v-walls)
        # 2p Ch6 (P0 walls rem)     ↔ MP Ch4
        # 2p Ch7 (P1 walls rem)     ↔ MP Ch5
        # 2p Ch8 (P0 dist to row 0) ↔ MP Ch6
        # 2p Ch9 (P1 dist to row N) ↔ MP Ch7

        # Pawn planes
        np.testing.assert_array_equal(t2p[:, :, 0], tmp[:, :, 0], "P0 pawn")
        np.testing.assert_array_equal(t2p[:, :, 1], tmp[:, :, 1], "P1 pawn")

        # Wall planes: 2p has per-player walls (ch2=P0-h, ch3=P0-v);
        # MP has shared (ch2=all-h, ch3=all-v). Since we gave all walls to P0,
        # P0-h == all-h and P0-v == all-v.
        np.testing.assert_array_equal(t2p[:, :, 2], tmp[:, :, 2], "h-walls")
        np.testing.assert_array_equal(t2p[:, :, 3], tmp[:, :, 3], "v-walls")

        # Walls remaining
        np.testing.assert_array_equal(
            t2p[:, :, 6], tmp[:, :, 4], "P0 walls rem")
        np.testing.assert_array_equal(
            t2p[:, :, 7], tmp[:, :, 5], "P1 walls rem")

        # Distance maps
        np.testing.assert_allclose(t2p[:, :, 8], tmp[:, :, 6], atol=1e-6,
                                   err_msg="P0 distance map")
        np.testing.assert_allclose(t2p[:, :, 9], tmp[:, :, 7], atol=1e-6,
                                   err_msg="P1 distance map")

        # Turn plane: MP ch8 should be 0.0 (current_player=0, 0/(2-1)=0)
        assert tmp[:, :, 8].max() == 0.0, "turn plane for cp=0"

    def test_parity_initial_state(self):
        self._compare(5, (4, 2), (0, 2), [], [], 3, 3, 3)

    def test_parity_with_walls(self):
        self._compare(5, (3, 1), (1, 3),
                      [(1, 0), (2, 2)], [(0, 1)],
                      1, 3, 3)

    def test_parity_9x9_no_walls(self):
        self._compare(9, (8, 4), (0, 4), [], [], 10, 10, 10)


# ── Augmentation guard test ──

class TestAugmentMP:
    def test_augment_mp_swaps_seats_2_and_3(self):
        from src.env.tensor_spec_mp import build_tensor_mp
        from src.mcts.self_play_mp import augment_mp
        bs, N = 5, 4
        # seats 2,3 distinguishable: different positions AND different walls-rem
        T = build_tensor_mp(bs, positions=[(4, 2), (0, 2), (2, 0), (2, 4)],
                            h_walls=set(), v_walls=set(),
                            remaining=[3, 3, 4, 1], max_walls=4,
                            goals=[("row", 0), ("row", 4), ("col", 4), ("col", 0)],
                            current_player=2)
        pol = np.zeros(44, np.float32)
        pol[3] = 1.0      # a RIGHT move
        val = np.array([0.1, 0.2, 0.3, 0.4], np.float32)
        aT, aPol, aVal = augment_mp(T, pol, val, N, bs)
        # seat-2 pawn plane must equal column-flip of ORIGINAL seat-3 plane
        np.testing.assert_array_equal(aT[:, :, 2], T[:, ::-1, 3])
        # value vector swapped on 2,3
        np.testing.assert_array_equal(aVal, np.array([0.1, 0.2, 0.4, 0.3], np.float32))
        # RIGHT move (3) -> LEFT (2)
        assert aPol[2] == 1.0
        # involution: mirror twice == identity
        bT, bPol, bVal = augment_mp(aT, aPol, aVal, N, bs)
        np.testing.assert_allclose(bT, T, atol=1e-6)
        np.testing.assert_array_equal(bVal, val)
