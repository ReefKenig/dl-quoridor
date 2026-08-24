"""
`get_valid_actions` fast path == exhaustive path-check, byte for byte.

Wall enumeration used to run `_all_paths` (one BFS per player) for all 128
candidate slots on a 9x9 board - up to 512 BFS per call, which is what starved
the GPU batcher in the opening. It now BFSes once per player and only re-checks
players whose route the candidate wall actually cuts.

That is a pure optimization, so the only thing worth testing is that it changes
nothing: `_reference_valid_actions` below is the original exhaustive loop, and
every test asserts array equality against it (order included, since MCTS indexes
into this array).
"""
import numpy as np
import pytest

from src.env.quoridor_env_mp import MOVE_MAP, QuoridorEnvMP


def _reference_valid_actions(env, state):
    """The pre-optimization implementation, verbatim."""
    if state.game_over:
        return np.array([], dtype=np.int64)
    cp = state.current_player
    pos = state.positions[cp]
    others = set(state.positions[i]
                 for i in range(state.num_players) if i != cp)
    h, v = state.h_walls, state.v_walls
    valid = []
    for tgt in env._pawn_moves(pos, others, h, v):
        valid.append(MOVE_MAP[(tgt[0] - pos[0], tgt[1] - pos[1])])
    if state.walls_remaining[cp] > 0:
        W = env.board_size - 1
        h_off, v_off = 12, 12 + W ** 2
        for r in range(W):
            for c in range(W):
                if env._valid_h(r, c, h, v):
                    if env._all_paths(state, h | {(r, c)}, v):
                        valid.append(h_off + r * W + c)
                if env._valid_v(r, c, h, v):
                    if env._all_paths(state, h, v | {(r, c)}):
                        valid.append(v_off + r * W + c)
    return np.array(valid, dtype=np.int64)


def _assert_same(env, state, ctx=""):
    fast = env.get_valid_actions(state)
    ref = _reference_valid_actions(env, state)
    assert np.array_equal(fast, ref), (
        f"{ctx}\n  fast={fast.tolist()}\n  ref ={ref.tolist()}\n"
        f"  only_fast={sorted(set(fast) - set(ref))}\n"
        f"  only_ref ={sorted(set(ref) - set(fast))}")


CASES = [
    (5, 2, 3), (5, 4, 4),
    (9, 2, 10), (9, 4, 5), (9, 4, 10),
]


@pytest.mark.parametrize("bs,n,walls", CASES)
def test_matches_reference_over_random_playouts(bs, n, walls):
    """Every position of several random games agrees with the reference."""
    env = QuoridorEnvMP(board_size=bs, num_players=n,
                        max_walls_per_player=walls, max_turns=400)
    rng = np.random.RandomState(0)
    checked = 0
    for g in range(6):
        s = env.reset()
        for ply in range(160):
            _assert_same(env, s, f"bs={bs} n={n} game={g} ply={ply}")
            checked += 1
            valid = env.get_valid_actions(s)
            if len(valid) == 0:
                break
            s, _, done, _ = env.step(s, int(rng.choice(valid)))
            if done:
                break
    assert checked > 100, f"only {checked} positions exercised"


@pytest.mark.parametrize("bs,n,walls", CASES)
def test_matches_reference_under_wall_heavy_play(bs, n, walls):
    """Bias play toward walls, so near-enclosure positions are actually hit.

    Uniform random play spends most walls early and then never places one; the
    interesting rejections live in cramped late boards.
    """
    env = QuoridorEnvMP(board_size=bs, num_players=n,
                        max_walls_per_player=walls, max_turns=400)
    rng = np.random.RandomState(7)
    for g in range(6):
        s = env.reset()
        for ply in range(160):
            _assert_same(env, s, f"wall-heavy bs={bs} n={n} game={g} ply={ply}")
            valid = env.get_valid_actions(s)
            if len(valid) == 0:
                break
            walls_av = [a for a in valid.tolist() if a >= 12]
            pick = walls_av if (walls_av and rng.rand() < 0.8) else valid.tolist()
            s, _, done, _ = env.step(s, int(rng.choice(pick)))
            if done:
                break


def test_rejects_a_wall_that_would_enclose_a_player():
    """A wall the reference rejects must still be rejected by the fast path.

    Guards the failure mode that matters: the filter skipping a BFS it needed.
    """
    env = QuoridorEnvMP(board_size=5, num_players=2,
                        max_walls_per_player=10, max_turns=200)
    rng = np.random.RandomState(3)
    saw_rejection = False
    for g in range(40):
        s = env.reset()
        for _ in range(60):
            cp = s.current_player
            if s.walls_remaining[cp] > 0:
                legal = set(env.get_valid_actions(s).tolist())
                W = env.board_size - 1
                for r in range(W):
                    for c in range(W):
                        for off, ok, hh, vv in (
                            (12, env._valid_h(r, c, s.h_walls, s.v_walls),
                             s.h_walls | {(r, c)}, s.v_walls),
                            (12 + W ** 2, env._valid_v(r, c, s.h_walls, s.v_walls),
                             s.h_walls, s.v_walls | {(r, c)}),
                        ):
                            if ok and not env._all_paths(s, hh, vv):
                                saw_rejection = True
                                assert off + r * W + c not in legal
            valid = env.get_valid_actions(s)
            if len(valid) == 0:
                break
            s, _, done, _ = env.step(s, int(rng.choice(valid)))
            if done:
                break
        if saw_rejection:
            break
    assert saw_rejection, "no enclosing wall ever arose; test proved nothing"


def test_path_blockers_match_can_move():
    """_path_blockers is the inverse of _can_move - check it directly.

    If these drift apart the filter silently skips real BFS checks, so pin the
    mapping rather than trusting the playout tests to stumble onto it.
    """
    env = QuoridorEnvMP(board_size=5, num_players=2, max_walls_per_player=3)
    for a, b in [((2, 2), (1, 2)), ((1, 2), (2, 2)),
                 ((2, 2), (2, 3)), ((2, 3), (2, 2))]:
        hb, vb = env._path_blockers([a, b])
        for r in range(-1, 5):
            for c in range(-1, 5):
                for is_h, slots in ((True, hb), (False, vb)):
                    h = {(r, c)} if is_h else set()
                    v = set() if is_h else {(r, c)}
                    blocked = not env._can_move(a, b, h, v)
                    assert blocked == ((r, c) in slots), (
                        f"step {a}->{b}, {'h' if is_h else 'v'}-wall ({r},{c}): "
                        f"_can_move blocked={blocked}, in blockers={(r, c) in slots}")
