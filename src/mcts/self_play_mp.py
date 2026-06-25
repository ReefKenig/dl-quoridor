"""
Vector value-target assignment for the maxn path.

KEY CHANGE vs scalar self_play.py:
  - value_target is a length-N vector, identical for all players' bookkeeping:
        vec[j] = +disc  if j == winner
                 -disc  otherwise
                 0       if winner is None (draw/timeout)
  - The per-position "mover perspective" disappears. The trajectory's
    `player` field is no longer used to sign the target; only `winner` is.
    (This is the simplification the vector design buys.)
"""
import numpy as np


def assign_vector_targets(trajectory, winner, num_players, discount=0.97):
    """trajectory: list of (state_tensor, policy, mover). Returns list of
    (state_tensor, policy, value_vec)."""
    n = len(trajectory)
    out = []
    for idx, (tensor, policy, _mover) in enumerate(trajectory):
        if winner is None:
            vec = np.zeros(num_players, dtype=np.float32)
        else:
            disc = discount ** (n - idx)
            vec = np.full(num_players, -disc, dtype=np.float32)
            vec[winner] = disc
        out.append((tensor, policy, vec))
    return out


def _seat_perm(num_players):
    """Left-right mirror swaps seats 2↔3 (horizontal goals); 0,1 (vertical) stay."""
    if num_players == 2:
        return [0, 1]
    if num_players == 4:
        return [0, 1, 3, 2]
    raise ValueError(f"augmentation supports N in (2,4), got {num_players}")


_MIRROR_MOVE = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 5, 6: 7, 7: 6,
                8: 9, 9: 8, 10: 11, 11: 10}


def augment_mp(tensor, policy, value_vec, num_players, board_size):
    """Left-right mirror augmentation for the MP tensor.
    Flips columns, permutes seat-grouped channels via π=[0,1,3,2],
    mirrors the policy, and permutes the value vector."""
    N, bs, W = num_players, board_size, board_size - 1
    pi = _seat_perm(N)
    t = tensor[:, ::-1, :].copy()          # flip columns
    out = t.copy()
    # layout: [0..N-1]=pawns, [N,N+1]=shared walls, [N+2..2N+1]=walls-rem,
    #         [2N+2..3N+1]=dist maps, [3N+2]=turn
    pawn0, rem0, dist0, turn_ch = 0, N + 2, 2 * N + 2, 3 * N + 2
    for new_i, old_i in enumerate(pi):
        out[:, :, pawn0 + new_i] = t[:, :, pawn0 + old_i]
        out[:, :, rem0 + new_i] = t[:, :, rem0 + old_i]
        out[:, :, dist0 + new_i] = t[:, :, dist0 + old_i]
    cp = int(round(float(t[0, 0, turn_ch]) * (N - 1))) if N > 1 else 0
    out[:, :, turn_ch] = (pi[cp] / (N - 1)) if N > 1 else 0.0

    fp = np.zeros_like(policy)
    h_off, v_off = 12, 12 + W * W
    for a in range(len(policy)):
        if a < 12:
            fp[_MIRROR_MOVE[a]] = policy[a]
        elif a < v_off:
            w = a - h_off
            r, c = w // W, w % W
            fp[h_off + r * W + (W - 1 - c)] = policy[a]
        else:
            w = a - v_off
            r, c = w // W, w % W
            fp[v_off + r * W + (W - 1 - c)] = policy[a]

    fv = np.asarray(value_vec, dtype=np.float32)[pi].copy()
    return out, fp, fv


def play_one_game(env, mcts, num_players, max_moves=200, discount=0.97,
                  temp_explore=1.0, explore_moves=15):
    state = env.reset()
    trajectory = []
    move_count = 0
    winner = None
    while True:
        if move_count >= max_moves:
            winner = None
            break
        temp = temp_explore if move_count < explore_moves else 0.3
        probs = mcts.search(env, state, temperature=temp)
        probs = probs / probs.sum()   # renormalize for fp precision
        mover = env.get_current_player(state)
        trajectory.append((env.state_to_tensor(state), probs, mover))
        action = int(np.random.choice(len(probs), p=probs))
        state, _, done, info = env.step(state, action)
        move_count += 1
        if done:
            winner = info.get("winner")
            break
    samples = assign_vector_targets(trajectory, winner, num_players, discount)
    aug = [augment_mp(t, p, v, num_players, env.board_size) for (t, p, v) in samples]
    return samples + aug, winner
