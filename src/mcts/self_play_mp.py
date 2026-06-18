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
    return assign_vector_targets(trajectory, winner, num_players, discount), winner
