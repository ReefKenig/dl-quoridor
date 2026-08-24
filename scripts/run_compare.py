"""
Test B: Same-weights comparison - negamax MCTS vs max^n MCTS.

Loads the real trained model.pt (scalar value head) and wraps it for both
the old negamax MCTS and the new max^n MCTS. Runs:
  1. Head-to-head (negamax vs max^n): expect ~50/50
  2. Both vs random: expect ~100% win rate for each

This confirms the max^n search swap preserves the model's strength.
"""
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
from src.mcts.mcts import MCTS, MCTSConfig as CfgOld
from src.model.network import QuoridorModel
from src.env.quoridor_env import QuoridorEnv
import logging
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.disable(logging.CRITICAL)


# --- Config ---
CHECKPOINT = Path(__file__).resolve().parent.parent / \
    "checkpoints" / "best" / "model.pt"
BOARD_SIZE = 5
IN_CHANNELS = 10
NUM_CHANNELS = 64
NUM_RES_BLOCKS = 4
ACTION_SPACE = 44
SIMS = 200
GAMES_H2H = 20
GAMES_VS_RANDOM = 20

env = QuoridorEnv(board_size=BOARD_SIZE, max_walls_per_player=3, max_turns=200)

# --- Load model ---
model = QuoridorModel(board_size=BOARD_SIZE, action_space_size=ACTION_SPACE,
                      in_channels=IN_CHANNELS, num_channels=NUM_CHANNELS,
                      num_res_blocks=NUM_RES_BLOCKS, device="cpu")
model.load(str(CHECKPOINT))
print(f"Loaded checkpoint: {CHECKPOINT.name}")


# --- Evaluators ---
def eval_negamax(state):
    """Old-style: returns (policy, scalar_value) in current-player perspective."""
    tensor = env.state_to_tensor(state)
    tensor = tensor[:, :, :10]  # strip turn channel; model trained on 10ch
    policy, value = model.predict(tensor)
    return policy, value  # model already returns scalar in mover perspective


def eval_maxn(state):
    """New-style: returns (policy, vector[N]) - absolute perspective."""
    tensor = env.state_to_tensor(state)
    tensor = tensor[:, :, :10]  # strip turn channel; model trained on 10ch
    policy, value_scalar = model.predict(tensor)
    # The scalar is from the current player's perspective.
    # Convert to absolute vector: if current_player==0, vec=[v, -v], else vec=[-v, v]
    cp = state.current_player
    vec = np.zeros(2, dtype=np.float32)
    vec[cp] = value_scalar
    vec[1 - cp] = -value_scalar
    return policy, vec


# --- MCTS instances ---
mcts_nega = MCTS(config=CfgOld(num_simulations=SIMS, c_puct=1.41, dirichlet_epsilon=0.0),
                 evaluate_fn=eval_negamax)
mcts_maxn = MCTSMaxN(config=MCTSConfig(num_simulations=SIMS, c_puct=1.41, dirichlet_epsilon=0.0),
                     evaluate_fn=eval_maxn, num_players=2)


def play_game(p0_search, p1_search, env, max_moves=200):
    """Play a game. Returns winner (0 or 1) or None for draw."""
    state = env.reset()
    for _ in range(max_moves):
        cp = env.get_current_player(state)
        search = p0_search if cp == 0 else p1_search
        probs = search(env, state, temperature=0.1)
        action = int(np.argmax(probs))
        state, _, done, info = env.step(state, action)
        if done:
            return info.get("winner")
    return None


def random_search(env, state, temperature=None):
    """Random policy for baseline."""
    valid = env.get_valid_actions(state)
    probs = np.zeros(env.action_space_size)
    probs[valid] = 1.0 / len(valid)
    return probs


# --- Test 1: Head-to-head ---
print(f"\n{'='*60}")
print(
    f"TEST B.1: Head-to-head (negamax vs max^n), {GAMES_H2H} games each side")
print(f"{'='*60}")

wins_nega = 0
wins_maxn = 0
draws = 0

for g in range(GAMES_H2H):
    # Alternate who goes first
    if g % 2 == 0:
        w = play_game(mcts_nega.search, mcts_maxn.search, env)
        if w == 0:
            wins_nega += 1
        elif w == 1:
            wins_maxn += 1
        else:
            draws += 1
    else:
        w = play_game(mcts_maxn.search, mcts_nega.search, env)
        if w == 0:
            wins_maxn += 1
        elif w == 1:
            wins_nega += 1
        else:
            draws += 1
    marker = "N" if (wins_nega > wins_maxn) else (
        "M" if wins_maxn > wins_nega else "=")
    print(
        f"  game {g+1:2d}: nega={wins_nega} maxn={wins_maxn} draw={draws} [{marker}]")

total_decided = wins_nega + wins_maxn
nega_pct = 100 * wins_nega / total_decided if total_decided > 0 else 0
maxn_pct = 100 * wins_maxn / total_decided if total_decided > 0 else 0
print(
    f"\n  RESULT: negamax {wins_nega}W ({nega_pct:.0f}%) | max^n {wins_maxn}W ({maxn_pct:.0f}%) | draws {draws}")
print("  EXPECTED: ~50/50 (same model, equivalent search)")

# --- Test 2: vs Random ---
print(f"\n{'='*60}")
print(f"TEST B.2: negamax vs random, {GAMES_VS_RANDOM} games")
print(f"{'='*60}")

nega_vs_random_wins = 0
for g in range(GAMES_VS_RANDOM):
    if g % 2 == 0:
        w = play_game(mcts_nega.search, random_search, env)
        if w == 0:
            nega_vs_random_wins += 1
    else:
        w = play_game(random_search, mcts_nega.search, env)
        if w == 1:
            nega_vs_random_wins += 1
print(
    f"  negamax wins: {nega_vs_random_wins}/{GAMES_VS_RANDOM} ({100*nega_vs_random_wins/GAMES_VS_RANDOM:.0f}%)")

print(f"\n{'='*60}")
print(f"TEST B.3: max^n vs random, {GAMES_VS_RANDOM} games")
print(f"{'='*60}")

maxn_vs_random_wins = 0
for g in range(GAMES_VS_RANDOM):
    if g % 2 == 0:
        w = play_game(mcts_maxn.search, random_search, env)
        if w == 0:
            maxn_vs_random_wins += 1
    else:
        w = play_game(random_search, mcts_maxn.search, env)
        if w == 1:
            maxn_vs_random_wins += 1
print(
    f"  max^n wins: {maxn_vs_random_wins}/{GAMES_VS_RANDOM} ({100*maxn_vs_random_wins/GAMES_VS_RANDOM:.0f}%)")

# --- Summary ---
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(
    f"  H2H:         negamax {nega_pct:.0f}% vs max^n {maxn_pct:.0f}%  (expect ~50/50)")
print(
    f"  vs Random:   negamax {100*nega_vs_random_wins/GAMES_VS_RANDOM:.0f}% | max^n {100*maxn_vs_random_wins/GAMES_VS_RANDOM:.0f}%  (expect ~100%)")
h2h_ok = abs(nega_pct - 50) < 30  # generous margin
vs_rand_ok = nega_vs_random_wins >= GAMES_VS_RANDOM * \
    0.8 and maxn_vs_random_wins >= GAMES_VS_RANDOM * 0.8
print(f"\n  VERDICT: {'PASS' if (h2h_ok and vs_rand_ok) else 'CHECK'} - ",
      end="")
if h2h_ok and vs_rand_ok:
    print("max^n preserves model strength")
else:
    if not h2h_ok:
        print("H2H too skewed; ", end="")
    if not vs_rand_ok:
        print("vs-random win rate too low", end="")
    print()
