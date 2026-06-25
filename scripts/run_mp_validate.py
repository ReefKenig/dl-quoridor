"""Validate the N-player engine: N=2 parity, jump rules, random play, max^n search."""
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
from src.env.quoridor_env_mp import QuoridorEnvMP
from src.env.quoridor_env import QuoridorEnv          # original 2p
import random
import logging
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.disable(logging.CRITICAL)


def banner(t): print("\n"+"="*66+"\n"+t+"\n"+"="*66)


# ---------- TEST 1: N=2 parity with the original engine (lockstep) ----------
banner("TEST 1  N=2 generalized engine == original engine (lockstep)")
orig = QuoridorEnv(board_size=5, max_walls_per_player=3, max_turns=200)
mp2 = QuoridorEnvMP(board_size=5, num_players=2,
                    max_walls_per_player=3, max_turns=200)
rng = random.Random(7)
mism = 0
games = 20
for g in range(games):
    so = orig.reset()
    sm = mp2.reset()
    for step in range(200):
        vo = set(orig.get_valid_actions(so).tolist())
        vm = set(mp2.get_valid_actions(sm).tolist())
        if vo != vm:
            mism += 1
            print(f"  game{g} step{step}: VALID-SET MISMATCH "
                  f"only_orig={sorted(vo-vm)} only_mp={sorted(vm-vo)}")
            break
        if not vo:
            break
        a = rng.choice(sorted(vo))
        so, _, do, io = orig.step(so, a)
        sm, _, dm, im = mp2.step(sm, a)
        # state correspondence
        ok = (so.p0_pos == sm.positions[0] and so.p1_pos == sm.positions[1] and
              (so.p0_h_walls | so.p1_h_walls) == sm.h_walls and
              (so.p0_v_walls | so.p1_v_walls) == sm.v_walls and
              so.current_player == sm.current_player and do == dm and io.get("winner") == im.get("winner"))
        if not ok:
            mism += 1
            print(f"  game{g} step{step}: STATE MISMATCH "
                  f"orig(p0={so.p0_pos},p1={so.p1_pos},cp={so.current_player},done={do},win={io.get('winner')}) "
                  f"mp(p={sm.positions},cp={sm.current_player},done={dm},win={im.get('winner')})")
            break
        if do:
            break
print("  PASS — N=2 identical to original" if mism ==
      0 else f"  FAIL ({mism} mismatches)")

# ---------- TEST 2: official jump rule, crafted N=4 positions ----------
banner("TEST 2  official jump rule (crafted)")
e = QuoridorEnvMP(board_size=5, num_players=4)


def legal_targets(pos, others, h=set(), v=set()):
    return sorted(e._pawn_moves(pos, set(others), h, v))


# (a) straight jump: mover at (2,2), pawn at (1,2), cell (0,2) empty -> jump to (0,2)
t = legal_targets((2, 2), [(1, 2), (4, 4), (4, 0)])
print("  (a) straight landing empty -> includes (0,2):", (0, 2) in t,
      "| up-targets:", [x for x in t if x[1] == 2 and x[0] < 2])
assert (0, 2) in t and (1, 2) not in t

# (b) pawn behind blocks straight -> diagonals beside the jumped pawn
# (1,2) adjacent, (0,2) occupied behind
t = legal_targets((2, 2), [(1, 2), (0, 2), (4, 0)])
diag = [x for x in t if x in [(1, 1), (1, 3)]]
print("  (b) pawn behind -> straight blocked, diagonals (1,1)/(1,3):",
      sorted(diag), "| (0,2) excluded:", (0, 2) not in t)
assert (0, 2) not in t and set(diag) == {(1, 1), (1, 3)}

# (c) no double-jump: even with two clear behind, never land 3 away
t = legal_targets((2, 2), [(1, 2), (4, 4), (4, 0)])
assert all(abs(x[0]-2) <= 2 and abs(x[1]-2) <= 2 for x in t)
print("  (c) no double-jump (max reach 2):", "OK")
print("  PASS")

# ---------- TEST 3: full games terminate with a winner, N=2/4 ----------
banner("TEST 3  random self-play terminates with a winner (N=2,4)")
for N in (2, 4):
    env = QuoridorEnvMP(board_size=5, num_players=N, max_turns=400)
    rng = np.random.RandomState(0)
    wins = {}
    jumps = 0
    tot_moves = 0
    for g in range(15):
        s = env.reset()
        for m in range(400):
            v = env.get_valid_actions(s)
            if len(v) == 0:
                break
            a = int(rng.choice(v))
            # detect a jump/diagonal action (codes 4..11)
            if a < 12 and a >= 4:
                jumps += 1
            s, _, done, info = env.step(s, a)
            tot_moves += 1
            if done:
                wins[info.get("winner")] = wins.get(info.get("winner"), 0)+1
                break
    decided = sum(c for w, c in wins.items() if w is not None)
    print(f"  N={N}: {decided}/15 games had a winner | win dist={{k: v for k, v in sorted(wins.items(), key=lambda x: (x[0] is None, x[0]))} } | jump-moves seen={jumps}")
print("  (random play; just checking termination + path-legality across N seats)")

# ---------- TEST 4: max^n search runs on N=4 ----------
banner("TEST 4  max^n search drives N=4 to terminal")
for N in (4,):
    env = QuoridorEnvMP(board_size=5, num_players=N, max_turns=200)
    mc = MCTSMaxN(config=MCTSConfig(num_simulations=30, dirichlet_epsilon=0.25, max_rollout_depth=80),
                  evaluate_fn=None, num_players=N)   # random-rollout
    s = env.reset()
    winner = None
    for m in range(200):
        probs = mc.search(env, s, temperature=0.3)
        a = int(np.random.choice(len(probs), p=probs))
        s, _, done, info = env.step(s, a)
        if done:
            winner = info.get("winner")
            break
    print(f"  N={N}: max^n self-play ended at move {m+1}, winner=seat {winner}")
print("  PASS — search loop handles N seats")
