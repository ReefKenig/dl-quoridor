"""
Quoridor Game Environment
==========================
Owner: Reef

Implement QuoridorEnvInterface for the 5x5 board.
MCTS and the training loop depend on this contract.

Checklist:
    [ ] action_space_size — calculate precisely for 5x5 (pawn moves + walls)
    [ ] get_valid_actions — must handle jumps, wall legality, connectivity
    [ ] step — apply action, check win condition, switch player
    [ ] clone_state — deep copy (MCTS clones thousands of times per search)
    [ ] get_current_player — 0 or 1
    [ ] reset — fresh 5x5 board
    [ ] state_to_tensor — 10-channel tensor, shape (5, 5, 10), values in [0, 1]

When done, run: python -m tests.test_mcts
If MCTS beats random at >80% win rate with your env, integration works.
"""

from src.env.env_interface import QuoridorEnvInterface

# TODO: Implement QuoridorEnv(QuoridorEnvInterface)
