"""
Dual-Headed Neural Network (Policy + Value)
=============================================
Owner: Rom

AlphaZero-style CNN/ResNet:
    Input:  (batch, board_h, board_w, 10) — 10-channel board tensor
    Output: policy (batch, action_space_size) — move probabilities
            value  (batch, 1)               — win probability [-1, 1]

Losses:
    Policy: Cross-entropy against MCTS visit distribution
    Value:  MSE against game outcome (+1 / -1)

Interface needed by self_play.py:
    model.predict(tensor)                          -> (policy_np, value_float)
    model.train_step(states, policies, values)     -> (loss_policy, loss_value)
    model.save(path) / model.load(path)

Start small for 5x5: 4 residual blocks, 64 channels.
"""

# TODO: Implement using PyTorch
