"""Imitation anchor (train_step's anchor_model / cfg.anchor_weight).

n4_9x9_v9 measured the failure this exists for: the warm-started racer held
its configured anchored-data share every iteration and still eroded to 0/80
vs greedy by iteration 28, because 65%-walled self-play gradient outweighs
any realistic sample mix. The anchor defends the prior in the loss instead.
"""
import numpy as np
import pytest

from src.mcts.training_mp import TrainingConfigMP
from src.model.network_mp import QuoridorModelMP


def _model(seed):
    import torch
    torch.manual_seed(seed)
    return QuoridorModelMP(board_size=5, action_space_size=44, num_channels=8,
                           num_res_blocks=1, num_players=2, lr=1e-2,
                           device="cpu")


def _batch(n=16, seed=0):
    rng = np.random.RandomState(seed)
    S = rng.rand(n, 5, 5, 9).astype(np.float32)
    P = np.full((n, 44), 1 / 44, dtype=np.float32)   # uniform MCTS targets
    V = np.zeros((n, 2), dtype=np.float32)
    return S, P, V


def _prob_of_anchor_argmax(model, anchor, S):
    probs, _ = model.predict(S[0])
    a_probs, _ = anchor.predict(S[0])
    return probs[int(np.argmax(a_probs))]


def test_a_heavy_anchor_pulls_the_policy_toward_the_reference():
    model, anchor = _model(0), _model(1)
    S, P, V = _batch()
    before = _prob_of_anchor_argmax(model, anchor, S)
    for _ in range(30):
        model.train_step(S, P, V, anchor_model=anchor, anchor_weight=10.0)
    after = _prob_of_anchor_argmax(model, anchor, S)
    assert after > before, (before, after)


def test_zero_weight_is_the_plain_step():
    """Same seeds, with and without a dormant anchor: identical losses."""
    S, P, V = _batch()
    a = _model(0).train_step(S, P, V)
    b = _model(0).train_step(S, P, V, anchor_model=_model(1), anchor_weight=0.0)
    assert a == pytest.approx(b)


def test_reported_policy_loss_excludes_the_anchor_term():
    """History comparability: loss_p is the MCTS-target CE either way."""
    S, P, V = _batch()
    lp_plain, _ = _model(0).train_step(S, P, V)
    lp_anch, _ = _model(0).train_step(S, P, V, anchor_model=_model(1),
                                      anchor_weight=5.0)
    assert lp_anch == pytest.approx(lp_plain)


def test_anchor_weight_without_a_checkpoint_is_refused():
    with pytest.raises(ValueError, match="init_checkpoint"):
        TrainingConfigMP(anchor_weight=0.2)
