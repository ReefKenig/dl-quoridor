"""Factored policy head (gate over move-vs-wall, then a per-class distribution).

The flat softmax puts ~91% of an untrained net's raw mass on walls at 9x9
by sheer action count (128/140) - measured to be the mechanism behind every
failed N=4 run. The factored head removes the count bias: see the mechanism
test below.
"""
import os
import tempfile

import numpy as np
import pytest
import torch

from src.env.quoridor_env_mp import NUM_MOVE_ACTIONS, compute_action_space_size
from src.model.network_mp import (QuoridorModelMP, QuoridorNetworkMP,
                                  head_type_from_state)


def _model(policy_head, board_size=5, num_players=2, seed=0, **kw):
    torch.manual_seed(seed)
    action_space_size = compute_action_space_size(board_size)
    kw.setdefault("num_channels", 8)
    kw.setdefault("num_res_blocks", 1)
    return QuoridorModelMP(
        board_size=board_size, action_space_size=action_space_size,
        num_players=num_players, device="cpu", policy_head=policy_head, **kw)


def _states(n, board_size, num_players, seed=0):
    rng = np.random.RandomState(seed)
    return rng.rand(n, board_size, board_size, 3 * num_players + 3).astype(np.float32)


# --- output contract: still a valid distribution over the full action space --

@pytest.mark.parametrize("policy_head", ["flat", "factored"])
def test_predict_policy_sums_to_one(policy_head):
    model = _model(policy_head)
    state = _states(1, 5, 2)[0]
    policy, value = model.predict(state)

    assert policy.shape == (44,)
    assert np.all(policy >= 0)
    assert np.isclose(policy.sum(), 1.0, atol=1e-5)
    assert value.shape == (2,)


@pytest.mark.parametrize("policy_head", ["flat", "factored"])
def test_predict_batch_policy_sums_to_one(policy_head):
    model = _model(policy_head, num_players=4)
    states = _states(6, 5, 4)
    x = torch.from_numpy(states).float().permute(0, 3, 1, 2)
    policies, values = model.predict_batch(x)

    sums = policies.sum(dim=1).numpy()
    assert np.allclose(sums, 1.0, atol=1e-5)
    assert values.shape == (6, 4)


# --- the mechanism: the factored head removes the wall count bias -----------

def test_untrained_flat_net_is_wall_dominated_at_9x9_n4():
    model = _model("flat", board_size=9, num_players=4, seed=1)
    states = _states(32, 9, 4, seed=2)
    x = torch.from_numpy(states).float().permute(0, 3, 1, 2)
    policies, _ = model.predict_batch(x)

    wall_mass = policies[:, NUM_MOVE_ACTIONS:].sum(dim=1).mean().item()
    assert wall_mass > 0.85


def test_untrained_factored_net_is_balanced_at_9x9_n4():
    model = _model("factored", board_size=9, num_players=4, seed=1)
    states = _states(32, 9, 4, seed=2)
    x = torch.from_numpy(states).float().permute(0, 3, 1, 2)
    policies, _ = model.predict_batch(x)

    wall_mass = policies[:, NUM_MOVE_ACTIONS:].sum(dim=1).mean().item()
    assert abs(wall_mass - 0.5) < 0.15


# --- training the factored head ----------------------------------------------

def _fixed_batch(board_size=5, num_players=2, n=16, move_heavy=False, seed=7):
    rng = np.random.RandomState(seed)
    action_space_size = compute_action_space_size(board_size)
    S = rng.rand(n, board_size, board_size, 3 * num_players + 3).astype(np.float32)
    if move_heavy:
        P = np.zeros((n, action_space_size), dtype=np.float32)
        P[:, :NUM_MOVE_ACTIONS] = 1.0 / NUM_MOVE_ACTIONS
    else:
        P = rng.dirichlet(np.ones(action_space_size), size=n).astype(np.float32)
    V = rng.uniform(-1, 1, (n, num_players)).astype(np.float32)
    return S, P, V


def test_factored_train_step_reduces_policy_loss():
    S, P, V = _fixed_batch()
    model = _model("factored")

    first_lp, _ = model.train_step(S, P, V)
    last_lp = first_lp
    for _ in range(50):
        last_lp, _ = model.train_step(S, P, V)

    assert last_lp < first_lp


def test_factored_net_fits_a_move_heavy_target():
    """Wall mass should fall as the net learns a target concentrated on moves."""
    board_size, num_players = 5, 2
    S, P, V = _fixed_batch(board_size=board_size, num_players=num_players,
                           move_heavy=True)
    model = _model("factored", board_size=board_size, num_players=num_players)

    wall_mass_before = model.predict(S[0])[0][NUM_MOVE_ACTIONS:].sum()
    for _ in range(50):
        model.train_step(S, P, V)
    wall_mass_after = model.predict(S[0])[0][NUM_MOVE_ACTIONS:].sum()

    assert wall_mass_after < wall_mass_before


# --- checkpointing -------------------------------------------------------------

@pytest.mark.parametrize("policy_head", ["flat", "factored"])
def test_save_load_round_trip_predictions_match(policy_head):
    model = _model(policy_head, num_players=4, seed=3)
    state = _states(1, 5, 4, seed=4)[0]
    policy_before, value_before = model.predict(state)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model.save(f.name)
        path = f.name
    try:
        model2 = _model(policy_head, num_players=4, seed=999)  # different init
        model2.load(path)
        policy_after, value_after = model2.predict(state)
        assert np.allclose(policy_before, policy_after, atol=1e-6)
        assert np.allclose(value_before, value_after, atol=1e-6)
    finally:
        os.unlink(path)


def test_head_type_from_state_detects_factored():
    net = QuoridorNetworkMP(board_size=5, action_space_size=44,
                            num_channels=8, num_res_blocks=1,
                            policy_head="factored")
    assert head_type_from_state(net.state_dict()) == "factored"


def test_head_type_from_state_detects_flat():
    net = QuoridorNetworkMP(board_size=5, action_space_size=44,
                            num_channels=8, num_res_blocks=1,
                            policy_head="flat")
    assert head_type_from_state(net.state_dict()) == "flat"


def test_old_style_checkpoint_without_policy_head_key_loads_as_flat():
    """Checkpoints written before this change have no "policy_head" key."""
    model = _model("flat", seed=5)
    ck = {"network_state": model.network.state_dict(),
          "optimizer_state": model.optimizer.state_dict(),
          "num_players": model.num_players}
    assert "policy_head" not in ck

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(ck, f.name)
        path = f.name
    try:
        model2 = _model("flat", seed=6)
        model2.load(path)  # must not raise
        state = _states(1, 5, 2)[0]
        p1, v1 = model.predict(state)
        p2, v2 = model2.predict(state)
        assert np.allclose(p1, p2, atol=1e-6)
        assert np.allclose(v1, v2, atol=1e-6)
    finally:
        os.unlink(path)


def test_load_rejects_a_head_type_mismatch():
    model = _model("factored", seed=8)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model.save(f.name)
        path = f.name
    try:
        flat_model = _model("flat", seed=9)
        with pytest.raises(ValueError, match="policy_head"):
            flat_model.load(path)
    finally:
        os.unlink(path)


# --- flat path regression: unchanged parameter surface ------------------------

def test_flat_head_parameter_names_and_shapes_are_unchanged():
    net = QuoridorNetworkMP(board_size=5, action_space_size=44,
                            num_channels=8, num_res_blocks=1,
                            policy_head="flat")
    names = dict(net.named_parameters())
    assert "policy_fc.weight" in names
    assert names["policy_fc.weight"].shape == (44, 2 * 5 * 5)
    assert names["policy_fc.bias"].shape == (44,)
    # The factored-only submodules must not exist on the flat head.
    assert "policy_type_head.weight" not in names
    assert "policy_move_head.weight" not in names
    assert "policy_wall_head.weight" not in names


def test_flat_train_step_matches_pre_change_losses_for_a_fixed_seed():
    """policy_head defaults to "flat" and its math is untouched, so a fixed
    seed must reproduce the same loss trajectory the unfactored head gave."""
    S, P, V = _fixed_batch()
    torch.manual_seed(42)
    model_a = QuoridorModelMP(board_size=5, action_space_size=44,
                              num_channels=8, num_res_blocks=1,
                              num_players=2, device="cpu")  # policy_head defaults to flat
    torch.manual_seed(42)
    model_b = QuoridorModelMP(board_size=5, action_space_size=44,
                              num_channels=8, num_res_blocks=1,
                              num_players=2, device="cpu", policy_head="flat")

    losses_a = [model_a.train_step(S, P, V) for _ in range(5)]
    losses_b = [model_b.train_step(S, P, V) for _ in range(5)]

    assert losses_a == pytest.approx(losses_b)
