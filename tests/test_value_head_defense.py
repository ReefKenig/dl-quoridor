"""Seat-0 value targets from clone-vs-clone games (cfg.clone_seat0_value_weight).

The policy anchor holds the racing prior; the erosion v10 measured surviving it
is in the value head, which clone games retrain on outcomes where seat-0 racing
loses. Anchored games keep every column -- there racing is what wins.
"""
import numpy as np
import pytest

from src.mcts.training_mp import (ReplayBufferMP, TrainingConfigMP,
                                  clone_seat0_value_weights)
from src.model.network_mp import QuoridorModelMP


def _model(seed, num_players=4):
    import torch
    torch.manual_seed(seed)
    return QuoridorModelMP(board_size=5, action_space_size=44, num_channels=8,
                           num_res_blocks=1, num_players=num_players, lr=1e-2,
                           device="cpu")


def _batch(n=16, num_players=4, seed=0):
    rng = np.random.RandomState(seed)
    S = rng.rand(n, 5, 5, 3 * num_players + 3).astype(np.float32)
    P = np.full((n, 44), 1 / 44, dtype=np.float32)
    V = rng.uniform(-1, 1, (n, num_players)).astype(np.float32)
    return S, P, V


# --- the weight mask ----------------------------------------------------------

def test_only_seat_zero_of_clone_samples_is_dropped():
    w = clone_seat0_value_weights(["self", "greedy", "self"], 4, 0.0)

    assert w.tolist() == [[0, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]]


def test_a_partial_weight_softens_rather_than_drops():
    w = clone_seat0_value_weights(["self", "past"], 4, 0.25)

    assert w[0, 0] == pytest.approx(0.25)
    assert w[1, 0] == 1.0            # only clone games are down-weighted


def test_the_off_setting_builds_no_mask_at_all():
    """1.0 is the plain step, not a mask of ones on the weighted path."""
    assert clone_seat0_value_weights(["self"] * 4, 4, 1.0) is None


def test_untagged_buffer_samples_count_as_clone_games():
    """`add` with no sources means self-play, which is what those samples are."""
    buf = ReplayBufferMP(100)
    buf.add([(np.zeros((3, 3, 1), np.float32), np.zeros(4, np.float32),
              np.zeros(4, np.float32)) for _ in range(4)])

    w = clone_seat0_value_weights(buf.sources_array(), 4, 0.0)

    assert w[:, 0].tolist() == [0, 0, 0, 0]


# --- the buffer hands the labels over ----------------------------------------

def test_sample_batch_can_return_the_drawn_sources():
    buf = ReplayBufferMP(100)
    sample = (np.zeros((3, 3, 1), np.float32), np.zeros(4, np.float32),
              np.zeros(4, np.float32))
    buf.add([sample] * 6, sources=["self"] * 3 + ["greedy"] * 3)

    S, P, V, src = buf.sample_batch(6, with_sources=True)

    assert len(src) == len(S) == 6
    assert sorted(src.tolist()) == ["greedy"] * 3 + ["self"] * 3


def test_sample_batch_keeps_its_three_value_shape_by_default():
    """Every other caller unpacks three; the labels are opt-in."""
    buf = ReplayBufferMP(100)
    buf.add([(np.zeros((3, 3, 1), np.float32), np.zeros(4, np.float32),
              np.zeros(4, np.float32))])

    assert len(buf.sample_batch(1)) == 3


# --- the loss ----------------------------------------------------------------

def test_uniform_weights_reproduce_the_plain_step():
    """Normalizing by the weights' own sum keeps loss_v on its scale."""
    S, P, V = _batch()
    plain = _model(0).train_step(S, P, V)
    weighted = _model(0).train_step(S, P, V,
                                    value_weights=np.ones_like(V))

    assert weighted == pytest.approx(plain, rel=1e-5)


def test_dropped_seat_zero_targets_do_not_train_the_head():
    """Seat 0 must stay put while the other seats still learn."""
    S, P, V = _batch(seed=3)
    weights = np.ones_like(V)
    weights[:, 0] = 0.0

    model = _model(0)
    before = np.array([model.predict(s)[1] for s in S])
    for _ in range(30):
        model.train_step(S, P, V, value_weights=weights)
    after = np.array([model.predict(s)[1] for s in S])

    err_before = np.abs(before - V).mean(axis=0)
    err_after = np.abs(after - V).mean(axis=0)
    assert err_after[1:].mean() < err_before[1:].mean()   # other seats learned
    assert err_after[0] >= err_before[0] * 0.9            # seat 0 did not


def test_an_all_zero_weighting_still_trains_the_policy():
    """A batch with no usable value target must not stall the whole step."""
    S, P, V = _batch()
    lp, lv = _model(0).train_step(S, P, V, value_weights=np.zeros_like(V))

    assert lv == 0.0
    assert lp > 0.0


# --- the config knob ----------------------------------------------------------

def test_the_knob_is_off_by_default():
    assert TrainingConfigMP().clone_seat0_value_weight == 1.0


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_a_weight_outside_zero_to_one_is_rejected(bad):
    with pytest.raises(ValueError, match="clone_seat0_value_weight"):
        TrainingConfigMP(clone_seat0_value_weight=bad)


def test_the_n4_variant_turns_it_on_and_n2_does_not():
    """A four-player intervention: at N=2 both seats are winnable."""
    import json

    from src.utils.config import resolve_run_config

    raw = json.loads(open("configs/config_9x9.json").read())
    assert resolve_run_config(raw, "n4")["clone_seat0_value_weight"] == 0.0
    assert resolve_run_config(raw, "n2")["clone_seat0_value_weight"] == 1.0
