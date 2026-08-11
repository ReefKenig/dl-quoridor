"""Champion (best.pt) persistence and resume behaviour.

Regression cover for the bug that produced 51 consecutive rejections across both
9x9 runs: `best.pt` was written only inside `if accepted:`, so a run that never
accepted never created it, and the resume path then silently re-anchored the
champion to the current learner. The gate was comparing the model against an
identical copy of itself while reporting a plausible-looking win rate.
"""
import os

import pytest

from src.mcts.training_mp import init_champion, load_champion


class FakeModel:
    """Minimal stand-in for QuoridorModelMP — only weight identity matters here."""

    def __init__(self, weights="init"):
        self.weights = weights

    def copy_weights_from(self, other):
        self.weights = other.weights

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.weights)

    def load(self, path):
        with open(path) as f:
            self.weights = f.read()


@pytest.fixture
def ckpt_dir(tmp_path):
    return str(tmp_path)


def test_best_pt_written_at_run_start(ckpt_dir):
    """The champion is durable from iteration 0, before anything is accepted."""
    model, best = FakeModel("iter0"), FakeModel("blank")
    init_champion(best, model, ckpt_dir)

    assert os.path.exists(os.path.join(ckpt_dir, "best.pt"))
    assert best.weights == "iter0"


def test_init_does_not_clobber_an_existing_champion(ckpt_dir):
    """Re-running init must not overwrite an accepted champion with a fresh model."""
    FakeModel("accepted-champion").save(os.path.join(ckpt_dir, "best.pt"))

    init_champion(FakeModel("fresh"), FakeModel("blank"), ckpt_dir)

    with open(os.path.join(ckpt_dir, "best.pt")) as f:
        assert f.read() == "accepted-champion"


def test_resume_with_best_pt_loads_from_disk(ckpt_dir):
    """Champion weights come from best.pt, not from the current learner."""
    FakeModel("champion").save(os.path.join(ckpt_dir, "best.pt"))
    model, best = FakeModel("learner-iter38"), FakeModel("blank")

    loaded = load_champion(best, model, ckpt_dir)

    assert loaded is True
    assert best.weights == "champion"


def test_resume_without_best_pt_warns_loudly(ckpt_dir):
    """The exact failure mode behind 51 straight rejections.

    Falling back to the learner is survivable, but it must not be silent: a
    champion equal to the candidate makes eval-vs-best structurally unwinnable.
    """
    messages = []
    model, best = FakeModel("learner-iter38"), FakeModel("blank")

    loaded = load_champion(best, model, ckpt_dir, log=messages.append)

    assert loaded is False
    assert any("missing on resume" in m for m in messages), messages
    # And it must persist, so the next restart is not silently re-anchored again.
    assert os.path.exists(os.path.join(ckpt_dir, "best.pt"))


def test_resume_after_fallback_is_stable(ckpt_dir):
    """Second restart loads the champion written by the first, not the learner."""
    load_champion(FakeModel("blank"), FakeModel("learner-A"), ckpt_dir,
                  log=lambda *_: None)

    best = FakeModel("blank")
    loaded = load_champion(best, FakeModel("learner-B"), ckpt_dir,
                           log=lambda *_: None)

    assert loaded is True
    assert best.weights == "learner-A"


# --- gate liveness (cfg.gate_arm_on_greedy) -----------------------------------
# The second instance of "the gate cannot fire" in this project, by a different
# route: 128 of 131 legal opening actions at 9x9 are walls, so an untrained
# champion spams walls and never advances. 134-151 of 160 gate games then ran
# past max_game_moves, decided stayed under the 25% floor, and the gate could not
# accept — which kept the champion at the random init. v7 accepted 0 times in 53
# iterations while win_vs_best read as high as 1.000.

class _Cfg:
    def __init__(self, **kw):
        self.gate_arm_on_greedy = True
        self.gate_arm_greedy_min = 0.0
        self.__dict__.update(kw)


def _armed(cfg, history):
    """The arming expression from training_loop_mp."""
    return not cfg.gate_arm_on_greedy or any(
        (r.get("win_vs_greedy") or 0.0) > cfg.gate_arm_greedy_min for r in history)


def test_a_fresh_run_starts_disarmed():
    assert not _armed(_Cfg(), [])


def test_the_gate_stays_held_while_the_learner_never_beats_greedy():
    # v7's shape: many evals, every one at zero.
    assert not _armed(_Cfg(), [{"win_vs_greedy": 0.0} for _ in range(13)])


def test_a_single_win_arms_the_gate():
    assert _armed(_Cfg(), [{"win_vs_greedy": 0.0}, {"win_vs_greedy": 0.05}])


def test_arming_survives_a_resume_and_is_not_re_disarmed():
    """Recovered from history rather than persisted separately: a later zero
    must not put the gate back to sleep and re-seed a champion."""
    history = [{"win_vs_greedy": 0.5}, {"win_vs_greedy": 0.0}]
    assert _armed(_Cfg(), history)


def test_rows_without_a_greedy_eval_do_not_arm():
    # eval_every skips greedy on most iterations; a missing key is not a win.
    assert not _armed(_Cfg(), [{"win_vs_best": 1.0}, {"win_vs_greedy": None}])


def test_the_bar_is_configurable():
    history = [{"win_vs_greedy": 0.1}]
    assert _armed(_Cfg(gate_arm_greedy_min=0.05), history)
    assert not _armed(_Cfg(gate_arm_greedy_min=0.25), history)


def test_disabling_the_feature_keeps_the_old_always_armed_behaviour():
    assert _armed(_Cfg(gate_arm_on_greedy=False), [])


# --- ship resolution (resolve_ship_checkpoint) ---------------------------------
# v8 shipped its iteration-20 model (60% vs greedy) while iteration 12's 81% was
# never written anywhere; greedy_peak.pt now holds the strongest greedy eval and
# outranks latest.pt whenever the gate never accepted.

import json

from src.utils.checkpoint import resolve_ship_checkpoint


def _write_run(ckpt_dir, history, files):
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump({"history": history}, f)
    for name in files:
        FakeModel(name).save(os.path.join(ckpt_dir, name))


def test_an_accepted_best_still_outranks_the_greedy_peak(ckpt_dir):
    _write_run(ckpt_dir,
               [{"iter": 4, "accepted": True, "win_vs_greedy": 0.4}],
               ["best.pt", "latest.pt", "greedy_peak.pt"])
    path, label = resolve_ship_checkpoint(ckpt_dir)
    assert path.endswith("best.pt") and "accepted at iter 4" in label


def test_with_no_accepts_the_greedy_peak_outranks_latest(ckpt_dir):
    _write_run(ckpt_dir,
               [{"iter": 12, "accepted": False, "win_vs_greedy": 0.812},
                {"iter": 20, "accepted": False, "win_vs_greedy": 0.60}],
               ["best.pt", "latest.pt", "greedy_peak.pt"])
    path, label = resolve_ship_checkpoint(ckpt_dir)
    assert path.endswith("greedy_peak.pt")
    assert "81.2%" in label and "iter 12" in label


def test_without_a_peak_file_latest_remains_the_fallback(ckpt_dir):
    _write_run(ckpt_dir,
               [{"iter": 20, "accepted": False, "win_vs_greedy": 0.6}],
               ["best.pt", "latest.pt"])
    path, label = resolve_ship_checkpoint(ckpt_dir)
    assert path.endswith("latest.pt") and "iter 20" in label
