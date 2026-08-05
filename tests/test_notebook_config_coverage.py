"""Every config setting must actually reach the training loop.

This is the failure that already cost this project a pair of 50-hour runs: the
v4/v5 runs were launched believing they carried a set of fixes and contained none
of them, because the notebooks read some settings and silently dropped others.
Nothing in the code errors when a key is ignored -- the run just quietly trains
under different settings than the config describes.

So: if configs/config_9x9.json names a TrainingConfigMP field, both 9x9 notebooks
must pass it. A new knob that nobody wires up fails here rather than after a
14-hour run.
"""
import dataclasses
import json
import pathlib

import pytest

from src.mcts.training_mp import TrainingConfigMP
from src.utils.config import resolve_run_config

NOTEBOOKS = ("notebooks/train_9x9_n2.ipynb", "notebooks/train_9x9_n4.ipynb")
CONFIG = "configs/config_9x9.json"


def _repo_root():
    return pathlib.Path(__file__).resolve().parent.parent


def _config_keys_that_are_training_fields(variant):
    raw = json.loads((_repo_root() / CONFIG).read_text())
    resolved = resolve_run_config(raw, variant)
    fields = {f.name for f in dataclasses.fields(TrainingConfigMP)}
    return sorted(k for k in resolved if k in fields)


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_every_configured_setting_reaches_the_training_config(notebook):
    text = (_repo_root() / notebook).read_text()
    keys = _config_keys_that_are_training_fields("n2")
    missing = [k for k in keys if f"{k}=" not in text]
    assert not missing, (
        f"{notebook} never passes {missing} to TrainingConfigMP — the run would "
        f"silently train with the dataclass default instead of the config value.")


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_the_notebook_is_valid_json(notebook):
    nb = json.loads((_repo_root() / notebook).read_text())
    assert nb["cells"] and nb["nbformat"] == 4


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_the_notebook_tracks_a_branch_that_exists(notebook):
    """`git checkout .` discards local edits first, so a stale BRANCH silently
    runs old code — which is how v4/v5 ran without their intended fixes."""
    text = (_repo_root() / notebook).read_text()
    assert 'BRANCH = \\"' in text


def test_the_levers_this_work_added_are_all_configured():
    """Named explicitly so removing one from the config is a visible decision."""
    keys = _config_keys_that_are_training_fields("n2")
    for lever in ("mcts_wall_candidates", "opponent_greedy_share",
                  "anchored_sample_share", "wall_mask_fraction",
                  "eval_minimax_games", "greedy_min_seat_after"):
        assert lever in keys, f"{lever} is not in {CONFIG}"


def test_both_variants_resolve_the_same_field_set():
    """A key present for one player count and absent for the other means one of
    the two runs quietly uses a default."""
    assert (_config_keys_that_are_training_fields("n2")
            == _config_keys_that_are_training_fields("n4"))
