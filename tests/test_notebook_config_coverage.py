"""Every config setting must actually reach the training loop.

This is the failure that already cost this project a pair of 50-hour runs: the
v4/v5 runs were launched believing they carried a set of fixes and contained none
of them, because the notebooks read some settings and silently dropped others.
Nothing in the code errors when a key is ignored -- the run just quietly trains
under different settings than the config describes.

So: if configs/config_9x9.json names a TrainingConfigMP field, both 9x9 notebooks
must pass it. A new knob that nobody wires up fails here rather than after a
14-hour run.

Name-matched keys are found automatically. Keys the config spells differently
from the dataclass field are invisible to that intersection, so they are listed
in RENAMED_CONFIG_KEYS and checked the same way; a config key that is neither a
field, a renamed pair, nor an explicit non-training setting fails the staleness
guard rather than going silently unchecked.
"""
import ast
import dataclasses
import json
import pathlib

import pytest

from src.mcts.training_mp import TrainingConfigMP
from src.utils.config import resolve_run_config

NOTEBOOKS = ("notebooks/train_9x9_n2.ipynb", "notebooks/train_9x9_n4.ipynb")
CONFIG = "configs/config_9x9.json"

# config key -> TrainingConfigMP field, for every pair whose names differ.
# The dataclass default is what a run silently gets when the wiring is dropped.
RENAMED_CONFIG_KEYS = {
    "num_simulations": "mcts_simulations",        # default 100 vs config 600
    "reward_decay": "discount",                   # default 0.97 vs config 0.99
    "training_epochs": "train_steps_per_iter",    # default 200 vs config 600
    "dirichlet_epsilon": "mcts_dirichlet_epsilon",
    "max_rollout_depth": "max_turns",             # default 300 vs config 200
    "dirichlet_alpha": "mcts_dirichlet_alpha",    # was inert until v8: MCTSConfig default 0.3
    "c_puct": "mcts_c_puct",                      # was inert until v8: MCTSConfig default 1.41
}

# Resolved config keys that deliberately do not reach TrainingConfigMP: they
# configure the model, the optimizer or the raw MCTSConfig instead.
NON_TRAINING_CONFIG_KEYS = frozenset({
    "is_poc", "num_channels", "num_res_blocks", "learning_rate", "weight_decay",
    "temperature",
})

# Settings where a hardcoded literal in the notebook would be an expensive,
# silent divergence from the config rather than a harmless one.
VALUE_MUST_COME_FROM_CONFIG = (
    "mcts_simulations", "discount", "train_steps_per_iter",
    "mcts_wall_candidates", "wall_mask_fraction", "opponent_greedy_share",
    "anchored_sample_share", "gate_arm_on_greedy", "clone_seat0_value_weight",
)


def _repo_root():
    return pathlib.Path(__file__).resolve().parent.parent


def _training_fields():
    return {f.name for f in dataclasses.fields(TrainingConfigMP)}


def _resolved(variant):
    raw = json.loads((_repo_root() / CONFIG).read_text())
    return resolve_run_config(raw, variant)


def _config_keys_that_are_training_fields(variant):
    resolved, fields = _resolved(variant), _training_fields()
    return sorted(k for k in resolved if k in fields)


def _code_cells(notebook):
    nb = json.loads((_repo_root() / notebook).read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _parsed_cells(notebook):
    """Cells that are plain Python; cells with IPython magics are skipped."""
    trees = []
    for src in _code_cells(notebook):
        try:
            trees.append(ast.parse(src))
        except SyntaxError:
            continue
    return trees


def _training_config_kwargs(notebook):
    """Keyword arguments of the notebook's real TrainingConfigMP(...) call.

    Parsed rather than substring-matched: a key must be *passed*, not mentioned.
    The largest call is the training one; the eval cells build small throwaways.
    """
    calls = [node for tree in _parsed_cells(notebook) for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and getattr(node.func, "id", None) == "TrainingConfigMP"]
    assert calls, f"{notebook} never constructs a TrainingConfigMP"
    best = max(calls, key=lambda c: len(c.keywords))
    return {kw.arg: kw.value for kw in best.keywords if kw.arg}


def _config_bindings(notebook):
    """Notebook constant name -> config key it is read from (`X = rc["key"]`)."""
    bindings = {}
    for tree in _parsed_cells(notebook):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            key = _config_key_read_by(node.value)
            if key is not None:
                bindings[target.id] = key
    return bindings


def _config_key_read_by(node):
    """The config key `rc["key"]` / `rc.get("key", ...)` reads, else None."""
    if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)):
        return node.slice.value
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)):
        return node.args[0].value
    return None


def _reads_config_key(value, bindings, key):
    """True when the kwarg value carries `key`'s configured value.

    Either a constant bound from the config (`SIMS = rc["num_simulations"]`) or
    an inline lookup, as the nested `cfg.get('mcts', {}).get(...)` cell does.
    """
    if isinstance(value, ast.Name):
        return bindings.get(value.id) == key
    return any(_config_key_read_by(n) == key for n in ast.walk(value))


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_every_configured_setting_reaches_the_training_config(notebook):
    passed = _training_config_kwargs(notebook)
    keys = _config_keys_that_are_training_fields("n2")
    missing = [k for k in keys if k not in passed]
    assert not missing, (
        f"{notebook} never passes {missing} to TrainingConfigMP — the run would "
        f"silently train with the dataclass default instead of the config value.")


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_renamed_config_keys_reach_their_dataclass_field(notebook):
    """The keys the name intersection cannot see, checked the same way.

    `num_simulations` is spelled `mcts_simulations` on the dataclass, so dropping
    it trains at the 100-sim default while the config still reads 600.
    """
    passed = _training_config_kwargs(notebook)
    resolved, bindings = _resolved("n2"), _config_bindings(notebook)
    broken = []
    for key, field in sorted(RENAMED_CONFIG_KEYS.items()):
        if key not in resolved:
            continue
        if field not in passed:
            broken.append(f"{field} (from {key!r}) is never passed")
        elif not _reads_config_key(passed[field], bindings, key):
            broken.append(f"{field} is passed but does not read {key!r}")
    assert not broken, (
        f"{notebook}: {broken}. These config keys are spelled differently on "
        f"TrainingConfigMP, so nothing else in this file can catch them.")


def test_the_renamed_key_mapping_is_not_stale():
    """A new config key must be a field, a renamed pair, or explicitly not a setting.

    Without this, adding a sixth differently-spelled key is silently unchecked.
    """
    resolved, fields = _resolved("n2"), _training_fields()
    unclassified = sorted(k for k in resolved if k not in fields
                          and k not in RENAMED_CONFIG_KEYS
                          and k not in NON_TRAINING_CONFIG_KEYS)
    assert not unclassified, (
        f"{unclassified} in {CONFIG} match no TrainingConfigMP field. Add each to "
        f"RENAMED_CONFIG_KEYS with the field it feeds, or to "
        f"NON_TRAINING_CONFIG_KEYS if it is not a training setting.")


def test_the_renamed_key_mapping_is_well_formed():
    resolved, fields = _resolved("n2"), _training_fields()
    for key, field in RENAMED_CONFIG_KEYS.items():
        assert key != field, f"{key} needs no mapping; it matches by name"
        assert field in fields, f"{field} is not a TrainingConfigMP field"
        assert key in resolved, f"{key} is no longer in {CONFIG}"
    assert not (NON_TRAINING_CONFIG_KEYS & fields), (
        "a TrainingConfigMP field is listed as a non-training setting")


def test_every_config_backed_field_is_covered_by_one_of_the_two_checks():
    """The two mechanisms together must cover every field the config feeds."""
    resolved, fields = _resolved("n2"), _training_fields()
    from_config = ({f for f in fields if f in resolved}
                   | {RENAMED_CONFIG_KEYS[k] for k in resolved
                      if k in RENAMED_CONFIG_KEYS})
    checked = (set(_config_keys_that_are_training_fields("n2"))
               | {RENAMED_CONFIG_KEYS[k] for k in resolved
                  if k in RENAMED_CONFIG_KEYS})
    assert from_config == checked, f"unchecked fields: {sorted(from_config - checked)}"


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_the_costly_settings_are_not_hardcoded(notebook):
    """Passing a literal satisfies "the key is present" while ignoring the config.

    The notebooks assign every setting to a module-level constant read from the
    resolved config, so a literal here means the config value is dead.
    """
    passed = _training_config_kwargs(notebook)
    bindings = _config_bindings(notebook)
    hardcoded = []
    for field in VALUE_MUST_COME_FROM_CONFIG:
        value = passed.get(field)
        if value is None:
            hardcoded.append(f"{field} is not passed at all")
        elif isinstance(value, ast.Constant):
            hardcoded.append(f"{field}={value.value!r} is a literal")
        elif isinstance(value, ast.Name) and value.id not in bindings:
            hardcoded.append(f"{field}={value.id} is not read from the config")
    assert not hardcoded, (
        f"{notebook}: {hardcoded}. These must come from the resolved config, "
        f"not from a value typed into the notebook.")


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
