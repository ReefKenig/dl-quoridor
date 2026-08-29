"""v13's factored policy head: config resolution and notebook wiring.

network_mp.py itself (QuoridorModelMP(policy_head=...), head_type_from_state)
is owned by a concurrent branch and is not imported here - these tests check
only that the config resolves the right value and that the notebooks read and
thread it, via the same source-inspection approach as
test_notebook_config_coverage.py.
"""
import ast
import json

import pytest

from src.utils.config import resolve_run_config
from tests.test_notebook_config_coverage import (NOTEBOOKS, _config_bindings,
                                                  _parsed_cells, _repo_root)

CONFIG = "configs/config_9x9.json"


def _cfg9():
    return json.loads((_repo_root() / CONFIG).read_text())


def test_n4_resolves_factored_policy_head():
    assert resolve_run_config(_cfg9(), "n4")["policy_head"] == "factored"


def test_n2_resolves_flat_policy_head():
    assert resolve_run_config(_cfg9(), "n2")["policy_head"] == "flat"


def test_n4_init_checkpoint_points_at_the_factored_pretrain():
    rc = resolve_run_config(_cfg9(), "n4")
    assert rc["init_checkpoint"].endswith("pretrain_n4_9x9_factored/pretrain.pt")


def _find_make_model_fn(notebook):
    """The `def make_model(): ...` FunctionDef feeding training_loop_mp, or None."""
    for tree in _parsed_cells(notebook):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "make_model":
                return node
    return None


def _quoridor_model_calls(node):
    """Every QuoridorModelMP(...) Call under an AST node."""
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "QuoridorModelMP"]


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_notebook_binds_policy_head_from_config(notebook):
    bindings = _config_bindings(notebook)
    assert bindings.get("POLICY_HEAD") == "policy_head", (
        f"{notebook} must bind POLICY_HEAD = rc.get('policy_head', ...) so the "
        f"model architecture follows the run's config.")


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_notebook_make_model_passes_policy_head(notebook):
    fn = _find_make_model_fn(notebook)
    assert fn is not None, f"{notebook} has no make_model() function"
    calls = _quoridor_model_calls(fn)
    assert calls, f"{notebook}: make_model() never constructs a QuoridorModelMP"
    call = calls[0]
    passed = {kw.arg: kw.value for kw in call.keywords if kw.arg}
    value = passed.get("policy_head")
    assert value is not None, (
        f"{notebook}: make_model()'s QuoridorModelMP(...) never passes "
        f"policy_head - a factored run would silently build a flat head.")
    assert isinstance(value, ast.Name) and value.id == "POLICY_HEAD", (
        f"{notebook}: make_model() passes policy_head={ast.dump(value)}, "
        f"not the POLICY_HEAD binding read from config.")


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_every_quoridor_model_construction_threads_policy_head(notebook):
    """Every QuoridorModelMP(...) built FROM the resolved config (i.e. that also
    passes board_size=BOARD) must carry policy_head, not just make_model()'s.

    Catches a stray construction - e.g. the smoke-test or final-eval cell -
    left on the old default, which would crash loading a factored checkpoint
    the training cell itself produced.
    """
    missing = []
    for tree in _parsed_cells(notebook):
        for call in _quoridor_model_calls(tree):
            kwargs = {kw.arg for kw in call.keywords if kw.arg}
            if "board_size" not in kwargs:
                continue
            if "policy_head" not in kwargs:
                missing.append(ast.dump(call.func))
    assert not missing, (
        f"{notebook}: QuoridorModelMP construction(s) without policy_head: "
        f"{missing}")
