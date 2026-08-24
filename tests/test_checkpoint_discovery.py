"""Checkpoint discovery for the held-out rescore (scripts/eval_all_checkpoints).

The hand-kept CHECKPOINTS list is how the v11 and pretrain rows sat un-scored in
a stash: every new run needed an edit nobody was reminded to make. Discovery
makes existing be enough - a run joins the table because its dir has a frozen
config and a checkpoint.
"""
import json
import os

from scripts.eval_all_checkpoints import discover_checkpoints


def _run(root, name, config=None, files=(), mtime=None):
    d = root / name
    d.mkdir()
    if config is not None:
        (d / "config.json").write_text(json.dumps(config))
    for f in files:
        p = d / f
        p.write_bytes(b"x")
        if mtime is not None:
            os.utime(p, (mtime, mtime))


CFG = {"num_players": 2, "board_size": 9}


def test_prefers_greedy_peak_over_every_other_checkpoint(tmp_path):
    _run(tmp_path, "a", CFG, ["greedy_peak.pt", "ship.pt", "best.pt", "latest.pt"])

    [(path, n, board)] = discover_checkpoints(root=str(tmp_path))

    assert path.endswith("a/greedy_peak.pt") and (n, board) == (2, 9)


def test_falls_back_down_the_preference_order(tmp_path):
    _run(tmp_path, "shipped", CFG, ["ship.pt", "best.pt"])
    _run(tmp_path, "probe", CFG, ["best.pt"])
    _run(tmp_path, "pretrain_x", CFG, ["pretrain.pt"])

    picked = {os.path.basename(os.path.dirname(p)): os.path.basename(p)
              for p, _, _ in discover_checkpoints(root=str(tmp_path))}

    assert picked == {"shipped": "ship.pt", "probe": "best.pt",
                      "pretrain_x": "pretrain.pt"}


def test_the_v7_override_beats_the_preference_order(tmp_path):
    """The canonical table scores n2_9x9_v7 on latest.pt, not its ship.pt."""
    _run(tmp_path, "n2_9x9_v7", CFG, ["ship.pt", "latest.pt"])

    [(path, _, _)] = discover_checkpoints(root=str(tmp_path))

    assert path.endswith("n2_9x9_v7/latest.pt")


def test_skips_are_loud_never_silent(tmp_path, capsys):
    _run(tmp_path, "no_config", None, ["ship.pt"])
    _run(tmp_path, "no_fields", {"num_workers": 8}, ["ship.pt"])
    _run(tmp_path, "no_checkpoint", CFG, [])
    _run(tmp_path, "n4_5x5_v2_killed", CFG, ["ship.pt"])   # EXCLUDED by name

    assert discover_checkpoints(root=str(tmp_path)) == []
    out = capsys.readouterr().out
    assert "no_fields" in out and "no_checkpoint" in out
    # No config at all is an artifact dir, not a run - nothing to report.
    assert "no_config" not in out and "killed" not in out


def test_newest_checkpoint_first(tmp_path):
    _run(tmp_path, "old", CFG, ["ship.pt"], mtime=1_000)
    _run(tmp_path, "new", CFG, ["ship.pt"], mtime=2_000)

    paths = [p for p, _, _ in discover_checkpoints(root=str(tmp_path))]

    assert [os.path.basename(os.path.dirname(p)) for p in paths] == ["new", "old"]
