"""The progress logger must never be able to kill a training run.

Several log strings contain em-dashes. `open()` without an explicit encoding
uses the locale's preferred encoding, which is ASCII under LC_ALL=C, so a
C-locale server would raise UnicodeEncodeError from inside the training loop and
lose a multi-day run. This has bitten this project before.
"""
import os

from src.utils.logger import make_progress_logger

# The exact non-ASCII text that appears in training_mp log lines.
EM_DASH_MSG = "WARNING: draw_rate 44% > 20% — games timing out at max_game_moves=160"


def test_em_dash_survives_a_round_trip(tmp_path, capsys):
    path = str(tmp_path / "games.log")

    make_progress_logger(path)(EM_DASH_MSG)

    with open(path, encoding="utf-8") as f:
        assert "—" in f.read()


def test_log_file_is_written_as_utf8_regardless_of_locale(tmp_path):
    """Pinned explicitly: the bytes must be UTF-8, not locale-dependent."""
    path = str(tmp_path / "games.log")

    make_progress_logger(path)(EM_DASH_MSG)

    raw = open(path, "rb").read()
    assert "—".encode("utf-8") in raw
    raw.decode("utf-8")          # must not raise


def test_multiline_messages_are_timestamped_per_line(tmp_path):
    path = str(tmp_path / "games.log")

    make_progress_logger(path)("first — line", "second — line")

    lines = [ln for ln in open(path, encoding="utf-8").read().splitlines() if ln]
    assert len(lines) == 2
    assert all(ln.startswith("20") for ln in lines), lines


def test_appends_rather_than_truncates(tmp_path):
    path = str(tmp_path / "games.log")
    log = make_progress_logger(path)

    log("one")
    log("two")

    assert len([ln for ln in open(path, encoding="utf-8") if ln.strip()]) == 2


def test_survives_a_stdout_that_cannot_encode(tmp_path, monkeypatch, capsys):
    """A pipe under a C locale raises on print. The run must continue anyway."""
    path = str(tmp_path / "games.log")
    calls = []

    def exploding_print(*args, **kwargs):
        if not calls:
            calls.append(1)
            raise UnicodeEncodeError("ascii", "—", 0, 1, "ordinal not in range")

    monkeypatch.setattr("builtins.print", exploding_print)

    make_progress_logger(path)(EM_DASH_MSG)   # must not raise

    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        assert "—" in f.read()
