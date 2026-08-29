from src.server import app as server_app
from src.server.app import app

BOARD_2P = "5x5"
BOARD_4P = "9x9"
SESSION_A = "session-a"
SESSION_B = "session-b"


def _reset_session(client, board_size, num_players, game_id):
    response = client.post(
        f"/api/{board_size}/reset",
        json={"num_players": num_players, "game_id": game_id},
    )
    assert response.status_code == 200
    return response.get_json()


def _assert_player_count(state, expected_players, expected_game_id):
    assert len(state["players"]) == expected_players
    assert state["num_players"] == expected_players
    assert state["game_id"] == expected_game_id


def test_game_state_is_isolated_per_client_session():
    client_a = app.test_client()
    client_b = app.test_client()

    state_a = _reset_session(client_a, BOARD_2P, num_players=2, game_id=SESSION_A)
    state_b = _reset_session(client_b, BOARD_4P, num_players=4, game_id=SESSION_B)

    _assert_player_count(state_a, expected_players=2, expected_game_id=SESSION_A)
    _assert_player_count(state_b, expected_players=4, expected_game_id=SESSION_B)

    current_state_a = client_a.get(
        f"/api/{BOARD_2P}/state?game_id={SESSION_A}"
    ).get_json()
    current_state_b = client_b.get(
        f"/api/{BOARD_4P}/state?game_id={SESSION_B}"
    ).get_json()

    _assert_player_count(
        current_state_a, expected_players=2, expected_game_id=SESSION_A
    )
    _assert_player_count(
        current_state_b, expected_players=4, expected_game_id=SESSION_B
    )


def test_session_map_is_bounded_by_max_sessions(monkeypatch):
    monkeypatch.setattr(server_app, "MAX_SESSIONS", 3)
    server_app.SESSION_GAMES.clear()

    for i in range(6):
        _reset_session(app.test_client(), BOARD_2P, num_players=2, game_id=f"cap-{i}")

    assert len(server_app.SESSION_GAMES) == 3
    # the three most recent survive; the oldest are evicted first
    assert set(server_app.SESSION_GAMES) == {"cap-3", "cap-4", "cap-5"}


def test_idle_sessions_expire(monkeypatch):
    monkeypatch.setattr(server_app, "SESSION_TTL_SECONDS", 0)
    server_app.SESSION_GAMES.clear()

    _reset_session(app.test_client(), BOARD_2P, num_players=2, game_id="stale")
    _reset_session(app.test_client(), BOARD_2P, num_players=2, game_id="fresh")

    assert "stale" not in server_app.SESSION_GAMES
    assert "fresh" in server_app.SESSION_GAMES


def test_state_does_not_create_an_evicted_session():
    server_app.SESSION_GAMES.clear()

    response = app.test_client().get(f"/api/{BOARD_2P}/state?game_id=evicted")

    assert response.status_code == 404
    assert response.get_json() == {"error": "No active game"}


def test_reset_normalizes_player_count_and_invalid_seat():
    server_app.SESSION_GAMES.clear()
    response = app.test_client().post(
        f"/api/{BOARD_2P}/reset",
        json={"num_players": 3, "human_seat": None, "game_id": "invalid"},
    )

    assert response.status_code == 200
    state = response.get_json()
    assert state["num_players"] == 2
    assert state["human_seat"] == 0


def test_reset_rollback_restores_session_state(monkeypatch):
    server_app.SESSION_GAMES.clear()

    def boom(_runtime, _state, _human_seat):
        raise RuntimeError("AI exploded")

    monkeypatch.setattr(server_app, "_advance_ai_until_human", boom)

    response = app.test_client().post(
        f"/api/{BOARD_2P}/reset",
        json={"num_players": 2, "human_seat": 2, "game_id": "reset-rollback"},
    )

    assert response.status_code == 500
    runtime = server_app.SESSION_GAMES["reset-rollback"]
    assert runtime["human_seat"] == 0
    assert runtime["state"].current_player == 0


def test_move_rollback_restores_pre_move_state(monkeypatch):
    server_app.SESSION_GAMES.clear()
    client = app.test_client()

    reset = client.post(
        f"/api/{BOARD_2P}/reset",
        json={"num_players": 2, "human_seat": 0, "game_id": "move-rollback"},
    )
    assert reset.status_code == 200

    runtime = server_app.SESSION_GAMES["move-rollback"]
    original_state = runtime["state"]

    def boom(_runtime, _state, _human_seat):
        raise RuntimeError("AI exploded")

    monkeypatch.setattr(server_app, "_advance_ai_until_human", boom)

    response = client.post(
        f"/api/{BOARD_2P}/move",
        json={
            "game_id": "move-rollback",
            "type": "pawn",
            "target": {"row": 3, "col": 2},
        },
    )

    assert response.status_code == 500
    runtime = server_app.SESSION_GAMES["move-rollback"]
    assert runtime["state"].current_player == original_state.current_player
    assert runtime["state"].positions == original_state.positions
