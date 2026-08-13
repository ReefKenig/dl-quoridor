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
