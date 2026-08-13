from src.server.app import app


def test_game_state_is_isolated_per_client_session():
    client_a = app.test_client()
    client_b = app.test_client()

    reset_a = client_a.post(
        "/api/5x5/reset", json={"num_players": 2, "game_id": "session-a"}
    )
    assert reset_a.status_code == 200
    state_a = reset_a.get_json()
    assert len(state_a["players"]) == 2
    assert state_a["game_id"] == "session-a"

    reset_b = client_b.post(
        "/api/9x9/reset", json={"num_players": 4, "game_id": "session-b"}
    )
    assert reset_b.status_code == 200
    state_b = reset_b.get_json()
    assert len(state_b["players"]) == 4
    assert state_b["game_id"] == "session-b"

    current_state_a = client_a.get("/api/5x5/state?game_id=session-a").get_json()
    assert len(current_state_a["players"]) == 2
    assert current_state_a["num_players"] == 2

    current_state_b = client_b.get("/api/9x9/state?game_id=session-b").get_json()
    assert len(current_state_b["players"]) == 4
    assert current_state_b["num_players"] == 4

    assert current_state_a["game_id"] == "session-a"
    assert current_state_b["game_id"] == "session-b"
