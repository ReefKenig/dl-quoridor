import os

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.env.quoridor_env_mp import (
    QuoridorEnvMP,
    MOVE_MAP,
    ACTION_TO_MOVE,
    compute_action_space_size,
)
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
from src.model.network_mp import QuoridorModelMP

torch.set_num_threads(1)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__name__), "."))
app = Flask(__name__, static_folder=os.path.join(root_dir, "static"))
CORS(app)

BOARD_SIZE = 5

# Pre-load models for each player count
models = {}
mcts_agents = {}

for n_players in [2, 3, 4]:
    action_size = compute_action_space_size(BOARD_SIZE)
    in_channels = 3 * n_players + 3
    m = QuoridorModelMP(
        board_size=BOARD_SIZE,
        action_space_size=action_size,
        num_channels=64,
        num_res_blocks=4,
        in_channels=in_channels,
        num_players=n_players,
    )
    ckpt_path = f"checkpoints_mp_n{n_players}/best.pt"
    if os.path.exists(ckpt_path):
        m.load(ckpt_path)
        print(f"Server: Loaded {n_players}-player model from {ckpt_path}")
    else:
        print(
            f"WARNING: No model at {ckpt_path} — AI will use random rollouts for {n_players}P")
    models[n_players] = m

# --- GLOBAL GAME STATE ---
current_game_state = None
current_env = None
current_num_players = 4


def _get_evaluate_fn(num_players):
    model = models[num_players]

    def evaluate_fn(state):
        tensor = current_env.state_to_tensor(state)
        return model.predict(tensor)

    return evaluate_fn


def _get_mcts(num_players, num_simulations=200):
    return MCTSMaxN(
        config=MCTSConfig(num_simulations=num_simulations),
        evaluate_fn=_get_evaluate_fn(num_players),
        num_players=num_players,
    )


# --- WEB ROUTES ---
@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/<path:path>")
def serve_static(path):
    return app.send_static_file(path)


# --- GAME API ROUTES ---
@app.route("/api/reset", methods=["POST"])
def reset_game():
    global current_game_state, current_env, current_num_players
    data = request.json or {}
    current_num_players = int(data.get("num_players", 4))
    current_num_players = max(2, min(4, current_num_players))

    current_env = QuoridorEnvMP(
        board_size=BOARD_SIZE, num_players=current_num_players)
    current_game_state = current_env.reset()

    return jsonify(_extract_state(current_env, current_game_state))


@app.route("/api/move", methods=["POST"])
def process_move():
    global current_game_state
    try:
        if current_game_state is None or current_env is None:
            return jsonify({"error": "Game not started. Call /api/reset first."}), 400

        data = request.json
        action_type = data.get("type", "pawn")
        target = data.get("target")

        human_action = None

        if action_type == "pawn":
            curr_pos = current_game_state.positions[0]
            dr = target["row"] - curr_pos[0]
            dc = target["col"] - curr_pos[1]
            human_action = MOVE_MAP.get((dr, dc))

        elif action_type == "h_wall":
            W = BOARD_SIZE - 1
            human_action = 12 + (target["row"] * W) + target["col"]

        elif action_type == "v_wall":
            W = BOARD_SIZE - 1
            human_action = 12 + (W ** 2) + (target["row"] * W) + target["col"]

        if human_action is None:
            return jsonify({"error": "Invalid move distance."}), 400

        valid_actions = current_env.get_valid_actions(current_game_state)
        if human_action not in valid_actions:
            return jsonify({"error": "Illegal move or wall placement blocked."}), 400

        # Execute human move
        current_game_state, reward, done, info = current_env.step(
            current_game_state, human_action)

        if done:
            return jsonify({
                "status": "game_over",
                "winner": info.get("winner"),
                "newState": _extract_state(current_env, current_game_state),
            })

        # AI players take turns until it's player 0's turn again (or game ends)
        mcts = _get_mcts(current_num_players, num_simulations=100)
        while current_game_state.current_player != 0 and not current_game_state.game_over:
            action_probs = mcts.search(
                current_env, current_game_state, temperature=0.0)
            ai_action = int(np.argmax(action_probs))
            current_game_state, reward, done, info = current_env.step(
                current_game_state, ai_action)

        status = "game_over" if current_game_state.game_over else "ongoing"
        return jsonify({
            "status": status,
            "winner": current_game_state.winner if current_game_state.game_over else None,
            "newState": _extract_state(current_env, current_game_state),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _decode_action(action, board_size):
    W = board_size - 1
    v_offset = 12 + W ** 2
    if action < 12:
        return "pawn", ACTION_TO_MOVE[action]
    elif action < v_offset:
        w = action - 12
        return "h_wall", (w // W, w % W)
    else:
        w = action - v_offset
        return "v_wall", (w // W, w % W)


def _extract_state(env, state):
    """Package the game state for JavaScript."""
    h_walls = [{"row": r, "col": c} for r, c in state.h_walls]
    v_walls = [{"row": r, "col": c} for r, c in state.v_walls]

    players = []
    for i in range(state.num_players):
        players.append({"row": int(state.positions[i][0]),
                        "col": int(state.positions[i][1])})

    valid_moves = []
    if not state.game_over and state.current_player == 0:
        for action in env.get_valid_actions(state):
            a_type, data = _decode_action(int(action), env.board_size)
            if a_type == "pawn":
                dr, dc = data
                valid_moves.append({
                    "type": "pawn",
                    "row": int(state.positions[0][0] + dr),
                    "col": int(state.positions[0][1] + dc),
                })
            else:
                valid_moves.append({
                    "type": a_type,
                    "row": int(data[0]),
                    "col": int(data[1]),
                })

    return {
        "players": players,
        "num_players": state.num_players,
        "current_player": state.current_player,
        "walls_remaining": list(state.walls_remaining),
        "h_walls": h_walls,
        "v_walls": v_walls,
        "valid_moves": valid_moves,
        "game_over": state.game_over,
        "winner": state.winner,
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
