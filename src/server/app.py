import os

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.env.quoridor_env import (
    MOVE_MAP,
    QuoridorEnv,
    compute_action_space_size,
    decode_action,
)
from src.mcts.mcts import MCTS, MCTSConfig
from src.model.network import QuoridorModel

torch.set_num_threads(1)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__name__), "."))
app = Flask(__name__, static_folder=os.path.join(root_dir, "static"))
CORS(app)

IS_POC = os.environ.get("IS_POC", "True").lower() == "true"
BOARD_SIZE = 5 if IS_POC else 9
ACTION_SIZE = compute_action_space_size(BOARD_SIZE)
MODEL_PATH = os.environ.get("MODEL_PATH", "runs/legacy_2p/best/model.pt")

env = QuoridorEnv(board_size=BOARD_SIZE)
model = QuoridorModel(
    board_size=BOARD_SIZE, action_space_size=ACTION_SIZE, in_channels=11
)

if os.path.exists(MODEL_PATH):
    model.load(MODEL_PATH)
    print(f"Server: Successfully loaded trained AI from {MODEL_PATH}")
else:
    print(f"WARNING: Could not find model at {MODEL_PATH}")


def evaluate_fn(state):
    tensor = env.state_to_tensor(state)
    return model.predict(tensor)


mcts_config = MCTSConfig(num_simulations=200)
mcts_agent = MCTS(config=mcts_config, evaluate_fn=evaluate_fn)

# --- GLOBAL GAME STATE ---
current_game_state = None


# --- WEB ROUTES ---
@app.route("/")
def inedx():
    return app.send_static_file("index.html")


@app.route("/<path:path>")
def serve_static(path):
    return app.send_static_file(path)


# --- GAME API ROUTES ---
@app.route("/api/5x5/reset", methods=["POST"])
def reset_game():
    global current_game_state
    current_game_state = env.reset()

    return jsonify(_extract_positions(env, current_game_state))


@app.route("/api/5x5/move", methods=["POST"])
def process_move():
    global current_game_state
    try:
        if current_game_state is None:
            return jsonify({"error": "Game state not found"})

        data = request.json
        action_type = data.get("type", "pawn")
        target = data.get("target")

        human_action = None

        if action_type == "pawn":
            curr_r, curr_c = current_game_state.p0_pos
            dr = target["row"] - curr_r
            dc = target["col"] - curr_c
            human_action = MOVE_MAP.get((dr, dc))

        elif action_type == "h_wall":
            W = env.board_size - 1
            human_action = 12 + (target["row"] * W) + target["col"]

        elif action_type == "v_wall":
            W = env.board_size - 1
            human_action = 12 + (W**2) + (target["row"] * W) + target["col"]

        if human_action is None:
            return jsonify({"error": "Invalid move distance."}), 400

        valid_actions = env.get_valid_actions(current_game_state)
        if human_action not in valid_actions:
            return jsonify({"error": "Illegal move or wall placement blocked."}), 400

        current_game_state, reward, done, _ = env.step(current_game_state, human_action)
        if done:
            return jsonify(
                {
                    "status": "game_over",
                    "winner": "human",
                    "newState": _extract_positions(env, current_game_state),
                }
            )

        action_probs = mcts_agent.search(env, current_game_state)
        ai_action = int(np.argmax(action_probs))
        current_game_state, reward, done, _ = env.step(current_game_state, ai_action)

        return jsonify(
            {
                "status": "game_over" if done else "ongoing",
                "newState": _extract_positions(env, current_game_state),
                "winner": "ai" if done else None,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _extract_positions(env, state):
    """Helper function to cleanly package the coordinates for JavaScript"""
    h_walls = [{"row": r, "col": c} for r, c in state.p0_h_walls | state.p1_h_walls]
    v_walls = [{"row": r, "col": c} for r, c in state.p0_v_walls | state.p1_v_walls]

    valid_moves = []
    if not state.game_over:
        for action in env.get_valid_actions(state):
            a_type, data = decode_action(int(action), env.board_size)
            if a_type == "pawn":
                dr, dc = data
                valid_moves.append(
                    {
                        "type": "pawn",
                        "row": int(state.p0_pos[0] + dr),
                        "col": int(state.p0_pos[1] + dc),
                    }
                )
            else:
                valid_moves.append(
                    {"type": a_type, "row": int(data[0]), "col": int(data[1])}
                )

    return {
        "player1": {"row": int(state.p0_pos[0]), "col": int(state.p0_pos[1])},
        "player2": {"row": int(state.p1_pos[0]), "col": int(state.p1_pos[1])},
        "h_walls": h_walls,
        "v_walls": v_walls,
        "valid_moves": valid_moves,
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
