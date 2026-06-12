import os

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.env.quoridor_env import MOVE_MAP, QuoridorEnv, compute_action_space_size
from src.mcts.mcts import MCTS, MCTSConfig
from src.model.network import QuoridorModel

torch.set_num_threads(1)

# Setup Flask to serve your HTML/CSS frontend directly
app = Flask(__name__, static_folder="static")
CORS(app)  # Prevents browser cross-origin blocking

IS_POC = os.environ.get("IS_POC", "True").lower() == "true"
BOARD_SIZE = 5 if IS_POC else 9
ACTION_SIZE = compute_action_space_size(BOARD_SIZE)
MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/best/model.pt")

# Initialize the Game Environment and Model in memory
env = QuoridorEnv(board_size=BOARD_SIZE)
model = QuoridorModel(
    board_size=BOARD_SIZE, action_space_size=ACTION_SIZE, in_channels=11
)

if os.path.exists(MODEL_PATH):
    model.load(MODEL_PATH)
    print(f"✅ SERVER: Successfully loaded trained AI from {MODEL_PATH}")
else:
    print(f"❌ WARNING: Could not find model at {MODEL_PATH}!")


def evaluate_fn(state):
    """Wraps the PyTorch model for the MCTS."""
    tensor = env.state_to_tensor(state)
    return model.predict(tensor)


# Initialize MCTS
mcts_config = MCTSConfig(num_simulations=200)
mcts_agent = MCTS(config=mcts_config, evaluate_fn=evaluate_fn)

# --- WEB ROUTES ---


# Route to serve HTML page
@app.route("/")
def index():
    return app.send_static_file("index.html")


# Route to serve CSS file
@app.route("/<path:path>")
def serve_static(path):
    return app.send_static_file(path)


# --- GAME API ROUTES


@app.route("/api/5x5/reset", methods=["POST"])
def reset_game():
    """Resets the board and returns the starting positions."""
    state = env.reset()

    # Extract the actual starting coordinates from your env state
    # TODO: Update variables based on how env tracks positions
    return jsonify(
        {
            "player1": {"row": 4, "col": 2},  # Human
            "player2": {"row": 0, "col": 2},  # AI
            "walls": [],
        }
    )


@app.route("/api/5x5/move", methods=["POST"])
def process_move():
    """Handles the human move, and immediately plays the AI move."""
    try:
        data = request.json
        target = data.get("target")

        # Translate human click(0-43)
        curr_r, curr_c = env.state[1]
        dr = target["row"] - curr_r
        dc = target["col"] - curr_c

        human_action = MOVE_MAP.get((dr, dc))
        if human_action is None:
            return jsonify({"error": "Invalid move distance."}), 400

        # Apply human move
        next_state, reward, done, _ = env.step(human_action)
        if done:
            return jsonify({"status": "game_over", "winner": "human"})

        # Apply AI move (MCTS)
        # Search the tree, get the action probabilities, and pick the best one
        action_probs = mcts_agent.search(next_state)
        ai_action = int(np.argmax(action_probs))
        next_state, reward, done, _ = env.step(ai_action)

        # Extract new positions
        new_p1_pos = {"row": int(next_state[1][0]), "col": int(next_state[1][1])}
        new_p2_pos = {"row": int(next_state[2][0]), "col": int(next_state[2][1])}

        return jsonify(
            {
                "status": "game_over" if done else "ongoing",
                "newState": {
                    "player1": new_p1_pos,
                    "player2": new_p2_pos,
                    "walls": [],  # Add wall coordinates ehre
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
