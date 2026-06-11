import os

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.env.quoridor_env import QuoridorEnv, compute_action_space_size
from src.model.network import QuoridorModel

# TODO: Import your MCTS and Evaluator here
# from src.mcts.mcts import MCTS, MCTSConfig

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

# TODO: Initialize MCTS agent here using the loaded model

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

        # Translate human click to an env action integer (0-43)
        # TODO: Write a helper function that takes target["row"] & target["col"] and figures out which action ID that corresponds to.
        human_action = 0

        # Apply human move
        next_state, reward, done, _ = env.step(human_action)

        if done:
            return jsonify({"status": "game_over", "winner": "human"})

        # Apply AI move (MCTS)
        # ai_action = mcts_agent.get_action(next_state)
        ai_action = 1  # Placeholder
        next_state, reward, done, _ = env.step(ai_action)

        # Extract the new board positions from next_state
        # TODO: Read the new coordinates from env to send to JavaScript
        new_p1_pos = {"row": 3, "col": 2}  # Placholder
        new_p2_pos = {"row": 1, "col": 2}  # Placholder

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
