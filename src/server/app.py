import os

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.env.quoridor_env import (
    MOVE_MAP,
    decode_action,
)
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig as MCTSConfigMaxN
from src.utils.config import DIFFICULTY_SETTINGS, DEFAULT_DIFFICULTY
from src.utils.model_registry import load_variant

torch.set_num_threads(1)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__name__), "."))
app = Flask(__name__, static_folder=os.path.join(root_dir, "static"))
CORS(app)

# ── Load models ──
# The registry pairs each checkpoint with architecture + tensor spec + wall
# count it was trained under; serving any of those from a different source is
# how a model ends up reading planes on a scale it never saw.
SUPPORTED_VARIANTS = [(5, 2), (5, 4), (9, 2), (9, 4)]


def _build_loaded_variant(board_size, num_players):
    env, model, spec = load_variant(board_size, num_players)

    def evaluate(state):
        tensor = env.state_to_tensor(state)
        return model.predict(tensor)

    # max^n at N=2 is bit-identical to negamax (scripts/run_reduction.py) and is
    # what every variant trained under, so one path serves all of them. The
    # negamax MCTSConfig has no wall_candidates field, so routing N=2 through it
    # silently searched 9x9 unrestricted — half the model's strength.
    def make_mcts(settings):
        cfg = MCTSConfigMaxN(num_simulations=settings["num_simulations"],
                             c_puct=settings["c_puct"],
                             dirichlet_epsilon=settings["dirichlet_epsilon"],
                             wall_candidates=spec.wall_candidates)
        return MCTSMaxN(config=cfg, evaluate_fn=evaluate, num_players=num_players)

    return {"env": env, "model": model, "spec": spec, "make_mcts": make_mcts}


AGENTS = {
    key: _build_loaded_variant(*key)
    for key in SUPPORTED_VARIANTS
}


# ── Game state ──
current_game_state = None
current_num_players = 2
current_env = AGENTS[(5, 2)]["env"]
current_mcts = AGENTS[(5, 2)]["make_mcts"](
    DIFFICULTY_SETTINGS[DEFAULT_DIFFICULTY])
current_temperature = DIFFICULTY_SETTINGS[DEFAULT_DIFFICULTY]["temperature"]


# --- WEB ROUTES ---
@app.route("/")
def inedx():
    return app.send_static_file("index.html")


@app.route("/<path:path>")
def serve_static(path):
    return app.send_static_file(path)


# --- GAME API ROUTES ---
@app.route("/api/<board_size>/reset", methods=["POST"])
def reset_game(board_size):
    global current_game_state, current_num_players, current_env, current_mcts, current_temperature
    data = request.json or {}
    num_players = data.get("num_players", 2)
    difficulty = data.get("difficulty", DEFAULT_DIFFICULTY)

    if difficulty not in DIFFICULTY_SETTINGS:
        difficulty = DEFAULT_DIFFICULTY
    settings = DIFFICULTY_SETTINGS[difficulty]
    current_temperature = settings["temperature"]

    try:
        bs = int(str(board_size).lower().split("x")[0])
    except (ValueError, IndexError):
        bs = 5

    current_num_players = 4 if num_players == 4 else 2
    variant_key = (bs, current_num_players)
    if variant_key not in AGENTS:
        variant_key = (5, current_num_players)

    agent = AGENTS[variant_key]
    current_env = agent["env"]
    current_mcts = agent["make_mcts"](settings)

    current_game_state = current_env.reset()
    return jsonify(_extract_positions(current_env, current_game_state))


@app.route("/api/<board_size>/state", methods=["GET"])
def get_state(board_size):
    if current_game_state is None:
        return jsonify({"error": "No active game"}), 404
    return jsonify(_extract_positions(current_env, current_game_state))


@app.route("/api/<board_size>/move", methods=["POST"])
def process_move(board_size):
    global current_game_state
    try:
        if current_game_state is None:
            return jsonify({"error": "Game state not found"})

        data = request.json
        action_type = data.get("type", "pawn")
        target = data.get("target")

        human_action = None

        if action_type == "pawn":
            curr_r, curr_c = current_game_state.positions[0]
            dr = target["row"] - curr_r
            dc = target["col"] - curr_c
            human_action = MOVE_MAP.get((dr, dc))

        elif action_type == "h_wall":
            W = current_env.board_size - 1
            human_action = 12 + (target["row"] * W) + target["col"]

        elif action_type == "v_wall":
            W = current_env.board_size - 1
            human_action = 12 + (W**2) + (target["row"] * W) + target["col"]

        if human_action is None:
            return jsonify({"error": "Invalid move distance."}), 400

        valid_actions = current_env.get_valid_actions(current_game_state)
        if human_action not in valid_actions:
            print(f"[MOVE REJECTED] action={human_action}, cp={current_game_state.current_player}, "
                  f"pos={current_game_state.positions}, valid={valid_actions.tolist()}")
            return jsonify({"error": "Illegal move or wall placement blocked."}), 400

        current_game_state, reward, done, _ = current_env.step(
            current_game_state, human_action)
        if done:
            return jsonify(
                {
                    "status": "game_over",
                    "winner": "human",
                    "newState": _extract_positions(current_env, current_game_state),
                }
            )

        ai_steps = []
        while current_env.get_current_player(current_game_state) != 0 and not current_game_state.game_over:
            action_probs = current_mcts.search(
                current_env, current_game_state, temperature=current_temperature)
            if current_temperature < 0.1:
                ai_action = int(np.argmax(action_probs))
            else:
                ai_action = int(np.random.choice(
                    len(action_probs), p=action_probs))
            current_game_state, reward, done, info = current_env.step(
                current_game_state, ai_action)
            ai_steps.append(_extract_positions(
                current_env, current_game_state))
            if done:
                return jsonify(
                    {
                        "status": "game_over",
                        "newState": _extract_positions(current_env, current_game_state),
                        "ai_steps": ai_steps,
                        "winner": "ai",
                    }
                )

        return jsonify(
            {
                "status": "ongoing",
                "newState": _extract_positions(current_env, current_game_state),
                "ai_steps": ai_steps,
                "winner": None,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _extract_positions(env, state):
    """Package game state for JavaScript."""
    bs = env.board_size
    positions = state.positions
    h_walls = list(state.h_walls)
    v_walls = list(state.v_walls)

    players = [{"row": int(p[0]), "col": int(p[1])} for p in positions]

    valid_moves = []
    if not state.game_over:
        current_pos = positions[env.get_current_player(state)]
        for action in env.get_valid_actions(state):
            a_type, data = decode_action(int(action), bs)
            if a_type == "pawn":
                dr, dc = data
                valid_moves.append({
                    "type": "pawn",
                    "row": int(current_pos[0] + dr),
                    "col": int(current_pos[1] + dc),
                })
            else:
                valid_moves.append(
                    {"type": a_type, "row": int(data[0]), "col": int(data[1])}
                )

    return {
        "players": players,
        "player1": players[0],
        "player2": players[1],
        "h_walls": [{"row": r, "col": c} for r, c in h_walls],
        "v_walls": [{"row": r, "col": c} for r, c in v_walls],
        "valid_moves": valid_moves,
        "walls_remaining": list(state.walls_remaining),
        "num_players": len(positions),
        "current_player": env.get_current_player(state),
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
