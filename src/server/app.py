import json
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
from src.env.quoridor_env_mp import QuoridorEnvMP
from src.mcts.mcts import MCTS, MCTSConfig
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig as MCTSConfigMaxN
from src.model.network import QuoridorModel
from src.model.network_mp import QuoridorModelMP

torch.set_num_threads(1)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__name__), "."))
app = Flask(__name__, static_folder=os.path.join(root_dir, "static"))
CORS(app)

# ── Load model registry ──
MODELS_PATH = os.path.join(root_dir, "runs", "MODELS.json")
with open(MODELS_PATH) as f:
    MODEL_REGISTRY = json.load(f)["models"]

BOARD_SIZE = 5
ACTION_SIZE = compute_action_space_size(BOARD_SIZE)
SIMS = 200

# ── Load 2-player model ──
# TODO: remove 2p_legacy from registry once server fully migrated to vector head
cfg_2p = MODEL_REGISTRY["2p_vector"]
env_2p = QuoridorEnv(board_size=BOARD_SIZE, max_walls_per_player=3)
model_2p = QuoridorModelMP(board_size=BOARD_SIZE, action_space_size=ACTION_SIZE,
                           in_channels=cfg_2p["in_channels"],
                           num_channels=64, num_res_blocks=4,
                           num_players=cfg_2p["num_players"])
if os.path.exists(cfg_2p["path"]):
    model_2p.load(cfg_2p["path"])
    print(f"[2P] Loaded: {cfg_2p['path']}")
else:
    print(f"[2P] WARNING: {cfg_2p['path']} not found")


def eval_2p(state):
    tensor = env_2p.state_to_tensor(state)
    policy, value_vec = model_2p.predict(tensor[:, :, :cfg_2p["in_channels"]])
    return policy, float(value_vec[state.current_player])


mcts_2p = MCTS(config=MCTSConfig(num_simulations=SIMS), evaluate_fn=eval_2p)

# ── Load 4-player model ──
cfg_4p = MODEL_REGISTRY["4p_ship"]
env_4p = QuoridorEnvMP(board_size=BOARD_SIZE, num_players=4,
                       max_turns=300, max_walls_per_player=4)
model_4p = QuoridorModelMP(board_size=BOARD_SIZE, action_space_size=ACTION_SIZE,
                           in_channels=cfg_4p["in_channels"],
                           num_channels=64, num_res_blocks=4,
                           num_players=cfg_4p["num_players"])
if os.path.exists(cfg_4p["path"]):
    model_4p.load(cfg_4p["path"])
    print(f"[4P] Loaded: {cfg_4p['path']}")
else:
    print(f"[4P] WARNING: {cfg_4p['path']} not found")


def eval_4p(state):
    tensor = env_4p.state_to_tensor(state)
    return model_4p.predict(tensor)


mcts_4p = MCTSMaxN(config=MCTSConfigMaxN(num_simulations=SIMS),
                   evaluate_fn=eval_4p, num_players=4)

# ── Game state ──
current_game_state = None
current_num_players = 2
current_env = env_2p
current_mcts = mcts_2p


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
    global current_game_state, current_num_players, current_env, current_mcts
    data = request.json or {}
    num_players = data.get("num_players", 2)

    if num_players == 4:
        current_num_players = 4
        current_env = env_4p
        current_mcts = mcts_4p
    else:
        current_num_players = 2
        current_env = env_2p
        current_mcts = mcts_2p

    current_game_state = current_env.reset()
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
            if current_num_players == 2:
                curr_r, curr_c = current_game_state.p0_pos
            else:
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

        # AI moves for all non-human players
        while current_env.get_current_player(current_game_state) != 0 and not current_game_state.game_over:
            action_probs = current_mcts.search(
                current_env, current_game_state, temperature=0.1)
            ai_action = int(np.argmax(action_probs))
            current_game_state, reward, done, info = current_env.step(
                current_game_state, ai_action)
            if done:
                return jsonify(
                    {
                        "status": "game_over",
                        "newState": _extract_positions(current_env, current_game_state),
                        "winner": "ai",
                    }
                )

        return jsonify(
            {
                "status": "ongoing",
                "newState": _extract_positions(current_env, current_game_state),
                "winner": None,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _extract_positions(env, state):
    """Package game state for JavaScript — supports both 2p and MP states."""
    bs = env.board_size

    # Handle both state formats
    if hasattr(state, "p0_pos"):
        # 2-player QuoridorState
        positions = [state.p0_pos, state.p1_pos]
        h_walls = list(state.p0_h_walls | state.p1_h_walls)
        v_walls = list(state.p0_v_walls | state.p1_v_walls)
    else:
        # N-player QuoridorStateMP
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
        "num_players": len(positions),
        "current_player": env.get_current_player(state),
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
