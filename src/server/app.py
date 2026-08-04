import json
import os

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.env.quoridor_env import (
    MOVE_MAP,
    compute_action_space_size,
    decode_action,
)
from src.env.quoridor_env_mp import QuoridorEnvMP
from src.mcts.mcts import MCTS, MCTSConfig
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig as MCTSConfigMaxN
from src.model.network_mp import QuoridorModelMP
from src.utils.config import DIFFICULTY_SETTINGS, DEFAULT_DIFFICULTY

torch.set_num_threads(1)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__name__), "."))
app = Flask(__name__, static_folder=os.path.join(root_dir, "static"))
CORS(app)

# ── Load model registry ──
MODELS_PATH = os.path.join(root_dir, "runs", "MODELS.json")
with open(MODELS_PATH) as f:
    MODEL_REGISTRY = json.load(f)["models"]

# Which registry entry + env settings back each (board_size, num_players) combo.
# 5x5 is the POC (64ch/4blocks, scaled-down walls); 9x9 is the full-size board
# (128ch/8blocks, official wall counts).
BOARD_VARIANTS = {
    (5, 2): {"registry_key": "2p_vector", "max_walls": 3, "max_turns": 300,
             "num_channels": 64, "num_res_blocks": 4},
    (5, 4): {"registry_key": "4p_ship", "max_walls": 4, "max_turns": 300,
             "num_channels": 64, "num_res_blocks": 4},
    (9, 2): {"registry_key": "2p_9x9", "max_walls": 10, "max_turns": 200,
             "num_channels": 128, "num_res_blocks": 8},
    (9, 4): {"registry_key": "4p_9x9", "max_walls": 5, "max_turns": 200,
             "num_channels": 128, "num_res_blocks": 8},
}


def _build_agent(board_size, num_players, spec):
    """Build one env + actor factory (MCTS over the trained net) for a combo.

    An actor is a callable (env, state) -> action.
    """
    cfg = MODEL_REGISTRY[spec["registry_key"]]
    action_size = compute_action_space_size(board_size)
    # spec_version from the registry: each checkpoint reads its distance planes
    # on the tensor scale it was trained under (see MODELS.json _tensor_spec_note).
    env = QuoridorEnvMP(board_size=board_size, num_players=num_players,
                        max_turns=spec["max_turns"],
                        max_walls_per_player=spec["max_walls"],
                        spec_version=cfg["tensor_spec"])
    model = QuoridorModelMP(board_size=board_size, action_space_size=action_size,
                            in_channels=cfg["in_channels"],
                            num_channels=spec["num_channels"],
                            num_res_blocks=spec["num_res_blocks"],
                            num_players=num_players)
    tag = f"[{board_size}x{board_size} N={num_players}]"
    if os.path.exists(cfg["path"]):
        model.load(cfg["path"])
        print(f"{tag} Loaded: {cfg['path']}")
    else:
        print(f"{tag} WARNING: {cfg['path']} not found")

    if num_players == 2:
        # Negamax at N=2: score the current player's own value component.
        def evaluate(state):
            policy, value_vec = model.predict(env.state_to_tensor(state))
            return policy, float(value_vec[state.current_player])

        def make_mcts(settings):
            return MCTS(config=MCTSConfig(
                num_simulations=settings["num_simulations"],
                c_puct=settings["c_puct"],
                dirichlet_epsilon=settings["dirichlet_epsilon"]),
                evaluate_fn=evaluate)
    else:
        def evaluate(state):
            return model.predict(env.state_to_tensor(state))

        def make_mcts(settings):
            # Narrow search to the walls that can change a distance to goal, so
            # a high sim count actually resolves the position instead of
            # spreading ~5 visits over 131 actions.
            return MCTSMaxN(config=MCTSConfigMaxN(
                num_simulations=settings["num_simulations"],
                c_puct=settings["c_puct"],
                dirichlet_epsilon=settings["dirichlet_epsilon"],
                wall_candidates=settings.get("wall_candidates", 16)),
                evaluate_fn=evaluate, num_players=num_players)

    def make_actor(settings):
        mcts = make_mcts(settings)
        temp = settings["temperature"]

        def act(e, s):
            probs = mcts.search(e, s, temperature=temp)
            if temp < 0.1:
                return int(np.argmax(probs))
            return int(np.random.choice(len(probs), p=probs))

        return act

    return {"env": env, "make_actor": make_actor}


AGENTS = {key: _build_agent(bs, np_, spec)
          for key in BOARD_VARIANTS
          for (bs, np_), spec in [(key, BOARD_VARIANTS[key])]}


# ── Game state ──
current_game_state = None
current_num_players = 2
current_env = AGENTS[(5, 2)]["env"]
current_actor = AGENTS[(5, 2)]["make_actor"](
    DIFFICULTY_SETTINGS[DEFAULT_DIFFICULTY])


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
    global current_game_state, current_num_players, current_env, current_actor
    data = request.json or {}
    num_players = data.get("num_players", 2)
    difficulty = data.get("difficulty", DEFAULT_DIFFICULTY)

    if difficulty not in DIFFICULTY_SETTINGS:
        difficulty = DEFAULT_DIFFICULTY
    settings = DIFFICULTY_SETTINGS[difficulty]

    # board_size arrives as e.g. "5x5" or "9x9"; take the leading dimension.
    try:
        bs = int(str(board_size).lower().split("x")[0])
    except (ValueError, IndexError):
        bs = 5
    num_players = 4 if num_players == 4 else 2
    if (bs, num_players) not in AGENTS:
        bs = 5

    agent = AGENTS[(bs, num_players)]
    current_num_players = num_players
    current_env = agent["env"]
    current_actor = agent["make_actor"](settings)

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
            ai_action = current_actor(current_env, current_game_state)
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
