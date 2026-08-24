import os
import time
import uuid
from collections import OrderedDict

import numpy as np
import torch
from flask import Flask, jsonify, request, session
from flask_cors import CORS

from src.env.quoridor_env import (
    MOVE_MAP,
    decode_action,
)
from src.mcts.mcts_maxn import MCTSConfig as MCTSConfigMaxN
from src.mcts.mcts_maxn import MCTSMaxN
from src.utils.config import DEFAULT_DIFFICULTY, DIFFICULTY_SETTINGS
from src.utils.model_registry import load_variant

torch.set_num_threads(1)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__name__), "."))
app = Flask(__name__, static_folder=os.path.join(root_dir, "static"))
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "quoridor-dev-secret")
CORS(app)

# ── Load models ──
# The registry pairs each checkpoint with architecture + tensor spec + wall
# count it was trained under; serving any of those from a different source is
# how a model ends up reading planes on a scale it never saw.
SUPPORTED_VARIANTS = [(5, 2), (5, 4), (9, 2), (9, 4)]
GAME_ID_HEADER = "X-Game-Id"


def _build_loaded_variant(board_size, num_players):
    env, model, spec = load_variant(board_size, num_players)

    def evaluate(state):
        tensor = env.state_to_tensor(state)
        return model.predict(tensor)

    # max^n at N=2 is bit-identical to negamax (scripts/run_reduction.py) and is
    # what every variant trained under, so one path serves all of them. The
    # negamax MCTSConfig has no wall_candidates field, so routing N=2 through it
    # silently searched 9x9 unrestricted - half the model's strength.
    def make_mcts(settings):
        cfg = MCTSConfigMaxN(
            num_simulations=settings["num_simulations"],
            c_puct=settings["c_puct"],
            dirichlet_epsilon=settings["dirichlet_epsilon"],
            wall_candidates=spec.wall_candidates,
        )
        return MCTSMaxN(config=cfg, evaluate_fn=evaluate, num_players=num_players)

    return {"env": env, "model": model, "spec": spec, "make_mcts": make_mcts}


AGENTS = {key: _build_loaded_variant(*key) for key in SUPPORTED_VARIANTS}


# ── Per-session game state ──
# Bounded on purpose: one board and one search instance are held per game id,
# so an unevicted map grows without limit on a long-lived process.
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "64"))
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", "3600"))
SESSION_GAMES = OrderedDict()


def _prune_sessions(keep):
    """Drop idle then least-recently-used games, never the one being served."""
    now = time.monotonic()
    for session_id, runtime in list(SESSION_GAMES.items()):
        if session_id != keep and now - runtime["last_seen"] > SESSION_TTL_SECONDS:
            del SESSION_GAMES[session_id]
    while len(SESSION_GAMES) > MAX_SESSIONS and next(iter(SESSION_GAMES)) != keep:
        SESSION_GAMES.popitem(last=False)


def _resolve_session_id(game_id=None):
    if game_id:
        session["game_id"] = game_id
    elif "game_id" not in session:
        session["game_id"] = str(uuid.uuid4())
    return session["game_id"]


def _get_request_game_id():
    payload = request.get_json(silent=True) or {}
    game_id = (
        payload.get("game_id")
        or request.args.get("game_id")
        or request.headers.get(GAME_ID_HEADER)
    )
    return game_id or _resolve_session_id()


def _make_session_runtime(board_size=5, num_players=2, difficulty=DEFAULT_DIFFICULTY):
    if difficulty not in DIFFICULTY_SETTINGS:
        difficulty = DEFAULT_DIFFICULTY
    settings = DIFFICULTY_SETTINGS[difficulty]
    current_num_players = 4 if num_players == 4 else 2
    variant_key = (board_size, current_num_players)
    if variant_key not in AGENTS:
        variant_key = (5, current_num_players)

    agent = AGENTS[variant_key]
    env = agent["env"]
    mcts = agent["make_mcts"](settings)
    return {
        "board_size": board_size,
        "num_players": current_num_players,
        "difficulty": difficulty,
        "env": env,
        "mcts": mcts,
        "temperature": settings["temperature"],
        "state": env.reset(),
    }


def _get_session_runtime(
    board_size=None, num_players=2, difficulty=DEFAULT_DIFFICULTY, game_id=None
):
    session_id = _resolve_session_id(game_id)
    runtime = SESSION_GAMES.get(session_id)
    if runtime is None:
        runtime = _make_session_runtime(
            board_size=board_size or 5,
            num_players=num_players,
            difficulty=difficulty,
        )
        SESSION_GAMES[session_id] = runtime
    runtime["last_seen"] = time.monotonic()
    SESSION_GAMES.move_to_end(session_id)
    _prune_sessions(session_id)
    return runtime


def _sync_session_runtime(
    board_size=None, num_players=2, difficulty=DEFAULT_DIFFICULTY, game_id=None
):
    runtime = _get_session_runtime(
        board_size=board_size,
        num_players=num_players,
        difficulty=difficulty,
        game_id=game_id,
    )
    if board_size is not None:
        requested_num_players = 4 if num_players == 4 else 2
        variant_key = (board_size, requested_num_players)
        if variant_key not in AGENTS:
            variant_key = (5, requested_num_players)

        settings = DIFFICULTY_SETTINGS[difficulty]
        agent = AGENTS[variant_key]
        runtime["board_size"] = board_size
        runtime["num_players"] = requested_num_players
        runtime["difficulty"] = difficulty
        runtime["env"] = agent["env"]
        runtime["mcts"] = agent["make_mcts"](settings)
        runtime["temperature"] = settings["temperature"]
        runtime["state"] = runtime["env"].reset()
    return runtime


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
    data = request.json or {}
    num_players = data.get("num_players", 2)
    difficulty = data.get("difficulty", DEFAULT_DIFFICULTY)
    game_id = (
        data.get("game_id")
        or request.args.get("game_id")
        or request.headers.get(GAME_ID_HEADER)
    )

    try:
        bs = int(str(board_size).lower().split("x")[0])
    except (ValueError, IndexError):
        bs = 5

    runtime = _sync_session_runtime(
        board_size=bs, num_players=num_players, difficulty=difficulty, game_id=game_id
    )
    response = _extract_positions(runtime["env"], runtime["state"])
    response["game_id"] = _resolve_session_id(game_id)
    return jsonify(response)


@app.route("/api/<board_size>/state", methods=["GET"])
def get_state(board_size):
    game_id = (
        request.args.get("game_id")
        or request.headers.get(GAME_ID_HEADER)
        or _get_request_game_id()
    )
    runtime = _get_session_runtime(
        board_size=5, num_players=2, difficulty=DEFAULT_DIFFICULTY, game_id=game_id
    )
    if runtime["state"] is None:
        return jsonify({"error": "No active game"}), 404
    response = _extract_positions(runtime["env"], runtime["state"])
    response["game_id"] = _resolve_session_id(game_id)
    return jsonify(response)


@app.route("/api/<board_size>/move", methods=["POST"])
def process_move(board_size):
    data = request.get_json(silent=True) or {}
    game_id = (
        data.get("game_id")
        or request.args.get("game_id")
        or request.headers.get(GAME_ID_HEADER)
        or _get_request_game_id()
    )
    runtime = _get_session_runtime(
        board_size=5, num_players=2, difficulty=DEFAULT_DIFFICULTY, game_id=game_id
    )
    state = runtime["state"]
    env = runtime["env"]
    mcts = runtime["mcts"]
    temperature = runtime["temperature"]

    try:
        if state is None:
            return jsonify({"error": "Game state not found"})

        data = request.json
        action_type = data.get("type", "pawn")
        target = data.get("target")

        human_action = None

        if action_type == "pawn":
            curr_r, curr_c = state.positions[0]
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

        valid_actions = env.get_valid_actions(state)
        if human_action not in valid_actions:
            print(
                f"[MOVE REJECTED] action={human_action}, cp={state.current_player}, "
                f"pos={state.positions}, valid={valid_actions.tolist()}"
            )
            return jsonify({"error": "Illegal move or wall placement blocked."}), 400

        state, reward, done, _ = env.step(state, human_action)
        runtime["state"] = state
        if done:
            response = {
                "status": "game_over",
                "winner": "human",
                "newState": _extract_positions(env, state),
                "game_id": _resolve_session_id(game_id),
            }
            return jsonify(response)

        ai_steps = []
        while env.get_current_player(state) != 0 and not state.game_over:
            action_probs = mcts.search(env, state, temperature=temperature)
            if temperature < 0.1:
                ai_action = int(np.argmax(action_probs))
            else:
                ai_action = int(np.random.choice(len(action_probs), p=action_probs))
            state, reward, done, info = env.step(state, ai_action)
            runtime["state"] = state
            ai_steps.append(_extract_positions(env, state))
            if done:
                response = {
                    "status": "game_over",
                    "newState": _extract_positions(env, state),
                    "ai_steps": ai_steps,
                    "winner": "ai",
                    "game_id": _resolve_session_id(game_id),
                }
                return jsonify(response)

        runtime["state"] = state
        response = {
            "status": "ongoing",
            "newState": _extract_positions(env, state),
            "ai_steps": ai_steps,
            "winner": None,
            "game_id": _resolve_session_id(game_id),
        }
        return jsonify(response)

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
                valid_moves.append(
                    {
                        "type": "pawn",
                        "row": int(current_pos[0] + dr),
                        "col": int(current_pos[1] + dc),
                    }
                )
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
