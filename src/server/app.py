import os
import numpy as np
from flask import Flask, request, jsonify
from src.model.network import QuoridorModel
from src.env.quoridor_env import compute_action_space_size

app = Flask(__name__)

IS_POC = os.environ.get("IS_POC", "True").lower() == "true"
BOARD_SIZE = 5 if IS_POC else 9
ACTION_SIZE = compute_action_space_size(BOARD_SIZE)
MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/best/model.pt")

model = QuoridorModel(board_size=BOARD_SIZE, action_space_size=ACTION_SIZE)

if os.path.exists(MODEL_PATH):
    model.load(MODEL_PATH)
    print(f"✅ SERVER: Successfully loaded trained AI from {MODEL_PATH}")
else:
    print(f"❌ WARNING: Could not find model at {MODEL_PATH}!")
    print("❌ SERVER IS RUNNING A RANDOM, UNTRAINED AI.")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        state_array = np.array(data["state"], dtype=np.float32)

        expected_shape = (BOARD_SIZE, BOARD_SIZE, 10)
        if state_array.shape != expected_shape:
            return (
                jsonify(
                    {"error": f"Expected {expected_shape}, got {state_array.shape}"}
                ),
                400,
            )

        policy, value = model.predict(state_array)

        return jsonify({"policy": policy.tolist(), "value": float(value)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
