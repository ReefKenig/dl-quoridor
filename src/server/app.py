import os
import numpy as np
import torch
from flask import Flask, request, jsonify
from src.model.network import QuoridorModel

app = Flask(__name__)

IS_POC = os.environ.get("IS_POC", "True").lower() == "true"
BOARD_SIZE = 5 if IS_POC else 9
ACTION_SIZE = 44 if IS_POC else 140
MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/best/model.pt")

model = QuoridorModel(board_size=BOARD_SIZE, action_space_size=ACTION_SIZE)

if os.path.exists(MODEL_PATH):
    model.load(MODEL_PATH)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        state_array = np.array(data['state'], dtype=np.float32)

        expected_shape = (BOARD_SIZE, BOARD_SIZE, 10)
        if state_array.shape != expected_shape:
            return jsonify({'error': f'Expected {expected_shape}, got {state_array.shape}'}), 400

        policy, value = model.predict(state_array)

        return jsonify({
            'policy': policy.tolist(),
            'value': float(value)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400
