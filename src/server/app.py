import sys
import os
import numpy as np
import torch
from flask import Flask, request, jsonify

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.model.network import QuoridorModel

app = Flask(__name__)

model = QuoridorModel(board_size=5, action_space_size=44)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        state_array = np.array(data['state'], dtype=np.float32)
        
        policy, value = model.predict(state_array)
        
        return jsonify({
            'policy': policy.tolist(),
            'value': float(value)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)