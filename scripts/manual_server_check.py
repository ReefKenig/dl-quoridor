import requests
import numpy as np


def test_predict():
    url = "http://127.0.0.1:5000/predict"

    dummy_board = np.zeros((5, 5, 10)).tolist()

    payload = {"state": dummy_board}

    print("Sending request to server...")
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        print("Success!")
        print(f"Policy length: {len(result['policy'])}")
        print(f"Value: {result['value']}")
    else:
        print(f"Failed! Error: {response.text}")


if __name__ == "__main__":
    test_predict()
