import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env.quoridor_env import QuoridorEnv 
from src.model.network import QuoridorModel  

def run_integration_test():
    print("--- Starting Integration Check ---")
    
    env = QuoridorEnv(board_size=5)
    
    state = env.reset()
    
    model = QuoridorModel(board_size=5, action_space_size=44)
    
    try:
        observation = env.state_to_tensor(state)
        
        print(f"Observation shape from Env: {observation.shape}")
        
        expected_shape = (5, 5, 10)
        if observation.shape != expected_shape:
            print(f"❌ Error: Expected shape {expected_shape}, but got {observation.shape}")
            return

        policy, value = model.predict(observation)
        
        print("✅ Model prediction successful!")
        print(f"Policy output size: {len(policy)}")
        print(f"Value output: {value:.4f}")
        
        if len(policy) == 44:
            print("Action Space size matches (44)")
        else:
            print(f"Action Space mismatch: Expected 44, got {len(policy)}")
            
    except Exception as e:
        print(f"Integration failed with error: {e}")

if __name__ == "__main__":
    run_integration_test()