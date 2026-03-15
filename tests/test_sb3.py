import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env.quoridor_env import QuoridorEnv

class SB3Wrapper(gym.Env):
    def __init__(self, board_size=5):
        super().__init__()
        self.env = QuoridorEnv(is_poc=True)
        self.board_size = board_size
        
        self.action_space = spaces.Discrete(self.env.action_space_size)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(board_size, board_size, 10), 
            dtype=np.float32
        )
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.env.reset()
        obs = self.env.state_to_tensor(self.state)
        return obs, {}

    def step(self, action):
        valid = self.env.get_valid_actions(self.state)
        if action not in valid:
            action = np.random.choice(valid)
            
        next_state, reward, done, info = self.env.step(self.state, action)
        self.state = next_state
        obs = self.env.state_to_tensor(self.state)
        return obs, float(reward), done, False, info

def run_sb3_check():
    print("Starting SB3 Compatibility Check")
    
    try:
        env = SB3Wrapper(board_size=5)
        
        model = PPO("MlpPolicy", env, verbose=1)
        print("SB3 Model initialized successfully")
        
        print("Running 100 timesteps")
        model.learn(total_timesteps=100)
        
        print("SB3 check finished successfully")
        
    except Exception as e:
        print(f"SB3 Check failed: {e}")

if __name__ == "__main__":
    run_sb3_check()