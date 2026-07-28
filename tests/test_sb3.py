import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from src.env.quoridor_env import QuoridorEnv


class SB3Wrapper(gym.Env):
    def __init__(self, board_size=5):
        super().__init__()
        self.env = QuoridorEnv(board_size=board_size)
        self.board_size = self.env.board_size

        self.action_space = spaces.Discrete(self.env.action_space_size)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(10, self.board_size, self.board_size),  # CHW for CnnPolicy
            dtype=np.float32
        )
        self.state = None

    def _get_obs(self):
        obs = self.env.state_to_tensor(self.state)
        return obs.transpose(2, 0, 1)  # HWC → CHW

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.env.reset()
        return self._get_obs(), {}

    def step(self, action):
        valid = self.env.get_valid_actions(self.state)
        if action not in valid:
            action = np.random.choice(valid)

        next_state, reward, done, info = self.env.step(self.state, action)
        self.state = next_state
        return self._get_obs(), float(reward), done, False, info


def test_sb3_ppo_cnn():
    """Verify SB3 PPO + CnnPolicy can init and run on the QuoridorEnv wrapper."""
    env = SB3Wrapper(board_size=5)

    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=100)


if __name__ == "__main__":
    test_sb3_ppo_cnn()
