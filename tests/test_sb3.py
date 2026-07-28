import numpy as np
import gymnasium as gym
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.env.quoridor_env import QuoridorEnv
from src.env.tensor_spec import OBS_CHANNELS


class BoardCNN(BaseFeaturesExtractor):
    """Small 3x3-conv extractor sized for a Quoridor board.

    SB3's default CnnPolicy extractor (NatureCNN) is built for Atari frames: its
    first layer is an 8x8 kernel at stride 4, which cannot consume a 5x5 board at
    all, and it asserts the observation is a uint8 image. Our observation is an
    11-channel float32 plane stack in [0, 1], so it needs padded 3x3 convs and no
    downsampling.
    """

    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_in = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_in, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))


class SB3Wrapper(gym.Env):
    def __init__(self, board_size=5):
        super().__init__()
        self.env = QuoridorEnv(board_size=board_size)
        self.board_size = self.env.board_size

        self.action_space = spaces.Discrete(self.env.action_space_size)
        self.observation_space = spaces.Box(
            low=0, high=1,
            # CHW for CnnPolicy. Width comes from the spec, not a literal, so it
            # tracks state_to_tensor instead of silently disagreeing with it.
            shape=(OBS_CHANNELS, self.board_size, self.board_size),
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

    model = PPO(
        "CnnPolicy", env, verbose=1,
        policy_kwargs=dict(
            features_extractor_class=BoardCNN,
            # Observations are already normalized to [0, 1]; SB3's default
            # image preprocessing would divide them by 255 again.
            normalize_images=False,
        ),
        # Keep the rollout buffer near the step budget so a 100-step smoke test
        # doesn't collect SB3's default 2048 steps first.
        n_steps=64, batch_size=32,
    )
    model.learn(total_timesteps=100)


if __name__ == "__main__":
    test_sb3_ppo_cnn()
