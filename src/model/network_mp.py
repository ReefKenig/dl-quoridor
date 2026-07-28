"""
Vector-value dual-headed network (maxn correct path).
Value head emits a length-`num_players` vector in [-1,1] (tanh per component).
NO sign-flipping anywhere. Reduces to the scalar 2p net only in interpretation,
not in shape: at num_players=2 the head outputs 2 numbers, not 1.
"""
import logging
from contextlib import nullcontext
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)

# Blackwell / Ampere+: TF32 matmul+conv is a large speedup over strict fp32 with
# negligible accuracy loss, and is batch-size agnostic. Enabled once at import,
# guarded so CPU-only hosts / non-CUDA builds are unaffected.
# NOTE: torch.backends.cudnn.benchmark is intentionally left OFF. Leaf-parallel
# self-play feeds VARIABLE batch sizes (wave sizes shrink on collisions), which
# would make benchmark re-autotune kernels on nearly every forward — a net loss,
# unlike the batch-size-agnostic TF32 flags.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels,
                               3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class QuoridorNetworkMP(nn.Module):
    def __init__(self, board_size=5, in_channels=11, num_channels=64,
                 num_res_blocks=4, action_space_size=44, num_players=2):
        super().__init__()
        self.board_size = board_size
        self.action_space_size = action_space_size
        self.num_players = num_players

        self.conv_input = nn.Conv2d(
            in_channels, num_channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_res_blocks)])

        self.policy_conv = nn.Conv2d(num_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(
            2 * board_size * board_size, action_space_size)

        self.value_conv = nn.Conv2d(num_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, num_channels)
        # CHANGED: scalar -> length num_players vector
        self.value_fc2 = nn.Linear(num_channels, num_players)

    def forward(self, x):
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.reshape(p.size(0), -1)
        policy = F.log_softmax(self.policy_fc(p), dim=1)
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))   # (B, num_players)
        return policy, value


class QuoridorModelMP:
    def __init__(self, board_size=5, action_space_size=44, num_channels=64,
                 num_res_blocks=4, lr=1e-3, weight_decay=1e-4, device="auto",
                 in_channels=None, num_players=2, value_loss_weight=1.0):
        # Default in_channels depends on number of players: channels = 3*N + 3
        if in_channels is None:
            in_channels = 3 * num_players + 3
        self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                                   else (device if device != "auto" else "cpu"))
        self.num_players = num_players
        self.value_loss_weight = value_loss_weight
        self.network = QuoridorNetworkMP(
            board_size=board_size, in_channels=in_channels, num_channels=num_channels,
            num_res_blocks=num_res_blocks, action_space_size=action_space_size,
            num_players=num_players,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=weight_decay)
        # Retained so a schedule can be expressed relative to the configured
        # starting rate without the caller having to thread it through.
        self.base_lr = lr
        self.board_size = board_size
        self.action_space_size = action_space_size

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _autocast(self):
        """bf16 autocast on CUDA, no-op on CPU. Only the network forward is wrapped;
        outputs are cast back to fp32 before leaving the model. log_softmax runs in
        fp32 under autocast automatically, so policy precision is preserved."""
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def predict(self, state_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (policy (A,), value (num_players,))."""
        self.network.eval()
        with torch.no_grad():
            x = torch.from_numpy(state_tensor).float().permute(
                2, 0, 1).unsqueeze(0).to(self.device)
            with self._autocast():
                log_policy, value = self.network(x)
            policy = torch.exp(log_policy.float()).squeeze(0).cpu().numpy()
            value_vec = value.float().squeeze(0).cpu().numpy()   # (num_players,)
        return policy, value_vec

    def train_step(self, states, target_policies, target_values):
        """target_values: (B, num_players)."""
        self.network.train()
        x = torch.from_numpy(states).float().permute(
            0, 3, 1, 2).to(self.device)
        pi = torch.from_numpy(target_policies).float().to(self.device)
        z = torch.from_numpy(target_values).float().to(
            self.device)  # (B, num_players)
        log_policy, value = self.network(x)
        loss_policy = -torch.sum(pi * log_policy) / pi.size(0)
        loss_value = F.mse_loss(value, z)           # MSE over the vector
        total = loss_policy + self.value_loss_weight * loss_value
        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()
        return loss_policy.item(), loss_value.item()

    def predict_batch(self, batch_tensor):
        """batch_tensor: (B, C, H, W) on device -> (policies (B,A), values (B,N))."""
        self.network.eval()
        with torch.no_grad():
            with self._autocast():
                log_policy, value = self.network(batch_tensor)
            return torch.exp(log_policy.float()), value.float()

    def save(self, path):
        torch.save({"network_state": self.network.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "num_players": self.num_players}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ck["network_state"])
        if "optimizer_state" in ck:
            self.optimizer.load_state_dict(ck["optimizer_state"])

    def copy_weights_from(self, other):
        self.network.load_state_dict(other.network.state_dict())
