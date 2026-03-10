"""
Dual-Headed Neural Network (Policy + Value)
=============================================
Owner: Rom

AlphaZero-style CNN/ResNet:
    Input:  (batch, board_h, board_w, 10) — 10-channel board tensor
    Output: policy (batch, action_space_size) — move probabilities
            value  (batch, 1)               — win probability [-1, 1]

Losses:
    Policy: Cross-entropy against MCTS visit distribution
    Value:  MSE against game outcome (+1 / -1)

Interface needed by self_play.py:
    model.predict(tensor)                          -> (policy_np, value_float)
    model.train_step(states, policies, values)     -> (loss_policy, loss_value)
    model.save(path) / model.load(path)

Start small for 5x5: 4 residual blocks, 64 channels.
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Single residual block: Conv → BN → ReLU → Conv → BN + skip"""

    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels,
                               3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class QuoridorNetwork(nn.Module):
    """
    AlphaZero-style dual-headed network for Quoridor.

    Architecture:
        Input Conv → N Residual Blocks → Policy Head + Value Head

    Input tensor: (batch, 10, board_size, board_size)
        Note: PyTorch uses NCHW format. The env produces (H, W, C),
        so we transpose in predict().
    """

    def __init__(
        self,
        board_size: int = 5,
        in_channels: int = 10,
        num_channels: int = 64,
        num_res_blocks: int = 4,
        action_space_size: int = 44,
    ):
        super().__init__()
        self.board_size = board_size
        self.action_space_size = action_space_size

        # Initial convolution: (10, H, W) -> (num_channels, H, W)
        self.conv_input = nn.Conv2d(
            in_channels, num_channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head: Conv 1x1 → BN → ReLU → Flatten → FC
        self.policy_conv = nn.Conv2d(num_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(
            2 * board_size * board_size, action_space_size)

        # Value head: Conv 1x1 → BN → ReLU → Flatten → FC → ReLU → FC → tanh
        self.value_conv = nn.Conv2d(num_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, num_channels)
        self.value_fc2 = nn.Linear(num_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, 10, board_size, board_size) float tensor

        Returns:
            policy: (batch, action_space_size) — log-softmax probabilities
            value:  (batch, 1) — scalar in [-1, 1]
        """
        # Shared trunk
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.reshape(p.size(0), -1)
        p = self.policy_fc(p)
        policy = F.log_softmax(p, dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value


class QuoridorModel:
    """
    High-level wrapper matching the interface expected by self_play.py:
        model.predict(tensor)  -> (policy_np, value_float)
        model.train_step(states, policies, values) -> (loss_p, loss_v)
        model.save(path) / model.load(path)
    """

    def __init__(
        self,
        board_size: int = 5,
        action_space_size: int = 44,
        num_channels: int = 64,
        num_res_blocks: int = 4,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.network = QuoridorNetwork(
            board_size=board_size,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            action_space_size=action_space_size,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.board_size = board_size
        self.action_space_size = action_space_size

        param_count = sum(p.numel() for p in self.network.parameters())
        logger.info(
            "QuoridorModel: %d params, device=%s, board=%dx%d, actions=%d",
            param_count, self.device, board_size, board_size, action_space_size,
        )

    def predict(self, state_tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Single-sample inference for MCTS. No gradient computation.

        Args:
            state_tensor: (board_size, board_size, 10) numpy array (HWC format)

        Returns:
            policy: (action_space_size,) numpy array of probabilities
            value: scalar float in [-1, 1]
        """
        self.network.eval()
        with torch.no_grad():
            # HWC -> CHW, add batch dim
            x = torch.from_numpy(state_tensor).float()
            x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)

            log_policy, value = self.network(x)

            policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
            value_scalar = value.item()

        return policy, value_scalar

    def train_step(
        self,
        states: np.ndarray,
        target_policies: np.ndarray,
        target_values: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Single training step on a batch.

        Args:
            states: (batch, board_size, board_size, 10) — HWC format
            target_policies: (batch, action_space_size) — MCTS visit distributions
            target_values: (batch,) — game outcomes in [-1, 1]

        Returns:
            (policy_loss, value_loss) as Python floats
        """
        self.network.train()

        # Convert to tensors, HWC -> NCHW
        x = torch.from_numpy(states).float().permute(
            0, 3, 1, 2).to(self.device)
        pi = torch.from_numpy(target_policies).float().to(self.device)
        z = torch.from_numpy(target_values).float(
        ).unsqueeze(1).to(self.device)

        # Forward
        log_policy, value = self.network(x)

        # Losses
        # Policy: cross-entropy = -sum(pi * log_policy) / batch
        loss_policy = -torch.sum(pi * log_policy) / pi.size(0)
        # Value: MSE
        loss_value = F.mse_loss(value, z)

        total_loss = loss_policy + loss_value

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return loss_policy.item(), loss_value.item()

    def save(self, path: str):
        """Save model weights and optimizer state."""
        torch.save(
            {
                "network_state": self.network.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info("Model saved to %s", path)

    def load(self, path: str):
        """Load model weights and optimizer state."""
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        logger.info("Model loaded from %s", path)

    def copy_weights_from(self, other: "QuoridorModel"):
        """Copy network weights from another model (for best-model tracking)."""
        self.network.load_state_dict(other.network.state_dict())
