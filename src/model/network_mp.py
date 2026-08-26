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

from src.env.constants import NUM_MOVE_ACTIONS

logger = logging.getLogger(__name__)

POLICY_HEADS = ("flat", "factored")

# Blackwell / Ampere+: TF32 matmul+conv is a large speedup over strict fp32 with
# negligible accuracy loss, and is batch-size agnostic. Enabled once at import,
# guarded so CPU-only hosts / non-CUDA builds are unaffected.
# NOTE: torch.backends.cudnn.benchmark is intentionally left OFF. Leaf-parallel
# self-play feeds VARIABLE batch sizes (wave sizes shrink on collisions), which
# would make benchmark re-autotune kernels on nearly every forward - a net loss,
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
                 num_res_blocks=4, action_space_size=44, num_players=2,
                 policy_head="flat"):
        super().__init__()
        if policy_head not in POLICY_HEADS:
            raise ValueError(f"policy_head must be one of {POLICY_HEADS}, got {policy_head!r}")
        self.board_size = board_size
        self.action_space_size = action_space_size
        self.num_players = num_players
        self.policy_head = policy_head

        self.conv_input = nn.Conv2d(
            in_channels, num_channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_res_blocks)])

        self.policy_conv = nn.Conv2d(num_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        policy_feat = 2 * board_size * board_size
        if policy_head == "flat":
            self.policy_fc = nn.Linear(policy_feat, action_space_size)
        else:
            # Move-vs-wall gate plus per-class placement logits, so the wall
            # class (128 of 140 raw actions at 9x9) can't dominate by count.
            self.policy_type_head = nn.Linear(policy_feat, 2)
            self.policy_move_head = nn.Linear(policy_feat, NUM_MOVE_ACTIONS)
            self.policy_wall_head = nn.Linear(
                policy_feat, action_space_size - NUM_MOVE_ACTIONS)

        self.value_conv = nn.Conv2d(num_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, num_channels)
        # CHANGED: scalar -> length num_players vector
        self.value_fc2 = nn.Linear(num_channels, num_players)

    def _policy_log_probs(self, p):
        """p: pooled policy-trunk features (B, 2*H*W). Returns log-probs over
        the full action space, summing to 1 in probability either way."""
        if self.policy_head == "flat":
            return F.log_softmax(self.policy_fc(p), dim=1)
        # Numerically stable composition: log(P(type) * P(action|type)), never
        # log(softmax(.)) directly.
        log_type = F.log_softmax(self.policy_type_head(p), dim=1)  # (B, 2)
        log_move = F.log_softmax(self.policy_move_head(p), dim=1)  # (B, 12)
        log_wall = F.log_softmax(self.policy_wall_head(p), dim=1)  # (B, A-12)
        log_move_full = log_type[:, 0:1] + log_move
        log_wall_full = log_type[:, 1:2] + log_wall
        return torch.cat([log_move_full, log_wall_full], dim=1)

    def forward(self, x):
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.reshape(p.size(0), -1)
        policy = self._policy_log_probs(p)
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))   # (B, num_players)
        return policy, value


def head_type_from_state(state_dict) -> str:
    """Infer "factored" vs "flat" from state-dict key names alone, for old
    checkpoints and loaders that only have the state dict, not the config."""
    if any(k.startswith("policy_type_head") for k in state_dict):
        return "factored"
    return "flat"


class QuoridorModelMP:
    def __init__(self, board_size=5, action_space_size=44, num_channels=64,
                 num_res_blocks=4, lr=1e-3, weight_decay=1e-4, device="auto",
                 in_channels=None, num_players=2, value_loss_weight=1.0,
                 policy_head="flat"):
        # Default in_channels depends on number of players: channels = 3*N + 3
        if in_channels is None:
            in_channels = 3 * num_players + 3
        self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                                   else (device if device != "auto" else "cpu"))
        self.num_players = num_players
        self.value_loss_weight = value_loss_weight
        self.policy_head = policy_head
        self.network = QuoridorNetworkMP(
            board_size=board_size, in_channels=in_channels, num_channels=num_channels,
            num_res_blocks=num_res_blocks, action_space_size=action_space_size,
            num_players=num_players, policy_head=policy_head,
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

    def train_step(self, states, target_policies, target_values,
                   anchor_model=None, anchor_weight=0.0, value_weights=None):
        """target_values: (B, num_players).

        anchor_model/anchor_weight: adds a cross-entropy pull toward a frozen
        reference policy on the same batch (KL up to a constant). The returned
        loss_policy stays the MCTS-target term alone, so history rows remain
        comparable with unanchored runs.

        value_weights: optional (B, num_players) weights on the value targets
        (training_mp.clone_seat0_value_weights). Normalized by their own sum, so
        uniform weights are the plain mean and loss_v keeps its scale.
        """
        self.network.train()
        x = torch.from_numpy(states).float().permute(
            0, 3, 1, 2).to(self.device)
        pi = torch.from_numpy(target_policies).float().to(self.device)
        z = torch.from_numpy(target_values).float().to(
            self.device)  # (B, num_players)
        log_policy, value = self.network(x)
        loss_policy = -torch.sum(pi * log_policy) / pi.size(0)
        if value_weights is None:
            loss_value = F.mse_loss(value, z)       # MSE over the vector
        else:
            w = torch.from_numpy(value_weights).float().to(self.device)
            total_w = w.sum()
            loss_value = ((w * (value - z) ** 2).sum() / total_w if total_w > 0
                          else torch.zeros((), device=self.device))
        total = loss_policy + self.value_loss_weight * loss_value
        if anchor_model is not None and anchor_weight > 0:
            # Through the public batch path: eval mode, no_grad and autocast
            # handled there, and the anchor forward runs at inference precision.
            anchor_probs, _ = anchor_model.predict_batch(x)
            total = total - anchor_weight * torch.sum(
                anchor_probs * log_policy) / pi.size(0)
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
                    "num_players": self.num_players,
                    "policy_head": self.policy_head}, path)

    def load(self, path, strict_head_check=True):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if strict_head_check:
            ckpt_head = ck.get("policy_head")
            if ckpt_head is None:
                # Old checkpoint, predates the "policy_head" key.
                ckpt_head = head_type_from_state(ck["network_state"])
            if ckpt_head != self.policy_head:
                raise ValueError(
                    f"checkpoint policy_head={ckpt_head!r} does not match "
                    f"model policy_head={self.policy_head!r}")
        # strict_head_check=False skips the check above; a real mismatch then
        # fails naturally in load_state_dict. Used by inspection/conversion tools.
        self.network.load_state_dict(ck["network_state"])
        if "optimizer_state" in ck:
            self.optimizer.load_state_dict(ck["optimizer_state"])

    def copy_weights_from(self, other):
        self.network.load_state_dict(other.network.state_dict())
