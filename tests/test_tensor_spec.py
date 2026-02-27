"""
Tensor Spec Tests
==================
Run: pytest tests/test_tensor_spec.py -v

Validates the reference tensor implementation.
When Reef implements state_to_tensor(), add a comparison test here
that verifies his output matches build_tensor() for the same state.
"""

import numpy as np
import pytest

from src.env.tensor_spec import build_tensor, validate_tensor_spec


def test_tensor_spec_validation():
    """Run the full self-consistency validation suite."""
    validate_tensor_spec()


def test_initial_state_shape_and_dtype():
    """Tensor for initial state has correct shape and dtype."""
    tensor = build_tensor(
        board_size=5,
        p0_pos=(4, 2), p1_pos=(0, 2),
        p0_h_walls=[], p0_v_walls=[],
        p1_h_walls=[], p1_v_walls=[],
        p0_walls_remaining=5, p1_walls_remaining=5,
        max_walls=5,
    )
    assert tensor.shape == (5, 5, 10)
    assert tensor.dtype == np.float32


def test_all_values_normalized():
    """All tensor values must be in [0, 1]."""
    tensor = build_tensor(
        board_size=5,
        p0_pos=(3, 1), p1_pos=(1, 3),
        p0_h_walls=[(1, 0), (2, 2)], p0_v_walls=[(0, 1)],
        p1_h_walls=[(3, 1)], p1_v_walls=[(1, 2)],
        p0_walls_remaining=2, p1_walls_remaining=3,
        max_walls=5,
    )
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_pawn_channels_single_cell():
    """Each pawn channel should have exactly one cell set to 1.0."""
    tensor = build_tensor(
        board_size=5,
        p0_pos=(4, 2), p1_pos=(0, 2),
        p0_h_walls=[], p0_v_walls=[],
        p1_h_walls=[], p1_v_walls=[],
        p0_walls_remaining=5, p1_walls_remaining=5,
        max_walls=5,
    )
    assert tensor[:, :, 0].sum() == 1.0
    assert tensor[:, :, 1].sum() == 1.0
    assert tensor[4, 2, 0] == 1.0
    assert tensor[0, 2, 1] == 1.0
