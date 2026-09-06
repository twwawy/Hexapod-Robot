"""Tests for the Jetson-only joint-sign conversion."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


CONTROLLER_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/hexapod_isaaclab/hexapod_isaaclab/controllers/firmware_controller_torch.py"
)
SPEC = importlib.util.spec_from_file_location("firmware_controller_torch", CONTROLLER_PATH)
assert SPEC is not None and SPEC.loader is not None
CONTROLLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROLLER)
jetson_tx_joint_targets = CONTROLLER.jetson_tx_joint_targets


def test_only_requested_joint_signs_are_flipped() -> None:
    source = torch.arange(1.0, 19.0).reshape(1, 6, 3)

    transmitted = jetson_tx_joint_targets(source)

    expected = source.clone()
    expected[:, 0:3, 1] *= -1.0
    expected[:, 3:6, 2] *= -1.0
    torch.testing.assert_close(transmitted, expected)
    torch.testing.assert_close(source, torch.arange(1.0, 19.0).reshape(1, 6, 3))


def test_rejects_non_leg_major_input() -> None:
    try:
        jetson_tx_joint_targets(torch.zeros(1, 18))
    except ValueError as error:
        assert "(6, 3)" in str(error)
    else:
        raise AssertionError("flat joint targets must be rejected")


def test_training_simulation_and_jetson_outputs_share_the_same_signs() -> None:
    state = CONTROLLER.initial_state(1, "cpu")
    output = CONTROLLER.initial_output(state)
    expected = jetson_tx_joint_targets(state.previous_joint)

    torch.testing.assert_close(output.model_joint_targets, expected)
    torch.testing.assert_close(output.servo_joint_targets, expected)


if __name__ == "__main__":
    test_only_requested_joint_signs_are_flipped()
    test_rejects_non_leg_major_input()
    test_training_simulation_and_jetson_outputs_share_the_same_signs()
    print("Jetson joint-sign tests passed")
