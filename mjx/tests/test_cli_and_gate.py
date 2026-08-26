from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rough_terrain_env import ACTION_CONTRACT_VERSION, default_config
from train_rough_terrain import (
    _apply_reward_weights,
    _arguments,
    _resolve_checkpoint,
)


class CliAndCheckpointGateTest(unittest.TestCase):
    def test_repeatable_reward_override_is_parsed(self) -> None:
        args = _arguments(
            [
                "--reward-weight",
                "upright=2.5",
                "--reward-weight",
                "touchdown_impact=-0.2",
            ]
        )
        self.assertEqual(
            args.reward_weights,
            {"upright": 2.5, "touchdown_impact": -0.2},
        )
        config = default_config()
        _apply_reward_weights(config, args.reward_weights)
        self.assertEqual(config.reward.upright, 2.5)
        self.assertEqual(config.reward.touchdown_impact, -0.2)

    def _assert_parser_error(self, value: str, substring: str) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            _arguments(["--reward-weight", value])
        self.assertIn(substring, stderr.getvalue())

    def test_unknown_reward_key_lists_valid_keys(self) -> None:
        self._assert_parser_error("not_a_reward=1", "valid keys:")
        self._assert_parser_error("not_a_reward=1", "touchdown_impact")

    def test_nonfloat_reward_weight_is_parser_error(self) -> None:
        self._assert_parser_error("upright=strong", "must contain a float")

    def _assert_legacy_checkpoint_rejected(
        self, observation_size: int, observation_contract: str
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            checkpoint = run_dir / "checkpoints" / "000000000001"
            checkpoint.mkdir(parents=True)
            (checkpoint / "ppo_network_config.json").write_text(
                json.dumps(
                    {
                        "action_size": 18,
                        "observation_size": {"shape": [observation_size]},
                        "network_factory_kwargs": {
                            "policy_hidden_layer_sizes": [256, 256, 128]
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "action_contract_version": ACTION_CONTRACT_VERSION,
                        "observation_contract_version": observation_contract,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "legacy checkpoints incompatible"
            ):
                _resolve_checkpoint(checkpoint, (256, 256, 128))

    def test_legacy_142d_checkpoint_has_explicit_rejection(self) -> None:
        self._assert_legacy_checkpoint_rejected(
            142, "firmware_state_collision_terrain_curriculum_v2"
        )

    def test_legacy_143d_checkpoint_has_explicit_rejection(self) -> None:
        self._assert_legacy_checkpoint_rejected(
            143, "firmware_state_collision_terrain_pitch_v3"
        )


if __name__ == "__main__":
    unittest.main()
