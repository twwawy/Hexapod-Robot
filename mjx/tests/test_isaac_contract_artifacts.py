"""Integrity tests for versioned MJX-to-Isaac reference artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "golden"
NPZ_PATH = GOLDEN_DIR / "isaac_contract_v1_flat_seed0.npz"
JSON_PATH = GOLDEN_DIR / "isaac_contract_v1_flat_seed0.json"
ASSET_PATH = GOLDEN_DIR / "asset_manifest_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GoldenContractArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        cls.trace = np.load(NPZ_PATH)

    def test_header_and_checksum(self) -> None:
        self.assertEqual(self.metadata["schema"], "hexapod_mjx_transition_v1")
        self.assertEqual(self.metadata["policy_steps"], 500)
        self.assertEqual(self.metadata["firmware_ticks_per_policy_step"], 4)
        self.assertEqual(self.metadata["npz"]["sha256"], _sha256(NPZ_PATH))

    def test_required_shapes_and_dtypes(self) -> None:
        expected = {
            "command": (500, 5),
            "action_requested": (500, 18),
            "action_applied": (500, 18),
            "q_des": (500, 18),
            "observation_pre": (500, 146),
            "observation_post": (500, 146),
            "firmware_tick/state/foot_memory": (500, 4, 6, 3),
            "firmware_tick/output/model_joint_targets": (500, 4, 6, 3),
        }
        for key, shape in expected.items():
            with self.subTest(key=key):
                self.assertEqual(self.trace[key].shape, shape)
        self.assertEqual(self.trace["action_requested"].dtype, np.float32)
        self.assertEqual(self.trace["firmware_tick/state/phase_index"].dtype, np.int32)
        self.assertEqual(self.trace["done"].dtype, np.bool_)

    def test_all_floating_arrays_are_finite(self) -> None:
        for key in self.trace.files:
            value = self.trace[key]
            if np.issubdtype(value.dtype, np.floating):
                with self.subTest(key=key):
                    self.assertTrue(np.isfinite(value).all())

    def test_scripted_action_contract(self) -> None:
        np.testing.assert_array_equal(
            self.trace["action_requested"][:50], np.zeros((50, 18), np.float32)
        )
        self.assertGreater(float(np.max(np.abs(self.trace["action_requested"][50:]))), 0.0)
        np.testing.assert_array_equal(
            self.trace["action_requested"], self.trace["action_applied"]
        )


class AssetManifestArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(ASSET_PATH.read_text(encoding="utf-8"))

    def test_asset_contract(self) -> None:
        self.assertEqual(self.manifest["schema"], "hexapod_mjx_asset_v1")
        self.assertAlmostEqual(
            self.manifest["contract"]["total_robot_mass_kg"], 10.0, places=6
        )
        self.assertEqual(len(self.manifest["joints"]), 18)
        self.assertEqual(
            self.manifest["contract"]["canonical_joint_order"][:3],
            ["RB_1", "RB_2", "RB_3"],
        )
        self.assertEqual(
            self.manifest["contract"]["canonical_joint_order"][-3:],
            ["LF_1", "LF_2", "LF_3"],
        )


if __name__ == "__main__":
    unittest.main()
