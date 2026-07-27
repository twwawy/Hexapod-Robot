from __future__ import annotations

import importlib
import sys

import pytest


def test_spider_mujoco_import_does_not_pull_spider_rl_root() -> None:
    sys.modules.pop("spider_mujoco", None)
    sys.modules.pop("spider_mujoco.hexapedal_direct", None)
    sys.modules.pop("spider_rl", None)

    importlib.import_module("spider_mujoco")

    assert "spider_rl" not in sys.modules


def test_generated_model_contract() -> None:
    pytest.importorskip("mujoco")

    from spider_mujoco.hexapedal_direct.model_builder import load_hexapedal_model

    model = load_hexapedal_model(validate_generated=True)

    assert model.xml_path.name == "hexapedal.xml"
    assert model.source_map_path.name == "source_map.yaml"
    assert len(model.joint_names) == 18
    assert model.desired_contact_site_names == (
        "LF_motor_horn_3_1_contact_site",
        "LM_motor_horn_3_1_contact_site",
        "LB_motor_horn_3_1_contact_site",
        "RF_motor_horn_3_1_contact_site",
        "RM_motor_horn_3_1_contact_site",
        "RB_motor_horn_3_1_contact_site",
    )

def test_source_asset_resolution_accepts_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from spider_mujoco.hexapedal_direct import model_builder

    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    urdf_dir = tmp_path / "urdf"
    urdf_dir.mkdir()
    urdf_path = urdf_dir / "HEXAPEDAL_URDF_fixed.urdf"
    urdf_path.write_text("<robot name='hexapedal' />\n", encoding="utf-8")

    monkeypatch.setenv(model_builder.URDF_ENV_VAR, str(urdf_path))
    monkeypatch.setenv(model_builder.MESH_DIR_ENV_VAR, str(mesh_dir))

    resolved_urdf, resolved_mesh_dir = model_builder.resolve_source_asset_paths()

    assert resolved_urdf == urdf_path.resolve()
    assert resolved_mesh_dir == mesh_dir.resolve()
