from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest



def test_env_registration_and_step() -> None:
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mujoco")

    import spider_mujoco  # noqa: F401

    env = gym.make("Hexapedal-MuJoCo-Direct-v0")
    try:
        obs, _ = env.reset(seed=123)
        assert obs.shape == (48,)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (48,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()

def test_resolve_model_path_regenerates_when_generated_assets_are_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from spider_mujoco.hexapedal_direct.env import HexapedalDirectEnv
    from spider_mujoco.hexapedal_direct import env as env_module

    xml_path = tmp_path / "hexapedal.xml"
    xml_path.write_text("<mujoco model='hexapedal_direct'/>\n", encoding="utf-8")

    fake_builder = SimpleNamespace(
        load_hexapedal_model=lambda validate_generated=True: (_ for _ in ()).throw(FileNotFoundError("missing assets")),
        write_hexapedal_assets=lambda: SimpleNamespace(xml_path=xml_path),
        XML_PATH=xml_path,
    )

    monkeypatch.setattr(env_module.importlib, "import_module", lambda name: fake_builder)

    resolved = HexapedalDirectEnv._resolve_model_path(tmp_path / "missing.xml")

    assert resolved == xml_path
