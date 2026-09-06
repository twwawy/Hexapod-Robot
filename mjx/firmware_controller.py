"""Build and call the STM32 high-level controller with MuJoCo ground truth."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np


MJX_DIR = Path(__file__).resolve().parent
REPOSITORY = MJX_DIR.parent
CORE = REPOSITORY / "SW/STM32/workspace/Hexapod/Core"
BRIDGE_SOURCE = MJX_DIR / "native/firmware_controller_bridge.c"
LIBRARY = MJX_DIR / "generated/libhexapod_firmware_controller.so"

CONTROLLER_SOURCES = tuple(
    CORE / "Src/high_control" / name
    for name in (
        "drone_controller.c",
        "gait_pose_controller.c",
        "workspace_limiter.c",
        "gait_manager.c",
        "foot_trajectory.c",
        "body_posture_controller.c",
        "leg_kinematics.c",
        "stance_trajectory.c",
        "swing_trajectory.c",
        "contact_adaptation.c",
    )
)
CONTROLLER_HEADERS = (
    *(CORE / "Inc/common").glob("*.h"),
    *(CORE / "Inc/high_control").glob("*.h"),
)


class _ControllerInput(ctypes.Structure):
    _fields_ = (
        ("target_vx_mps", ctypes.c_float),
        ("target_wz_radps", ctypes.c_float),
        ("target_roll_rad", ctypes.c_float),
        ("target_pitch_rad", ctypes.c_float),
        ("body_position_world", ctypes.c_float * 3),
        ("attitude_rad", ctypes.c_float * 3),
        ("foot_contact", ctypes.c_uint8 * 6),
    )


class _ControllerOutput(ctypes.Structure):
    _fields_ = (
        ("joint_angle_rad", ctypes.c_float * 18),
        ("foot_target_body", ctypes.c_float * 18),
        ("applied_twist", ctypes.c_float * 4),
        ("gait_progress", ctypes.c_float * 6),
        ("gait_state", ctypes.c_uint8 * 6),
        ("ik_valid", ctypes.c_uint8 * 6),
        ("gait_enabled", ctypes.c_uint8),
        ("gait_accepted", ctypes.c_uint8),
        ("posture_accepted", ctypes.c_uint8),
    )


@dataclass(frozen=True)
class FirmwareControllerState:
    joint_angles: np.ndarray
    foot_targets: np.ndarray
    applied_twist: np.ndarray
    gait_progress: np.ndarray
    gait_state: np.ndarray
    ik_valid: np.ndarray
    gait_enabled: bool
    gait_accepted: bool
    posture_accepted: bool


def _build_library() -> Path:
    sources = (BRIDGE_SOURCE, *CONTROLLER_SOURCES)
    dependencies = (*sources, *CONTROLLER_HEADERS)
    missing = [path for path in dependencies if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing firmware controller source: {missing[0]}")

    rebuild = not LIBRARY.exists() or any(
        path.stat().st_mtime > LIBRARY.stat().st_mtime for path in dependencies
    )
    if rebuild:
        LIBRARY.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "cc",
            "-std=c11",
            "-O2",
            "-fPIC",
            "-shared",
            "-Wall",
            "-Wextra",
            "-Wno-stringop-overread",
            "-I",
            str(CORE / "Inc"),
            *(str(path) for path in sources),
            "-lm",
            "-o",
            str(LIBRARY),
        ]
        print("Building STM32 high-control bridge...", flush=True)
        subprocess.run(command, check=True)
    return LIBRARY


class FirmwareController:
    """Stateful wrapper around the exact STM32 high-control C modules."""

    def __init__(self) -> None:
        self._library = ctypes.CDLL(str(_build_library()))
        self._library.FirmwareController_Create.restype = ctypes.c_void_p
        self._library.FirmwareController_Destroy.argtypes = (ctypes.c_void_p,)
        self._library.FirmwareController_Step.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_ControllerInput),
            ctypes.POINTER(_ControllerOutput),
        )
        self._library.FirmwareController_Step.restype = ctypes.c_int
        self._handle = self._library.FirmwareController_Create()
        if not self._handle:
            raise RuntimeError("Failed to initialize the firmware controller")

    def close(self) -> None:
        if self._handle:
            self._library.FirmwareController_Destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "FirmwareController":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def step(
        self,
        *,
        target_vx: float,
        target_wz: float,
        body_position: np.ndarray,
        attitude: np.ndarray,
        contacts: np.ndarray,
        target_roll: float = 0.0,
        target_pitch: float = 0.0,
    ) -> FirmwareControllerState:
        if not self._handle:
            raise RuntimeError("Firmware controller is closed")

        controller_input = _ControllerInput(
            target_vx,
            target_wz,
            target_roll,
            target_pitch,
            (ctypes.c_float * 3)(*body_position),
            (ctypes.c_float * 3)(*attitude),
            (ctypes.c_uint8 * 6)(*contacts.astype(np.uint8)),
        )
        output = _ControllerOutput()
        if not self._library.FirmwareController_Step(
            self._handle, ctypes.byref(controller_input), ctypes.byref(output)
        ):
            raise RuntimeError("Firmware controller step failed")

        return FirmwareControllerState(
            joint_angles=np.ctypeslib.as_array(output.joint_angle_rad).copy(),
            foot_targets=np.ctypeslib.as_array(output.foot_target_body).copy().reshape(6, 3),
            applied_twist=np.ctypeslib.as_array(output.applied_twist).copy(),
            gait_progress=np.ctypeslib.as_array(output.gait_progress).copy(),
            gait_state=np.ctypeslib.as_array(output.gait_state).copy(),
            ik_valid=np.ctypeslib.as_array(output.ik_valid).astype(bool).copy(),
            gait_enabled=bool(output.gait_enabled),
            gait_accepted=bool(output.gait_accepted),
            posture_accepted=bool(output.posture_accepted),
        )
