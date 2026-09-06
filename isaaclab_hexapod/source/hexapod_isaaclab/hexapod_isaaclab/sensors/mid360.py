"""Livox MID-360 geometry and Isaac Sim sensor helpers.

The full-CAD MJCF export merges fixed URDF links into the floating base.  The
sensor therefore remains attached to the base rigid body with the optical
origin recovered from the fixed-joint chain in the current Xacro.
"""

from __future__ import annotations

import math


MID360_POSITION_BODY = (
    0.018070710783485648,
    0.03671389869660539,
    0.30472693304105325,
)
"""Optical origin from the current fixed URDF chain, in body-frame metres."""

MID360_MOUNT_RPY_BODY_RAD = (math.pi, -math.pi / 6.0, math.pi / 2.0)
"""Inverted Livox frame: roll 180, pitch -30, yaw +90 degrees."""


def _quaternion_from_rpy_wxyz(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    """Convert URDF fixed-axis RPY to a WXYZ quaternion."""

    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


MID360_ROTATION_BODY_WXYZ = _quaternion_from_rpy_wxyz(
    *MID360_MOUNT_RPY_BODY_RAD
)
"""Sensor-to-body rotation copied from URDF joint ``강체 202``."""

MID360_HORIZONTAL_FOV_DEG = (-180.0, 180.0)
MID360_VERTICAL_FOV_DEG = (-7.0, 52.0)
MID360_MIN_RANGE_M = 0.1
MID360_MAX_RANGE_M = 40.0
MID360_FRAME_RATE_HZ = 10
MID360_POINT_RATE_HZ = 200_000
MID360_CHANNELS = 40

# A full 200 kpoint/s pattern in every cloned environment is wasteful for the
# 32 x 24 locomotion elevation map.  This spacing preserves the physical FOV
# and 40-line density while keeping the batched training sensor tractable.
MID360_TRAINING_HORIZONTAL_RES_DEG = 4.0


def author_rtx_mid360_prim(prim_path: str, *, stage=None):
    """Author the configured MID-360 ``OmniLidar`` prim on the open stage.

    Isaac Sim 5.1 does not ship a Livox profile.  The RTX Core profile below
    matches the published range, accuracy, wavelength, 40-line density,
    10 Hz frame rate and 200 kpoint/s rate.  It uses a rotary 40-emitter proxy;
    it does not claim to reproduce Livox's proprietary non-repetitive pattern.
    """

    import numpy as np
    import omni.usd
    from pxr import Gf, UsdGeom, Vt

    if stage is None:
        stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("cannot create MID-360 without an open USD stage")
    prim = stage.DefinePrim(prim_path, "OmniLidar")
    if not prim.ApplyAPI("OmniSensorGenericLidarCoreAPI"):
        raise RuntimeError("failed to apply RTX generic LiDAR API")

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*MID360_POSITION_BODY))
    rotation = MID360_ROTATION_BODY_WXYZ
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Quatf(rotation[0], Gf.Vec3f(*rotation[1:]))
    )

    elevations = np.linspace(
        MID360_VERTICAL_FOV_DEG[0],
        MID360_VERTICAL_FOV_DEG[1],
        MID360_CHANNELS,
        dtype=np.float32,
    ).tolist()
    # 5,000 azimuth ticks/s x 40 emitters = 200,000 points/s.
    report_rate_hz = MID360_POINT_RATE_HZ // MID360_CHANNELS
    tick_period_ns = 1_000_000_000 // report_rate_hz
    fire_times = np.linspace(
        0, tick_period_ns - 1, MID360_CHANNELS, dtype=np.uint32
    ).tolist()

    attributes = {
        "omni:sensor:marketName": "MID-360",
        "omni:sensor:modelName": "Livox MID-360",
        "omni:sensor:modelVendor": "Livox",
        "omni:sensor:modelVersion": "1.0",
        "omni:sensor:tickRate": float(MID360_FRAME_RATE_HZ),
        "omni:sensor:Core:scanType": "ROTARY",
        "omni:sensor:Core:auxOutputType": "BASIC",
        "omni:sensor:Core:rotationDirection": "CW",
        "omni:sensor:Core:rayType": "IDEALIZED",
        "omni:sensor:Core:intensityProcessing": "NORMALIZATION",
        "omni:sensor:Core:nearRangeM": MID360_MIN_RANGE_M,
        "omni:sensor:Core:farRangeM": MID360_MAX_RANGE_M,
        "omni:sensor:Core:rangeResolutionM": 0.002,
        "omni:sensor:Core:rangeAccuracyM": 0.02,
        "omni:sensor:Core:minReflectance": 0.1,
        "omni:sensor:Core:minReflectionRangeM": MID360_MAX_RANGE_M,
        "omni:sensor:Core:waveLengthNm": 905.0,
        "omni:sensor:Core:pulseTimeNs": 6,
        "omni:sensor:Core:maxReturns": 1,
        "omni:sensor:Core:scanRateBaseHz": MID360_FRAME_RATE_HZ,
        "omni:sensor:Core:reportRateBaseHz": report_rate_hz,
        "omni:sensor:Core:numberOfEmitters": MID360_CHANNELS,
        "omni:sensor:Core:numberOfChannels": MID360_CHANNELS,
        "omni:sensor:Core:validStartAzimuthDeg": 0.0,
        "omni:sensor:Core:validEndAzimuthDeg": 360.0,
        "omni:sensor:Core:skipDroppingInvalidPoints": True,
        "omni:sensor:Core:emitterState:s001:azimuthDeg": Vt.FloatArray(
            [0.0] * MID360_CHANNELS
        ),
        "omni:sensor:Core:emitterState:s001:elevationDeg": Vt.FloatArray(
            elevations
        ),
        "omni:sensor:Core:emitterState:s001:fireTimeNs": Vt.UIntArray(
            fire_times
        ),
        "omni:sensor:Core:emitterState:s001:channelId": Vt.UIntArray(
            list(range(1, MID360_CHANNELS + 1))
        ),
    }
    missing: list[str] = []
    for name, value in attributes.items():
        attribute = prim.GetAttribute(name)
        if not attribute.IsValid() or not attribute.Set(value):
            missing.append(name)
    if missing:
        raise RuntimeError(f"MID-360 RTX attributes could not be authored: {missing}")

    return prim


def create_rtx_mid360(prim_path: str):
    """Author and construct the one-robot RTX MID-360 proxy."""

    import omni.usd
    from isaacsim.sensors.rtx import LidarRtx
    from pxr import Sdf, Usd

    author_rtx_mid360_prim(prim_path)
    lidar = LidarRtx(prim_path=prim_path, name="livox_mid360_rtx")

    # Isaac Sim 5.1's LidarRtx constructor authors GenericModelOutput only,
    # while both RTX point-cloud converters also consume RtxSensorMetadata.
    # Add the missing AOV before any annotator/writer is attached.
    stage = omni.usd.get_context().get_stage()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        render_vars = stage.DefinePrim("/Render/Vars", "Scope")
        del render_vars  # path existence is the only requirement
        metadata_path = "/Render/Vars/RtxSensorMetadata"
        metadata = stage.DefinePrim(metadata_path, "RenderVar")
        metadata.CreateAttribute(
            "sourceName", Sdf.ValueTypeNames.String
        ).Set("RtxSensorMetadata")
        render_product = stage.GetPrimAtPath(lidar.get_render_product_path())
        if not render_product.IsValid():
            raise RuntimeError("MID-360 RTX render product was not created")
        render_product.GetRelationship("orderedVars").AddTarget(metadata_path)
    return lidar


def initialize_rtx_mid360(lidar, *, debug_vis: bool = True) -> None:
    """Initialize point-cloud output after the simulation has reset."""

    lidar.initialize()
    # Keep a per-frame point array for numerical health diagnostics.  The
    # buffered writer below separately accumulates a complete scan for display.
    lidar.attach_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")
    if debug_vis:
        # The buffered writer accumulates a complete 10 Hz revolution and is
        # materially easier to see than the per-render-frame point fragment.
        lidar.attach_writer("RtxLidarDebugDrawPointCloudBuffer")
