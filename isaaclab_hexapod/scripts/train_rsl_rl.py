#!/usr/bin/env python3
"""Bootstrap Isaac Lab's RSL-RL trainer with the local Hexapod task."""

from __future__ import annotations

import builtins
import importlib
from pathlib import Path
import runpy
import sys


ISAACLAB_ROOT = Path("/home/huro/IsaacLab")
UPSTREAM_TRAIN = ISAACLAB_ROOT / "scripts/reinforcement_learning/rsl_rl/train.py"
UPSTREAM_SCRIPT_DIR = str(UPSTREAM_TRAIN.parent)
if UPSTREAM_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, UPSTREAM_SCRIPT_DIR)

original_import = builtins.__import__
hexapod_registered = False


def import_with_hexapod(name, globals=None, locals=None, fromlist=(), level=0):
    """Register the external task immediately after Isaac Lab task imports."""
    global hexapod_registered
    module = original_import(name, globals, locals, fromlist, level)
    if name == "isaaclab_tasks" and not hexapod_registered:
        hexapod_registered = True
        importlib.import_module("hexapod_isaaclab")
    return module


builtins.__import__ = import_with_hexapod
try:
    runpy.run_path(str(UPSTREAM_TRAIN), run_name="__main__")
finally:
    builtins.__import__ = original_import
