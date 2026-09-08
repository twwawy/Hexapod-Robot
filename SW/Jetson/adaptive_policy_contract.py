"""Only adaptive 24-D v4; 18-D legacy replay lives in root mjx separately."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'mjx'))
from adaptive_contract import ACTION_CONTRACT, ACTION_SIZE, XY_LIMIT_M

def zero_action():
    import jax.numpy as jp
    return jp.zeros(ACTION_SIZE)

def load_policy(path):
    from adaptive_gait_policy import load_policy as load
    return load(path)  # validates action/observation versions AND source hashes
