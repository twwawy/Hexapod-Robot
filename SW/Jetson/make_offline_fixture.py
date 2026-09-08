"""Synthetic dense-flat sensor fixture for offline zero-action plumbing only."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'mjx'))
import firmware_mjx_controller as fw
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=Path('/tmp/hexapod-flat-input.npz'))
a=p.parse_args()
feet=np.asarray(fw.BASE_FEET)
axis=np.arange(-1.5,1.5,.025)
x,y=np.meshgrid(axis,axis,indexing='ij')
points=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
o=dict(session_id=42,sequence=7,plan_id=11,swing_mask=0x15,gait=0,planned_gait=0,
       flags=1,feet=feet.tolist(),start=feet.tolist(),posture_command=[0.,0.,0.],body_height=0.,
       elapsed=0.,contacts=63,raw_contacts=63,command=[.025,0.,0.])
position=np.array((0.,0.,-float(fw.BASE_FOOT_Z)+.032))
np.savez(a.output,observation_json=json.dumps(o),position=position,body_rotation=np.eye(3),
         com_world=position,points=points,valid=np.ones(len(points),dtype=bool),time=1.)
print(a.output)
