"""Offline/hardware-input bridge: coherent observation + odom + deskewed world returns.

No simulator raycast or GT enters this bridge. Live Livox/odom acquisition and
SPI device scheduling are injected by the caller; this module does not arm motors.
"""
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'mjx'))
import jax
import jax.numpy as jp
import numpy as np
import adaptive_gait_controller as adaptive
from adaptive_gait_env import AdaptiveGaitEnv, ACTOR_SIZE
from adaptive_gait_perception import MapState, initial_map, sample, GRID_N, RESOLUTION, MAX_AGE
import hybrid_gait_supervisor as supervisor
import wave_gait_scheduler as scheduler
from adaptive_execution_plan import AdaptiveExecutionPlan, LegPlan
from adaptive_spi_protocol import encode_execution

@dataclass(frozen=True)
class PlannerData:
    qpos: object
    xmat: object
    site_xpos: object
    subtree_com: object
    time: object
    def replace(self, **kwargs):
        return replace(self, **kwargs)

class HardwareGeometry:
    """Reuse the exact JAX geometry/supervisor without constructing a MuJoCo model."""
    _candidates = AdaptiveGaitEnv._candidates
    _phase_check = AdaptiveGaitEnv._phase_check
    _landing_plan = AdaptiveGaitEnv._landing_plan
    _root_id = 0
    _foot_site_ids = jp.arange(6)
    perception = 'lidar'
    gait_mode = 'hybrid'
    dt = .02
    def _query(self, grid, xy, now, *, privileged=False):
        if privileged:
            raise ValueError('Hardware geometry has no oracle')
        return sample(grid, xy, now)

def update_world_returns(grid, points, valid, position, now):
    """Fixed-shape point buffer with validity mask; self-filter/deskew upstream.

    Input frame MUST be odom/world, not the LiDAR frame. No unknown fill.
    """
    points, valid = jp.asarray(points), jp.asarray(valid, dtype=bool)
    valid &= jp.all(jp.isfinite(points), axis=1)
    center = jp.round(jp.asarray(position[:2])/RESOLUTION)*RESOLUTION
    shift = jp.rint((center-grid.center)/RESOLUTION).astype(jp.int32)
    i,j = jp.meshgrid(jp.arange(GRID_N),jp.arange(GRID_N),indexing='ij')
    oi,oj = i+shift[0],j+shift[1]
    overlap = (oi>=0)&(oi<GRID_N)&(oj>=0)&(oj<GRID_N)
    oi,oj = jp.clip(oi,0,GRID_N-1),jp.clip(oj,0,GRID_N-1)
    height = jp.where(overlap,grid.height[oi,oj],0.)
    stamp = jp.where(overlap,grid.timestamp[oi,oj],-1e6)
    spread = jp.where(overlap,grid.spread[oi,oj],0.)
    ij = jp.floor((points[:,:2]-center)/RESOLUTION+GRID_N/2).astype(jp.int32)
    valid &= jp.all((ij>=0)&(ij<GRID_N),axis=1)
    ij=jp.clip(ij,0,GRID_N-1); ids=ij[:,0]*GRID_N+ij[:,1]
    high=jp.full(GRID_N*GRID_N,-jp.inf).at[ids].max(jp.where(valid,points[:,2],-jp.inf)).reshape(GRID_N,GRID_N)
    low=jp.full(GRID_N*GRID_N,jp.inf).at[ids].min(jp.where(valid,points[:,2],jp.inf)).reshape(GRID_N,GRID_N)
    seen=jp.isfinite(high); fresh=(now-stamp>=0)&(now-stamp<=MAX_AGE)
    high=jp.where(fresh,jp.maximum(high,height),high)
    low=jp.where(fresh,jp.minimum(low,height-spread),low)
    return MapState(jp.where(seen,high,height),jp.where(seen,now,stamp),
                    jp.where(seen,high-low,spread),center,jp.sum(valid))

class AdaptiveRuntime:
    def __init__(self):
        self.geometry=HardwareGeometry()
        self.supervisor=supervisor.initial_supervisor()
        self.sequence=0
        self.session=None

    def prepare(self, observation, grid, position, body_rotation, com_world, now,
                *, action=None, policy=None, actor_observation=None):
        """now and map timestamps share one monotonic odometry clock.

        A nonzero policy needs the full v4 actor vector from the deployment state
        estimator. Missing velocity/history features are never silently filled.
        STM32 independently rejects stale source-observation IDs.
        """
        o=observation
        if not o['session_id'] or not o['flags']&1:
            raise ValueError('STM32 has no active advertised phase; send a NOP')
        if self.session != o['session_id']:
            self.session=o['session_id'];self.sequence=0;self.supervisor=supervisor.initial_supervisor()
        if policy is not None:
            if actor_observation is None or np.shape(actor_observation)!=(ACTOR_SIZE,):
                raise ValueError(f'Policy requires complete {ACTOR_SIZE}-D v4 actor observation')
            action=policy({'state':jp.asarray(actor_observation)},jax.random.PRNGKey(self.sequence))[0]
        action=jp.zeros(24) if action is None else jp.asarray(action)
        if action.shape!=(24,) or not np.all(np.isfinite(action)):
            raise ValueError('PolicyAction24 requires 24 finite values')
        # Model axes -> hardware controller forward/left/up axes.
        rotation=jp.asarray(body_rotation)@jp.array(((0.,-1.,0.),(1.,0.,0.),(0.,0.,1.)))
        position=jp.asarray(position)
        cs=adaptive.initial_state()._replace(foot_memory=jp.asarray(o['feet']),
            posture_command=jp.asarray(o['posture_command']),height_applied=jp.asarray(o['body_height']))
        mask=int(o['swing_mask'])
        planned_gait=int(o.get('planned_gait',o['gait']))
        phase=(0 if mask==0x15 else 1) if planned_gait==0 else int(np.where(np.array((0,5,1,3,2,4))==int(np.log2(mask)))[0][0])
        sched=cs.scheduler._replace(mode=jp.asarray(o['gait']),phase=jp.asarray(phase),
            running=jp.asarray(bool(o['flags']&8)),elapsed=jp.asarray(o['elapsed']))
        cs=cs._replace(scheduler=sched)
        pre=jp.asarray(o['feet']).at[:,2].add(-o['body_height'])
        body=adaptive.fw._rotate_inverse(pre,cs.posture_command)
        model=jp.stack((body[:,1],-body[:,0],body[:,2]),axis=-1)
        feet=position+model@rotation.T
        data=PlannerData(jp.concatenate((position,jp.array((1.,0.,0.,0.)))),rotation[None],feet,
                         jp.asarray(com_world)[None],jp.asarray(now))
        contacts=jp.array([bool(o['contacts']&(1<<i)) for i in range(6)])
        info=dict(controller_state=cs,lidar_map=grid,confirmed_contacts=contacts,
                  command=jp.array((o['command'][0],o['command'][2],0.,0.,0.)),supervisor=self.supervisor)
        plan=self.geometry._landing_plan(data,info,action)
        self.supervisor=supervisor.SupervisorState(plan['return_time'],plan['wave_time'],plan['decision'])
        mode=int(plan['mode']); desired_mask=int(sum((1<<i)*int(v) for i,v in enumerate(np.asarray(scheduler.swing_mask(mode,phase if mode==o['gait'] else 0)))))
        execute=bool(plan['permit']) and mode==planned_gait and desired_mask==mask
        # Snapshots already contain pre-posture target coordinates; do not add XY/Z residual again.
        legs=tuple(LegPlan(tuple(map(float,target)),float(height),float(apex),float(transfer))
            for target,height,apex,transfer in zip(np.asarray(plan['selected_pre']),np.asarray(plan['selected_clearance']),
                                                  np.asarray(plan['selected_apex_phase']),np.asarray(plan['selected_transfer'])))
        posture=np.asarray(plan['posture']);speed=float(plan['speed_scale'])
        execution=AdaptiveExecutionPlan(o['session_id'],o['sequence'],self.sequence,o['plan_id'],mask,mode,
            float(plan['period']),float(posture[0]),float(posture[1]),float(posture[2]),
            (float(np.clip(o['command'][0]*speed,-.1,.1)),0.,0.,float(np.clip(o['command'][2]*speed,-np.deg2rad(18),np.deg2rad(18)))),legs,execute)
        self.sequence=(self.sequence+1)&0xffff
        return execution,plan

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='NPZ: observation_json, position, body_rotation, com_world, points, valid, time')
    parser.add_argument('--output',type=Path,required=True,help='128-byte command; offline only')
    args=parser.parse_args()
    with np.load(args.input,allow_pickle=False) as data:
        o=json.loads(str(data['observation_json']))
        grid=update_world_returns(initial_map(jp.asarray(data['position'][:2])),data['points'],data['valid'],data['position'],float(data['time']))
        execution,plan=AdaptiveRuntime().prepare(o,grid,data['position'],data['body_rotation'],data['com_world'],float(data['time']))
    args.output.write_bytes(encode_execution(execution))
    print(json.dumps(dict(contract=adaptive.ACTION_CONTRACT,execute=execution.execute,gait=execution.requested_gait,
                         stride=float(plan['stride']),decision=int(plan['decision']),output=str(args.output))))

if __name__=='__main__':
    main()
