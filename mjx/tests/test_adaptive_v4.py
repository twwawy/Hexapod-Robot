"""Fast v4 projection, cross-language trajectory/wire and latch checks. No rollout."""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'mjx'),str(ROOT/'SW/Jetson')]
import jax.numpy as jp
import adaptive_gait_controller as controller
import adaptive_foothold_estimator as estimator
import hybrid_gait_supervisor as supervisor
import wave_gait_scheduler as scheduler
from adaptive_execution_plan import AdaptiveExecutionPlan, LegPlan
from adaptive_spi_protocol import encode_execution, crc

class Vec(c.Structure):
    _fields_=[('x',c.c_float),('y',c.c_float),('z',c.c_float)]

class V4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='hexapod-golden-')
        core=ROOT/'SW/STM32/workspace/Hexapod/Core'
        wrapper=Path(cls.temp.name)/'wrapper.c'
        wrapper.write_text('''#include "communication/adaptive_spi_protocol.h"
#include "high_control/rl_controller.h"
#include "high_control/gait_manager.h"
#include "high_control/foot_trajectory.h"
#include <math.h>
int decode(const unsigned char *f,float *out) {
 RobotAdaptiveExecutionPlan_t p;
 if(!AdaptiveSpi_DecodeExecution(f,&p)||!RlController_ExecutionValuesValid(&p)) return 0;
 out[0]=p.leg[0].landing.x;out[1]=p.leg[0].landing.z;
 out[2]=p.body_height_offset_m;out[3]=p.phase_duration_s;
 RlController_Handle_t h;RlController_BeginSession(&h,p.session_id);
 RlController_SetPlan(&h,p.plan_id,p.swing_mask,true);
 RlController_RecordObservation(&h,p.observation_sequence,100);
 if(RlController_SubmitExecution(&h,&p,110)!=RL_SUBMIT_ACCEPTED) return 0;
 if(RlController_SubmitExecution(&h,&p,111)!=RL_SUBMIT_SEQUENCE) return 0;
 if(RlController_GetExecution(&h,&p,211)) return 0;
 return 1;
}
int transition_validation(void) {
 GaitManager_Handle_t h;GaitManager_Init(&h);bool contact[6]={true,true,true,true,true,true};
 GaitManager_SetAdaptiveTiming(&h,true,1.0f);GaitManager_SetPattern(&h,ROBOT_GAIT_WAVE);
 h.run_enable=true;h.start_wait_count=ROBOT_GAIT_START_DELAY_CYCLES;
 RobotGaitPhase_t g=GaitManager_StepContacts(&h,true,true,false,false,ROBOT_TRIPOD_NORMAL,0,contact,contact);
 if(h.active_pattern!=ROBOT_GAIT_TRIPOD || !g.next_phase_preview || g.next_phase_pattern!=ROBOT_GAIT_WAVE) return 0;
 GaitManager_StepContacts(&h,true,true,false,false,ROBOT_TRIPOD_NORMAL,0,contact,contact);
 if(h.active_pattern!=ROBOT_GAIT_TRIPOD) return 0;
 GaitManager_StepContacts(&h,true,true,true,true,ROBOT_TRIPOD_NORMAL,0,contact,contact);
 return h.active_pattern==ROBOT_GAIT_WAVE && h.initialized;
}
float baseline(int wave) {return wave?ROBOT_ADAPTIVE_WAVE_BASE_S:ROBOT_ADAPTIVE_TRIPOD_BASE_S;}
int duration_latch(void) {
 GaitManager_Handle_t h;GaitManager_Init(&h);bool contact[6]={true,true,true,true,true,true};
 GaitManager_SetAdaptiveTiming(&h,true,.75f);
 for(int i=0;i<40&&!h.initialized;i++) GaitManager_StepContacts(&h,true,true,true,true,ROBOT_TRIPOD_NORMAL,0,contact,contact);
 if(!h.initialized||fabsf(h.active_duration_s-.75f)>1e-6f) return 0;
 GaitManager_SetAdaptiveTiming(&h,true,1.3f);GaitManager_SetPattern(&h,ROBOT_GAIT_WAVE);
 GaitManager_StepContacts(&h,true,true,true,true,ROBOT_TRIPOD_NORMAL,0,contact,contact);
 return fabsf(h.active_duration_s-.75f)<1e-6f && h.active_pattern==ROBOT_GAIT_TRIPOD;
}
''')
        library=Path(cls.temp.name)/'golden.so'
        files=['high_control/swing_trajectory.c','communication/adaptive_spi_protocol.c','high_control/rl_controller.c','high_control/gait_manager.c']
        subprocess.run(['gcc','-shared','-fPIC','-std=c11','-O1','-I'+str(core/'Inc'),str(wrapper),
                        *[str(core/'Src'/f) for f in files],'-lm','-o',str(library)],check=True)
        cls.lib=c.CDLL(str(library))
        cls.lib.SwingTrajectory_CalculateAdaptive.argtypes=[c.c_float,c.POINTER(Vec),c.POINTER(Vec),c.c_float,c.c_float,c.c_float]
        cls.lib.SwingTrajectory_CalculateAdaptive.restype=Vec
        cls.lib.baseline.argtypes=[c.c_int];cls.lib.baseline.restype=c.c_float
    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_cross_language_swing_golden(self):
        start=np.tile([.22,.08,-.287],(6,1)); end=start+np.array([.06,-.04,.08])
        for apex,transfer in ((.3,.35),(.4,.55),(.5,.5),(.7,.65)):
            for progress in (0,.1,.25,.4,.5,.75,.9,1):
                expected=np.asarray(controller.planned_swing(progress,jp.asarray(start),jp.asarray(end),.08,apex,transfer))[0]
                actual=self.lib.SwingTrajectory_CalculateAdaptive(progress,c.byref(Vec(*start[0])),c.byref(Vec(*end[0])),.08,apex,transfer)
                np.testing.assert_allclose([actual.x,actual.y,actual.z],expected,atol=3e-7)

    def test_decoder_and_final_projection_extremes(self):
        reference=jp.zeros((6,3));xy=estimator.local_candidate_xy(reference,jp.eye(2))
        safe=jp.ones((6,25),dtype=bool)
        for axis,extent in ((0,.06),(1,.04)):
            for sign in (-1,1):
                action=jp.zeros(24).at[axis].set(sign)
                residual=controller.decode(action)[0]
                self.assertAlmostEqual(float(residual[0,axis]),sign*extent,places=6)
                index,valid=estimator.project_local_candidates(xy,safe,residual)
                self.assertTrue(bool(valid[0]))
                self.assertAlmostEqual(float(xy[0,index[0],axis]),sign*extent,places=6)

    def test_real_geometry_projection_and_z_ownership(self):
        from adaptive_runtime import HardwareGeometry, PlannerData
        from adaptive_gait_perception import initial_map
        import firmware_mjx_controller as fw
        cs=controller.initial_state()
        origin=jp.array((0.,0.,-fw.BASE_FOOT_Z+.032))
        model=jp.stack((fw.BASE_FEET[:,1],-fw.BASE_FEET[:,0],fw.BASE_FEET[:,2]),axis=-1)
        feet=origin+model
        data=PlannerData(jp.concatenate((origin,jp.array((1.,0.,0.,0.)))),jp.eye(3)[None],feet,origin[None],jp.asarray(1.))
        grid=initial_map(jp.zeros(2))._replace(timestamp=jp.ones((64,64)))
        geometry=HardwareGeometry();info=dict(controller_state=cs,lidar_map=grid)
        ref=feet.at[:,2].add(-.032);basis=jp.array(((0.,-1.),(1.,0.)))
        xy=estimator.local_candidate_xy(ref,basis)
        plan=estimator.evaluate_candidates(geometry,data,info,xy,feet,basis,jp.zeros(6))
        # At least one actual reachable leg must exercise each extreme, not decoder-only.
        for axis,extent in ((0,.06),(1,.04)):
            for sign in (-1,1):
                request=ref[:,:2]+jp.tile(jp.eye(2)[axis]*sign*extent,(6,1))@basis
                index,valid=estimator.project_local_candidates(xy,plan['safe'],request)
                delta=(xy[jp.arange(6),jp.maximum(index,0)]-ref[:,:2])@jp.linalg.inv(basis)
                self.assertTrue(bool(jp.any(valid & (jp.abs(delta[:,axis]-sign*extent)<1e-6))))
        changed=estimator.evaluate_candidates(geometry,data,info,xy,feet,basis,jp.full(6,.04),apex_delta=.1,transfer_delta=-.1)
        np.testing.assert_allclose(plan['world'][...,2],changed['world'][...,2])
        raised=estimator.evaluate_candidates(geometry,data,dict(info,lidar_map=grid._replace(height=grid.height+.01)),xy,feet,basis,jp.zeros(6))
        np.testing.assert_allclose(raised['world'][...,2],.01,atol=1e-6)

    def test_wire_c_validation_and_timeout(self):
        plan=AdaptiveExecutionPlan(42,7,9,11,0x15,0,.75,.08,-.10,.02,(.05,0,0,.02),
            tuple(LegPlan((.26,.10,-.24),.08,.4,.55) for _ in range(6)))
        wire=encode_execution(plan);out=(c.c_float*4)()
        self.assertEqual(self.lib.decode((c.c_ubyte*128).from_buffer_copy(wire),out),1)
        np.testing.assert_allclose(list(out),[.26,-.24,.02,.75],atol=1e-6)
        damaged=bytearray(wire);damaged[40]^=1
        self.assertEqual(self.lib.decode((c.c_ubyte*128).from_buffer_copy(damaged),out),0)

    def test_phase_baselines_duration_latch(self):
        for mode in (0,1):
            self.assertEqual(self.lib.baseline(mode),1.)
            self.assertEqual(float(supervisor.phase_duration(1.,mode)),1.)
        self.assertEqual(self.lib.duration_latch(),1)
        self.assertEqual(self.lib.transition_validation(),1)

if __name__=='__main__':
    unittest.main()
