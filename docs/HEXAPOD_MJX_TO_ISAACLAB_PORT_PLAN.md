# Hexapod MJX → Isaac Sim / Isaac Lab parity-first 이식 계획

작성일: 2026-08-27 (Asia/Seoul)  
대상 저장소: `/home/huro/Hexapod-Robot`  
대상 브랜치: `codex/cartesian-residual-rl`  
계획 작성 시 원격 최신 commit: `b63e296`  
로컬 Isaac Lab: `/home/huro/IsaacLab`, `VERSION=2.3.0`, checkout `cbf51abb5e`  
로컬 Isaac Sim: 5.1 계열 설치 확인  

> 최종 목표: MJX의 firmware-residual 환경을 Isaac Lab `DirectRLEnv`로 옮기되, RL을 붙이기 전에 asset, controller, observation, reward, termination의 계약을 각각 검증한다. “비슷하게 걷는다”는 합격 기준이 아니다.

---

## 1. 절대 원칙

1. **현재 MJX를 먼저 동결한다.** Source commit, generated scene hash, config, fixed seed, action sequence를 golden manifest에 기록한다.
2. **Isaac Lab upstream은 수정하지 않는다.** `/home/huro/IsaacLab`은 이미 dirty checkout이다. Hexapod는 repository 안의 외부 extension으로 만든다.
3. **CAD collision을 학습에 사용하지 않는다.** Visual mesh와 primitive training collision을 분리한다.
4. **firmware gait/IK/state machine을 재설계하지 않는다.** `firmware_mjx_controller.py`의 수식, 상태, 연산 순서를 Torch로 옮긴다.
5. **parity 전에 RL과 DR을 켜지 않는다.** Randomization은 모델링 오류를 숨긴다.
6. **현재 동작의 문제도 우선 복제한다.** 현재 ramp/stairs pitch와 pitch feedforward는 음수이고 roll command는 0이다. 뒤로 기울이기/roll curriculum 개선은 parity 후 새 contract version으로 한다.

### 금지 사항

- parity 전 PPO 또는 domain randomization 실행
- generic height scanner 결과를 15-D terrain observation에 연결
- Isaac articulation 내부 joint 순서를 신뢰
- `robot.data.root_vel_w`를 그대로 사용
- imported CAD collision과 primitive collision을 중복 사용
- 이식 중 reward, command sign, collision mode를 임의 수정
- 긴 closed-loop trajectory 차이만으로 controller parity 실패 판정

---

## 2. Source of truth

| 영역 | 파일 |
|---|---|
| URDF 정리 | `mjx/prepare_urdf.py` |
| MuJoCo root/home/joint actuator | `mjx/prepare_scene.py` |
| 최종 mass/inertia/collision | `mjx/prepare_rl_scene.py`, `mjx/generated/hexapod_rl.xml` |
| firmware 수식/state machine/IK | `mjx/firmware_mjx_controller.py` |
| env 순서/obs/reward/done | `mjx/rough_terrain_env.py` |
| terrain level | `mjx/terrain_curriculum.py` |
| actuator 상수 | `mjx/servo_model.py` |
| DR 범위 | `mjx/domain_randomization.py` |
| curriculum promotion | `mjx/train_competence_curriculum.py` |

Golden manifest에 반드시 저장할 값:

```text
git commit 및 dirty status
Python/JAX/MuJoCo/MJX versions
Isaac Lab commit/version 및 dirty status
Isaac Sim version
hexapod.urdf / hexapod_rl.xml SHA256
golden schema version
seed / terrain level / DR / collision mode / action source
```

---

## 3. 고정할 MJX 계약

### 3.1 시간과 step 순서

| 항목 | 값 |
|---|---:|
| physics dt | `0.0025 s` |
| policy/environment dt | `0.0200 s` |
| Isaac `decimation` | `8` |
| firmware dt | `0.0050 s` |
| firmware ticks / policy step | `4` |
| physics steps / policy step | `8` |
| action | `18-D` |
| observation | `146-D` |

현재 MJX의 한 step:

```text
pre-step physics state/contact
→ action clip 및 action-delay
→ pitch_ff / swing_boost
→ 같은 pre-step pose/contact로 firmware 5 ms tick × 4
→ 마지막 firmware q_des
→ q_des 유지, physics 2.5 ms step × 8
→ post-step contact/observation/reward/done
```

Isaac Lab에서는 `_pre_physics_step()`에서 firmware tick 4회와 q_des 계산까지 하고, `_apply_action()`은 같은 q_des를 8 physics step 동안 유지한다. `_apply_action()`에서 firmware state를 진행하면 안 된다.

### 3.2 contract와 canonical order

```text
ACTION_CONTRACT_VERSION = stm32_firmware_adaptive_swing_residual_v3
OBSERVATION_CONTRACT_VERSION = firmware_state_collision_terrain_command5_pitch_v3
```

Canonical leg/joint/action order:

```text
RB1 RB2 RB3 RM1 RM2 RM3 RF1 RF2 RF3
LB1 LB2 LB3 LM1 LM2 LM3 LF1 LF2 LF3
```

즉 leg order는 `RB → RM → RF → LB → LM → LF`, 각 leg 내부는 `1 → 2 → 3`이다. Legacy `tripod_controller.py`의 선언 순서가 다르게 보이더라도 Torch array 계약은 `firmware_mjx_controller.py`의 `LEG_ROOTS`, `LEG_ANGLES`, `MODEL_SIGNS` 순서를 따른다.

### 3.3 좌표계와 home

```text
MODEL_FORWARD = (0, -1, 0)
MODEL_LATERAL = (1, 0, 0)
home root position = (0, 0, 0.287006)
home root quaternion wxyz = (sqrt(0.5), 0, 0, sqrt(0.5))
```

CAD/model frame이 world z축으로 +90° 회전되어 model-local `-Y`가 world `+X` 전진이다.

Isaac Lab에서는 명시적으로 다음 property를 사용한다.

```python
robot.data.root_link_pos_w
robot.data.root_link_quat_w       # wxyz
robot.data.root_link_lin_vel_w
robot.data.root_link_ang_vel_w
```

Isaac Lab 2.3의 `root_vel_w`, `root_lin_vel_w`, `root_ang_vel_w`는 COM velocity alias이므로 MJX root-link parity에 사용하지 않는다.

### 3.4 raw joint sign / axis

Firmware servo angle → model joint angle:

```text
RB/RM/RF: (q1, -q2, +q3)
LB/LM/LF: (q1, +q2, -q3)
```

Servo home은 모든 leg `(0°, 30°, 50°)`, model home은 right `(0°, -30°, +50°)`, left `(0°, +30°, -50°)`다.

최종 MJCF joint axis:

| Joint | axis |
|---|---|
| 모든 `_1` | `(0, 0, 1)` |
| RB_2 / RB_3 | `(+.707,+.707,0)` / `(-.707,-.707,0)` |
| RM_2 / RM_3 | `(0,+1,0)` / `(0,-1,0)` |
| RF_2 / RF_3 | `(-.707,+.707,0)` / `(+.707,-.707,0)` |
| LB_2 / LB_3 | `(-.707,+.707,0)` / `(+.707,-.707,0)` |
| LM_2 / LM_3 | `(0,+1,0)` / `(0,-1,0)` |
| LF_2 / LF_3 | `(+.707,+.707,0)` / `(-.707,-.707,0)` |

USD import 후 각 joint를 `+0.1 rad` 움직이는 finite-motion test로 child link 방향까지 검증한다.

### 3.5 actuator/dynamics

| 항목 | 값 |
|---|---:|
| total robot mass | `10.0 kg` |
| joint limit | `±2.356194 rad` |
| firmware joint rate | `315.8 deg/s` |
| Kp / Kd | `500 / 10` |
| torque limit | `±14.709975 Nm` |
| armature | `0.02 kg·m²` |
| joint viscous damping | `0.15 Nms/rad` |
| gearbox Coulomb friction | `0.8 Nm` |

Isaac Lab 2.3 / Isaac Sim 5.1 시작 설정:

```python
ImplicitActuatorCfg(
    joint_names_expr=JOINT_ORDER,
    stiffness=500.0,
    damping=10.0,
    effort_limit_sim=14.709975,
    armature=0.02,
    friction=0.8,
    dynamic_friction=0.8,
    viscous_friction=0.15,
)
```

Drive Kd 10과 joint viscous damping 0.15는 별도다. `SERVO_NO_LOAD_SPEED`는 metadata에만 있고 현재 MJX의 hard velocity clamp가 아니므로 parity 중 Isaac에 새 hard limit을 추가하지 않는다.

### 3.6 collision

- torso box MJCF half extent `(0.17, 0.15, 0.045)`; Isaac full size `(0.34, 0.30, 0.09)`
- coxa capsule radius `0.028 m`
- femur capsule radius `0.026 m`
- tibia capsule radius `0.023 m`
- foot sphere radius `0.032 m`

Capsule endpoint/local pose는 `hexapod_rl.xml`에서 자동 추출한다. 기본 `collision_mode=lower_leg`:

```text
terrain collision enabled: torso, tibia, foot
terrain collision disabled: coxa, femur
self collision: disabled
firmware contact: foot sphere ↔ world/terrain contact만 leg bool로 축약
```

### 3.7 146-D observation slice

| Slice | Dim | 내용 | 변환 |
|---|---:|---|---|
| `[0:5]` | 5 | command `[vx,wz,height,pitch,roll]` | 그대로 |
| `[5:8]` | 3 | root local linear velocity | inverse rotate |
| `[8:11]` | 3 | root world angular velocity | `×0.2` |
| `[11:14]` | 3 | local gravity | inverse rotate `(0,0,-1)` |
| `[14:16]` | 2 | relative roll, pitch | home quaternion 기준 |
| `[16:34]` | 18 | joint position error | `q-home_q` |
| `[34:52]` | 18 | joint velocity | `×0.1` |
| `[52:70]` | 18 | foot positions in controller body frame | `(6,3)` flatten |
| `[70:76]` | 6 | foot contact | float32 |
| `[76:91]` | 15 | terrain relative height | fixed 5×3 |
| `[91:97]` | 6 | gait progress | 그대로 |
| `[97:103]` | 6 | gait state | `state/2` |
| `[103:107]` | 4 | applied twist | 그대로 |
| `[107:113]` | 6 | IK valid | float32 |
| `[113:119]` | 6 | policy valid | float32 |
| `[119:125]` | 6 | foot limited | float32 |
| `[125:127]` | 2 | gait/posture accepted | float32 |
| `[127:145]` | 18 | last applied action | delay 후 action |
| `[145:146]` | 1 | pitch feedforward | rad |

Foot frame 변환:

```text
relative_world = foot_link_pos_w - root_link_pos_w
model_body = inverse_rotate(root_link_quat_w, relative_world)
controller_foot = [dot(model_body, MODEL_FORWARD),
                   dot(model_body, MODEL_LATERAL),
                   model_body.z]
```

### 3.8 15-point terrain ray

Controller sample 좌표:

```text
forward = (-0.10, 0.15, 0.40, 0.65, 0.90)
lateral = (-0.24, 0.00, 0.24)
```

정확한 순서:

```text
(-.10,-.24), (.15,-.24), (.40,-.24), (.65,-.24), (.90,-.24),
(-.10, .00), (.15, .00), (.40, .00), (.65, .00), (.90, .00),
(-.10, .24), (.15, .24), (.40, .24), (.65, .24), (.90, .24)
```

RayCaster model-local start는 `model_x=lateral`, `model_y=-forward`다. Generic grid 대신 custom `Hexapod15PointPatternCfg`를 만들고 `ray_alignment="yaw"`를 사용한다. Forward 9-point subset은 `forward >= 0.40`인 ray다.

### 3.9 reward

| term | weight | term | weight |
|---|---:|---|---:|
| velocity | `+2.5` | yaw | `+0.25` |
| upright | `+1.0` | height | `+0.6` |
| progress | `+1.0` | stability | `+0.5` |
| joint_margin | `+1.0` | action_rate | `-0.02` |
| residual | `-0.005` | swing_height | `-0.10` |
| early_swing_contact | `-1.0` | vertical_velocity | `-0.10` |
| lateral_velocity | `-0.10` | joint_velocity | `-0.04` |
| torque | `-0.03` | torque_saturation | `-0.25` |
| gait_rejected | `-0.50` | posture_rejected | `-0.50` |
| policy_rejected | `-2.0` | foot_limited | `-2.0` |
| body_contact | `-2.0` | self_collision | `-0.5` |
| foot_clearance_terrain | `-2.0` | edge_margin | `-0.50` |
| touchdown_impact | `-0.05` |  |  |

```text
running_reward = sum(scaled_terms) * 0.02
reward = clip(running_reward + ascent_bonus + success_bonus + failure_penalty,
              -50, +50)
ascent coefficient = 8.0
success bonus = +30.0
failure penalty = -30.0
```

Raw term과 scaled term을 모두 구현/기록한다. Torque는 우선 `robot.data.applied_torque`를 사용하되 implicit actuator의 실제 clipped solver torque인지 audit한다.

### 3.10 done/success

Failure OR:

```text
any IK invalid
joint margin <= 1 deg
root linear speed > 1.5 m/s
root angular speed > 6 rad/s
max joint speed > 20 rad/s
abs(roll or pitch) > 45 deg
clearance < 0.14 m
torso contact
non-finite q/qd/q_des
```

Success AND:

```text
root world x >= terrain goal_x
final support height reached (ramp/stairs)
roll/pitch target error < 12 deg
not failure
```

Failure/success는 `terminated`, episode length는 `truncated`로 분리한다.

### 3.11 현재 command 의미 — 이식 중 변경 금지

```text
flat/rough: height=0, pitch=0, roll=0
ramp: height∈[-0.05,0], pitch=-slope, roll=0
stairs: height∈[-0.05,0], pitch∈[-25°,-5°], roll=0
terrain-rise pitch_ff: negative
```

현재 주석 기준 음수 pitch는 uphill forward lean이다. Backward lean/roll 학습은 parity 후 observation/action contract version을 올리고 MJX golden부터 다시 만든다.

---

## 4. 목표 repository 구조

```text
/home/huro/Hexapod-Robot/
├── mjx/
│   ├── export_isaac_contract.py
│   ├── export_asset_manifest.py
│   └── contracts/{schema_v1.json,golden/flat_seed0_v1.{json,npz}}
└── isaaclab_hexapod/
    ├── README.md
    ├── pyproject.toml
    ├── config/extension.toml
    ├── scripts/
    │   ├── build_asset.py
    │   ├── inspect_asset.py
    │   ├── replay_golden.py
    │   ├── parity_report.py
    │   ├── zero_action.py
    │   └── rsl_rl/{train.py,play.py}
    ├── data/{asset_manifest_v1.json,generated/,usd/}
    ├── docs/parity_report.md
    └── source/hexapod_isaaclab/hexapod_isaaclab/
        ├── assets/{hexapod_asset_cfg.py,joint_contract.py}
        ├── controllers/{firmware_controller_torch.py,firmware_types.py}
        ├── contracts/{observation_contract.py,reward_contract.py,frame_contract.py}
        ├── terrains/{hexapod_terrain_cfg.py,hexapod_terrain_generator.py,ray_pattern.py}
        └── tasks/direct/hexapod/{hexapod_env_cfg.py,hexapod_env.py,agents/}
```

Isaac Lab 내부에는 파일을 추가하지 않는다.

## 5. Phase 0 — MJX golden contract 생성

### 5.1 다음 세션 첫 명령

```bash
cd /home/huro/Hexapod-Robot
git pull origin codex/cartesian-residual-rl
git rev-parse HEAD
git status --short
sha256sum mjx/generated/hexapod.urdf mjx/generated/hexapod_rl.xml
cat /home/huro/IsaacLab/VERSION
git -C /home/huro/IsaacLab rev-parse HEAD
git -C /home/huro/IsaacLab status --short
```

Isaac Lab dirty tree는 정리하지 않고 상태만 manifest에 기록한다.

### 5.2 `export_isaac_contract.py`

첫 golden:

```text
schema = hexapod_mjx_transition_v1
seed = 0
terrain_level = 0
DR = off
collision_mode = lower_leg
duration = 10 s = 500 policy steps
action source = deterministic scripted action
```

Exporter가 seed로 bounded action sequence를 만들고 NPZ에 저장한다. 시작 1초는 zero action, 이후에는 leg별 XYZ가 모두 활성화되는 낮은 진폭의 deterministic sequence를 사용한다. Isaac은 policy를 실행하지 않고 저장 action을 replay한다. 별도 zero-action golden은 추가 가능하지만 residual path를 자극하는 golden이 기본 정답지다.

각 transition `k`:

```text
t[k]
pre/root_pose, pre/root_velocity, pre/joint_q, pre/joint_qd, pre/contact
command[k]
action_requested[k]
action_applied[k]
firmware_tick/state/*[k,4,...]
firmware_tick/output/*[k,4,...]
q_des[k,18]
post/root_pose, post/root_velocity, post/joint_q, post/joint_qd, post/contact
observation_pre[k,146]
observation_post[k,146]
reward_raw/*[k]
reward_scaled/*[k]
reward_ascent/success/failure[k]
reward_total[k]
done[k]
```

FirmwareState 전 field:

```text
first_step, throttle_filter, yaw_filter, yaw_reference,
position_reference, previous_twist, gait_applied,
phase_index, phase_time, airborne_seen, landed,
gait_initialized, gait_running, stop_pending,
foot_memory, swing_start, previous_leg_state,
adapted_stance, custom_swing, posture_command,
last_ik, previous_joint, residual_filter
```

FirmwareOutput 전 field:

```text
model_joint_targets, servo_joint_targets, foot_targets_body,
applied_twist, gait_progress, gait_state, swing_height_command,
ik_valid, policy_valid, foot_limited,
gait_enabled, gait_accepted, posture_accepted
```

추가 debug:

```text
support_height, terrain_15, pitch_ff, swing_boost,
foot_world, foot_controller_body,
torso_contact, self_collision,
actuator_force, joint_limit_margin
```

Boundary 규칙:

- `observation_pre[k]`는 requested action 적용 전 observation이다.
- `action_applied[k]`는 clip/action delay 후 firmware에 실제 전달된 action이다.
- firmware 4 tick은 같은 pre-step pose/contact를 받는다.
- `q_des[k]`는 4번째 tick의 `model_joint_targets`다.
- post observation/reward/done은 8 physics substeps 뒤 값이다.
- float32/int32/bool을 유지하고 JSON 경유 float64 승격을 금지한다.
- quaternion은 wxyz다.

### 5.3 `export_asset_manifest.py`

`prepare_rl_scene()` 후 최종 `MjModel`에서 직접 JSON으로 저장:

```text
body name/parent/local pose
mass/inertial COM pos/inertial quat/diagonal inertia
joint name/parent-child/axis/limit/qposadr/dofadr
actuator name/joint/Kp/Kd/force range
collision name/body/type/local pose/dimensions/friction/mask
site name/body/local position
home root pose/joint q/actuator target
total robot mass
```

URDF inertia를 다시 계산하지 않는다. 10 kg로 rescale된 최종 MjModel이 source of truth다.

### 5.4 Phase 0 gate

- [ ] manifest에 source/config/hash가 있음
- [ ] NPZ row=500, schema key/shape 일치
- [ ] obs=146, action/q_des=18, firmware tick=4
- [ ] NaN/Inf 없음
- [ ] 동일 환경 재-export 시 command/action/controller arrays 동일
- [ ] 아직 Isaac/RL 실행 안 함

---

## 6. Phase 1 — External Isaac Lab project scaffold

`/home/huro/IsaacLab/isaaclab.sh -n`의 External project template을 사용하거나 같은 구조를 수동 생성한다.

```text
project path = /home/huro/Hexapod-Robot/isaaclab_hexapod
package = hexapod_isaaclab
workflow = Direct
```

Editable install:

```bash
/home/huro/IsaacLab/isaaclab.sh -p -m pip install -e \
  /home/huro/Hexapod-Robot/isaaclab_hexapod/source/hexapod_isaaclab
```

RSL-RL scaffold는 만들 수 있으나 Phase 7 전에는 실행하지 않는다. Phase 1 gate는 extension import다.

---

## 7. Phase 2 — URDF → USD asset parity

### 7.1 변환 전략

1. `prepare_urdf.py`로 plain URDF 생성
2. 별도 script에서 CAD `<collision>`을 제거한 visual-only URDF 생성
3. `UrdfConverterCfg`로 USD 변환
4. asset manifest의 primitive collision 추가
5. final MJX mass/COM/inertia를 rigid body별 적용
6. joint drive/dynamics/filtering 적용

권장 converter 설정:

```python
UrdfConverterCfg(
    asset_path=visual_only_urdf,
    usd_dir=...,
    usd_file_name="hexapod_mjx_parity.usd",
    fix_base=False,
    merge_fixed_joints=True,
    force_usd_conversion=True,
    collision_from_visuals=False,
    self_collision=False,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=0.0,
            damping=0.0,
        ),
    ),
)
```

최종 gain은 importer default가 아니라 `ArticulationCfg.actuators`에서 적용한다.

### 7.2 primitive와 filtering

- Capsule은 MJCF `fromto`에서 midpoint, length, orientation을 계산한다.
- MJCF capsule segment와 PhysX capsule height 정의 차이를 확인한다.
- box half/full extent를 혼동하지 않는다.
- `activate_contact_sensors=True`를 robot spawn에 설정한다.
- self collision off
- torso/tibia/foot terrain collision on
- coxa/femur terrain collision off

### 7.3 canonical index

```python
JOINT_ORDER = [
    "RB_1", "RB_2", "RB_3",
    "RM_1", "RM_2", "RM_3",
    "RF_1", "RF_2", "RF_3",
    "LB_1", "LB_2", "LB_3",
    "LM_1", "LM_2", "LM_3",
    "LF_1", "LF_2", "LF_3",
]
joint_ids, names = robot.find_joints(JOINT_ORDER, preserve_order=True)
assert names == JOINT_ORDER
```

이후 모든 gather/scatter는 `joint_ids`를 사용한다.

### 7.4 Phase 2 tests

- [ ] moving joint 정확히 18개
- [ ] canonical name/order 정확
- [ ] finite-motion axis direction 일치
- [ ] limit ±2.356194
- [ ] side별 home sign 일치
- [ ] root home quaternion/height 일치
- [ ] body별 mass/COM/inertia 및 총 10 kg 일치
- [ ] CAD collision prim 0개
- [ ] primitive type/radius/count 일치
- [ ] lower-leg filtering/self-collision off
- [ ] Kp/Kd/effort/armature/friction/viscous friction 적용

이 단계에서 RL을 실행하지 않는다.

---

## 8. Phase 3 — `firmware_controller_torch.py` 1:1 포팅

### 8.1 구현 규칙

- `torch.float32`, `torch.int32`, `torch.bool` 명시
- state tensor 첫 dimension=`num_envs`
- JAX `where`의 shape/dtype 유지
- angle wrap/clip/epsilon/rounding 그대로 유지
- C `lroundf` half-away-from-zero를 Python `round`로 교체 금지
- IK invalid fallback → last valid → joint rate limit 순서 유지
- gait/contact/IK를 Isaac API로 재작성 금지

고정 상수:

```text
firmware dt=.005, gait phase=.5
max linear=.28, max yaw=45 deg/s
swing nominal/min/max=.06/.04/.25
swing radial=.07
early landing=.50
late landing/inward=.20/.16
joint limit=135 deg, joint rate=315.8 deg/s
links=.074/.121/.230
root distance=.1845, foot radius=.218728, foot z=-.287006
workspace margin=.001
residual scale=(.04,.02,.02)
height offset max=.10
```

### 8.2 controller-only parity

Golden의 매 5 ms input을 Torch에 직접 넣어 모든 FirmwareState/FirmwareOutput field를 비교한다. Isaac physics는 사용하지 않는다.

| 종류 | tolerance |
|---|---|
| bool/int | exact |
| command/action | exact |
| timer/gait progress | `atol=1e-7` |
| foot target/IK/q_des | `atol=2e-6`, `rtol=1e-6` |
| 기타 float state | `atol=1e-6`, `rtol=1e-6` |

CPU float32와 CUDA float32 둘 다 검사한다. 실패 시 DirectRLEnv 작업으로 넘어가지 않는다.

---

## 9. Phase 4 — Flat `DirectRLEnv`

### 9.1 config

```python
@configclass
class HexapodFlatEnvCfg(DirectRLEnvCfg):
    decimation = 8
    action_space = 18
    observation_space = 146
    state_space = 0
    sim = SimulationCfg(dt=0.0025, render_interval=8)
```

처음에는 `num_envs=1`, `replicate_physics=False`, DR off. 통과 후 32 env로 확장한다.

### 9.2 `_pre_physics_step()`

1. requested action clone/clip
2. action delay buffer와 applied action
3. `root_link_*`, canonical joint q/qd, foot contact 읽기
4. controller frame attitude/velocity
5. pitch_ff/swing_boost
6. command delay
7. 같은 pose/contact로 Torch firmware 4회
8. 마지막 `model_joint_targets`를 canonical q_des로 저장

### 9.3 `_apply_action()`

```python
robot.set_joint_position_target(q_des, joint_ids=joint_ids)
```

Firmware state를 여기서 진행하지 않는다.

### 9.4 contact와 첫 실행

Foot contact sensor body도 canonical leg order로 gather한다. Firmware bool은 foot sphere와 terrain contact만 포함한다. Isaac 초기 bool은 `net_forces_w.norm > 1e-6`로 만들고 raw force를 함께 기록한다. Tibia contact는 physics에는 영향을 주지만 firmware contact bool에는 포함하지 않는다.

실행 순서:

1. spawn 후 physics 없이 pose inspection
2. zero command/action 1 step
3. forward command + zero action 500 steps
4. golden action replay 500 steps

Policy checkpoint는 사용하지 않는다.

---

## 10. Phase 5 — Observation / reward / done parity

Pure Torch 함수로 분리:

```text
build_observation
compute_reward_terms
scale_reward_terms
compute_failures
compute_success
terrain_pitch_ff
swing_boost
```

Golden state tensor를 직접 넣어 physics 없이 먼저 비교한다.

MJX foot vertical velocity는 site Jacobian 결과다. Isaac에서는 link origin velocity를 그대로 쓰지 않고 `v_foot = v_link_origin + omega × r_endpoint`로 foot sphere center velocity를 계산한다.

Parity report를 분리한다.

### A. Functional replay

동일 tensor input으로 controller, observation slice, raw/scaled reward, done reason을 strict 비교한다.

### B. Closed-loop physics

Solver가 다르므로 다음을 보고한다.

```text
1/5/10/25/50 policy step root/joint error
first contact sequence divergence
first done divergence
10 s displacement/support height/mean velocity
reward term별 누적합
gait rejection/policy rejection/foot limited rate
```

초기 gate:

| 비교 | tolerance |
|---|---|
| controller | `2e-6` abs |
| observation functional | `2e-5` abs |
| reward functional | `2e-5` abs |
| bool/int/done reason | exact |
| mass/inertia | relative `1e-6` |
| one-step joint q / qd | `1e-3 rad` / `1e-2 rad/s` |
| one-step root pos/attitude | `1e-3 m` / `1e-3 rad` |

Tolerance 완화는 원인과 before/after를 `docs/parity_report.md`에 남긴다.

---

## 11. Phase 6 — Terrain과 15-point height contract

Functional parity와 flat closed-loop gate를 모두 통과한 뒤에만 terrain을 붙인다.

### 11.1 MJX level을 그대로 유지

| level | type | parameter |
|---:|---|---|
| 0 | flat | 0 |
| 1 | rough | max height 0.025 m |
| 2 | rough | max height 0.050 m |
| 3 | ramp | 8 deg |
| 4 | ramp | 15 deg |
| 5 | stairs | 7 steps, riser 0.05 m |
| 6 | stairs | 7 steps, riser 0.10 m |
| 7 | stairs | 7 steps, riser 0.15 m |
| 8 | stairs | 7 steps, riser 0.20 m |
| 9 | stairs | 10 steps, riser 0.05 m |
| 10 | stairs | 10 steps, riser 0.10 m |
| 11 | stairs | 10 steps, riser 0.15 m |
| 12 | stairs | 10 steps, riser 0.20 m |

공통 geometry도 그대로 고정한다.

```text
terrain start x     = 0.55 m
half width          = 0.60 m
rough tiles         = 8 x 4
rough tile depth    = 0.22 m
rough total length  = 1.76 m
ramp length         = 1.20 m
stair tread         = 0.25 m
top plateau         = 0.50 m
```

`TerrainGeneratorCfg`에 level별 sub-terrain cfg를 만들되, generator가 만드는 모양이 위 수치와 달라지면 MJX geometry를 직접 mesh로 생성한다. 이름이 같은 Isaac 기본 terrain을 대충 대응시키지 않는다. seed와 terrain parameter를 episode metadata에 남긴다.

### 11.2 RayCaster의 정확한 15개 ray

generic grid scanner 대신 custom pattern을 만든다.

```python
forward = (-0.10, 0.15, 0.40, 0.65, 0.90)
lateral = (-0.24, 0.00, 0.24)

# observation order: lateral outer, forward inner
points = [(lat, -fwd, 0.0) for lat in lateral for fwd in forward]
```

위 `(model_x, model_y) = (lateral, -forward)` 매핑과 `ray_alignment="yaw"`를 고정한다. RayCast origin height, hit height, root reference height를 각각 로그에 남겨 부호와 offset을 분리해 확인한다. `forward >= 0.40`인 9개 ray가 현재 forward-terrain 집계 대상이다.

### 11.3 Terrain 검증 순서

level을 섞지 않고 다음 순서로 한 단계씩 gate한다.

```text
flat -> rough 0.025 -> rough 0.050 -> ramp 8 -> ramp 15
     -> 7-step stairs 0.05/0.10/0.15/0.20
     -> 10-step stairs 0.05/0.10/0.15/0.20
```

각 level에서 확인할 것은 geometry snapshot, 15 ray 값, support height, foot clearance, terrain pitch feedforward, reward/done이다. 이 단계에서도 PPO와 DR은 끈다.

주의: 현재 command contract는 ramp에서 `height in [-0.05, 0]`, `pitch=-slope`, stairs에서 `height in [-0.05, 0]`, `pitch in [-25,-5] deg`, roll=0이며 uphill pitch feedforward도 음수다. 이것이 사용자가 기대하는 “뒤로 기울기”와 직관적으로 다르더라도 이식 중에 바꾸지 않는다. 먼저 현재 MJX와 parity를 만든 뒤 별도의 contract version에서 명령 부호와 범위를 바꾸고 golden을 다시 생성한다.

---

## 12. Phase 7 — PPO와 curriculum 연결

모든 parity gate 이후에만 `DirectRLEnv`를 학습 환경으로 노출한다.

### 12.1 첫 PPO

1. flat, DR off, zero-init/new policy로 작은 학습 실행
2. MJX teacher checkpoint를 직접 Isaac policy에 끼우지 말고 observation/action contract가 같은지 먼저 확인
3. 동일 network shape이면 checkpoint converter를 별도 도구로 만들고 tensor name/shape/checksum manifest를 저장
4. teacher-student distillation이 필요하면 frozen teacher output과 student output을 canonical action order에서 비교
5. flat 통과 후 rough, ramp, stairs 순서로 curriculum 개방

MJX의 competence progression 의미를 유지한다.

```text
stage budget       = 명시한 timesteps
promote threshold  = success-rate 기준
checkpoint select  = best-safe 우선, 정의를 metadata에 고정
level              = stage 시작 시 고정
resume             = checkpoint + optimizer + curriculum state 모두 복원
```

Isaac 학습 기본 실행 예시는 구현 후 다음 형태로 제공한다.

```bash
cd /home/huro/Hexapod-Robot/isaaclab_hexapod
/home/huro/IsaacLab/isaaclab.sh -p scripts/train.py \
  --task Hexapod-Firmware-Flat-Direct-v0 \
  --num_envs 2048 \
  --seed 8 \
  --headless \
  --wandb_project hexapod-isaac-ppo
```

실제 flag는 구현된 runner config와 일치하게 문서화하며, 위 명령을 미리 존재하는 인터페이스로 간주하지 않는다.

### 12.2 영상과 W&B

현재 운영 요구를 그대로 따른다.

- stage가 끝날 때 그 stage의 best-safe checkpoint 영상 하나만 저장한다.
- progress/eval마다 영상 생성하지 않는다.
- parity는 `hexapod-isaac-parity`, 학습은 새 project `hexapod-isaac-ppo`를 기본값으로 한다.
- W&B 핵심 chart만 유지한다: reward total, success/failure, level/stage, forward velocity/command error, upright/height, policy rejection, foot limited, body contact, terrain clearance, PPO loss/entropy/KL, throughput.
- reward term 전체와 debug tensor는 artifact/local parquet에 저장하되 dashboard 기본 panel로 자동 생성하지 않는다.

---

## 13. Phase 8 — Domain randomization

flat부터 stairs까지 deterministic parity와 PPO baseline이 확보된 뒤 켠다. 한 번에 모두 켜지 않고 아래 순서로 누적한다.

| order | randomization | MJX range/behavior |
|---:|---|---|
| 1 | friction | `[0.40, 1.25]` |
| 2 | body mass + inertia | 같은 body multiplier `[0.80, 1.20]` |
| 3 | Kp | nominal의 `[0.90, 1.10]` |
| 4 | armature | nominal의 `[0.80, 1.20]` |
| 5 | viscous damping | nominal의 `[0.80, 1.20]` |
| 6 | initial root position | each axis `+-0.01 m` |
| 7 | initial orientation | each axis `+-3 deg` |
| 8 | initial joint q | `+-0.05 rad` |
| 9 | action delay | 0--2 policy ticks |
| 10 | push | `+-0.5 m/s`, every 4--8 s |
| 11 | sensor noise | MJX와 동일 분포를 명시한 뒤 추가 |

각 randomization sample은 env reset 시 저장하고, 실패 episode를 같은 sample로 재생할 수 있게 한다. mass만 바꾸고 inertia를 놓치거나, Kp만 바꾸고 effort saturation을 바꾸는 식의 숨은 차이를 금지한다. 항목 하나를 추가할 때 deterministic baseline 회귀와 해당 DR distribution 통계 둘 다 확인한다.

---

## 14. Test와 parity matrix

테스트 파일은 역할별로 작게 유지한다.

| test | 핵심 assertion |
|---|---|
| `test_asset_manifest.py` | 18 joint order/name/axis/sign, body mass/COM/inertia, collision primitive, total 10 kg |
| `test_joint_mapping.py` | servo/model/Isaac round-trip과 home pose |
| `test_firmware_controller_parity.py` | 4-tick state/output field 전체, CPU와 CUDA |
| `test_observation_contract.py` | 146-D shape와 slice별 값/순서 |
| `test_reward_done_contract.py` | raw/scaled term, total, success/failure reason |
| `test_terrain_rays.py` | 15-point 좌표/순서/height와 forward 9-ray 집계 |
| `test_zero_action_env.py` | reset, one policy step, q_des hold 8 physics ticks |
| `test_golden_replay.py` | 500-step functional parity와 closed-loop divergence report |

CI를 두 층으로 나눈다.

- 일반 Python/Torch CI: asset manifest, mapping, controller functional, observation/reward/done.
- Isaac launch가 필요한 GPU CI: USD inspection, DirectRLEnv zero action, RayCaster, closed-loop replay.

Golden 파일 자체도 header/schema/checksum test를 넣어 잘못된 reference를 조용히 읽지 못하게 한다.

---

## 15. 다음 세션에서 바로 실행할 작업 순서

### Batch A — MJX reference를 얼리기

1. 새 branch/worktree 상태와 dependency version 기록
2. `export_asset_manifest.py` 구현
3. `export_isaac_contract.py` 구현
4. seed 0, flat, DR off, scripted 500-step golden 생성
5. golden schema/checksum test 추가

완료 산출물:

```text
mjx/golden/isaac_contract_v1_flat_seed0.npz
mjx/golden/isaac_contract_v1_flat_seed0.json
mjx/golden/asset_manifest_v1.json
```

### Batch B — Asset만 이식

1. `/home/huro/IsaacLab/isaaclab.sh -n`으로 external project skeleton 생성
2. prepared URDF를 USD로 변환
3. primitive collision와 final dynamics 적용
4. joint mapping/pose inspection 도구 및 test 작성
5. USD asset manifest 생성 후 MJX manifest와 diff

### Batch C — Firmware functional parity

1. Torch dataclass와 constants 작성
2. state machine, gait, foot target, IK, joint target 순서로 포팅
3. golden 4-tick replay test
4. CPU/CUDA 결과 비교

### Batch D — Flat DirectRLEnv

1. timing과 actuator 설정
2. state/contact adapter
3. zero action 1-step 및 500-step 실행
4. q_des가 8 physics substep 동안 유지되는지 trace

### Batch E — Obs/reward/done과 flat parity

1. pure Torch observation/reward/done
2. golden functional test
3. closed-loop divergence report
4. flat gate 통과 또는 mismatch ledger 작성

### Batch F — Terrain, PPO, DR

1. terrain geometry와 exact 15-ray pattern
2. level별 deterministic replay
3. flat PPO와 curriculum
4. stage별 best 영상 한 개
5. DR를 표 순서대로 누적

각 batch가 끝날 때 아래 네 가지가 없으면 다음 batch로 넘어가지 않는다.

```text
artifact + automated test + mismatch report + reproduction command
```

---

## 16. Definition of Done

이식 완료는 “Isaac에서 걷는다”가 아니라 다음 조건 전부를 뜻한다.

- MJX golden schema와 asset manifest가 versioned artifact로 존재한다.
- USD의 joint order/axis/sign, collision, mass/COM/inertia, actuator가 manifest와 일치한다.
- Torch firmware의 모든 state/output이 4-tick golden replay tolerance를 통과한다.
- 146-D observation의 각 slice가 exact order와 frame으로 일치한다.
- reward raw term, scaling, total, success/failure reason이 functional parity를 통과한다.
- DirectRLEnv가 `0.0025 / 8 / 0.02 / 0.005` timing contract를 지킨다.
- flat closed-loop 차이를 정량화하고 허용 범위 밖 mismatch를 모두 설명한다.
- 13개 terrain level과 15-point ray order가 MJX contract에 대응한다.
- PPO 전에 deterministic terrain gate가 통과한다.
- DR는 parity 뒤에만 활성화되며 sample replay가 가능하다.
- stage별 영상은 best-safe 하나만, W&B는 핵심 chart만 만든다.
- 새 command/posture semantics가 필요하면 기존 contract를 덮어쓰지 않고 v2 golden으로 분기한다.

이 순서를 지키면 다음 세션은 Batch A의 exporter 두 개부터 구현하면 된다. 첫 구현 대상은 Isaac 환경이 아니라 **MJX golden contract와 final-model asset manifest**다.
