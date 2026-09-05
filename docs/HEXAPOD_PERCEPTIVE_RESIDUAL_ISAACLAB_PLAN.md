# Hexapod Perceptive Residual Locomotion 구현 계획

> 2026-09-06 구조 변경: 최신 결정은
> [LiDAR·IMU → geometric foothold correction → RL residual 계획](HEXAPOD_LIDAR_FOOTHOLD_RESIDUAL_PLAN.md)을 따른다.
> 첫 단계는 LiDAR 기반 elevation map과 착지 후보 선택이며, D435IF RGB는 후속 후보 점수에 사용한다.
> 아래 LiDAR+Depth 동시 fusion과 기존 actor 차원은 이전 계획 기록이며 새 구현의 확정 계약이 아니다.

작성일: 2026-08-27 (Asia/Seoul)  
대상 브랜치: `codex/cartesian-residual-rl`  
대상 환경: Isaac Lab 2.3.0 / Isaac Sim 5.1  
상위 목표: MID-360 + Depth + IMU 기반 계단/불규칙 지형 보행의 Jetson + STM32 배포

## 1. 결정 사항

최종 시스템은 raw point cloud나 depth image를 PPO MLP에 직접 넣지 않는다.

```text
MID-360 point cloud ───┐
                       ├─ gravity-aligned local elevation map ─ CNN ─ terrain latent
Depth image → points ──┘                         │
                                                ├─ geometric pitch_ff/swing_boost
IMU gyro/projected gravity ─ proprioception ────┤
joint/contact/controller state ─────────────────┤
command ────────────────────────────────────────┘
                                                ↓
                                      18-D Cartesian residual
                                                ↓
                              STM32 gait/posture/safety/IK
                                                ↓
                                           q_des(18)
```

핵심 결정은 다음과 같다.

1. STM32-equivalent gait, workspace protection, IK, joint rate limit은 유지한다.
2. elevation map을 기본 제어기의 `pitch_ff`/`swing_boost`와 RL actor가 함께 사용한다.
3. actor는 실제 로봇에서 측정 가능한 정보만 사용한다.
4. critic, reward, curriculum, supervised perception label에는 simulator privileged
   information을 허용한다.
5. 센서 map은 roll/pitch를 제거하고 yaw/heading을 유지한 gravity-aligned local frame을
   기준으로 한다.
6. LiDAR/Depth map은 10--30 Hz, policy는 50 Hz, firmware는 200 Hz, physics는 400 Hz의
   multi-rate contract로 구현한다.
7. 처음에는 perfect raycast, 마지막에만 MID-360 scan/noise/latency를 근사한다.

## 2. 현재 코드의 지형 정답 누설

`mjx/rough_terrain_env.py`의 `_terrain_height()`는 simulator가 알고 있는 지형
geometry를 직접 조회한다. 이 값은 현재 세 경로로 배포 대상 동작에 들어간다.

| 경로 | 현재 구현 | 문제 |
|---|---|---|
| actor observation | `_terrain_features()`의 15-D `[76:91]` | 실제 센서 없이 GT 높이 사용 |
| base posture | `_terrain_pitch_ff()` → `pitch_ff` | 기본 제어기가 계단 형상을 이미 앎 |
| base swing | `_terrain_swing_boost()` → `swing_boost` | 기본 제어기가 앞 장애물 최고점을 이미 앎 |

따라서 센서만 추가하고 이 세 경로를 유지하면 perceptive policy가 아니다. 기존 경로는
다음 두 모드로 명시적으로 분리한다.

```text
legacy_gt_mode:
  GT 15-point height → pitch_ff/swing_boost/legacy actor
  목적: MJX↔Isaac parity, locomotion feasibility teacher

sensor_mode:
  LiDAR+Depth+IMU elevation map → pitch_ff/swing_boost/sensor actor
  목적: 최종 학습과 배포
```

`sensor_mode`에서는 actor와 base controller가 `_terrain_height()`, terrain level,
stair riser/tread 등 GT 지형 파라미터에 접근하면 실패다. 다만 reward, termination,
critic privileged observation, map supervision label, evaluation metric은 학습 중 GT를
사용할 수 있다.

### 누설 방지 테스트

센서 mode에서 다음 테스트를 자동화한다.

1. sensor tensors를 고정한다.
2. simulator GT map/terrain parameters를 서로 다른 값으로 바꾼다.
3. actor observation, `pitch_ff`, `swing_boost`, action, q_des가 bitwise 또는 지정
   tolerance 안에서 동일한지 확인한다.
4. critic/reward/debug metric만 GT 변화에 반응하는지 확인한다.

이 테스트가 actor 또는 controller 경로의 GT 재유입을 막는 최종 gate다.

## 3. Robot asset과 sensor frame

### 3.1 MID-360

`HW/urdf/urdf/HEXAPEDAL_URDF.xacro`에는 다음 CAD link와 fixed joint가 존재한다.

```xml
<link name="MID-360_4_2_1_1"/>

<joint name="강체 196" type="fixed">
  <origin xyz="-0.038 0.024785 0.054373" rpy="0 0 0"/>
  <parent link="M_column_3_1"/>
  <child link="MID-360_4_2_1_1"/>
</joint>
```

이 origin은 base/body 기준이 아니라 `M_column_3_1` 기준이다. 따라서 최종 LiDAR
extrinsic은 fixed-joint chain 전체를 합성해 얻는다.

```text
T_body_lidar = T_body_M_column_3_1 · T_M_column_3_1_MID360 · T_CAD_sensor
```

CAD mesh origin과 optical/ray origin이 같다고 가정하지 않는다. USD에 다음 frame을
분리한다.

```text
MID-360_4_2_1_1    # visual/inertial CAD link
mid360_sensor_frame # ray/point-cloud origin and axes
```

`HEXAPEDAL_URDF.gazebo`에는 MID-360 material/friction만 있고 LiDAR `<sensor>` plugin은
없다. Isaac sensor는 별도 `RayCasterCfg` 또는 후반 RTX LiDAR config로 생성한다.

### 3.2 Depth camera

현재 Xacro/mesh/gazebo tree에서 `D435`, `RealSense`, `camera`, `depth` 이름의 명시적
sensor link를 확인하지 못했다. 따라서 임의 CAD link를 카메라라고 단정하지 않는다.

1. 실물 CAD/조립도에서 camera optical center와 orientation을 확인한다.
2. `depth_camera_link`와 optical convention을 갖는 `depth_camera_frame`을 Xacro에
   명시한다.
3. 확정 전에는 config의 provisional extrinsic으로만 두고 checkpoint metadata에
   `depth_extrinsic_provisional=true`를 기록한다.

Depth 위치 미확정은 robot/USD와 flat controller parity를 막지 않지만 sensor-fusion
gate 이후로 넘어가는 것은 막는다.

### 3.3 IMU

IMU도 CAD visual과 measurement frame을 분리한다.

```text
imu_link          # physical mounting body
imu_sensor_frame  # gyro/accel axes
```

실기에서는 STM32가 roll/pitch/angular-rate를 직접 받아 200 Hz posture stabilization에
사용하고, timestamp가 포함된 IMU 상태를 Jetson으로 전달하는 구성을 우선한다. Isaac
actor에는 body angular velocity와 projected gravity를 먼저 넣는다. raw linear
acceleration은 충격/진동 민감도를 확인한 뒤 ablation으로 추가한다.

### 3.4 Asset gate

- USD articulation root가 정확히 하나다.
- 18개 canonical joint와 mass/collision parity가 유지된다.
- body→LiDAR/Depth/IMU transform을 URDF와 USD에서 계산해 translation/rotation diff를
  저장한다.
- 각 sensor axis를 1 m debug ray/axis marker로 시각화한다.
- sensor CAD collision은 학습 collision에서 제외한다.
- sensor frame hash와 extrinsic source가 asset manifest에 들어간다.

## 4. Gravity-aligned local elevation map contract

### 4.1 좌표계

frame을 다음과 같이 정의한다.

- `L`: LiDAR sensor frame
- `C`: depth optical frame
- `B`: robot controller/body frame
- `G`: gravity-aligned, heading-preserving local map frame

LiDAR point와 depth point를 먼저 body frame으로 변환한다.

```text
p_B = R_BS p_S + t_BS
```

IMU/projected gravity로 body roll/pitch를 제거하되 yaw는 robot heading과 함께 유지한다.

```text
p_G = R_GB p_B
map +x = robot forward
map +y = robot left
map +z = opposite gravity / physical height
```

`R_GB`는 IMU quaternion을 그대로 복사하지 않고 quaternion convention, home frame,
heading extraction을 unit test한 함수에서 계산한다.

### 4.2 초기 grid

```text
x range      = [-0.40, 1.20] m
y range      = [-0.60, 0.60] m
resolution   = 0.05 m
shape        = 32 x 24 x 3
```

채널은 다음으로 시작한다.

| channel | 의미 | 초기 집계 |
|---:|---|---|
| 0 | support-relative elevation | cell hit z의 robust median 또는 upper-ground estimate |
| 1 | confidence/validity | hit count와 sensor agreement를 `[0,1]`로 정규화 |
| 2 | roughness/slope | valid neighbor의 local height variation |

unknown cell을 height 0으로만 채우지 않는다. height와 confidence를 함께 유지하고,
invalid/occluded cell이 평지로 오인되지 않게 한다.

### 4.3 LiDAR + Depth fusion

초기 fusion은 학습 가능한 end-to-end mapper가 아니라 deterministic GPU tensor 연산으로
구현한다.

1. sensor별 range/validity filtering
2. timestamp에 맞는 IMU pose로 motion/gravity compensation
3. `L/C → B → G` transform
4. common ROI crop
5. cell scatter/reduction
6. sensor별 confidence 산출
7. fused elevation/confidence/roughness 생성

LiDAR는 넓은 범위, Depth는 전방 근거리 foothold 영역의 밀도를 담당한다. 두 센서가
충돌하면 confidence와 timestamp를 보존하고 조용히 평균내지 않는다.

### 4.4 Multi-rate state

| component | nominal rate |
|---|---:|
| physics | 400 Hz (`dt=0.0025`) |
| firmware controller | 200 Hz (`dt=0.005`) |
| policy | 50 Hz (`dt=0.020`) |
| fused terrain map | 10--30 Hz |

policy는 last-valid map과 다음 metadata를 함께 받는다.

```text
map_timestamp
map_age
map_valid_fraction
lidar_age / depth_age / imu_age
stale/dropout flags
```

stale timeout을 넘으면 base controller는 `pitch_ff/swing_boost`를 안전한 기본값으로
완만하게 복귀시키고 actor terrain latent는 masked fallback을 사용한다. 오래된 map을
새 map처럼 계속 쓰지 않는다.

## 5. Controller terrain interface

기존 `_pitch_ff()`와 `_swing_boost()`의 수식과 limit/filter는 첫 sensor cutover에서
유지한다. 입력 source만 바꾼다.

```text
legacy_gt_mode: forward_heights = simulator GT 9 samples
sensor_mode:    forward_heights = elevation map의 같은 ROI/sample
```

이를 위해 controller가 simulator/env를 직접 참조하지 않도록 interface를 분리한다.

```python
TerrainControlInput(
    forward_heights,
    support_height,
    confidence,
    age,
    valid,
)
```

낮은 confidence/stale input에서는 boost를 무조건 키우지 않는다. 보수적 fallback,
command 감속, 정지 중 어느 전략을 쓸지는 hardware safety gate에서 결정한다.

## 6. Observation과 network contract

### 6.1 Legacy teacher contract

기존 `146-D`는 parity/teacher 전용으로 동결한다.

```text
[76:91]  GT terrain 15-D
[145]    GT terrain에서 계산된 pitch_ff
```

기존 checkpoint는 이 계약을 유지한 teacher로만 사용한다.

### 6.2 Sensor actor contract

기존 146-D에서 GT terrain 15-D를 제거하면 base vector는 131-D다. 이 131-D 안의
`pitch_ff`는 sensor map에서 계산된 값일 때만 배포 가능하다.

초기 actor 계약:

```text
deployable base vector         131-D
terrain latent                  64-D
actor effective input          195-D
action                          18-D Cartesian residual
```

raw map은 `32 x 24 x 3` tensor group으로 유지하고 CNN encoder가 32/64-D latent를
만든다. 처음에는 64-D, 이후 32-D와 ablation한다.

IMU 최소 입력:

```text
body angular velocity          3
projected gravity              3
```

raw accelerometer 3-D는 optional contract version으로 추가한다. 단순 차원 추가로 기존
checkpoint를 조용히 restore하지 않는다.

### 6.3 Asymmetric actor-critic

Actor observation:

```text
command + deployable proprio/controller state + sensor terrain latent/map metadata
```

Critic privileged observation:

```text
actor observation
+ GT elevation map
+ true root linear velocity
+ true contact forces/contact state
+ terrain parameters
+ sensor noise/latency realization
```

Isaac Lab 2.3 `DirectRLEnv`의 `policy`/`critic` space와 RSL-RL `obs_groups`를 사용한다.

```python
obs_groups = {
    "policy": ["policy", "images"],
    "critic": ["policy", "privileged"],
}
```

CNN encoder가 필요한 actor/critic architecture는 기본 MLP config에 억지로 flatten하지
말고 custom encoder 또는 지원되는 multimodal policy config로 명시한다.

### 6.4 Checkpoint migration

142-D/146-D teacher와 195-D sensor actor는 observation contract가 다르다.

- old actor를 direct restore하지 않는다.
- 가능한 경우 old normalizer/actor의 공통 slice를 명시적 converter로 옮긴다.
- 권장 경로는 GT teacher action distillation → sensor student pretraining → asymmetric
  PPO fine-tuning이다.
- checkpoint에는 actor/critic observation schema, map shape/channel, extrinsic hash,
  sensor latency/noise config를 저장한다.

## 7. Perception 학습

Isaac GT map은 actor input이 아니라 supervision label로 사용한다.

```text
LiDAR + Depth + IMU → predicted elevation/confidence
GT terrain map      → training-only label
```

초기 loss:

```text
L_height = masked L1 or Huber(pred_height, GT_height)
L_valid  = BCE(pred_confidence, GT_observable/valid)
L_smooth = edge-aware local regularization
```

계단 edge를 평활화해 없애지 않도록 smooth loss는 GT edge/invalid mask로 제한한다.

검증 split은 동일 계단의 seed만 바꾸는 수준을 넘어서 riser, tread, 시작 거리, yaw,
lighting/material(Depth phase), sensor extrinsic/noise를 분리한다.

초기 perfect-ray gate의 목표값은 다음으로 두고 실제 sensor model에서 재조정한다.

```text
flat valid-cell height MAE       <= 0.01 m
stairs valid-cell height MAE     <= 0.02 m
riser height median error        <= 0.02 m
map frame tilt residual          <= 1 deg
```

## 8. Isaac sensor 구현 순서

고정 버전은 로컬 Isaac Lab 2.3.0이다. `main`/`develop` 온라인 문서의 API가 로컬과
같다고 가정하지 않는다. 로컬 소스에는 `RayCasterCfg`, `RayCasterCameraCfg`,
`CameraCfg`, `TiledCameraCfg`가 모두 존재한다.

1. `RayCasterCfg`: perfect/sparse LiDAR-style hit points와 GT height scanner 비교
2. `RayCasterCameraCfg`: geometry-only depth와 map pipeline 검증
3. `CameraCfg`: 실제 depth output과 intrinsics/unprojection 검증
4. 필요한 경우 pinned version에서 benchmark 후 `TiledCameraCfg` 선택
5. RTX LiDAR: MID-360 FOV/scan pattern/range/intensity가 필요한 후반 fidelity phase

처음부터 RTX LiDAR와 rendered depth를 대규모 PPO에 넣지 않는다. sensor throughput,
memory, update rate를 1/32/256/2048 env에서 측정한 뒤 vectorization 전략을 결정한다.

## 9. Curriculum

### Track A — Locomotion feasibility teacher

GT 정보 사용을 허용하지만 배포 불가 teacher임을 metadata에 기록한다.

```text
A0 flat
A1 3--5 cm step
A2 5--10 cm stairs
A3 10--20 cm target stairs
A4 randomized riser/tread staircase
```

먼저 완벽한 지형 정보로 robot/controller/action space가 20 cm 계단을 오를 수 있는지
검증한다.

### Track B — Sensor map

```text
B0 perfect gravity-aligned rays
B1 sparse LiDAR-style rays
B2 Depth fusion
B3 FOV/occlusion
B4 noise/dropout/latency/extrinsic jitter
```

각 단계에서 GT map과 sensor map diff를 저장한다.

### Track C — Sensor actor

```text
C0 flat, sensor present, residual near zero
C1 low step
C2 low stairs
C3 10--20 cm stairs
C4 random staircase
```

각 level의 승급은 locomotion success뿐 아니라 map validity/MAE와 GT leakage test까지
통과해야 한다.

## 10. Domain randomization

기존 dynamics DR 뒤에 perception DR를 하나씩 누적한다.

1. range/depth noise
2. missing LiDAR points / invalid depth pixels
3. frame/packet dropout
4. LiDAR/Depth/IMU latency
5. update-rate jitter
6. body motion/occlusion
7. LiDAR/Camera/IMU extrinsic translation/rotation error
8. depth material/lighting sensitivity가 필요한 camera phase

모든 reset에서 실제 sample을 기록한다. sensor latency는 observation만 늦추고 GT label,
physics state까지 함께 늦추는 잘못된 구현을 금지한다.

## 11. Jetson + STM32 배포 contract

### Jetson Orin Nano

```text
MID-360 + Depth + timestamped IMU/joint/contact
→ elevation map
→ terrain encoder + actor @ 50 Hz
→ 18-D Cartesian residual + timestamp/validity
```

### STM32

```text
RC/velocity command + IMU posture feedback + residual
→ gait state machine
→ workspace/safety/IK/joint-rate limit @ 200 Hz
→ 18 servos
```

통신 message에는 contract version, sequence, source timestamp, residual validity, map age를
포함한다. Jetson timeout/invalid residual에서는 STM32가 zero residual과 안전 보행/정지로
복귀해야 한다. Jetson이 motor-level safety를 소유하지 않는다.

## 12. Repository 구조

```text
isaaclab_hexapod/
├── data/
│   ├── usd/
│   ├── sensor_extrinsics.yaml
│   └── perception_contract.json
├── scripts/
│   ├── inspect_sensor_frames.py
│   ├── collect_perception_dataset.py
│   ├── compare_gt_sensor_maps.py
│   └── benchmark_sensors.py
└── source/hexapod_isaaclab/hexapod_isaaclab/
    ├── assets/
    │   └── hexapod_sensor_frames.py
    ├── controllers/
    │   ├── firmware_controller_torch.py
    │   └── terrain_control.py
    ├── perception/
    │   ├── frames.py
    │   ├── elevation_map.py
    │   ├── fusion.py
    │   └── encoder.py
    └── tasks/direct/hexapod/
        ├── hexapod_env.py
        ├── hexapod_env_cfg.py
        ├── observations.py
        └── rewards.py
```

## 13. 구현 phase와 gate

### P0 — 기존 reference 동결

현재 golden contract, asset manifest, W&B metadata를 유지한다. 완료됨.

### P1 — Asset/sensor frame

USD single-articulation 문제를 해결하고 MID-360 CAD chain을 보존한다. Depth/IMU frame의
provisional/confirmed 상태를 manifest에 기록한다.

Gate: asset parity + transform diff + axis visualization.

### P2 — Flat zero-residual controller

Torch firmware controller를 이식하고 flat에서 GT terrain 보조 없이 zero residual로
걷는다.

Gate: controller functional parity + 500-step q_des hold + flat stability.

### P3 — GT locomotion teacher

legacy GT map으로 low step부터 20 cm stairs까지 locomotion feasibility를 확인한다.

Gate: level별 best checkpoint/video + success/failure ledger. 배포 불가 표시 필수.

### P4 — Perfect sensor map

RayCaster LiDAR와 IMU gravity alignment로 elevation map을 만들고 GT와 비교한다.

Gate: frame/MAE/riser 기준과 multi-rate timestamp test.

### P5 — Depth fusion

Depth extrinsic을 확정하고 RayCasterCamera, 이후 Camera depth를 fusion한다.

Gate: occlusion/invalid/confidence와 throughput benchmark.

### P6 — Perception pretraining

GT label로 elevation encoder/map confidence를 supervised pretrain한다.

Gate: held-out terrain metric과 qualitative map artifact.

### P7 — GT cutover

actor 15-D GT terrain과 GT 기반 `pitch_ff/swing_boost`를 제거하고 sensor interface로
교체한다.

Gate: automated GT leakage test 통과.

### P8 — Asymmetric PPO / distillation

sensor actor + privileged critic을 연결하고 GT teacher action을 distill한 뒤 PPO로
fine-tune한다.

Gate: flat regression, low-step teacher gap, actor-only export test.

### P9 — Random stairs + sensor DR

riser/tread와 sensor noise/latency/dropout/extrinsic을 단계적으로 randomize한다.

Gate: unseen terrain/noise seeds, stale sensor safety, ablation report.

### P10 — Jetson/STM32 HIL

timestamped residual protocol, timeout fallback, actor export, recorded sensor replay를
검증한다.

Gate: 동일 recorded input의 Isaac/Jetson actor output diff와 STM32 fallback test.

## 14. 첫 구현 작업 순서

현재 바로 수행할 일은 다음 다섯 개다.

1. USD 중복 articulation root 수정과 flat smoke 완료
2. main Xacro fixed chain에서 body→MID-360 transform을 추출하는 manifest 도구 작성
3. `mid360_sensor_frame`, provisional `depth_camera_frame`, `imu_sensor_frame` config 작성
4. Torch firmware flat zero-residual controller 이식
5. GT terrain access를 actor/controller/critic/reward별로 분리하는 interface와 leakage
   test skeleton 작성

센서 ray나 CNN 구현은 1--5가 끝난 뒤 시작한다.

## 15. Definition of Done

- Robot/USD가 single articulation이고 sensor frame extrinsic이 추적 가능하다.
- Flat zero residual에서 STM32-equivalent controller가 안정적으로 동작한다.
- GT teacher로 목표 20 cm stairs의 기구학/제어 feasibility가 확인된다.
- LiDAR + Depth + IMU map이 gravity-aligned frame과 multi-rate contract를 지킨다.
- sensor mode actor와 base controller에는 GT terrain access가 없다.
- actor는 131-D deployable base + terrain latent contract를 사용한다.
- critic만 privileged GT terrain/simulator state를 사용한다.
- map supervised pretraining과 asymmetric PPO가 재현 가능하다.
- random staircase와 sensor DR에서 held-out 평가를 통과한다.
- Jetson actor timeout 시 STM32가 zero residual 안전 fallback을 수행한다.
- actor/checkpoint/export artifact에 observation, map, extrinsic, timing version이 기록된다.

## 16. 참고 자료

- [Isaac Lab Ray Caster](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/ray_caster.html)
- [Isaac Lab sensors API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html)
- [Isaac Lab RL wrappers / observation groups](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html)
- [Isaac Sim RTX LiDAR](https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/isaacsim_sensors_rtx_lidar.html)

온라인 `main`/`develop` 문서는 방향 확인용이다. 실제 구현 API와 config는 고정된 로컬
Isaac Lab 2.3.0 source/commit을 source of truth로 사용한다.
