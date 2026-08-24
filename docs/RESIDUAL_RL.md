# Hexapod Classical Whole-Body + Residual RL

현재 기준 계약은 다음 두 버전이다.

- action: `classical_wbc_cartesian_body6d_residual_v1` (24-D)
- observation: `body_state_command3_coarse9_touchdown6_v2` (113-D)

기존 22-D/110-D checkpoint는 의미와 shape가 모두 달라 재사용할 수 없다.

## 1. 제어 구조

원본 제어기 2차 완성본의 책임 분리를 MJX에도 유지한다.

```text
forward/lateral/yaw-rate command
  -> dead zone + slew limit
  -> world x/y Position PI + Heading Hold PI
  -> Final Body Twist
  -> Tripod phase manager
       stance: -v - omega x p PULL
       swing: quintic-time-scaled cubic Bezier + radial offset
       transition: 0.5 s 이후 swing 3발의 airborne→contact가 모두 확인될 때만 교대
  -> Early/Late contact adaptation
  -> bounded foot residual
  -> single attitude PI
  -> inverse body translation + roll/pitch/yaw overlay
  -> six-leg workspace candidate accept/hold
  -> final numeric workspace projection
  -> URDF analytical IK
  -> joint jump hold + joint rate limit + actuator force cap
```

정책이 바꿀 수 없는 값은 gait phase, stride ownership, frequency, nominal swing height,
radial offset, contact state, IK와 safety다. 즉 zero residual도 완전한 보행 제어기다.
정책은 어려운 지형에서 그 출력을 제한된 범위로 보완한다.

MJX policy step은 50 Hz(`ctrl_dt=0.02 s`)이고 실제 제어기 목표는 원본과 같이
200 Hz(`5 ms`)다. 학습 action은 0.15초 LPF를 거쳐 연속적인 body request가 되며,
서보 목표에는 전압 최고 무부하 기준 `315.8 deg/s` 제한을 둔다.

## 2. Action 24-D

| Slice | 의미 | 정책 권한 |
|---|---|---|
| `0:18` | 6 legs × foot XYZ | swing XYZ, stance Z-only |
| `18:21` | body forward/lateral/height | 각 `±0.05/±0.05/±0.10 m` |
| `21:24` | body roll/pitch/yaw | `±45/±45/±25 deg` |

Stance X/Y residual은 코드에서 정확히 0이다. Body residual은 nominal 보폭이나
착지점을 수정하지 않고 IK 직전에 전체 발 좌표를 역변환한다.

```text
p_body_cmd = R_body_request^T (p_nominal+foot_residual - t_body_request)
```

Roll/Pitch/Yaw는 단일 자세 PI가 각속도를 만들고 이를 적분한다. 6개 다리 후보가
모두 workspace와 `±135 deg` 관절 제한을 만족할 때만 세 회전축을 함께 적용한다.
불가능한 후보는 projection으로 억지 적용하지 않고 직전 승인 body pose를 유지한다.
이 경로 덕분에 계단에서 chassis pitch/roll/height를 실제 학습할 수 있다.

우선순위는 다음과 같다.

```text
Safety > Contact adaptation > bounded residual > nominal controller
```

Early landing은 한 번 airborne이 된 swing 발의 재접촉에만 적용한다. Late landing은
원본값처럼 `0.20 m/s` 아래쪽과 그 `0.8`배 속도로 다리 안쪽을 함께 탐색한다.
착지가 늦으면 다음 tripod를 먼저 들지 않고 현재 phase를 유지한다. 몸체 위치 추정의
stance anchor도 simulator world position을 읽지 않고 직전 추정 위치와 URDF FK만으로
갱신하므로 학습 시 배포 불가능한 위치 정보가 새지 않는다.

## 3. Observation 113-D

| 구성 | 차원 |
|---|---:|
| forward/lateral/yaw command | 3 |
| body local velocity / angular velocity / gravity | 9 |
| joint position / scaled velocity | 36 |
| body-frame foot position | 18 |
| hysteretic foot contact | 6 |
| heading-aligned 3×3 terrain grid | 9 |
| six nominal-touchdown heights | 6 |
| gait sin/cos | 2 |
| previous applied action | 24 |
| 합계 | 113 |

Terrain height는 접촉 중인 stance foot 아래 지형 median에 대한 상대값이다. Coarse
grid는 `forward=(0.05, 0.35, 0.65) m`, `lateral=(-0.22, 0, 0.22) m`이고, 추가
6개 값은 고전제어기가 계산한 nominal touchdown 위치다.

## 4. 평지 → 울퉁불퉁 → 계단 20 cm

`train_competence_curriculum.py`는 checkpoint가 없으면 먼저 짧은 flat baseline을
학습한다(기본 1M step). 이후 mixed terrain에서 rough를 stairs보다 먼저 노출하고,
마지막 level의 계단 전체 누적 상승만 20 cm에 도달한다.

| Level | 계단 전체 상승 | ramp 상승 | flat | blocks | rough | stairs |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2–4 cm | 4–8 cm | .40 | .10 | .15 | .00 |
| 1 | 4–8 cm | 8–12 cm | .25 | .20 | .20 | .05 |
| 2 | 8–12 cm | 12–16 cm | .20 | .20 | .20 | .15 |
| 3 | 12–16 cm | 16–20 cm | .15 | .20 | .15 | .25 |
| 4 | 16–20 cm | 20–24 cm | .10 | .20 | .15 | .30 |

6단 stair lane의 Level 4 최대 단차는 `0.20 / 6 = 0.0333 m`이고 최상단 높이는
정확히 `0.20 m`다. `--terrain-total-rise`가 전체 높이 override이며,
`--terrain-step-height`는 한 단을 직접 지정하는 고급 옵션이다. 두 옵션은 동시에
사용할 수 없고 어떤 경우에도 전체 상승 `0.20 m`를 넘으면 시작 전에 오류가 난다.

## 5. 학습

가상환경과 GPU를 먼저 확인한다.

```bash
cd /home/huro/Hexapod-Robot
source /home/huro/.venvs/hexapod-mjx/bin/activate
unset LD_LIBRARY_PATH
python -c "import jax; print(jax.devices())"
```

전체 자동 순서(짧은 평지 → rough/mixed → 20 cm stairs):

```bash
python SW/mjx/train_competence_curriculum.py \
  --run-name controller-body6d \
  --flat-baseline-timesteps 1000000 \
  --stages 8 --stage-timesteps 5000000 \
  --level-progression sequential --wandb \
  -- --num-envs 2048 --num-evals 20 --terrain-randomize
```

이미 새 24-D/113-D flat checkpoint가 있으면 자동 baseline 대신 그것을 사용한다.

```bash
python SW/mjx/train_competence_curriculum.py \
  --run-name controller-body6d-resume \
  --init-checkpoint SW/mjx/runs/command/<run>/checkpoints \
  --stages 8 --stage-timesteps 5000000 --wandb \
  -- --num-envs 2048 --num-evals 20 --terrain-randomize
```

빠른 smoke와 계약 검증:

```bash
python SW/mjx/train_command_curriculum.py \
  --smoke --smoke-steps 100 --run-name body6d-command-smoke

python SW/mjx/train_rough_terrain.py \
  --smoke --smoke-steps 100 --terrain-layout mixed --terrain-level 4 \
  --terrain-randomize --run-name body6d-terrain-smoke

python -m unittest SW.mjx.tests.test_rough_terrain_contract -v
```

## 6. 주요 metric

- tracking: `velocity_error_mps`, `yaw_error_rps`, `position_error_m`,
  `heading_error_rad`
- body 6-DOF: `applied_body_*`, `posture_command_accepted`,
  `body_residual_filter_error`
- gait/controller: `effective_stride_m`, `contact_early_landing`, `contact_lost`
- safety: `projection_cost`, `torque_rms_nm`, `torque_saturation`,
  `self_collision`, body contact와 termination
- curriculum: `terrain_success`, `curriculum_stage`, 실제 stair total rise

쉬운 level은 residual penalty가 강하고, stairs가 어려워질수록 동일한 물리 action
범위를 더 자유롭게 쓸 수 있게 penalty만 완화한다. Curriculum 중 action scale이나
observation 의미는 바꾸지 않는다.
