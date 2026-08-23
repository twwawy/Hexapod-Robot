# Hexapod Classical-first Residual RL

> Canonical contract: `cartesian_gait_residual_v2` action +
> `body_state_coarse9_touchdown6_v1` observation

이 문서가 현재 MJX residual RL의 설계·관측·보상·학습·실행 방법에 대한 최신 기준이다.
이전 6-D/7-D custom policy와 현재 Brax PPO checkpoint는 호환되지 않는다.

## 1. 제어 계약

```text
command
  -> classical tripod nominal foot targets
  -> phase-masked 22-D residual
  -> airborne/contact adaptation
  -> reachable-workspace projection
  -> analytical IK
  -> joint speed/position + fixed ±8 Nm actuator safety
```

우선순위는 `safety > contact > RL residual > nominal gait`다. Action은 22-D를
유지한다.

```text
[Δx, Δy, Δz] × 6 legs + stride + frequency + swing-height + radial
```

- swing: XYZ residual 허용
- stance: Z-only, X/Y는 정확히 0
- action physical range는 curriculum 중 바꾸지 않는다.
- 쉬운 stage에서는 residual penalty를 강하게 하고 어려워질수록 완화한다.
- action=0이면 NumPy classical nominal과 JAX nominal target이 `1e-4 rad` 이내로
  일치해야 한다.

## 2. Contact와 body-height safety

Swing 시작 직후 남아 있는 접촉은 early landing이 아니다. Leg마다 `airborne`을
latched state로 보관한다.

```text
stance -> swing/contact -> swing/no-contact(airborne=true)
       -> swing/contact(early landing, measured foot hold)
```

```text
early_landing = swing & airborne & contact
```

접촉 추정기는 실제 collision과 clearance를 함께 사용한다. Clearance contact는
35 mm에서 진입하고 45 mm에서 해제하는 hysteresis를 사용한다.

Body height/reward/termination은 root XY 바로 아래 높이가 아니다. 현재 contact 중인
stance feet 아래 지형 높이의 median을 support height로 사용한다.

```text
h_support = median(terrain_height(contact & stance feet))
h_body = root_z - h_support
```

유효한 stance contact가 하나도 없을 때만 root XY terrain을 fallback으로 사용한다.

## 3. Observation 110-D

전체 크기는 110으로 유지하지만 terrain 15-D의 의미가 개선됐다.

| 구성 | 차원 |
| --- | ---: |
| command | 2 |
| body local velocity / angular velocity / gravity | 9 |
| joint position / scaled velocity | 36 |
| body-frame foot positions | 18 |
| hysteretic foot contacts | 6 |
| heading-aligned coarse terrain grid | 9 |
| six nominal-touchdown terrain heights | 6 |
| gait sin/cos | 2 |
| previous applied action | 22 |
| 합계 | 110 |

Coarse grid는 `forward=(0.05, 0.35, 0.65) m`,
`lateral=(-0.22, 0, 0.22) m`의 3×3이다. 나머지 6개는 classical controller가
계산한 각 leg의 nominal touchdown 위치에서 지형 높이를 직접 샘플한다. 두 feature
모두 support height에 대한 상대 높이다.

Observation 크기가 이전과 같아도 의미가 달라졌으므로
`body_state_coarse9_touchdown6_v1` metadata가 없는 checkpoint는 transfer하지 않는다.

## 4. Multi-terrain model

기본 terrain layout은 `mixed`다. 하나의 mesh-free XML에 평행한 여섯 lane을 둔다.

```text
flat | curb | ramp | irregular blocks | stairs | rough patch
```

각 vectorized env는 reset마다 lane을 다시 샘플한다. Training wrapper는
`full_reset=True`라 episode reset 때 새 patch와 새 command를 실제로 뽑는다. XML을
재생성하지 않으므로 GPU batching은 그대로 유지된다.

| Level | flat | curb | ramp | blocks | stairs | rough |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | .70 | .20 | .10 | 0 | 0 | 0 |
| 1 | .40 | .20 | .20 | .10 | .05 | .05 |
| 2 | .25 | .15 | .20 | .15 | .15 | .10 |
| 3 | .15 | .15 | .20 | .20 | .15 | .15 |
| 4 | .10 | .15 | .20 | .20 | .20 | .15 |

`--terrain-layout stairs`는 비교 실험용 기존 stairs-only scene이다.

## 5. Command sampling

Command는 episode의 고정 순서를 암기하지 못하도록 1.5–4.0초마다 무작위로 다시
샘플한다. Curriculum stage는 허용 가능한 speed/yaw 범위만 정하며, stage 안의
command 순서는 고정하지 않는다.

## 6. Reward와 competence

Tracking/upright/height/progress와 residual/action-rate/slip/torque/projection/contact
cost를 각각 W&B에 기록한다. `terrain_success`는 episode 종료 시 다음 조건을 모두
만족하면 1이다.

- 조기 termination 없음
- forward velocity error < 0.08 m/s
- world-forward progress > 0.5 m

Adaptive launcher는 evaluation success가 0.8보다 높으면 다음 level로 올리고,
0.5보다 낮으면 level을 내린다. 각 stage는 동일한 action/observation 계약을 쓰며
직전 Brax checkpoint로 policy/normalizer/value를 초기화한다.

## 7. Domain randomization

Level 4의 `--terrain-randomize`는 vectorized env마다 friction, mass/inertia,
joint damping, position-servo realization을 다르게 만든다. Actuator gain/bias만
`0.85–1.0×`로 바꾸며 `actuator_forcerange`는 항상 `[-8, 8] Nm`로 고정한다.

## 8. PPO 기본값

| Task | γ | unroll |
| --- | ---: | ---: |
| flat command | 0.97 | 20 (0.4 s) |
| mixed terrain | 0.99 | 32 (0.64 s) |

비교 실험은 `--discounting 0.97|0.99`와 `--unroll-length 20|32|50`으로 실행한다.

## 9. 검증

Smoke는 zero action 100 step과 bounded random action 100 step을 각각 실행한다.

```bash
cd ~/Hexapod-Robot

~/.venvs/hexapod-mjx/bin/python SW/mjx/train_command_curriculum.py \
  --smoke --smoke-steps 100 --run-name command-smoke

~/.venvs/hexapod-mjx/bin/python SW/mjx/train_rough_terrain.py \
  --smoke --smoke-steps 100 --terrain-layout mixed --terrain-level 4 \
  --terrain-randomize --run-name terrain-smoke

~/.venvs/hexapod-mjx/bin/python -m unittest \
  SW.mjx.tests.test_rough_terrain_contract -v
```

Contract test는 action/observation size, phase mask, airborne transition,
contact hysteresis, support median, workspace projection, yaw frame invariance,
NumPy↔JAX zero-action parity, 실제 env reset/JIT/full gait cycle을 검증한다.

## 10. GPU 학습

Flat walking+turning부터 fresh로 학습한다.

```bash
cd ~/Hexapod-Robot

~/.venvs/hexapod-mjx/bin/python SW/mjx/train_command_curriculum.py \
  --run-name flat-transfer-source \
  --timesteps 50000000 --num-envs 2048 --num-evals 100 \
  --wandb --wandb-project hexapod-command-curriculum
```

Run directory는 같은 이름을 넣어도 timestamp와 seed가 붙어 항상 새로 생성된다.

```text
SW/mjx/runs/command/flat-transfer-source_<UTC timestamp>_seed0/
```

Flat checkpoint에서 mixed terrain을 초기화한다. `checkpoints/` root를 주면 가장 큰
numeric checkpoint를 자동 선택한다. Terrain reward가 달라 critic은 기본적으로 새로
초기화하며, 같은 task를 이어갈 때만 `--init-value-function`을 사용한다.

```bash
~/.venvs/hexapod-mjx/bin/python SW/mjx/train_rough_terrain.py \
  --run-name terrain-transfer-level0 \
  --terrain-layout mixed --terrain-level 0 \
  --init-checkpoint SW/mjx/runs/command/<flat-run>/checkpoints \
  --timesteps 50000000 --num-envs 2048 --num-evals 100 \
  --wandb --wandb-project hexapod-rough-terrain
```

Competence-based 전체 curriculum:

```bash
~/.venvs/hexapod-mjx/bin/python SW/mjx/train_competence_curriculum.py \
  --run-name mixed-competence \
  --stages 8 --stage-timesteps 5000000 \
  --start-level 0 --max-level 4 \
  --init-checkpoint SW/mjx/runs/command/<flat-run>/checkpoints \
  --wandb \
  -- --num-envs 2048 --num-evals 20 --terrain-randomize
```

학습별 W&B scalar는 `eval/episode_reward`, 각 `eval/episode_reward/*`,
`eval/episode_terrain_success`, `best/*`이며 `train/global_step`을 공통 x축으로 쓴다.

Flat command policy는 매 evaluation마다 동일한 scripted command로 Stage 0/1/2를
독립 reset하여 다음 값을 추가 기록한다.

```text
eval/stage0|1|2/reward_mean
eval/stage0|1|2/velocity_error_mps
eval/stage0|1|2/yaw_error_rps
eval/stage0|1|2/survival_fraction
```

학습 command는 계속 1.5–4초 random resampling을 사용하고, 위 비교 평가와 영상만
고정 script를 사용한다. `eval/episode_reward`가 NEW_BEST일 때 command run은
`videos/`에 Stage 0, Stage 1, Stage 2, 전체 curriculum GIF를 각각 저장하고 W&B의
`best/video_stage0_forward`, `best/video_stage1_limited_yaw`,
`best/video_stage2_full_command`, `best/video_curriculum_full`에 올린다. Terrain run은
`videos/best_policy.gif`와 `best/video` 한 개를 유지한다.
