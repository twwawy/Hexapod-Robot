# Classical-first Residual RL v2

## 목표와 contract

1순위는 지형 변화 대응이며, Tripod A/B 위상·nominal stance/swing·analytical 3DOF
IK는 classical controller가 계속 소유한다. Policy는 그 controller가 알 수 없는
terrain/contact mismatch만 Cartesian residual로 보정한다.

```text
command → classical tripod nominal foot target
        → phase-masked RL residual
        → contact adaptation
        → workspace projection
        → analytical IK → joint/rate limit → position actuator
```

안전 우선순위는 `joint/rate safety > contact adaptation > RL residual > nominal gait`다.
현재 interface는 `cartesian_gait_residual_v2`다. 기존 22-D v1 checkpoint는 action의
물리적 의미가 달라졌으므로 resume하지 말고 fresh training을 시작한다.

## Action 22-D

Policy의 tensor shape은 바꾸지 않는다.

| index | policy action | command task authority | terrain task authority |
| --- | --- | --- | --- |
| `0:18` | six local foot `(x,y,z)` | swing: ±10/±8 mm, Z −5/+20 mm; stance: Z ±3 mm, XY=0 | swing: ±25/±15 mm, Z −10/+50 mm; stance: Z ±8 mm, XY=0 |
| `18` | stride | `0.8…1.2×` | `0.8…1.2×` |
| `19` | frequency | `0.85…1.15×` | `0.85…1.15×` |
| `20` | global swing height | `65…90 mm` | `50…110 mm` |
| `21` | radial offset | `5…18 mm` | `5…25 mm` |

`a=0`은 모든 learned Cartesian/gait correction이 0이므로 nominal controller와
동일하다. 비대칭 Z는 affine mapping이 아니라 sign-dependent mapping을 사용해 이
zero-action invariant를 유지한다.

```text
swing:  Δp = [Δx, Δy, Δz]
stance: Δp = [0, 0, Δz]
```

## Observation 110-D

차원은 그대로 110이다. 달라진 것은 frame contract다.

- foot 18-D는 `R_WBᵀ (p_foot^W - p_base^W)`인 body frame vector다.
- terrain 15-D는 body forward heading을 XY 평면에 투영한 5×3 forward/lateral grid다.
  Roll/pitch가 sampling plane을 기울이지 않으며, robot yaw가 바뀌면 grid도 함께 돈다.
- contact 6-D는 foot collision을 우선 사용하고, terrain-clearance estimator를 fallback으로 쓴다.

따라서 world에서 로봇과 발을 함께 yaw 회전한 동등 상태는 같은 foot observation을
만들어야 한다.

## Deterministic contact/safety layer

- swing early landing: measured current foot target hold. RL command보다 우선한다.
- stance lost contact: 최대 10 mm의 downward search만 적용한다.
- reachable workspace: `d=sqrt((sqrt(x²+y²)-L1)²+z²)`를 안전 annulus `0.112…0.345 m`로
  projection한 뒤 IK에 넣는다.
- `projection_cost = mean(||p_requested - p_projected||²)`는 impossible Cartesian
  command의 직접 metric이다. IK cosine clipping은 float-error guard만 남긴다.
- joint ±135°, 240 deg/s target-rate limit, actuator force limit은 safety layer다.

## Reward v2

tracking(`velocity`, `yaw`, `upright`, `height`, `progress`)과 비용을 분리한다.

- `swing_residual`, `stance_residual`: stance 비용이 더 크며 nominal stance를 보존한다.
- `gait_residual`, `foot_action_rate`, `gait_action_rate`: gait parameter의 50 Hz 흔들림을
  더 강하게 억제한다.
- `projection`, `torque`, `slip`, `body_contact`, `joint_velocity`: hardware transfer와
  recovery를 위한 normalized cost다.

torque는 `mean((actuator_force / 8 Nm)²)`이고, slip은 contact foot의 XY site velocity를
`0.30 m/s` 기준으로 normalize한다. torso contact는 큰 penalty지만 즉시 termination은
아니다. 심한 tilt, low root clearance, NaN만 termination한다.

## Terrain curriculum / run-level randomization

terrain level은 `0: 0–20 mm`, `1: 20–35 mm`, `2: 35–50 mm`, `3: 20–60 mm + friction`,
`4: level 3 + mass/actuator/damping` 순서다. `--terrain-randomize`는 한 run마다 scene
height/depth/friction을 sample하고 level 4에서는 mass, actuator force, damping도 sample한다.
sampled 값은 `run_metadata.json`에 저장되므로 결과를 정확히 재현할 수 있다.

per-reset noise/latency randomization은 config에 보존하지만, deterministic v2 baseline이
안정적이라는 확인 전에는 비활성화한다.

## Run layout and launch

trainer는 자동으로 다음을 만든다.

```text
SW/mjx/runs/<task>/<timestamp>_seed<seed>/
├── checkpoints/
├── monitor/
├── best_policy.gif
├── config.json
└── run_metadata.json
```

`best_policy.gif`는 새 `eval/episode_reward` 최고점마다 deterministic 10 s / 20 fps로
교체되며, `--wandb` run에서는 `best/video`로 함께 업로드된다.

```bash
# one policy for flat walking + yaw
python SW/mjx/train_command_curriculum.py \
  --run-name command-v2-seed0 --num-envs 2048 --num-evals 50 --wandb

# terrain level 3: stairs and friction vary per run
python SW/mjx/train_rough_terrain.py \
  --run-name terrain-v2-l3-seed0 --terrain-level 3 --terrain-randomize \
  --num-envs 2048 --num-evals 50 --wandb
```

Use `--smoke` before PPO. It compiles reset/step and executes ten bounded random
actions; it fails if observation, state, or reward contains NaN.
