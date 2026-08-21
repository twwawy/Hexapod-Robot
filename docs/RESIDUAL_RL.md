# Hexapod Cartesian Residual RL

이 문서는 `Hexapod-Robot`의 **유일한 현재 Residual RL 명세**다. 목표는 기존 tripod 보행기를 버리는 것이 아니라, 그 보행기가 실패하기 쉬운 순간만 작은 학습 보정으로 다루는 것이다.

## 제어 계약

```text
RC / velocity command
  → command filter + nominal tripod gait + swing Bezier
  → nominal foot targets p_nom
  → RL foot residual Δp_RL
  → contact adaptation / hard safety
  → deterministic posture layer
  → linearized IK + joint hard limits + PD torque
```

우선순위는 다음과 같다.

```text
joint/workspace safety > contact adaptation > RL residual > nominal gait
```

현재 MJX baseline의 posture layer는 identity다. 아직 calibration된 posture PI gain이 없으므로, roll/pitch/height를 RL action으로 우회하지 않는다. 실제 IMU 기반 posture PI를 붙일 때도 이 위치에 classical layer로 추가한다.

## 1차 action space: 6-D swing Δz

정책 action은 아래 순서의 6차원이다.

```text
a = [Δz_LF, Δz_LM, Δz_LB, Δz_RF, Δz_RM, Δz_RB]
```

정규화 정책 값은 `[-1, 1]`로 clamp한 뒤 실제 길이 단위로 변환된다.

```text
Δz_i = 0.03 m × clip(a_i, -1, 1)
p_cmd,i = p_nom,i + [0, 0, Δz_i]ᵀ
```

- 허용 범위: leg당 ±3 cm
- swing leg에만 적용한다. stance leg의 residual은 정확히 0이다.
- XY foothold, step period, swing phase, body-height/roll/pitch trim에는 RL 권한이 없다.
- swing 시작 직후의 정상적인 접촉은 25% phase까지 허용하고, 그 이후 early landing이 감지되면 현재 foot body-frame 위치를 유지한다. 이 경우 해당 RL residual은 적용되지 않는다.
- 그 뒤에도 workspace clamp, Jacobian IK, joint hard limit가 항상 적용된다.

따라서 action의 물리적 의미는 단순하다. `+0.03 m`는 “해당 swing 발을 3 cm 더 들어라”다. 이 6-D 버전이 안정된 다음에만 `Δx, Δy, Δz`의 18-D Cartesian residual을 별도 실험으로 확장한다.

## Observation

현재 관측은 62차원이다.

| 항목 | 차원 | 내용 |
|---|---:|---|
| command | 3 | `[vx_cmd, vy_cmd, wz_cmd]` |
| body linear velocity | 3 | body frame 선속도 |
| IMU-like attitude | 3 | roll, pitch, root-height error |
| body angular velocity | 3 | body frame 각속도 |
| joint position | 18 | 18개 관절각 |
| joint velocity | 18 | 18개 관절속도 |
| foot contact | 6 | 발별 binary contact |
| gait phase | 2 | global phase의 `sin`, `cos` |
| previous action | 6 | 이전 bounded-action 입력 |

`phase`와 `contact`는 반드시 유지한다. 같은 관절 자세라도 swing/stance 문맥에 따라 residual의 의미가 달라지고, contact layer가 policy보다 우선해야 하기 때문이다.

## Reward

기본 reward는 velocity/yaw tracking, attitude, height, slip, torque cost, body contact를 사용한다. Residual policy가 nominal gait를 덮어쓰지 않도록 물리 길이 단위의 두 penalty를 포함한다.

```text
r_residual = -λ_r mean_i(Δz_i²)
r_rate     = -λ_rate mean_i((Δz_i - Δz_i,prev)²)
```

현재 계수는 `residual_penalty=24`, `action_rate_penalty=12`다. 이는 초기값이며, 명령 추종을 망치지 않는 선에서 residual을 0 근처로 유지하도록 둔 값이다. 실험 로그에서는 아래 두 항목을 함께 본다.

- `residual_cost`: 평균 `Δz²`
- `action_rate_cost`: 연속 action의 평균 차이 제곱

## Checkpoint 호환성

이 명세 이전의 7-D policy는 보폭·주기·몸체 trim을 조절했다. 새 policy는 action 6-D, observation 62-D, residual interface `swing_delta_z_v1`이므로 이전 checkpoint를 resume할 수 없다. 학습기는 불일치를 명시적으로 거부한다.

새 실험은 반드시 `fresh`로 시작한다.

```bash
cd ~/Hexapod-Robot
./Hexapod-MJX-가이드/빠른학습.sh fresh
```

긴 실행 예시는 다음과 같다.

```bash
./Hexapod-MJX-가이드/큰병렬.sh fresh \
  --num-envs 256 \
  --rollout-steps 256 \
  --num-updates 1000 \
  --minibatch-size 2048
```

학습 wrapper는 best/latest checkpoint, metrics JSON, replay MP4를 `SW/mjx/artifacts/residual_rl_runs/` 아래에 남긴다. 생성물은 Git에서 제외된다.

## 검증 순서

1. **Import/compile**: MJX 환경 생성과 한 policy step이 동작하는지 확인한다.
2. **Zero residual**: action=0에서 nominal gait가 기존처럼 동작하는지 확인한다.
3. **Mask**: stance 다리에 큰 action을 넣어도 foot target이 변하지 않는지 확인한다.
4. **Early landing**: swing contact에서 현재 foot target이 유지되는지 확인한다.
5. **PPO smoke**: `fresh` 소규모 run으로 checkpoint metadata가 `swing_delta_z_v1`, action 6, observation 62인지 확인한다.
6. **본 학습**: reward만이 아니라 velocity tracking, slip, residual/action-rate cost, body contact를 함께 비교한다.

## 확장 원칙

18-D Cartesian residual로 확장하기 전 아래 기준을 만족해야 한다.

- 6-D Δz policy가 평지와 낮은 장애물에서 nominal보다 안정적으로 개선된다.
- residual/action-rate cost가 지속적으로 큰 상태가 아니다.
- stance slip과 body contact가 증가하지 않는다.
- zero-residual baseline과 학습 policy 모두 repeatable하게 재생된다.

확장 시에도 action 순서는 발 단위 `[Δx, Δy, Δz] × 6`, stance mask, contact override, workspace/joint safety를 그대로 보존한다.
