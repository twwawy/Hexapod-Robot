# Residual 강화학습 설계: 평지 명령 커리큘럼 / 계단 지형

## 제어 구조

강화학습 정책은 기존 Tripod 제어기를 대체하지 않고 다음과 같이 보정한다.

```text
목표 속도 [vx, yaw rate] + 로봇 상태 + 전방 지형 높이
                         ↓
                 Residual Policy
            ┌────────────┴────────────┐
     다리별 발끝 Δp(6×3)       Gait 보정(4)
            └────────────┬────────────┘
                         ↓
기본 Stance/Swing 궤적 + 보정 → 3DOF IK → 18개 위치 actuator
```

기존 제어기의 발끝 목표를 `p_base`라 하면 최종 목표는 다음과 같다.

```text
p_cmd,i = p_base,i + scale * Δp_RL,i
```

정책이 출력하는 발끝 보정 한계는 다리 좌표계 기준으로 각각 ±40 mm, ±30 mm,
±90 mm이다. 따라서 학습 초기의 임의 행동이 기본 제어기를 완전히 무너뜨리지
않는다.

## Action 22차원

| 범위 | 의미 |
|---|---|
| 0:18 | RF, RM, RB, LF, LM, LB 발끝 XYZ residual |
| 18 | 보폭 scale: 0.5~1.5 |
| 19 | gait frequency scale: 0.65~1.35 |
| 20 | Swing Height: 0.025~0.17 m |
| 21 | Swing 방사/stance 폭 offset: 0~0.06 m |

정책 행동은 `[-1, 1]`로 제한한다. 위상 자체를 매 주기 직접 출력하게 하지 않고
주파수만 제한적으로 바꾸므로 Tripod A/B 순서와 기본 안정성은 유지된다.

## Observation

- 목표 전진 속도와 목표 Yaw rate
- 몸체 선속도·각속도와 중력 방향
- 18개 관절 위치·속도
- 6개 발 위치와 접촉 추정값
- 로봇 앞 15개 지점의 상대 지형 높이
- gait phase의 sin/cos
- 이전 action

## Reward와 종료

주 보상은 목표 선속도·Yaw rate 추종이다. Upright, 몸체 높이 및 전진 진행을
보상하고 action 변화, 큰 residual, 수직·횡방향 속도, 관절 속도와 IK workspace
초과를 페널티로 둔다. 몸체가 지면에 너무 가까워지거나 크게 기울면 episode를
종료한다.

## 두 개의 분리된 학습 진입점

### 1. 평지 보행 + 회전: 하나의 command curriculum

`train_command_curriculum.py`는 `hexapod_flat_rl.xml`을 사용한다. 한 1,000-step
episode 안에서 아래 순서로 명령을 바꾸므로, 보행과 회전을 서로 다른 policy나
서로 다른 action space로 나누지 않는다.

| 구간 | step | 전진 명령 | yaw 명령 | 목적 |
| --- | ---: | --- | --- | --- |
| 0 | `0–249` | `0.03–0.08 m/s` | `0 rad/s` | nominal tripod 안정화 |
| 1 | `250–499` | `0.05–0.12 m/s` | `±0.15 rad/s` | 완만한 곡선 보행 |
| 2 | `500–999` | `0.03–0.18 m/s` | `±0.35 rad/s` | 전체 보행+회전 추종 |

각 구간 전환 시 speed/yaw command만 다시 sample한다. policy가 보는 observation,
22D action, nominal controller, reward 함수는 처음부터 끝까지 같다.

```bash
python SW/mjx/train_command_curriculum.py --wandb \
  --wandb-project hexapod-command-curriculum
```

### 2. 계단·험지: 별도 terrain task

`train_rough_terrain.py`는 `hexapod_rl.xml`의 50 mm 연속 계단과 전방 height scan을
사용한다. 이 run은 평지 command curriculum과 checkpoint와 W&B project를 분리한다.

```bash
python SW/mjx/train_rough_terrain.py --wandb \
  --wandb-project hexapod-rough-terrain
```

다음 확장은 terrain task에만 적용한다.

1. 20~50 mm 랜덤 단차
2. 계단 높이·폭·마찰과 로봇 질량 domain randomization
3. 외란, 센서 노이즈, actuator 지연
