# STM32 펌웨어 기반 계단 residual 강화학습

## 기준 제어기

학습 환경의 base controller는 `tripod_controller.py`의 독립 Python 보행기가
아니다. 실제 STM32 모듈을 호출하는 `native/firmware_controller_bridge.c`와 같은
순서·상태·상수를 `firmware_mjx_controller.py`에 JAX로 옮긴 것이다.

```text
목표 vx/wz
  → 5 Hz command LPF
  → Position/Heading feedback + rate limit
  → gait workspace preview
  → contact-gated Tripod / Early·Late Landing
  → PULL/Bezier foot trajectory
  → roll/pitch posture feedback
  → bounded RL foot residual
  → workspace gate / IK hold / 315.8 deg/s joint limit
  → MuJoCo ±8 Nm position actuators
```

한 RL step은 20 ms이며 내부에서 펌웨어 주기 5 ms를 네 번 실행한다. 발 접촉은
높이 proxy가 아니라 MuJoCo의 `foot_collision`–world collision pair만 사용한다.

## Action 18-D

다리 순서는 펌웨어와 같은 `RF, RM, RB, LF, LM, LB`이다.

| 범위 | 의미 | 한계 |
|---|---|---|
| `0:18` | 다리별 local XYZ 발끝 residual | X ±40 mm, Y ±30 mm, Z ±90 mm |

- Swing 다리에는 XYZ를 적용한다.
- Stance와 Late-Landing 다리에는 Z만 적용해 지지발 XY 미끄럼 명령을 막는다.
- residual은 0.10초 low-pass filter를 지난다.
- 도달 불가능한 residual은 다리별로 거부되고 펌웨어 nominal 목표를 사용한다.
- 정책은 gait phase/frequency, swing height 0.20 m, radial offset 0.07 m,
  contact state machine, 자세 PI, IK 또는 관절 속도 제한을 바꿀 수 없다.

계약 이름은 `stm32_firmware_cartesian_foot_residual_v1`이다. 과거 22-D checkpoint는
호환되지 않으므로 새 checkpoint 디렉터리에서 학습해야 한다.

## Observation 142-D

- 목표 전진 속도와 yaw rate
- 몸체 선속도·각속도·중력 방향과 펌웨어 좌표계 roll/pitch
- 18개 관절 위치·속도
- 여섯 발의 controller-body 위치와 실제 collision contact
- heading 기준 전방 15개 지형 높이
- 펌웨어 gait progress/state와 실제 적용 twist
- IK valid, residual valid, foot projection, gait/posture acceptance
- 이전 action

계약 이름은 `firmware_state_collision_contact_stairs_v1`이다.

## 제어기 발산 종료 조건

기존 환경은 약 69°까지 기울어져도 계속 진행했고 `termination=-2`에도 `dt`를
곱해 실제 페널티가 약 -0.04에 불과했다. 이제 다음 중 하나가 발생하면 즉시
failure로 종료하며 **시간 간격을 곱하지 않은 -30**을 준다.

- 펌웨어 최종 IK가 한 다리라도 invalid
- 펌웨어 관절 목표가 ±135° 한계의 1° 이내
- root 선속도 > 1.5 m/s 또는 각속도 > 6 rad/s
- 관절 속도 > 20 rad/s
- roll 또는 pitch > 45°
- 지형 기준 몸체 clearance < 0.14 m
- torso–world collision
- qpos/qvel/controller target에 NaN 또는 Inf

`gait_accepted`, `posture_accepted`, policy residual 거부 및 workspace projection은
그 전 단계의 연속 페널티로 기록한다. 이 신호만으로 바로 종료하지는 않아 정책이
자세를 회복할 기회를 남긴다.

`run_firmware_base.py`도 기본적으로 같은 controller-failure 조건에서 폭발 전에
멈춘다. 실패 장면을 끝까지 관찰해야 할 때만 `--allow-unsafe`를 붙인다. 기준 scene과
RL level 4는 모두 7단 전체 높이 20 cm를 사용한다.

## 계단 상승 보상과 성공

- 목표 속도 추종, upright, 지형 기준 몸체 높이, 낮은 각속도와 관절 여유를 보상한다.
- torque/saturation, joint velocity, lateral/vertical velocity, residual 크기와 변화,
  controller/policy rejection, body/self collision을 페널티로 둔다.
- 새 최고 계단 높이에 처음 도달할 때만 ascent bonus를 지급해 접촉 높이 떨림으로
  보상을 반복 획득할 수 없게 한다.
- 마지막 계단 상단을 몸체가 통과하고 최상단 지지가 확인되며 roll/pitch가 20°
  이내이면 success 종료와 +30 보너스를 준다.
- 기본 episode는 40초(2000 × 20 ms)로 전체 7단 계단을 오를 시간을 확보한다.

## 실행

```bash
cd /home/huro/Hexapod-Robot
PY=/home/huro/.venvs/hexapod-mjx/bin/python

# 환경/계약만 검증
$PY mjx/train_rough_terrain.py --terrain-level 4 --smoke

# 한 난이도만 직접 학습
$PY mjx/train_rough_terrain.py \
  --run-name firmware-stairs-level4 \
  --terrain-level 4 \
  --timesteps 50000000 --num-envs 2048 --num-evals 20 \
  --wandb --wandb-project hexapod-firmware-stairs
```

실제 학습은 GPU JAX backend를 요구한다. `--allow-cpu`는 아주 작은 디버깅에만
사용한다. 난이도는 계단 한 칸 높이가 아니라 **7단 전체 상승 높이** 기준이다.
`level 0=평지`, `level 1/2/3/4=전체 5/10/15/20 cm`이고 최고 level 4의 riser는
`20/7 ≈ 2.86 cm`다. 모든 level에서 action/observation 계약은 같아 checkpoint를
안전하게 이어받을 수 있다.

원래 사용하던 competence launcher 형태로 평지 baseline부터 전체 계단까지 돌리는
권장 명령은 다음과 같다.

```bash
$PY mjx/train_competence_curriculum.py \
  --run-name firmware-stairs-safe \
  --flat-baseline-timesteps 1000000 \
  --stages 8 --stage-timesteps 5000000 \
  --start-level 1 --max-level 4 \
  --level-progression competence \
  --wandb --wandb-project hexapod-firmware-stairs \
  -- --num-envs 2048 --num-evals 20
```

`competence`는 `eval/episode_terrain_success`가 0.80 이상이면 level을 올리고
0.50 미만이면 내린다. 매 stage를 무조건 올리려면
`--level-progression sequential`을 쓴다. 기존 checkpoint에서 시작할 때는 launcher에
`--init-checkpoint mjx/runs/terrain/<run>/checkpoints`를 지정한다. 로더는 18-D,
142-D, network layer 및 semantic contract가 모두 맞을 때만 승계를 허용한다.

## W&B, checkpoint, 영상 산출물

최초 한 번만 다음을 실행한다. `wandb`는 `requirements-train.txt`에 포함되어 있다.

```bash
$PY -m pip install -r mjx/requirements-train.txt
/home/huro/.venvs/hexapod-mjx/bin/wandb login
```

각 실행은 `mjx/runs/terrain/<name>_<timestamp>_seed<seed>/` 아래에 독립 저장된다.

- `checkpoints/`: 매 evaluation의 일반 Brax checkpoint
- `monitor/best_checkpoint.json`: NEW_BEST와 정확히 같은 `checkpoints/<step>` 경로
- `monitor/best_score.json`, `latest_metrics.json`, `metrics_history.jsonl`: 로컬 점수 기록
- `videos/best_stageXX_levelY.gif`: NEW_BEST 정책 영상
- `videos/stage_final_stageXX_levelY.gif`: 각 curriculum stage의 최종 정책 영상
- `videos/progress/`: 기본 0/25/50/75/100% 시점 정책 영상
- `run_metadata.json`: 펌웨어/action/observation/terrain/PPO/checkpoint 계약

W&B에서는 모든 stage가 launcher의 `--run-name` group 아래 별도 run으로 묶인다.
공통 x축은 `train/global_step`이며 `eval/*`, `best/*`, `stage/*`, `progress/*`를
기록한다. 영상은 공통 `progress/video`와
`progress/video_stageXX_levelY_p000` 같은 고유 key 양쪽에 저장된다. 네트워크가 없는
서버에서는 `--wandb --wandb-mode offline`으로 실행한 뒤 `wandb sync`를 사용한다.
offline 원본은 각 run directory의 `wandb/` 아래에 남는다.

기본 영상은 각 20초다. 개수·길이는 `--progress-video-count`,
`--progress-video-duration`, `--best-video-duration`, `--stage-video-duration`으로
조절하고, 필요할 때 각각 `--no-progress-video`, `--no-best-video`,
`--no-stage-video`로 끌 수 있다.
