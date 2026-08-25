# STM32 펌웨어 기반 지형 curriculum residual 강화학습

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
  → MuJoCo DS51150 position actuators (12.6 V, ±14.709975 Nm stall)
```

한 RL step은 20 ms이며 내부에서 펌웨어 주기 5 ms를 네 번 실행한다. 발 접촉은
높이 proxy가 아니라 MuJoCo의 `foot_collision`–world collision pair만 사용한다.

## 로봇 질량과 DS51150 서보 모델

RL용 `mjx/generated/hexapod_rl.xml`을 만들 때 원본 CAD 링크의 질량과 관성을 같은
비율로 스케일해 로봇 전체 질량을 정확히 `10.0 kg`으로 맞춘다. 이 값은 현재 학습
plant의 목표 질량이며, 완성된 실물 로봇의 실측 질량을 의미하지 않는다.

| 항목 | 값 | 근거 |
|---|---:|---|
| supply | 12.6 V | DS51150-270 고전압 사양 |
| gear ratio | 357:1 | 제조사 사양 |
| stall torque | 150 kgf·cm = 14.709975 Nm | 제조사 사양, actuator hard cap |
| no-load speed | 0.19 s/60° = 315.8 deg/s | 제조사 사양, 목표각 rate limit |
| position gain | `kp=500`, `kv=10` | calibration prior |
| output armature | `0.02 kg·m²` | calibration prior |
| joint damping | `0.15 Nms/rad` | calibration prior |
| gear friction loss | `0.8 Nm` | calibration prior |

기어 출력의 armature, damping, dry friction을 joint에 포함해 토크 포화 시 관절이
무마찰 자유 힌지처럼 퍼지는 기존 모델을 방지한다. 단, 마지막 네 값은 제조사
데이터가 아니므로 실기에서 무부하 전류·step response·외력 backdrive를 측정한 뒤
`servo_model.py`만 수정해 재식별해야 한다. 학습 metadata에도 이 전체 계약을 남긴다.

`test_servo_model.py`는 생성 scene의 10 kg 질량, 18개 actuator의 gain/force range,
joint armature/damping/friction과 5초 powered-home 자세 유지를 회귀 검사한다.

## Action 18-D

다리 순서는 펌웨어와 같은 `RF, RM, RB, LF, LM, LB`이다.

| 범위 | 의미 | 한계 |
|---|---|---|
| `0:18` | 다리별 local XYZ 발끝 residual | X ±40 mm, Y ±20 mm, Z ±90 mm |

- Swing 다리에는 XYZ를 적용한다.
- Stance와 Late-Landing 다리에는 Z만 적용해 지지발 XY 미끄럼 명령을 막는다.
- residual은 0.10초 low-pass filter를 지난다.
- IK 2-link 작업공간은 경계 안쪽 1 mm에서 제한한다.
- 도달 불가능한 residual은 다리별로 거부되고 펌웨어 nominal 목표를 사용한다.
- 정책은 gait phase/frequency, swing height 0.20 m, radial offset 0.07 m,
  contact state machine, 자세 PI, IK 또는 관절 속도 제한을 바꿀 수 없다.

계약 이름은 `stm32_firmware_cartesian_foot_residual_v2`이다. v1과 tensor shape은
같으므로 초기 policy로 불러올 수 있지만, Y residual 폭과 IK 여유는 새 값으로 적용된다.
과거 22-D checkpoint는 호환되지 않는다.

## Observation 142-D

- 목표 전진 속도와 yaw rate
- 몸체 선속도·각속도·중력 방향과 펌웨어 좌표계 roll/pitch
- 18개 관절 위치·속도
- 여섯 발의 controller-body 위치와 실제 collision contact
- heading 기준 전방 15개 지형 높이
- 펌웨어 gait progress/state와 실제 적용 twist
- IK valid, residual valid, foot projection, gait/posture acceptance
- 이전 action

계약 이름은 `firmware_state_collision_terrain_curriculum_v2`이다. 관측 배열의
위치와 크기는 기존 stairs v1과 같아서 검증된 18-D/142-D checkpoint는 초기화에
사용할 수 있다.

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
멈춘다. 실패 장면을 끝까지 관찰해야 할 때만 `--allow-unsafe`를 붙인다.

## 지형 curriculum

| Level | 지형 | 상세 |
|---:|---|---|
| 0 | 평지 | 고정 최소 budget만 학습하고 종료 |
| 1 | 울퉁불퉁 | 최대 2.5 cm 높이 타일 |
| 2 | 강한 울퉁불퉁 | 최대 5 cm 높이 타일 |
| 3 | 경사면 | 8° 연속 경사 |
| 4 | 가파른 경사면 | 15° 연속 경사 |
| 5~8 | 기존 계단 curriculum | 7단 전체 높이 5/10/15/20 cm |
| 9~12 | 연속 10단 계단 | 한 riser 5/10/15/20 cm |

최종 level 12는 **20 cm riser가 10번 연속**되며 최상단은 바닥에서 2 m다.
Level 0은 기본 262,144 step(기본 PPO 설정의 4 update)만 실행한다. Level 1 이후는
각 stage 평가의 `eval/episode_terrain_success`가 기본 0.80 이상일 때만 다음 level로
올라가며, 미달하면 같은 level을 새 stage에서 반복한다. `--max-stages-per-level 3`을
주면 한 level을 최대 세 번만 시도한 뒤 다음 level로 강제 승급한다. stage 사이에
최고 평가 checkpoint를 이어가려면 `--checkpoint-selection best`를 사용한다.

## 보상과 성공

- 목표 속도 추종, upright, 지형 기준 몸체 높이, 낮은 각속도와 관절 여유를 보상한다.
- 관절 한계 근접 보상 가중치를 높이고 policy residual 거부와 workspace projection은
  각각 더 강하게 감점해, 제한에 걸리는 동작보다 작은 안전 residual을 우선 학습한다.
- torque/saturation, joint velocity, lateral/vertical velocity, residual 크기와 변화,
  controller/policy rejection, body/self collision을 페널티로 둔다.
- 경사면/계단에서 새 최고 높이에 처음 도달할 때만 ascent bonus를 지급해 접촉 높이 떨림으로
  보상을 반복 획득할 수 없게 한다.
- 지형 끝을 몸체가 통과하고 경사면/계단의 최종 높이 지지가 확인되며 roll/pitch가 20°
  이내이면 success 종료와 +30 보너스를 준다.
- episode는 평지 20초, level 1~8은 50초, 연속 10단 level 9~12는 100초가 기본이다.

## 실행

```bash
cd /home/huro/Hexapod-Robot
PY=/home/huro/.venvs/hexapod-mjx/bin/python

# 환경/계약만 검증
$PY mjx/train_rough_terrain.py --terrain-level 1 --smoke
$PY mjx/train_rough_terrain.py --terrain-level 4 --smoke
$PY mjx/train_rough_terrain.py --terrain-level 12 --smoke

# 한 난이도만 직접 학습
$PY mjx/train_rough_terrain.py \
  --run-name firmware-final-stairs \
  --terrain-level 12 \
  --timesteps 50000000 --num-envs 2048 \
  --num-evals 10 --num-eval-envs 32 \
  --wandb --wandb-project hexapod-firmware-terrain
```

실제 학습은 GPU JAX backend를 요구한다. `--allow-cpu`는 아주 작은 디버깅에만
사용한다. 모든 level에서 action/observation tensor 계약은 같아 checkpoint를
안전하게 이어받는다.

원래 사용하던 competence launcher 형태로 평지 baseline부터 전체 계단까지 돌리는
권장 명령은 다음과 같다.

```bash
$PY mjx/train_competence_curriculum.py \
  --run-name firmware-terrain-final-stairs \
  --flat-baseline-timesteps 262144 \
  --stages 36 --stage-timesteps 5000000 \
  --start-level 1 --max-level 12 \
  --level-progression competence \
  --wandb --wandb-project hexapod-firmware-terrain \
  -- --num-envs 2048 --num-evals 10 --num-eval-envs 32
```

`num-envs`는 총 sample 수가 아니라 병렬 rollout 폭이다. 기본 PPO 설정에서는
2048이 한 번의 rollout batch이고, 1024로 낮추면 같은 65,536 sample을 두 번에
나누어 계산하므로 학습량이 절반이 되지 않는다. rough terrain은 32개 box 대신
단일 heightfield collision을 사용해 MJX contact graph 크기를 제한한다. 동일 shape의
재시작은 기본적으로 `~/.cache/hexapod-mjx/jax`의 persistent XLA cache를 재사용한다.
기본 `--collision-mode lower_leg`는 계단에 필요한 foot/tibia/torso 접촉을 유지하되
제한된 residual 정책에서 불필요하게 큰 링크 간 self-collision graph는 만들지 않는다.
전체 링크와 self-collision이 필요한 검증 run에서만 `--collision-mode full`을 쓴다.
순수 throughput 확인에서는 `--no-eval --no-stage-video`를 사용하면 학습 뒤의 긴
episode evaluation과 렌더링을 건너뛸 수 있다. Curriculum 실행에는 평가 결과가
필요하므로 `--no-eval`을 전달하면 안 된다.

`competence`는 `eval/episode_terrain_success`가 0.80 이상이면 level을 올리고,
미달하면 같은 level을 반복한다. 매 stage를 무조건 올리려면
`--level-progression sequential`을 쓴다. 기존 checkpoint에서 시작할 때는 launcher에
`--init-checkpoint mjx/runs/terrain/<run>/checkpoints`를 지정한다. 로더는 18-D,
142-D, network layer 및 semantic contract가 모두 맞을 때만 승계를 허용한다.

현재 검증된 level 4 actor에서 10 kg/DS51150 동역학으로 새 run을 시작하는 명령은
다음과 같다. 기본값인 `--no-init-value-function`으로 critic은 새 plant에 맞춰 다시
학습하고 actor만 초기화한다.

```bash
/home/huro/bin/hexapod-mjx-python mjx/train_competence_curriculum.py \
  --run-name firmware-terrain-10kg-level4 \
  --seed 8 --flat-baseline-timesteps 0 \
  --start-level 4 --max-level 12 \
  --stages 36 --stage-timesteps 5000000 \
  --level-progression competence --max-stages-per-level 3 \
  --checkpoint-selection best \
  --init-checkpoint mjx/runs/terrain/firmware-terrain-ik-safe-max3-v3-stage07-level4_20260825-151107_seed8/checkpoints/000003407872 \
  --wandb --wandb-project hexapod-firmware-terrain \
  -- --num-envs 1024 --num-evals 4 --num-eval-envs 32
```

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
- `videos/best_stageXX_levelY.gif`: `--best-video` 사용 시 NEW_BEST 정책 영상
- `videos/stage_final_stageXX_levelY.gif`: 각 curriculum stage의 최종 정책 영상
- `videos/progress/`: `--progress-video` 사용 시 0/25/50/75/100% 시점 정책 영상
- `run_metadata.json`: 펌웨어/action/observation/terrain/PPO/checkpoint 계약

W&B에서는 모든 stage가 launcher의 `--run-name` group 아래 별도 run으로 묶인다.
공통 x축은 `train/global_step`이며 `eval/*`, `best/*`, `stage/*`, `progress/*`를
기록한다. 영상은 공통 `progress/video`와
`progress/video_stageXX_levelY_p000` 같은 고유 key 양쪽에 저장된다. 네트워크가 없는
서버에서는 `--wandb --wandb-mode offline`으로 실행한 뒤 `wandb sync`를 사용한다.
offline 원본은 각 run directory의 `wandb/` 아래에 남는다.

학습 중 best/progress 영상은 성능을 위해 기본적으로 꺼져 있고 stage 최종 영상만
생성한다. `--best-video`, `--progress-video`로 켤 수 있다. 영상은 각 20초다.
개수·길이는 `--progress-video-count`,
`--progress-video-duration`, `--best-video-duration`, `--stage-video-duration`으로
조절하며 `--no-stage-video`로 최종 영상도 끌 수 있다.
