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
  → bounded RL foot residual + phase-gated adaptive swing height
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
| `0:18` | 다리별 local XYZ action | Swing X/Y ±100 mm; Z는 높이 4~25 cm |

- Swing X/Y는 Cartesian residual로 적용한다.
- Swing Z는 `-1/0/+1 → 4/6/25 cm` 높이 명령이며, phase envelope를 곱해
  이륙점과 착지점의 Z offset은 항상 0이다.
- Stance Z는 ±100 mm까지 요청할 수 있고 XY는 0이다. 최종 workspace/IK gate는 그대로다.
- Late-Landing에는 RL residual을 적용하지 않고 펌웨어 하강 탐색만 사용한다.
- residual은 0.10초 low-pass filter를 지난다.
- IK 2-link 작업공간은 경계 안쪽 1 mm에서 제한한다.
- 도달 불가능한 residual은 다리별로 거부되고 펌웨어 nominal 목표를 사용한다.
- 정책은 범위 내 swing X/Y·높이 및 stance Z residual을 선택할 수 있다. gait
  phase/frequency, radial offset 0.07 m, contact state machine, 자세 PI, IK 또는
  관절 속도 제한은 바꿀 수 없다.

계약 이름은 `stm32_firmware_adaptive_swing_residual_100mm_v4`이다. tensor 크기는 계속
18-D지만 v2의 Z는 Cartesian endpoint offset이고 v3의 Z는 phase-gated 높이 명령이라
의미가 다르다. v4는 v3보다 XY/stance Z scale이 커졌다. 따라서 v1/v2/v3 및 과거 22-D checkpoint는 현재 v4로 직접 복원하지 않는다.

## Observation 146-D

현재 계약은 `firmware_state_collision_terrain_command5_pitch_v3`이다.

| slice | 내용 |
|---|---|
| `0:5` | 전진·yaw·height·pitch·roll command |
| `5:16` | 몸체 속도·각속도·중력·상대 roll/pitch |
| `16:52` | 18개 관절 위치·속도 |
| `52:76` | 발 controller-body 위치·collision contact |
| `76:91` | heading 기준 지형 높이 15개 |
| `91:103` | gait progress/state |
| `103:107` | 적용 twist |
| `107:127` | IK valid·policy valid·projection·gait/posture acceptance |
| `127:145` | 이전 action |
| `145:146` | pitch feedforward |

루트 학습 환경은 GT 지형을 쓰는 teacher/개발 경로다. LiDAR 입력 뷰어는 격리된 v3 소스에서
`76:91`을 센서 값으로 바꾸고 GT feedforward를 끈다. v4 학습과 stage31 v3 재생을 구분한다.
142-D 관측은 아래 이전 teacher 이관 설명의 레거시 계약이다.

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
| 0 | 평지 | 최소 baseline 후 competence 미달 시 한 번 더 시도 |
| 1 | 울퉁불퉁 | 최대 2.5 cm 높이 타일 |
| 2 | 강한 울퉁불퉁 | 최대 5 cm 높이 타일 |
| 3 | 경사면 | 8° 연속 경사 |
| 4 | 가파른 경사면 | 15° 연속 경사 |
| 5~10 | 7단 계단 | 한 riser 5/6.5/8/10/15/20 cm |
| 11~16 | 연속 10단 계단 | 한 riser 5/6.5/8/10/15/20 cm |

최종 level 16은 **20 cm riser가 10번 연속**되며 최상단은 바닥에서 2 m다.
Level 0은 기본 262,144 step(기본 PPO 설정의 4 update) baseline으로 시작하고,
competence 미달이면 한 번 더 시도한다. Level 0~3(평지, 두 rough, 8° 경사)은
level당 최대 2회, Level 4~16(15° 경사부터 최종 계단)는 최대 4회 시도한다.
각 stage의 `eval/episode_terrain_success`가 기본 0.80 이상이면 제한에 닿기 전에도
즉시 다음 level로 올라간다. `--max-stages-per-level N`을 명시하면 이 2/4 규칙을
전체 level 공통 N회로 override한다. stage 사이에 최고 평가 checkpoint를
이어가려면 `--checkpoint-selection best`를 사용한다.

## 보상과 성공

- 자동 ramp/계단 curriculum의 pitch command는 0이고, 전방 0.40/0.65/0.90 m
  지형 높이로 계산한 `pitch_ff`만 지형 자세 목표로 사용한다. 명시적 pitch command는
  외부 명령을 위한 additive offset으로 남겨 두며, 자동 curriculum에서 같은 경사를
  `pitch_cmd + pitch_ff`로 두 번 더하지 않는다.
- 0.5초 EMA 전진속도를 `0.04 m/s` 폭으로 엄격하게 목표 속도에 추종시키고,
  `vx / vcmd`로 정규화한 전진 progress를 직접 보상한다. 목표 미달 속도는 별도
  제곱 페널티를 받는다.
- upright, 지형 기준 몸체 높이, 낮은 각속도와 관절 여유 같은 양의 보조 보상은
  전진속도 gate를 통과한 만큼만 지급한다. 따라서 제자리 보행은 생존 보상을
  누적할 수 없다.
- command 활성화 후 3초 동안 `root_x + 0.5 * support_height`가 2 cm 이상
  증가하지 않으면 no-progress 실패로 종료하고 시간 간격을 곱하지 않은 -10을 준다.
- 관절 한계 근접 보상 가중치를 높이고 policy residual 거부와 workspace projection은
  각각 더 강하게 감점해, 제한에 걸리는 동작보다 작은 안전 residual을 우선 학습한다.
- torque/saturation, joint velocity, lateral/vertical velocity, residual 크기와 변화,
  controller/policy rejection, body/self collision을 페널티로 둔다.
- Swing 높이 자체와 공중에 뜬 뒤 Swing 전반부에 다시 충돌하는 toe scuff를 별도
  페널티로 둬, 평지에서는 낮게 들고 장애물 통과에 필요한 경우에만 높이를 사용한다.
- 경사면/계단에서 새 최고 높이에 처음 도달할 때만 ascent bonus를 지급해 접촉 높이 떨림으로
  보상을 반복 획득할 수 없게 한다.
- 지형 끝을 몸체가 통과하고 경사면/계단의 최종 높이 지지가 확인되며 roll/pitch가 20°
  이내이면 success 종료와 +30 보너스를 준다.
- episode는 평지 20초, rough/ramp/7단 계단은 50초, 연속 10단 계단은 100초가 기본이다.

## 실행

```bash
cd /home/huro/Hexapod-Robot
PY=/home/huro/.venvs/hexapod-mjx/bin/python

# 환경/계약만 검증
$PY mjx/train_rough_terrain.py --terrain-level 1 --smoke
$PY mjx/train_rough_terrain.py --terrain-level 4 --smoke
$PY mjx/train_rough_terrain.py --terrain-level 16 --smoke

# 한 난이도만 직접 학습
$PY mjx/train_rough_terrain.py \
  --run-name firmware-final-stairs \
  --terrain-level 16 \
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
  --stages 44 --stage-timesteps 5000000 \
  --start-level 1 --max-level 16 \
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
146-D, network layer 및 action/observation semantic contract가 모두 맞을 때만 승계를 허용한다.
이전 run의 stage 번호까지 이어 표시하려면 다음 stage 번호를 `--start-stage`로
지정한다. 첫 stage는 기본적으로 actor만 복원하고 critic은 새 지형에서 다시 학습한다.

현재 v4는 residual scale이 바뀌므로 기존 v3 actor를 직접 이어받지 않고 새 run을 시작한다.

```bash
/home/huro/bin/hexapod-mjx-python mjx/train_competence_curriculum.py \
  --run-name firmware-terrain-adaptive-swing \
  --seed 8 --flat-baseline-timesteps 262144 \
  --start-level 1 --max-level 16 \
  --stages 44 --stage-timesteps 5000000 \
  --level-progression competence \
  --checkpoint-selection best \
  --wandb --wandb-project hexapod-firmware-terrain \
  -- --num-envs 1024 --num-evals 4 --num-eval-envs 32
```

### Teacher-student 보행 보존 학습 — 이전 v3 이관 기록

아래는 142-D→146-D 및 v2/v3 teacher 이관 경로의 기록이다. 현재 v4에 이전 가중치를 바로 적용하는
명령으로 해석하지 않는다. v4와 호환되는 teacher/변환 계약을 마련해야 한다. LiDAR sensor student와
GT supervision의 후속 구조는 [최신 설계](../docs/HEXAPOD_LIDAR_FOOTHOLD_RESIDUAL_PLAN.md)를 따른다.

Command5/posture 조건을 유지하면서 기존 142-D 보행을 보존할 때는 frozen teacher
manifest를 launcher에 전달한다. Adaptive-swing v3 teacher는 18-D 전체 action을,
Cartesian v2 teacher는 의미가 호환되는 다리별 X/Y 12개 action만 지도한다. 현재
146-D observation은 `[0:2] + [5:145]`로 legacy 142-D teacher observation에
투영하며, teacher checkpoint를 일반 `--init-checkpoint` gate로 복원하지 않는다.

```bash
/home/huro/bin/hexapod-mjx-python mjx/train_competence_curriculum.py \
  --run-name firmware-terrain-command5-teacher-student-v1 \
  --run-root /home/huro/Hexapod-Robot/mjx/runs \
  --seed 8 \
  --flat-baseline-timesteps 262144 \
  --start-level 1 --max-level 16 \
  --stages 44 --stage-timesteps 5000000 \
  --level-progression competence --promote-threshold 0.80 \
  --checkpoint-selection best \
  --teacher-manifest mjx/teacher_manifests/walking-teachers-v1.json \
  --teacher-huber-delta 0.10 \
  --wandb --wandb-project hexapod-firmware-terrain \
  -- --num-envs 2048 --num-evals 4 --num-eval-envs 32 \
     --dr-bank-size 16 --collision-mode lower_leg \
     --best-video --no-stage-video --no-progress-video
```

Fresh curriculum의 actor/normalizer는 level-0 v3 teacher의 142-D 입력 layer를
146-D로 확장해 시작한다. 새 height/pitch/roll/pitch-FF 입력 weight는 0이고 critic은
현재 reward에 맞춰 새로 시작한다. 이후 stage는 current student checkpoint를 정상
승계하면서 level별 teacher loss만 바뀐다. `best` pointer는 manifest 사용 시 기본
best-safe 제한(policy rejection 1%, foot-limited 1%, failure 5%)을 통과한 평가만
가리킨다. 모든 teacher 경로와 weight는 각 run의 `run_metadata.json` 및 curriculum
history에 기록된다.

## W&B, checkpoint, 영상 산출물

최초 한 번만 다음을 실행한다. `wandb`는 `requirements-train.txt`에 포함되어 있다.

```bash
$PY -m pip install -r mjx/requirements-train.txt
/home/huro/.venvs/hexapod-mjx/bin/wandb login
```

각 실행은 `mjx/runs/terrain/<name>_<timestamp>_seed<seed>/` 아래에 독립 저장된다.

- `checkpoints/`: 매 evaluation의 일반 Brax checkpoint
- `monitor/best_checkpoint.json`: NEW_BEST와 정확히 같은 `checkpoints/<step>` 경로
- `monitor/level_best_checkpoint.json`: safety gate와 무관한 현재 terrain level 최고 checkpoint
- `monitor/best_score.json`, `level_best_score.json`, `latest_metrics.json`, `metrics_history.jsonl`: 로컬 점수 기록
- `videos/best_levelY.gif`: 새 level-best 평가 직후 교체되는 해당 level 최고 영상
- `videos/stage_final_stageXX_levelY.gif`: `--stage-video`를 명시한 경우의 최종 정책 영상
- `videos/progress/`: `--progress-video`를 명시한 경우의 중간 정책 영상
- `run_metadata.json`: 펌웨어/action/observation/terrain/PPO/checkpoint 계약

W&B에서는 모든 stage가 launcher의 `--run-name` group 아래 별도 run으로 묶인다.
공통 x축은 `train/global_step`이다. 차트에는 reward/success, failure·safety rate,
진행·속도·자세 reward, teacher distillation 오차처럼 stage 판단에 필요한 핵심 metric만
기록하며 전체 metric은 로컬 `monitor/metrics_history.jsonl`에 보존한다. 새 level-best가
나오면 즉시 `level/best_video`와 `level/best_video_levelY`에 올리고, 같은 파일을
`policy-video` artifact로도 큐잉한다. `monitor/artifacts.jsonl`의
`wandb_queued=true`와 콘솔의 `wandb_best_video_queued`로 업로드 요청 여부를 확인한다.
네트워크가 없는
서버에서는 `--wandb --wandb-mode offline`으로 실행한 뒤 `wandb sync`를 사용한다.
offline 원본은 각 run directory의 `wandb/` 아래에 남는다.

기본 영상 정책은 각 evaluation의 새 level-best가 나올 때 동일한 level별 GIF를
원자적으로 교체하는 것이다. stage 종료 시에는 누락된 경우 한 번 더 생성한다.
stage-final, progress, teacher 영상은 기본적으로 꺼져 있으며 필요한 경우에만 각각
`--stage-video`, `--progress-video`, `--teacher-video`로 켠다. 영상은 각 20초다.
개수·길이는 `--progress-video-count`,
`--progress-video-duration`, `--best-video-duration`, `--stage-video-duration`으로
조절하며 `--no-stage-video`로 최종 영상도 끌 수 있다.
