# Hexapod-Robot

6족 로봇의 CAD/URDF, MuJoCo MJX 보행 학습, LiDAR 높이 지도 뷰어와 Isaac Lab 이식 코드를 관리한다.
**현재 뷰어 제어는 펌웨어 기본 보행 + 학습된 residual + Safety/IK**다. 기본 제어기는 관측 여부와
무관하게 gait·접촉·자세를 계속 처리한다. LiDAR 착지 후보를 발의 절대 경로로 주입하는 전환은 제거했다.

다음 학습은 **MJX에서 보폭·주기·몸체 자세·스윙 높이의 residual을 학습하고 Isaac Lab으로 sim-to-sim 이전**하는 순서로 진행한다.
[파라미터 학습 설계](docs/HEXAPOD_MJX_ADAPTIVE_GAIT_LEARNING_PLAN.md)에 따라 **23-D MJX 환경·LiDAR actor/GT critic PPO·재생기**를 추가했다.
새 가중치는 아직 학습하지 않았으며 실행 검증은 사용자가 진행한다. 아래 기본 실행기는 기존 stage31 비교 모드다.

## 새 23-D 보행 파라미터 모드

LiDAR 기울기는 **45°로 롤백**했다. 새 모드는 기본 제어기를 유지하며 다리별 착지 XY·스윙 여유 높이,
몸체 pitch/roll/height·보폭·주기를 조절한다. 미관측 다리는 기본 궤적으로 시작하고 관측된 목표는 스윙 시작에 고정한다.

```bash
cd /home/huro/Hexapod-Robot
source /home/huro/.venvs/hexapod-mjx/bin/activate

# 먼저 기본 제어기 반복 보행 확인 (새 정책 없이 action 0)
bash scripts/view_foothold_planner.sh --controller adaptive --terrain flat --perception blind

# LiDAR 기반 착지/단차 보정 확인 (5 cm × 7단, action 0)
bash scripts/view_foothold_planner.sh --controller adaptive --terrain steps --perception lidar

# 사용자 확인 후 새 LiDAR 정책 학습 시작
bash scripts/train_adaptive_gait.sh --perception lidar --terrain-level 0 \
  --num-envs 64 --timesteps 10000000 --output mjx/runs/adaptive-lidar-flat
```

방향키로 계속 걷고, Space 정지, Enter 일시정지, H 초기화, C 지도 지우기, M 지도 표시, G LiDAR FOV, P trace 저장이다.
흐린 청록 점은 지도, 진한 녹색은 후보, 빨간 점은 수락한 목표다. 학습 가중치 없이도 LiDAR 기준값은 적용된다.
FOV는 기본 표시되며 주황색 하단(-7°)·파란색 상단(+52°) 경계와 센서 원점을 그린다.
`--fov-display-radius 1.2`로 표시 반경을 조절한다. 이는 실제 측정 거리가 아닌 시야각 안내선이다.
기존 18-D stage31 가중치는 새 모드에 호환되지 않는다.

[새 모드 실행·학습·환경 구성·확인 항목](docs/HEXAPOD_MJX_ADAPTIVE_GAIT_USAGE.md)에
checkpoint 재생, 계단 학습, teacher 이관, 관측 계약과 현재 제한을 정리했다.

## 바로 실행

```bash
cd /home/huro/Hexapod-Robot
source /home/huro/.venvs/hexapod-mjx/bin/activate
bash scripts/view_foothold_planner.sh --terrain steps
```

기존 창을 닫고 다시 실행한다. MuJoCo/MJX·JAX·Brax·Orbax가 있는 `~/.venvs/hexapod-mjx`를 사용한다.
스크립트가 해당 Python을 직접 선택하므로 activate는 선택 사항이다. 첫 실행에는 JAX 컴파일 시간이 필요하다.

기본 가중치는 `progress-v2-stage31-level6_20260828-111825_seed40`의 checkpoint `000001703936`이다.
학습용 skeleton 모델을 사용하고, 같은 창에서 방향키로 12×12 m 코스의 계단·경사·플랫폼을 살펴본다.

| 키 | 기본 동역학 모드 |
|---|---|
| ↑ / ↓ | 전후진 속도 ±0.04 m/s, 범위 ±0.12 m/s |
| ← / → | 회전 속도 ±0.15 rad/s, 범위 ±0.3 rad/s |
| Space / Enter | 속도 0 / 물리 일시정지·재개 |
| PageUp / PageDown | 몸체 높이 trim ±2 cm, 범위 -5~+10 cm |
| H / R | 상태·지도 초기화 / 인식 재시도·종료 상태 reset |
| 1~6 | 다리별 후보 상세 표시 |
| M / L / G | 높이 지도 / 점군 / MID-360 FOV 표시 |
| K / C / P | scan 전환 / 지도 초기화·속도 0 / 진단 저장 |
| F / T / V | 추적 / 위에서 보기 / 전체 코스 |

방향키는 속도를 증감하므로 키를 놓아도 계속 걷는다. 흰 점은 실제 제어 목표, 색상 점과 경로는
LiDAR 착지 후보다. **후보를 못 찾아도 기본 gait가 계속 진행하며, 후보가 실제 착지 목표를 뜻하지 않는다.**

## 현재 보행과 센서 입력

```text
명령 + 몸체/접촉 상태 → 펌웨어 nominal gait ─┐
                                           ├→ Safety / IK → MJX 동역학
LiDAR 높이 지도 + proprioception → RL residual ┘

LiDAR 지도 → 착지 후보 표시
LiDAR 샘플 ↔ simulator GT → 오차 비교/저장, 후속 학습의 정답
```

- 첫걸음에서 정책의 15개 높이 샘플이 모두 미관측이면 **NOMINAL / RL=0**으로 걷는다.
- 관측이 생기면 **NOMINAL + RL RESIDUAL**이다. 목표 gain은 `residual-scale × 관측 비율`이며 0.5초 시정수로 변화를 완화한다. 이 비율은 임시 가용성 기준이다.
- 미관측 입력은 중립값 0이며 지도에 관측으로 채우지 않는다. 관측이 모두 사라지면 residual/filter를 0으로 지운다.
- 원래 v3 펌웨어의 swing/stance/late landing, residual 의미, foot memory, IK·관절 속도 제한을 유지한다. 외부 경로·odom stance anchor를 넣지 않는다.
- GT 기반 pitch feedforward/swing boost는 끈다. 지형 GT 비교값은 actor나 제어 보정에 입력하지 않는다.

기존 가중치는 GT 높이 입력으로 학습됐다. LiDAR 공백/오차에 적응한 새 정책을 학습한 것은 아니며,
현재 입력 교체와 gain 조절은 통합 실험이다. 실제 등반 성공은 사용자가 검증한다.
LiDAR+IMU odometry 대신 현재 뷰어의 몸체 상태는 MuJoCo 상태를 사용한다.

## 비교 실행과 확인

```bash
# 같은 MJX 동역학에서 기본 제어기만 실행
bash scripts/view_foothold_planner.sh --terrain steps --residual-scale 0

# residual을 줄여 비교
bash scripts/view_foothold_planner.sh --terrain steps --residual-scale 0.25

# 평지
bash scripts/view_foothold_planner.sh --terrain flat

# 이전 기구학 착지 경로 모드; 동역학 비교와 구분
bash scripts/view_foothold_planner.sh --terrain steps --controller nominal

# 별도 v3 정책 재생: 저장된 지형 설정과 GT 입력
bash scripts/view_trained_policy.sh
```

사용자는 먼저 scale 0에서 연속 스윙을 보고, residual 적용 후 `IK valid`, `residual IK valid`,
`reach limited`, 흰 목표와 실제 발을 비교한다. K/C로 입력을 비웠을 때도 보행이 이어지는지 확인한다.
이번 수정은 코드 문법·diff만 확인했으며, GUI·추론·보행 테스트와 재학습은 실행하지 않았다.

P로 `mjx/generated/foothold_preview/`에 다음 자료를 저장한다.

| 파일 | 내용 |
|---|---|
| `latest_plan.json` | 제어 모드·gain·action·IK 상태·후보·현재 목표·LiDAR/GT 오차 |
| `latest_map.npz` | 높이·valid·시각·점군 |
| `latest_lidar_gt_pair.npz` | 동일 XY의 15개 LiDAR/GT 높이·valid·age·입력 pose/시각 |
| `scene_manifest.json` | 모델·checkpoint·센서·제어 설정 |

GT 오차는 관측된 샘플만 비교하며, 없으면 `n/a`다. 기준은 2 cm 코스 raster의 bilinear 높이라
물리 표면 모서리의 삼각형 보간과 차이가 날 수 있다. P는 최근 샘플을 덮어쓰며 대규모 학습 데이터셋 수집은 별도다.

## 센서·지형·모델

| 항목 | 뷰어 기본값 |
|---|---|
| LiDAR measured TF | 밑면 중심에서 높이 215 mm, 전방 13.529 mm, 전방 기울기 45°로 복원 |
| base→LiDAR | XYZ `(0,-0.013529,0.1642) m`, RPY `(0,45°,-90°)` |
| MID-360 proxy | 수평 360°, 수직 -7°~+52°, 거리 0.1~8 m, 720×64 angular rays |
| 높이 지도 | odom 정렬 8×8 m rolling map, 4 cm 셀, 60초 유지 |
| 불투명도 | 지도 16%, 점군 22%, 후보 마커 불투명 |
| 코스 | 12×12 m, 정면 4 cm 계단 6단·경사·플랫폼·돌출물·바위 |
| 로봇 | 학습 box/capsule/sphere skeleton; 구형 발 반지름 32 mm |

measured TF는 CAD의 URDF 장착 chain과 별도다. Livox 비반복 스캔 패턴을 정확히 재현하지 않는다.
물리·raycast는 같은 2 cm heightfield를 사용한다. 별도 정책 재생기의 기록 지형은 6.5 cm 계단 7단이다.
CAD 비교는 `--controller nominal --robot-model mesh`로 열 수 있다.

## 코드·가중치 버전 구분

| 경로 | 역할·계약 |
|---|---|
| `scripts/view_foothold_planner.sh` 기본 모드 | 격리된 v3/18-D action·146-D observation + LiDAR 입력과 residual gain |
| `scripts/view_trained_policy.sh` | stage31 가중치와 저장 설정으로 v3 재생, GT 입력 사용 |
| `--controller adaptive` / `scripts/train_adaptive_gait.sh` | 신규 23-D 파라미터 action, actor 641-D / critic 764-D; 아직 새 학습 결과 없음 |
| 루트 `mjx/` 학습 | v4/18-D action·146-D observation, swing X/Y·stance Z 최대 ±100 mm, curriculum 0~16 |
| `isaaclab_hexapod/` | v4 Torch 제어기·센서·지형·학습 scaffold와 USD 이식 |
| `SW/mjx/` | 과거 24-D/113-D 실험; 아래 레거시 설명 참고 |

v3와 v4는 차원이 같아도 residual scale이 달라 직접 checkpoint를 공유하지 않는다.
뷰어는 `0805164` 코드/URDF를 격리해 사용한다. 당시 미커밋 소스가 저장되지 않아 W&B 영상의 정확한 재현은 미확인이다.
Isaac handoff의 과거 안전 평가와 최신 v4 호환성은 별도로 기록하며, 기존 MJX 가중치를 자동 로드하지 않는다.

현재 루트 MJX 모델의 물리 기준은 10 kg 질량 정규화, DS51150-270 12.6 V servo,
14.709975 Nm 정격 모델, 315.8 deg/s 관절 속도 제한이다. `kp=500`, `kv=10`, armature 0.02,
damping 0.15, friction loss 0.8은 실측 식별 전 사용하는 모델 prior다.

```bash
# 루트 MJX의 v4 새 학습 예시; stage31 v3 재생과 구분
/home/huro/bin/hexapod-mjx-python mjx/train_competence_curriculum.py \
  --run-name firmware-terrain-residual-v4 --seed 8 \
  --flat-baseline-timesteps 262144 --start-level 1 --max-level 16 \
  --stages 44 --stage-timesteps 5000000 --level-progression competence \
  --checkpoint-selection best --wandb --wandb-project hexapod-firmware-terrain \
  -- --num-envs 1024 --num-evals 4 --num-eval-envs 32
```

## 문서와 업데이트 기록

- [문서 안내·최신 상태](docs/README.md)
- [뷰어 실행·환경·사용자 확인 순서](docs/HEXAPOD_FOOTHOLD_PREVIEW_USAGE.md)
- [LiDAR 입력, GT 정답, residual 학습 구조](docs/HEXAPOD_LIDAR_FOOTHOLD_RESIDUAL_PLAN.md)
- [2026-09-06 수정 및 미커밋 업데이트 정리](docs/HEXAPOD_UPDATE_2026-09-06.md)
- [현재 MJX 학습 계약](mjx/RL_DESIGN.md), [기본 펌웨어 설명](mjx/FIRMWARE_BASE.md)
- [Isaac Lab 실행 안내](isaaclab_hexapod/README.md), [이식 기록](isaaclab_hexapod/PORT_RESULT_AND_USAGE.md)
- [stage31 가중치 출처](mjx/policies/progress-v2-stage31-level6/README.md)

## 레거시 `SW/mjx/` 경로

아래 24-D/113-D `SW/mjx/` 설명은 과거 classical whole-body residual 실험을
재현하기 위한 참고 자료다. 현재 루트 `mjx/`의 18-D/146-D v4 firmware residual이나
10 kg/DS51150 동역학 계약과 checkpoint를 섞어 사용하지 않는다.

Canonical contract: `classical_wbc_cartesian_body6d_residual_v1` action(24-D) +
`gt_attitude_collision_contact6_coarse9_touchdown6_v3` observation(113-D)

```text
forward/lateral/yaw command → Position/Heading PI → final body twist
        → controller-owned Tripod PULL/Bezier trajectory
        → phase/contact-masked 18-D foot residual + contact adaptation
        → attitude PI + 6-D body pose residual → six-leg workspace gate
        → URDF analytical IK → joint speed/position + fixed ±8 Nm legacy actuator
```

- Action 24-D: 다리 순서 `RF, RM, RB, LF, LM, LB`의 발 `[Δx, Δy, Δz] × 6` + 몸체 병진/회전 6-D
- Swing leg는 XYZ residual을, stance leg는 Z-only residual을 쓴다. stance 착지 XY는 항상 정확히 0이다.
- 마지막 6-D는 몸체 forward/lateral/height `±50/±50/±100 mm`, roll/pitch/yaw `±45/±45/±25 deg`이며 0.15초 filter와 전체 workspace 승인을 거친다.
- gait phase/stride/frequency와 nominal swing height `0.20 m`, radial offset `0.07 m`는 정책이 아니라 제어기가 소유한다.
- 0.5초 phase가 끝나도 swing 3발 착지가 확인되지 않으면 다음 tripod를 들지 않고 late-landing 탐색을 계속한다.
- 접촉은 압력센서·발높이 추정 없이 MuJoCo foot–world terrain collision만 사용하고,
  roll/pitch/yaw는 MuJoCo root ground truth를 사용한다.
- 우선순위는 `safety > contact > RL residual > nominal gait`다. early landing이면 contact/safety 계층이 현재 발 위치를 유지해 residual을 무시한다.
- 이전 6-D/7-D/22-D residual checkpoint는 action/observation 계약이 달라 재사용할 수 없다. `fresh`로 새 학습을 시작해야 한다.

설계·관측·보상·실행 방법은 [docs/RESIDUAL_RL.md](docs/RESIDUAL_RL.md)에만 최신 기준으로 정리한다.

## 주요 위치

- `SW/mjx/tripod_core.py`: source-controller PI, Tripod PULL/Bezier, contact phase, body overlay, IK safety
- `SW/mjx/rough_terrain_env.py`: 24-D residual action, 113-D observation, reward와 termination
- `SW/mjx/train_command_curriculum.py`: flat walking+turning curriculum 학습 진입점
- `SW/mjx/train_rough_terrain.py`: mixed terrain 학습(task 엔진, command 학습도 여기서 재사용)
- `SW/mjx/train_competence_curriculum.py`: evaluation success로 level을 올리고 내리는 전체 curriculum launcher
- `SW/mjx/best_policy_video.py` / `SW/mjx/visualize_residual_policy.py`: 최고 policy GIF 렌더 / viewer replay
- `Hexapod-MJX-가이드/residual_rl_run.sh`: 레거시 7-D residual(`train_residual_ppo.py`) 학습·재개·영상 wrapper
- `HW/`: 실제 로봇의 URDF, CAD, PCB, 부품 자료

## 빠른 시작

저장소를 클론하고 가상환경을 준비한다. 아래 문서와 스크립트는 저장소 위치를
`~/Hexapod-Robot`으로 가정하므로 다른 경로에 클론했다면 `HEXAPOD_ROBOT_REPO`
환경 변수로 지정한다.

```bash
git clone https://github.com/seoneum/hexapod-residual-rl-mjx.git ~/Hexapod-Robot
cd ~/Hexapod-Robot

# 최초 1회: 가상환경 + 의존성
python3 -m venv ~/.venvs/hexapod-mjx
~/.venvs/hexapod-mjx/bin/python -m pip install -r SW/mjx/requirements-train.txt
# NVIDIA GPU(CUDA 12) 머신에서만 CPU JAX를 CUDA wheel로 교체
~/.venvs/hexapod-mjx/bin/python -m pip install --upgrade "jax[cuda12]==0.6.2"
~/.venvs/hexapod-mjx/bin/python -c "import jax; print(jax.devices())"  # GpuDevice 확인
```

### W&B 준비 (최초 1회)

학습 스크립트는 `--wandb`를 붙일 때 로그인된 W&B 계정으로 기록한다.

```bash
~/.venvs/hexapod-mjx/bin/pip install wandb   # requirements-train.txt에는 미포함이라 별도 설치
~/.venvs/hexapod-mjx/bin/wandb login         # https://wandb.ai/authorize 의 API key 입력
```

### 실행 명령

가장 작은 검증 실행(wrapper 사용):

```bash
cd ~/Hexapod-Robot
./Hexapod-MJX-가이드/빠른학습.sh fresh   # fresh 대신 이어서(resume)는 인자 생략
```

Smoke 테스트(환경·계약 검증, 학습 없음 수준의 짧은 rollout):

```bash
PY=~/.venvs/hexapod-mjx/bin/python

$PY SW/mjx/train_command_curriculum.py \
  --smoke --smoke-steps 100 --run-name command-smoke

$PY SW/mjx/train_rough_terrain.py \
  --smoke --smoke-steps 100 --terrain-layout mixed --terrain-level 4 \
  --terrain-randomize --run-name terrain-smoke

$PY -m unittest discover -s SW/mjx/tests -v
```

본 학습 — flat walking+turning부터 fresh로 시작한다:

```bash
$PY SW/mjx/train_command_curriculum.py \
  --run-name flat-transfer-source \
  --timesteps 50000000 --num-envs 2048 --num-evals 100 \
  --wandb --wandb-project hexapod-command-curriculum
```

Flat checkpoint로 mixed terrain을 초기화한다:

```bash
$PY SW/mjx/train_rough_terrain.py \
  --run-name terrain-transfer-level0 \
  --terrain-layout mixed --terrain-level 0 \
  --init-checkpoint SW/mjx/runs/command/<flat-run>/checkpoints \
  --timesteps 50000000 --num-envs 2048 --num-evals 100 \
  --wandb --wandb-project hexapod-rough-terrain
```

짧은 flat baseline부터 시작해 rough→총 20 cm 계단으로 진행하는 전체 curriculum:

```bash
$PY SW/mjx/train_competence_curriculum.py \
  --run-name mixed-body6d \
  --flat-baseline-timesteps 1000000 \
  --stages 8 --stage-timesteps 5000000 \
  --level-progression sequential \
  --wandb \
  -- --num-envs 2048 --num-evals 20 --terrain-randomize --best-video
```

Checkpoint·monitor·GIF는 run directory(`SW/mjx/runs/<task>/<name>_<timestamp>_seed<seed>/`)에
저장되고, 같은 `--run-name`을 다시 써도 timestamp 때문에 섞이지 않는다.
레거시 7-D residual 워크플로(`residual_rl_run.sh`)는
`./Hexapod-MJX-가이드/residual_rl_run.sh --fresh`로 실행할 수 있다.

전체 명령 옵션과 평가 기준은 [Residual RL 가이드](docs/RESIDUAL_RL.md)를 따른다.

## W&B 보는법

`--wandb`로 실행하면 브라우저에서 대시보드로 결과를 볼 수 있다.

- 프로젝트 URL: `https://wandb.ai/<내-entity>/<project>`
  - flat command 학습 → project `hexapod-command-curriculum`
  - mixed terrain 학습 → project `hexapod-rough-terrain`
  - competence launcher → baseline과 모든 stage를 project
    `hexapod-rough-terrain`, group `<run-name>`에 함께 기록
  - legacy wrapper(`residual_rl_run.sh`) → project `hexapod-residual-rl`
  - entity 없이 로그인했다면 URL에서 entity 부분이 개인 계정명이 된다.
- 공통 x축은 `train/global_step`이다. Panels에서 x축을 이 값으로 고정하면
  eval 주기가 달라도 run 간 비교가 어긋나지 않는다.

주요 지표:

| 지표 | 의미 |
| --- | --- |
| `eval/episode_reward` | 종합 reward. 학습이 잘 되고 있는지 첫 번째로 보는 곡선 |
| `eval/episode_reward/*` | reward 구성요소(tracking/upright/cost 등) 분해 |
| `eval/episode_terrain_success` | episode 종료 시 성공(termination 없음 + 속도 오차 < 0.08 m/s + progress > 0.5 m) 비율 |
| `eval/stage0|1|2/reward_mean` 외 | flat run의 고정 script 평가(속도 오차, yaw 오차, torque 등) |
| `best/*` | NEW_BEST 시점 스냅샷 지표 |
| `best/video_stage0_forward` ~ `best/video_curriculum_full` | flat run 최고 policy GIF (stage별 + 전체) |
| `best/video` | terrain run 최고 policy GIF (`videos/best_policy.gif`와 동일) |
| `progress/video` | 각 flat/stage run의 0·25·50·75·100% 시점 20초 GIF 추세 |

위 curriculum 명령은 flat baseline과 terrain stage 0~7을 W&B project
`hexapod-rough-terrain`의 group `mixed-body6d` 아래 별도 run으로 묶는다. 각 run의 `progress/video` panel에는 학습
0/25/50/75/100% 시점 영상이 순서대로 쌓이며, 로컬 원본은 각 run의
`videos/progress/`에 남는다. 기본은 run당 5개·각 20초이고
`--progress-video-count`, `--progress-video-duration`으로 조절할 수 있다. NEW_BEST
영상도 별도 `best/*` 키로 계속 저장된다.

Adaptive launcher는 `eval/episode_terrain_success`(evaluation success)가 0.8보다
크면 level을 올리고 0.5보다 낮으면 내린다. Competence curriculum 진행 상황도 이
지표로 판단한다.

네트워크 접속이 없는 서버에서는 offline으로 기록한 뒤 나중에 올릴 수 있다.

```bash
$PY SW/mjx/train_command_curriculum.py ... --wandb --wandb-mode offline
wandb sync SW/mjx/runs/command/<run-dir>   # 또는 ./wandb/offline-run-* 디렉터리
```

W&B 기록 없이 로컬 파일만 남기려면 `--wandb` 플래그를 빼면 된다. monitor JSON과
GIF는 run directory에 그대로 저장된다.

## 참고 자료

`완전 튜토리얼.md`는 초기 MuJoCo/Isaac 학습을 위한 배경 자료다. 현재 MJX residual 구현의 명세나 실행 기준은 [docs/RESIDUAL_RL.md](docs/RESIDUAL_RL.md)다.
## 전체 프로젝트 개요

인하대학교 로봇연구회가 2024년부터 개발하고 있는 6족 보행 로봇 프로젝트이다. Jetson Orin Nano Super가 인지와 자율주행을 담당하고, STM32 NUCLEO-F446RE가 센서 수집과 200 Hz 실시간 보행 제어를 담당하는 구조를 목표로 한다.

### 저장소 구성

| 경로 | 내용 |
|---|---|
| [HW](HW/README.md) | 기구, PCB, 부품, URDF와 제작 파일 |
| [SW/Controller](SW/Controller/Controller_Architecture.md) | MATLAB/Simulink 보행 제어기 |
| [SW/STM32](SW/STM32/STM32F446RE%20설정%20정리본.md) | STM32 설정과 펌웨어 |
| [SW/Jetson](SW/Jetson/README.md) | Jetson 상위 제어 소프트웨어 |

### 주요 문서

- [제어기 Architecture](SW/Controller/Controller_Architecture.md)
- [제어기 상세 설계](SW/Controller/Controller_detail.md)
- [좌표계와 관절 정의](SW/Controller/좌표축/README.md)
- [드론 조종기 입력](SW/Controller/드론%20조종기%20입력/README.md)
- [STM32F446RE 설정](SW/STM32/STM32F446RE%20설정%20정리본.md)
- [하드웨어 부품 목록](HW/parts.md)
