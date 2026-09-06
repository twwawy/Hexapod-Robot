# Hexapod-Robot

6족 로봇의 하드웨어 자산과 MuJoCo MJX 기반 보행 실험을 함께 관리하는 저장소다. 현재 강화학습의 기준 경로는 **classical tripod gait + Cartesian residual RL**이다.

> 이 작업 브랜치의 STM32 펌웨어 직접 이식 기반 지형 curriculum 학습은 루트
> `mjx/`에 있다. 실제 펌웨어의 gait/contact/posture/IK를 base controller로
> 고정하고 안전-gated 18-D 발끝 residual만 학습한다. 실행 명령, W&B,
> 평지→울퉁불퉁→경사면→10×20cm 최종 계단, best/stage/progress 영상과 checkpoint 규칙은
> [mjx/RL_DESIGN.md](mjx/RL_DESIGN.md)를 따른다.

최신 인식·보행 구조는 **LiDAR + IMU 상태 추정 → local elevation map → geometric
foothold correction → RL residual → Safety / IK**다. D435IF RGB는 후속 단계에서
지도 위 traversability/semantic score로 착지 후보를 평가하는 데 추가한다.
구현 단계와 인터페이스는
[LiDAR 착지점·residual 설계](docs/HEXAPOD_LIDAR_FOOTHOLD_RESIDUAL_PLAN.md)를 따른다.

## 착지점 뷰어에서 학습 정책으로 이동

기존 착지점 뷰어의 기본 제어는 **stage31 학습 정책 + MJX 동역학**이다.
`progress-v2-stage31-level6_20260828-111825_seed40`의 최고 점수 checkpoint
`000001703936`과 관측 정규화 통계를 불러온다.

```bash
cd /home/huro/Hexapod-Robot
source /home/huro/.venvs/hexapod-mjx/bin/activate
bash scripts/view_foothold_planner.sh --terrain steps
```

같은 창 안에서 학습용 링크 로봇을 방향키로 조종하며 12×12 m 코스의 계단·경사로·플랫폼·
징검다리를 지나고, LiDAR 점군·높이 지도·착지 후보를 살펴볼 수 있도록 구성했다.
처음에는 이동 명령 0, **`NOMINAL / RL=0`**으로 시작한다. 첫걸음의 착지 구간이 미관측이면
펌웨어 기본 gait로 걷는다. 관측된 착지점이 확보된 다리는 다음 스윙 시작 때
**`FOOTHOLD + RL`**로 전환한다. 공중에 있는 발의 목표는 지도 갱신만으로 바꾸지 않는다.

| 키 | 기본 학습 정책 모드의 동작 |
|---|---|
| ↑ / ↓ | 전후진 command ±0.04 m/s; 범위 ±0.12 m/s |
| ← / → | yaw command ±0.15 rad/s; 범위 ±0.3 rad/s |
| Space | 이동 command 0; 물리와 자세 제어는 계속 실행 |
| Enter | 동역학 일시정지 / 재개 |
| PageUp / PageDown | 몸체 높이 보정 ±2 cm; 범위 -5~+10 cm |
| H | 로봇·정책·지도 초기화 |
| 1~6 | 다리별 LiDAR 후보 상세 선택 |
| M / L / G | 높이 지도 / 점군 / MID-360 FOV 표시 |
| P / C / K | 지도·계획 저장 / 지도 초기화 / LiDAR scan 전환 |
| F / T / V | 추적 카메라 / 위에서 보기 / 전체 코스 |

**흰 점은 현재 발 제어 목표**, 큰 색상 점과 경로는 착지 계획이다. 스윙 시작 때 선택한
관측 경로를 고정한다. 주황 경로는 nominal 예측이다. 미관측 다리는 기존 펌웨어 궤적과 **RL=0**, 관측 다리는
**nominal gait → LiDAR 착지점 보정 → 제한된 stage31 residual → Safety/IK → MJX 동역학**을 사용한다.
알려진 장애물이나 도달 불가능한 착지점은 다음 스윙을 hold한다. 단순 미관측은 hold 사유가 아니다.
146-D 관측 중 지형 15개 항목은 LiDAR 관측 셀만 사용한다. 미관측 값은 중립값 0이며,
지도 자체를 채우지 않는다. 시뮬레이터 지형 기반 pitch feedforward/swing boost는 끈다.
이 결합은 원래 학습 정책 그대로의 재생과 다르며, 실행·계단 등반 검증은 사용자가 수행한다.
후진·회전은 이 run의 학습 command 범위 밖이며, 횡이동 A/D는 지원하지 않는다.

센서는 밑면 중심에서 높이 **215 mm**, 전방 **13.529 mm**, 위를 향해 전방으로 **45°** 기울어진
TF를 사용한다. MID-360 FOV는 수평 360°, 수직 -7°~+52°다. 지도는 8×8 m, 4 cm 셀,
60초 관측 유지이며 기본 불투명도는 지도 16%, 점군 22%다.

기본 모드의 넓은 지형은 기존 장애물 배치에서 만든 **2 cm heightfield**다.
물리 접촉과 LiDAR가 같은 표면을 사용하고, 정책에는 이 표면의 높이를 보간해 제공한다.
정면은 기존 탐색 코스의 4 cm 계단 6단이며, 별도 run 재현 환경의 6.5 cm 계단 7단과 구분한다.

### 환경과 비교 모드

가상환경은 `~/.venvs/hexapod-mjx`이며 스크립트가 직접 사용하므로 activate는 선택 사항이다.
**JAX/MJX/Brax가 설치된 기존 학습 환경**이 필요하다. 첫 실행에는 JAX 컴파일 시간이 걸린다.
GPU 계산은 별도 프로세스에서 수행하며, CPU 뷰어와 LiDAR worker는 같은 로봇 자세를 받아 표시한다.
실제 하드웨어로 명령을 보내지는 않는다.

```bash
# 기존 LiDAR 후보 기반 기구학 제어와 비교
bash scripts/view_foothold_planner.sh --terrain steps --controller nominal

# 평지에서 학습 정책 확인
bash scripts/view_foothold_planner.sh --terrain flat

# 지도와 점군을 더 옅게 표시
bash scripts/view_foothold_planner.sh --terrain steps --map-alpha 0.10 --lidar-point-alpha 0.15

# 별도 창에서 run에 기록된 6.5 cm 계단 7단 재현
bash scripts/view_trained_policy.sh
```

학습 정책 모드의 로봇/제어기는 기록된 v3 코드 버전을 별도로 추출해 사용한다.
`--revision`과 CAD 외형 비교는 `--controller nominal`에만 적용한다.
학습 당시 미커밋 소스가 없어 W&B 영상과 완전히 같은 재현은 미확인이며,
사용자 요청에 따라 이번 통합본의 실행·정책 추론·계단 보행 검증은 수행하지 않았다.

자세한 구성·조작·확인 항목은 [착지점 뷰어 안내](docs/HEXAPOD_FOOTHOLD_PREVIEW_USAGE.md),
가중치 출처는 [학습 정책 안내](mjx/policies/progress-v2-stage31-level6/README.md)를 참고한다.

## 현재 MJX 기준

현재 학습 경로는 루트 `mjx/`이며 다음 물리 계약을 사용한다.

| 항목 | 현재 값 |
|---|---|
| RL 로봇 총질량 | 정확히 `10.0 kg` (`prepare_rl_scene.py`에서 원본 링크 질량·관성을 비례 스케일) |
| 액추에이터 | DS51150-270, 12.6 V 기준 geared position servo |
| 정격 모델 | 357:1, 150 kgf·cm = `14.709975 Nm`, 0.19 s/60° = `315.8 deg/s` |
| MuJoCo servo prior | `kp=500`, `kv=10`, armature `0.02 kg·m²`, damping `0.15 Nms/rad`, friction loss `0.8 Nm` |
| 정책 계약 | 18-D adaptive-swing foot residual v3 / 142-D observation |

토크·기어비·무부하 속도는 DS51150 제조사 사양을 사용한다. `kp`, `kv`, armature,
damping, friction loss는 제조사가 제공한 식별값이 아니라 실기 벤치시험 전 사용하는
명시적 초기값이다. 실제 로봇 자체의 실측 질량을 10 kg로 단정하는 값도 아니다.

기본 Swing은 6 cm이며 정책의 다리별 Z action이 Swing 중간 높이를 4~25 cm로
조절한다. 위상 envelope 때문에 이륙점과 착지점의 Z는 바뀌지 않는다. Stance Z는
±20 mm로 제한하고 Late Landing은 펌웨어가 단독으로 제어한다.

기존 v2 Level 4 actor는 Z 출력이 Cartesian 위치 offset이라 v3 높이 명령과 의미가
다르므로 직접 restore하지 않는다. 새 동역학과 adaptive swing은 level 0부터 새로
학습한다. 환경 생성 시 `mjx/generated/hexapod_rl.xml`은 자동 재생성된다.

```bash
cd /home/huro/Hexapod-Robot
/home/huro/bin/hexapod-mjx-python mjx/train_competence_curriculum.py \
  --run-name firmware-terrain-adaptive-swing \
  --seed 8 \
  --flat-baseline-timesteps 262144 \
  --start-level 1 --max-level 12 \
  --stages 44 --stage-timesteps 5000000 \
  --level-progression competence \
  --checkpoint-selection best \
  --wandb --wandb-project hexapod-firmware-terrain \
  -- --num-envs 1024 --num-evals 4 --num-eval-envs 32
```

자세한 active contract, curriculum, 테스트와 실행 옵션은
[mjx/RL_DESIGN.md](mjx/RL_DESIGN.md)와 [mjx/FIRMWARE_BASE.md](mjx/FIRMWARE_BASE.md)를
기준으로 한다.

## 레거시 `SW/mjx/` 경로

아래 24-D/113-D `SW/mjx/` 설명은 과거 classical whole-body residual 실험을
재현하기 위한 참고 자료다. 현재 루트 `mjx/`의 18-D/142-D firmware residual이나
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
