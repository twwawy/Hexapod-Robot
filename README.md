# Hexapod-Robot

6족 로봇의 하드웨어 자산과 MuJoCo MJX 기반 보행 실험을 함께 관리하는 저장소다. 현재 강화학습의 기준 경로는 **classical tripod gait + Cartesian residual RL**이다.

## 현재 기준 경로

Canonical contract: `classical_wbc_cartesian_body6d_residual_v1` action(24-D) +
`gt_attitude_collision_contact6_coarse9_touchdown6_v3` observation(113-D)

```text
forward/lateral/yaw command → Position/Heading PI → final body twist
        → controller-owned Tripod PULL/Bezier trajectory
        → phase/contact-masked 18-D foot residual + contact adaptation
        → attitude PI + 6-D body pose residual → six-leg workspace gate
        → URDF analytical IK → joint speed/position + fixed ±8 Nm safety
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
