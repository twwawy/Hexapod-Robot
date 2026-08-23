# Hexapod-Robot

6족 로봇의 하드웨어 자산과 MuJoCo MJX 기반 보행 실험을 함께 관리하는 저장소다. 현재 강화학습의 기준 경로는 **classical tripod gait + Cartesian residual RL**이다.

## 현재 기준 경로

```text
command → nominal tripod gait → nominal foot targets
        → RL Δz (swing legs only) → contact/safety → posture layer
        → linearized IK → joint limits → PD torque
```

- RL action: 다리 순서 `LF, LM, LB, RF, RM, RB`의 6차원 swing-foot `Δz`
- RL은 stance 발, 보폭, 착지 XY, gait timing, body pose를 직접 바꾸지 않는다.
- early landing은 contact/safety 계층이 현재 발 위치를 유지해 RL residual을 무시한다.
- 기존 7-D residual checkpoint는 action/observation 계약이 달라 재사용할 수 없다. `fresh`로 새 학습을 시작해야 한다.

설계·관측·보상·실행 방법은 [docs/RESIDUAL_RL.md](docs/RESIDUAL_RL.md)에만 최신 기준으로 정리한다.

## 주요 위치

- `SW/mjx/hexapod_mjx/residual_controller.py`: nominal gait, residual, contact safety, IK
- `SW/mjx/hexapod_mjx/residual_env.py`: MJX observation, reward, termination
- `SW/mjx/train_residual_ppo.py`: PPO 학습 및 checkpoint 계약
- `SW/mjx/visualize_residual_policy.py`: policy replay/render
- `Hexapod-MJX-가이드/residual_rl_run.sh`: 학습·재개·영상 생성 wrapper
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

$PY -m unittest SW.mjx.tests.test_rough_terrain_contract -v
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

Evaluation success 기준으로 level을 올리고 내리는 전체 curriculum:

```bash
$PY SW/mjx/train_competence_curriculum.py \
  --run-name mixed-competence \
  --stages 8 --stage-timesteps 5000000 \
  --start-level 0 --max-level 4 \
  --init-checkpoint SW/mjx/runs/command/<flat-run>/checkpoints \
  --wandb \
  -- --num-envs 2048 --num-evals 20 --terrain-randomize
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
