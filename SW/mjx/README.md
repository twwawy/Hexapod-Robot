# Hexapod-Robot
This is a hexapod walking robot project developed by the Inha University Robot Research Society since 2024. It focuses on stable autonomous locomotion across diverse terrains.

> Integration note: this README was imported from `~/Downloads/mjx`. In this repository the directory is `SW/mjx/`, not `<repo>/mjx/`; use the exact commands in `docs/Hexapod_MJX_Obsidian_Study_Vault.md` for the current virtualenv and paths.

## MuJoCo/MJX: 로봇 모델 띄우기

기존 ROS Xacro/URDF 모델을 독립 실행 가능한 MJCF 장면으로 변환한 뒤 MuJoCo
뷰어에서 표시합니다. ROS 설치는 필요하지 않습니다. 장면에는 바닥, floating
base와 문서에 정의된 기본 자세(각 다리 관절 `0°`, `30°`, `50°`)가 포함됩니다.
18개 위치 actuator가 해당 자세를 유지하며, 원시 URDF에서 좌우 반전된 관절축은
자동으로 서보 각도에 맞게 변환됩니다.

```bash
source /home/huro/.venvs/hexapod-mjx/bin/activate
cd /home/huro/Hexapod-Robot
python -m pip install -r SW/mjx/requirements.txt
python SW/mjx/view_robot.py
```

창을 띄울 수 없는 서버에서는 PNG로 렌더링할 수 있습니다.

```bash
MUJOCO_GL=egl python SW/mjx/view_robot.py --headless
```

생성된 `SW/mjx/generated/hexapod_scene.xml`은 원본 Xacro나 변환 코드가 변경되면
자동으로 다시 생성됩니다. 화면 확인용 CAD 장면과 별도로, 강화학습에서는 GPU
병렬화에 적합한 primitive 충돌 형상의 `hexapod_rl.xml`을 사용합니다.

## 기존 설계 기반 Tripod 보행

`Controller_detail.md`의 200 Hz Tripod Gait, Stance PULL, Quintic-scaled
Bezier Swing 및 3DOF IK를 MuJoCo 위치 actuator에 연결한 첫 보행 데모입니다.

```bash
python SW/mjx/run_controller.py
```

처음 1초 동안 기본 자세를 유지한 뒤 0.06 m/s 전진 명령을 9초 동안 실행합니다.
초기 안전 검증값은 Swing Height 0.06 m, 방사 오프셋 0.01 m입니다. 설계 문서의
Simulink 검증값은 실제 동역학에서 공격적인 설정이므로 궤적 시험 시에만 다음처럼
적용합니다.

```bash
python SW/mjx/run_controller.py --speed 0.14 \
  --swing-height 0.25 --radial-offset 0.07
```

화면 없는 환경의 빠른 검증:

```bash
MUJOCO_GL=egl python SW/mjx/run_controller.py --headless --duration 5
```

후진 시험은 `--speed -0.06`으로 실행할 수 있습니다.

## MJX Residual RL: 평지 명령과 mixed-terrain curriculum

기본 Tripod/IK 제어기 위에서 **swing XYZ / stance Z-only** 권한으로만 보정하는
`cartesian_gait_residual_v2` 22차원 residual 환경입니다. contact adaptation과
workspace projection이 policy보다 먼저 적용됩니다. 110-D observation의 terrain
부분은 3×3 coarse grid와 6개 nominal touchdown 높이로 구성됩니다. 평지에서
보행·회전을 먼저 학습한 뒤 flat checkpoint로 flat/curb/ramp/blocks/stairs/rough가
같은 XML에 들어 있는 mixed terrain을 초기화합니다. 자세한 최신 명세는
[`docs/RESIDUAL_RL.md`](../../docs/RESIDUAL_RL.md)를 참고합니다.

Flat command Stage 2는 최대 `0.21 m/s`를 학습합니다. `phase_time=0.50 s`는
유지하고, forward+yaw를 합친 가장 빠른 다리 기준 horizontal stroke를 120 mm로
제한합니다. Actuator hard clamp ±8 Nm와 함께 torque saturation(6.8 Nm 초과), slip,
self-collision을 별도 reward/metric으로 감시합니다. Terrain 속도 상한은 0.18 m/s입니다.

학습 전에 계단 장면과 residual이 0인 기본 보행을 눈으로 확인합니다. 기본
제어기가 계단에서 보이는 진행 거리와 자세가 이후 학습 결과의 비교 기준입니다.

```bash
python SW/mjx/view_rl_scene.py
```

GUI가 없는 머신에서는 움직이는 GIF를 생성할 수 있습니다.

```bash
MUJOCO_GL=egl python SW/mjx/view_rl_scene.py --headless
```

```bash
python -m pip install -r SW/mjx/requirements-train.txt
# NVIDIA GPU/CUDA 12 머신에서 CPU JAX를 CUDA wheel로 교체
python -m pip install --upgrade "jax[cuda12]==0.6.2"
# pip CUDA와 /usr/local/cuda 라이브러리가 섞이지 않도록 현재 셸에서 제거
unset LD_LIBRARY_PATH
python -c "import jax; print(jax.devices())"
python SW/mjx/train_command_curriculum.py \
  --smoke --smoke-steps 100 --run-name command-v2-smoke
python SW/mjx/train_command_curriculum.py \
  --run-name flat-transfer-source --num-evals 100 --wandb

# mixed terrain smoke
python SW/mjx/train_rough_terrain.py \
  --smoke --smoke-steps 100 --terrain-layout mixed --terrain-level 4 \
  --terrain-randomize --run-name terrain-v2-smoke

# 위 flat run의 checkpoint로 terrain policy를 초기화한다.
python SW/mjx/train_rough_terrain.py \
  --run-name terrain-transfer-level0 --terrain-layout mixed --terrain-level 0 \
  --init-checkpoint SW/mjx/runs/command/<flat-run>/checkpoints \
  --num-evals 100 --wandb

# 평가 성공률로 level을 올리고 내리는 전체 curriculum
python SW/mjx/train_competence_curriculum.py \
  --run-name mixed-competence --stages 8 --stage-timesteps 5000000 \
  --init-checkpoint SW/mjx/runs/command/<flat-run>/checkpoints --wandb \
  -- --num-envs 2048 --num-evals 20 --terrain-randomize
```

실제 학습 전 장치 출력에 `GpuDevice`가 있어야 합니다. 이 프로젝트가 고정한 JAX
0.6.2 조합은 위 CUDA 12 wheel을 사용합니다. 기본 학습 설정은 2048개 병렬 환경과
5천만 environment step입니다.

각 run은 `SW/mjx/runs/<task>/<name>_<timestamp>_seed<seed>/`에 checkpoint, `monitor/`,
`videos/`, `config.json`, `run_metadata.json`을 함께 저장합니다. Flat command run은
매 evaluation마다 Stage 0/1/2를 독립 reset·고정 command로 평가하고, NEW_BEST에서
`best_stage0_forward.gif`, `best_stage1_limited_yaw.gif`,
`best_stage2_full_command.gif`, `best_curriculum_full.gif` 네 파일을 갱신합니다.
W&B에도 각각 `best/video_stage0_forward`, `best/video_stage1_limited_yaw`,
`best/video_stage2_full_command`, `best/video_curriculum_full`로 올라갑니다. Terrain
run은 `videos/best_policy.gif`와 W&B `best/video` 한 개를 유지합니다.

같은 `--run-name`을 다시 사용해도 timestamp suffix 때문에 기존 checkpoint/monitor와
섞이지 않습니다. `--best-video-path`를 생략하면 새 run directory 안에 저장됩니다.

Command 영상은 기본 Stage 0/1/2/full이 각각 `10/10/12/22초`이며 모든 frame에
stage, `v_cmd/v`, `yaw_cmd/yaw` overlay가 들어갑니다. Full 영상은 stage 전환을
0.8초 banner로 표시합니다. 길이는 `--best-video-stage0-duration` 등으로, 공통 품질은
`--best-video-fps`, `--best-video-width`, `--best-video-height`로 바꿉니다. 영상이
필요 없는 짧은 debug run에는 `--no-best-video`를 사용합니다. 렌더링 실패는
`best_video_error`만 출력하며 학습·checkpoint 저장을 멈추지 않습니다. X11 DISPLAY가
없는 tmux/SSH 실행에서는 trainer가 `MUJOCO_GL=egl`을 자동 선택해 GPU offscreen
렌더링으로 GIF를 저장합니다. 설치/드라이버가 다른 경우에만 실행 전에 명시적으로
`MUJOCO_GL=egl`(GPU) 또는 사용 가능한 다른 MuJoCo backend를 지정합니다.
