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

## MJX Residual RL: 평지 명령 커리큘럼과 계단 지형

기본 Tripod/IK 제어기 위에서 **swing XYZ / stance Z-only** 권한으로만 보정하는
`cartesian_gait_residual_v2` 22차원 residual 환경입니다. contact adaptation과
workspace projection이 policy보다 먼저 적용됩니다. 평지에서 **보행 → 완만한 회전
→ 전체 보행+회전**을 하나의 1,000-step curriculum으로 먼저 학습하고, 계단 지형은
별도 명령으로 학습합니다. 기존 v1 22-D checkpoint는 resume하지 않습니다. 자세한 구조는
[`RL_DESIGN.md`](RL_DESIGN.md)와 Obsidian 가이드를 참고합니다.

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
python SW/mjx/train_command_curriculum.py --smoke --run-name command-v2-smoke
python SW/mjx/train_command_curriculum.py --run-name command-v2-seed0 --num-evals 50 --wandb

# 계단 지형은 평지 커리큘럼과 별도 run/checkpoint로 실행한다.
python SW/mjx/train_rough_terrain.py --smoke --run-name terrain-v2-smoke
python SW/mjx/train_rough_terrain.py \
  --run-name terrain-v2-level3-seed0 --terrain-level 3 --terrain-randomize \
  --num-evals 50 --wandb
```

실제 학습 전 장치 출력에 `GpuDevice`가 있어야 합니다. 이 프로젝트가 고정한 JAX
0.6.2 조합은 위 CUDA 12 wheel을 사용합니다. 기본 학습 설정은 2048개 병렬 환경과
5천만 environment step입니다.

각 run은 `SW/mjx/runs/<task>/<timestamp>_seed<seed>/`에 checkpoint, `monitor/`,
`config.json`, `run_metadata.json`을 함께 저장합니다. 기본으로
`--best-video`가 켜져 있어 `eval/episode_reward`가 새 최고점일 때마다 그 policy의
deterministic 10초 GIF를 `<run-dir>/best_policy.gif`에 자동으로 저장하고 이전
최고 영상을 교체합니다. W&B run에서는 같은 파일을 `best/video`로도 업로드합니다.

```bash
# 예: run별 최고 policy 영상의 위치를 명시한다. (생략하면 run-dir/best_policy.gif)
python SW/mjx/train_command_curriculum.py \
  --run-name command-v2-seed0 \
  --best-video \
  --best-video-path SW/mjx/runs/command/command-v2-seed0/best_policy.gif \
  --wandb
```

영상은 기본 `10초 / 20 fps / 640x360`이고 `--best-video-duration`,
`--best-video-fps`, `--best-video-width`, `--best-video-height`로 바꿉니다. 영상이
필요 없는 짧은 debug run에는 `--no-best-video`를 사용합니다. 렌더링 실패는
`best_video_error`만 출력하며 학습·checkpoint 저장을 멈추지 않습니다. X11 DISPLAY가
없는 tmux/SSH 실행에서는 trainer가 `MUJOCO_GL=egl`을 자동 선택해 GPU offscreen
렌더링으로 GIF를 저장합니다. 설치/드라이버가 다른 경우에만 실행 전에 명시적으로
`MUJOCO_GL=egl`(GPU) 또는 사용 가능한 다른 MuJoCo backend를 지정합니다.
