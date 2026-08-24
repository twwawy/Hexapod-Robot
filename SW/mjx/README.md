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

Position/Heading PI, Tripod PULL/Bezier, contact adaptation, 단일 자세 PI와 IK가
보행 전체를 소유하고 policy는 **swing XYZ / stance Z-only + body 6-DOF**만 보정하는
`classical_wbc_cartesian_body6d_residual_v1` 24차원 환경입니다. 113-D observation의
terrain 부분은 3×3 coarse grid와 6개 nominal touchdown 높이로 구성됩니다. 평지에서
짧게 baseline/residual을 잡고 rough를 거친 뒤 전체 상승 최대 20 cm 계단으로 갑니다.
접촉은 압력센서나 발높이 proxy 없이 MuJoCo foot–world terrain collision만 사용하고,
roll/pitch/yaw는 IMU 대신 MuJoCo root ground truth를 사용합니다.
자세한 최신 명세는
[`docs/RESIDUAL_RL.md`](../../docs/RESIDUAL_RL.md)를 참고합니다.

Flat command는 Stage 0/1/2에서 각각 최대 `0.10/0.18/0.27 m/s`를 학습합니다.
`phase_time=0.50 s`는
유지하고, forward+lateral+yaw를 합친 가장 빠른 다리 기준 horizontal stroke를 140 mm로
제한합니다. 고속에서는 gait가 낼 수 있는 범위 안으로 yaw command를 자동 축소합니다.
Actuator hard clamp ±8 Nm와 함께 torque saturation(6.8 Nm 초과), slip,
self-collision을 별도 reward/metric으로 감시합니다. 보폭·주파수·Swing Height
`0.20 m`·Radial Offset `0.07 m`는 policy가 아닌 제어기 소유입니다. 마지막 6-D
body residual은 기본 0.15초 filter와 전체 workspace 승인 단계를 거칩니다. 평지 contact는 건조 아스팔트 nominal
`friction=0.8`이고 `--flat-friction`으로 바꿀 수 있습니다. Terrain 속도 상한은
0.18 m/s입니다.

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
  --smoke --smoke-steps 100 --run-name command-body6d-smoke
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

# checkpoint가 없으면 짧은 flat baseline부터 자동 실행하는 전체 curriculum
python SW/mjx/train_competence_curriculum.py \
  --run-name mixed-body6d --flat-baseline-timesteps 1000000 \
  --stages 8 --stage-timesteps 5000000 --level-progression sequential --wandb \
  -- --num-envs 2048 --num-evals 20 --terrain-randomize --best-video
```

Mixed terrain level은 경사면과 계단을 함께 올립니다. `--terrain-randomize` 사용 시
각 stage는 해당 범위에서 재현 가능한 지형 하나를 뽑고, 각 environment reset은 그
지형의 lane을 확률적으로 선택합니다.

| Level | 계단 전체 상승 | 경사면 상승 높이 | yaw 제한 | 계단 lane 확률 |
|---|---:|---:|---:|---:|
| 0 | 2–4 cm | 4–8 cm | 0.00 rad/s | 0% |
| 1 | 4–8 cm | 8–12 cm | 0.05 rad/s | 5% |
| 2 | 8–12 cm | 12–16 cm | 0.10 rad/s | 15% |
| 3 | 12–16 cm | 16–20 cm | 0.20 rad/s | 25% |
| 4 | 16–20 cm | 20–24 cm | 0.35 rad/s | 30% |

학습 없이 모든 level의 ramp/stairs를 직진 명령으로 짧게 확인하고 한 W&B run에
10개 영상을 모으려면 다음 preview를 사용합니다. `--checkpoint`를 생략하면
zero-residual 기본 보행을 사용합니다.

```bash
python SW/mjx/preview_terrain_curriculum.py \
  --checkpoint SW/mjx/runs/terrain/<terrain-run>/checkpoints \
  --wandb
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
competence run은 각 stage의 로컬 `videos/best_policy.gif`를 유지하고 W&B에는
`best/video_stage00_level0`, `best/video_stage01_level1`처럼 stage/level이 명시된
키로 새 best를 즉시 갱신합니다. GIF overlay에도 curriculum stage, terrain level,
실제 계단 전체 상승 높이가 표시됩니다. 비교 가능한 ascent 영상을 위해 mixed-terrain best
video는 항상 stairs lane과 직진 명령(`v=0.08 m/s`, `yaw=0`)으로 렌더링합니다.

추세 영상은 기본으로 모든 run의 0/25/50/75/100%에서 각각 20초씩 저장됩니다.
Flat baseline과 competence stage 0~7은 W&B project `hexapod-rough-terrain`의 같은
group 안에 있는 독립 run이고, 각 run의 `progress/video` history에서 시점별 영상을 넘겨볼 수 있습니다. 로컬 파일은
`videos/progress/`에 보존됩니다. 개수와 길이는 `--progress-video-count 5`,
`--progress-video-duration 20`으로 바꿀 수 있고, 필요 없으면 `--no-progress-video`를
trainer 인자(`--` 뒤)에 지정합니다. NEW_BEST 영상은 이와 별도로 유지됩니다.

`--level-progression sequential`은 Stage 0/1/2/3/4를 Level 0/1/2/3/4로
확실히 올린 뒤 Level 4를 유지합니다. 기본값 `competence`는 평가 성공률이 0.8을
넘을 때만 승급하므로 어려운 level에 도달하지 못할 수 있습니다.

같은 `--run-name`을 다시 사용해도 timestamp suffix 때문에 기존 checkpoint/monitor와
섞이지 않습니다. `--best-video-path`를 생략하면 새 run directory 안에 저장됩니다.

Command NEW_BEST 영상은 기본 Stage 0/1/2/full이 각각 `20초`이며 모든 frame에
stage, `v_cmd/v`, `yaw_cmd/rate`, `GT R/P/Y`, 여섯 발 collision contact overlay가
들어갑니다. Full 영상은 stage 전환을
0.8초 banner로 표시합니다. 길이는 `--best-video-stage0-duration` 등으로, 공통 품질은
`--best-video-fps`, `--best-video-width`, `--best-video-height`로 바꿉니다. 영상이
필요 없는 짧은 debug run에는 `--no-best-video`를 사용합니다. 렌더링 실패는
`best_video_error`만 출력하며 학습·checkpoint 저장을 멈추지 않습니다. X11 DISPLAY가
없는 tmux/SSH 실행에서는 trainer가 `MUJOCO_GL=egl`을 자동 선택해 GPU offscreen
렌더링으로 GIF를 저장합니다. 설치/드라이버가 다른 경우에만 실행 전에 명시적으로
`MUJOCO_GL=egl`(GPU) 또는 사용 가능한 다른 MuJoCo backend를 지정합니다.
