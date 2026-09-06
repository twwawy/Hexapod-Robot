# Hexapod MJX → Isaac Sim / Isaac Lab 이식 결과와 사용법

2026-09-06 기록 정리: 이 문서의 실행·검사 결과는 아래 날짜의 기존 기록이다.
기존 소스·USD·handoff 변경을 Git에 함께 반영했으나 최신 통합본으로 재검증한 결과는 아니다.
현재 MuJoCo residual 보행과 센서/GT 역할은 [최신 설계](../docs/HEXAPOD_LIDAR_FOOTHOLD_RESIDUAL_PLAN.md),
전체 변경 범위는 [업데이트 기록](../docs/HEXAPOD_UPDATE_2026-09-06.md)을 따른다.

작성 기준: 2026-08-29
대상 브랜치: `codex/cartesian-residual-rl`
환경: Isaac Sim 5.1 + Isaac Lab 2.3.0 + RSL-RL 3.1.2

## 현재 결과

전체 CAD 자산과 최신 MJX handoff를 연결했고, USD와 sensor/action/observation 계약의
정적 검증을 완료했다. RTX LiDAR가 포함된 장면 runtime 검증은 GPU가 노출되는 일반
데스크톱 세션에서 실행해야 한다.

- MJCF → USD 재생성 후 중복 articulation root를 제거했다.
- USD는 articulation root 1개와 canonical 18 joints를 정적으로 통과했다.
- STM32-equivalent controller를 batch-first Torch로 이식했다.
- Torch controller는 frozen MJX trace 500 policy steps, 2,000 firmware ticks와 parity를
  통과했다. 최대 joint target 오차는 `3.70e-6 rad`다.
- 400 Hz physics, 200 Hz firmware, 50 Hz policy timing을 고정했다.
- 기존 146-D observation 계약과 GT terrain 15-D 제거 함수를 구현했다.
- sensor actor `131 proprio + 64 terrain latent = 195-D`, asymmetric critic `225-D` 계약을
  구현했다.
- MID-360 RayCaster, provisional Depth RayCasterCamera, body IMU, distal-leg contact sensor
  config를 추가했다.
- LiDAR/Depth point fusion, gravity/heading alignment, `32 x 24 x 3` elevation map과 64-D CNN
  encoder를 추가했다.
- golden replay task와 experimental perceptive task를 Gym에 등록했다.
- Isaac bundled Python에 `gymnasium==1.2.1`, `rsl-rl-lib==3.1.2`, `onnxscript>=0.5` 및
  Hexapod extension을 editable install했다.
- 사용자가 지정한 W&B run `hexapod-real/g7db7prs`의 config/summary/history를 내려받았다.

### 2026-08-29 최신 학습/실물 mesh 동기화

- Isaac 자산은 이제 `hexapod_full_mesh_mjx_parity.usd`를 사용한다.
- 새 URDF의 133개 STL mesh는 visual로 보존한다. 여기에는
  `livox_mid_360_1`과 MID-360 FOV 외형이 포함된다.
- CAD mesh collision은 끄고 MJX와 같은 torso box 및 다리 capsule/sphere 25개만
  collision으로 사용한다. 외형을 복원하면서 학습 contact 계약은 바꾸지 않는다.
- 전체 질량 10 kg, 18개 joint, 단일 articulation root 계약은 유지한다.
- `data/training/latest_mjx_training.json`이 현재 source contract, curriculum 0~16,
  최신 run/evaluation, lineage checkpoint, 마지막 safety-gated checkpoint를 구분해
  기록한다.
- 최신 시도는 stage 48 / level 9 (`7 x 15 cm`)이고 strict safety gate 실패이며,
  stage 자체 checkpoint도 없다. 따라서 stage 47 lineage checkpoint를 자동 배포하거나
  Isaac 초기화에 사용하지 않는다.
- 현재 action contract는 `stm32_firmware_adaptive_swing_residual_100mm_v4`로 올렸다.
  swing X/Y와 stance Z는 축별 최대 100 mm이며, swing Z의 4~25 cm 높이 mapping은
  유지한다. 기존 v3 학습 checkpoint는 scale이 달라 자동 로드하지 않는다.

동기화·자산 생성·현재 지형 GUI 실행은 다음과 같다.

```bash
cd /home/huro/Hexapod-Robot
/home/huro/.venvs/hexapod-mjx/bin/python \
  isaaclab_hexapod/scripts/sync_latest_mjx_training.py
./isaaclab_hexapod/scripts/build_asset.sh
./isaaclab_hexapod/scripts/run_realtime.sh
```

GUI 실행기는 기본적으로 handoff의 최신 level 9 계단을 만들며, 예를 들어 level 5로
바꾸려면 `--terrain-level 5`를 붙인다. 이 실행기는 모델/지형 확인용이다. 최신 MJX
reward/termination 전체 이식과 sensor encoder pretraining이 완료되기 전에는 Isaac PPO
본 학습 task로 간주하지 않는다.

실시간 확인은 full-CAD robot 1대만 생성한다. RTX 3090 부하 측정용 RSL-RL 실행은
다음처럼 512 environments에서 시작하고 `nvidia-smi`를 보며 조절한다.

```bash
HEXAPOD_NUM_ENVS=512 ./isaaclab_hexapod/scripts/train_perceptive_gpu80.sh
watch -n 1 nvidia-smi
```

약 80%는 고정 quota가 아니라 측정 목표다. 70% 미만이면 768/1024, OOM 또는 95%
이상이면 384/256으로 바꾼다. 현재 명령은 개발 reward scaffold의 통합/부하 확인용이고,
최신 MJX reward/termination 전체 이식 전에는 본 학습 결과로 사용하지 않는다.

구현 계획 전체는
[`docs/HEXAPOD_PERCEPTIVE_RESIDUAL_ISAACLAB_PLAN.md`](../docs/HEXAPOD_PERCEPTIVE_RESIDUAL_ISAACLAB_PLAN.md)에 있다.

## 검증 결과

버전 관리되는 보고서는 다음과 같다.

| 보고서 | 결과 |
|---|---|
| `data/usd/asset_inspection.json` | root 1개, 18 joints, PASS |
| `data/torch_controller_parity.json` | 2,000 ticks, max `q_des` error `3.70e-6`, PASS |
| `data/config_validation.json` | sensor/API/shape/GT leak/RSL group, PASS |

한 번에 다시 확인하려면 다음 명령을 사용한다. 이 명령은 Gym environment나 PhysX scene을
생성하지 않으므로 locomotion smoke test가 아니다.

```bash
cd /home/huro/Hexapod-Robot
./isaaclab_hexapod/scripts/verify_static.sh
```

현재 Codex 실행 sandbox에서는 NVIDIA driver가 Isaac 프로세스에 노출되지 않아 startup
log에 `NVML_ERROR_DRIVER_NOT_LOADED`가 출력됐다. USD 읽기와 config 검증은 PASS지만,
full-mesh 장면의 PhysX step과 RTX point cloud runtime은 이 환경에서 통과했다고 주장하지
않는다.

## 설치

현재 머신에는 다음 경로를 사용했다.

```text
Isaac Sim 5.1  /home/huro/isaac-sim-5.1
Isaac Lab 2.3  /home/huro/IsaacLab
MJX Python     /home/huro/.venvs/hexapod-mjx/bin/python
```

재설치는 다음으로 충분하다.

```bash
cd /home/huro/Hexapod-Robot
ISAAC_SIM_ROOT=/home/huro/isaac-sim-5.1 \
ISAACLAB_ROOT=/home/huro/IsaacLab \
./isaaclab_hexapod/scripts/install.sh
```

Isaac Lab 전체 optional dependency 설치는 현재 PyPI의 `pin/pytransform3d` 조합에서 긴
resolver backtracking이 발생했다. 이 프로젝트는 필요한 버전만 pin하여 설치한다.

## 자산 재생성

```bash
cd /home/huro/Hexapod-Robot
ISAACLAB_ROOT=/home/huro/IsaacLab \
MJX_PYTHON=/home/huro/.venvs/hexapod-mjx/bin/python \
./isaaclab_hexapod/scripts/build_asset.sh
```

`build_asset.sh`는 MJX용 Isaac asset을 다시 export하고 USD를 생성한 다음 articulation/joint
검사를 자동 수행한다.

## Task와 계약

### `Hexapod-Firmware-Flat-Direct-v0`

- 18-D Cartesian residual action interface
- 146-D legacy observation interface
- frozen golden `q_des` replay
- asset/timing/joint-order 확인용

### `Hexapod-Perceptive-Direct-v0`

- online batch Torch firmware controller, policy step당 firmware tick 4회
- LiDAR + optional Depth + IMU + contact
- actor 195-D, critic 225-D
- actor에서 simulator GT terrain `[76:91]` 제거
- sensor `pitch_ff/swing_boost` 연결 전에는 둘 다 0으로 안전하게 유지
- RSL-RL PPO config: actor/critic `256 → 256 → 128`, W&B logger

두 번째 task는 개발용으로 등록됐지만 아직 본 학습을 시작하면 안 된다. 현재 reward는
joint-target tracking 최소 구현이고, CNN encoder도 supervised map pretraining checkpoint를
아직 로드하지 않는다. runtime sensor prim, contact ordering, RayCaster ground mesh는 GPU가
노출되는 일반 데스크톱 세션에서 확인해야 한다.

## 센서 좌표와 안전 gate

- MID-360 body transform: `(0.018071, 0.036714, 0.304727) m`
- 실시간 1-robot 실행은 `/World/Robot/hexapod/hexapod/Sensors/LivoxMID360`에
  `OmniLidar`를 붙이고 RTX point cloud debug draw를 켠다.
- Isaac Sim 5.1 설치본에는 Livox 제조사 프로파일이 없으므로 RTX 설정은 공식 사양의
  360° x 59°, 0.1~40 m, 40-line, 10 Hz, 200 kpoint/s를 맞춘 rotary proxy다.
  Livox 고유 non-repetitive firing pattern을 정확히 재현하는 설정은 아니다.
- batched RL에서는 같은 좌표/FOV/범위를 쓰는 RayCaster를 사용하며 수평 4°로
  downsample한다. RTX LiDAR는 1대 GUI 확인에만 생성한다.
- 모델의 진행 방향 `-Y`를 sensor/map `+X`로 변환한다.
- map 범위: `x=[-0.4,1.2] m`, `y=[-0.6,0.6] m`, resolution `0.05 m`
- map channel: elevation, confidence, roughness
- Depth extrinsic은 CAD/Xacro에서 식별되지 않아 `DEPTH_EXTRINSIC_CONFIRMED=False`다.
- Depth sensor는 기본 비활성화되어 있다. 실측 optical transform 확정 전 sensor-policy
  checkpoint 생성에 사용하면 안 된다.

## W&B 기준 데이터

최신 기준 파일은 `data/wandb/level5_stage06_g7db7prs.json`이다.

```text
run     hurolilys-inha-university/hexapod-real/g7db7prs
name    firmware-terrain-command5-level5-to-final-stage06-level5_20260827-035106_seed15
state   finished
stage   6
level   5
reward  42.41857147216797
```

이 run은 `best_video=True`였지만 summary에 best checkpoint/video가 없고 W&B 동기화 로그도
media/artifact 0개였다. 따라서 config와 metric은 가져왔지만 checkpoint나 영상을 W&B에서
복원할 수 없다. 로컬 checkpoint를 쓰거나 다음 학습부터 checkpoint를 W&B Artifact로
업로드해야 한다.

다른 run을 내려받는 방법:

```bash
/home/huro/.venvs/hexapod-mjx/bin/python \
  isaaclab_hexapod/scripts/fetch_wandb_reference.py \
  --run ENTITY/PROJECT/RUN_ID \
  --output isaaclab_hexapod/data/wandb/RUN_ID.json
```

## 다음 실행 순서

사용자가 검토·실행할 때는 다음 순서가 안전하다.

1. GPU가 보이는 일반 터미널에서 `nvidia-smi` 확인
2. `./isaaclab_hexapod/scripts/verify_static.sh`
3. 요청 시에만 `./isaaclab_hexapod/scripts/zero_action.sh`로 golden replay smoke
4. online zero-residual flat parity
5. Depth optical extrinsic 확정
6. GT elevation label로 CNN encoder supervised pretraining
7. sensor map으로 `pitch_ff/swing_boost` 계산 연결 및 GT leak test
8. reward/termination/terrain curriculum 이식
9. asymmetric RSL-RL PPO 학습

이번 변경에서는 3번 이후를 실행하지 않았다.
