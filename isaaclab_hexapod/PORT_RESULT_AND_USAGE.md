# Hexapod MJX → Isaac Sim / Isaac Lab 이식 결과와 사용법

작성 기준: 2026-08-27  
대상 브랜치: `codex/cartesian-residual-rl`  
환경: Isaac Sim 5.1 + Isaac Lab 2.3.0 + RSL-RL 3.1.2

## 현재 결과

요청대로 물리 smoke test는 실행하지 않고, 실행 전에 준비·정적 검증할 수 있는 항목을
완성했다.

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

현재 Codex 실행 sandbox에서는 NVIDIA driver가 노출되지 않아 Isaac startup log에
`NVML_ERROR_DRIVER_NOT_LOADED`가 출력됐다. USD 읽기와 config 검증은 PASS였지만 실제 GPU
PhysX/RayCaster 동작을 검증한 것으로 보지 않는다.

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
아직 로드하지 않는다. 사용자 요청에 따라 smoke test를 생략했으므로 runtime sensor prim,
contact ordering, RayCaster ground mesh도 확인 전이다.

## 센서 좌표와 안전 gate

- MID-360 body transform: `(-0.017929, 0.004714, 0.393473) m`
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
