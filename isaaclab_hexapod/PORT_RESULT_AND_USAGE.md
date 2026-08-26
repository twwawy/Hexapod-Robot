# Hexapod MJX → Isaac Lab 1차 이식 결과와 사용법

작성 기준: 2026-08-27  
대상 브랜치: `codex/cartesian-residual-rl`  
원본 계획: `docs/HEXAPOD_MJX_TO_ISAACLAB_PORT_PLAN.md`

## 1. 현재 결과

이번 작업은 최종 RL 학습보다 재현 가능한 세팅과 계약 고정에 집중했다.

- MJX 최종 자산에서 질량, 관성, 충돌체, 18개 관절 정보를 추출한 manifest를 생성했다.
- flat/seed 0 조건의 10초 golden trace를 생성했다.
- 정책 주기 20 ms, 물리 주기 2.5 ms, firmware tick 5 ms 계약을 고정했다.
- Isaac Lab 외부 extension과 Gym task `Hexapod-Firmware-Flat-Direct-v0`를 추가했다.
- MJCF를 Isaac Sim USD로 변환하는 스크립트와 생성 USD를 추가했다.
- W&B 기준 run의 config, summary, sampled history를 내려받아 버전 관리 데이터로 저장했다.
- Isaac Sim에서 자산 로딩까지 확인했으나, importer가 `worldBody`와 로봇을 두 개의
  articulation root로 만든 문제 때문에 DirectRLEnv 1-step 완료는 아직 확인되지 않았다.
  중복 root를 제거하는 후처리는 `scripts/build_asset.py`에 추가했지만, 재생성 실행은
  중단 요청 시점에 멈춰 최종 검증하지 않았다.

즉, 현재 상태는 **Phase 0 계약 + 자산 변환 + DirectRLEnv replay 스캐폴드**이며,
온라인 Torch firmware controller와 PPO 학습 이식 완료 상태는 아니다.

## 2. 생성된 계약과 데이터

### MJX golden contract

- `mjx/golden/asset_manifest_v1.json`
  - canonical leg/joint 순서: `RB, RM, RF, LB, LM, LF`, 각 3축
  - 관절 수: 18
  - 최종 model total mass: 약 10 kg
- `mjx/golden/isaac_contract_v1_flat_seed0.json`
  - schema, shape, dtype, checksum metadata
- `mjx/golden/isaac_contract_v1_flat_seed0.npz`
  - 500 policy steps, 총 10초
  - action: `(500, 18)`
  - observation: `(500, 146)`
  - q_des: `(500, 18)`
  - policy step당 firmware tick 4회
  - 처음 1초는 zero action, 이후 deterministic bounded action
  - 126개 array를 `float32`, `int32`, `bool`로 고정

### W&B reference

저장 위치: `isaaclab_hexapod/data/wandb/reference_run.json`

- project: `hurolilys-inha-university/hexapod-firmware-terrain`
- run id: `g568d0hq`
- run name: `firmware-terrain-adaptive-swing-v3-fixed-stairs-stage24-level8_20260826-154912_seed33`
- state: `finished`
- 수집 내용: config, summary, history sample 6개

W&B에는 이 run의 checkpoint가 artifact/file로 업로드되어 있지 않다. summary에는
로컬 checkpoint 경로만 기록되어 있으므로 다른 머신에서 W&B만으로 checkpoint를
복원할 수 없다. 현재 포트는 W&B 모델을 실행한 것이 아니라 W&B 실험 조건과 결과
메타데이터를 참조 데이터로 가져온 상태다.

다시 수집하려면 W&B 로그인 후 다음을 실행한다.

```bash
cd /path/to/Hexapod-Robot
/home/huro/.venvs/hexapod-mjx/bin/python \
  isaaclab_hexapod/scripts/fetch_wandb_reference.py
```

다음 학습부터 checkpoint를 W&B Artifact로 업로드해야 이식 환경에서 직접 내려받아
policy parity를 확인할 수 있다.

## 3. 로컬 실행 환경

이번 세팅에서 사용한 버전은 다음과 같다.

- Ubuntu 22.04
- NVIDIA RTX 3090 / driver 580.173.02
- Isaac Sim 5.1: `/home/huro/isaac-sim-5.1`
- Isaac Lab 2.3.0: `/home/huro/IsaacLab`
- MJX Python: `/home/huro/.venvs/hexapod-mjx/bin/python`

Isaac Lab은 `_isaac_sim`이 Isaac Sim 5.1을 가리키도록 연결하고, Isaac Sim Python에
Isaac Lab과 이 extension을 editable install하는 방식으로 구성했다.

## 4. 최초 설치

경로가 다른 머신에서는 환경변수로 덮어쓴다.

```bash
cd /home/huro/IsaacLab
test -e _isaac_sim || ln -s /home/huro/isaac-sim-5.1 _isaac_sim
TERM=xterm ./isaaclab.sh -i none
```

`flatdict` build isolation 오류가 나면 다음 순서로 설치한다.

```bash
cd /home/huro/IsaacLab
TERM=xterm ./isaaclab.sh -p -m pip install flatdict==4.0.1 --no-build-isolation
TERM=xterm ./isaaclab.sh -p -m pip install \
  -e source/isaaclab --no-build-isolation --no-deps
```

## 5. 데이터 재생성

```bash
cd /path/to/Hexapod-Robot

/home/huro/.venvs/hexapod-mjx/bin/python mjx/export_asset_manifest.py
JAX_PLATFORMS=cpu /home/huro/.venvs/hexapod-mjx/bin/python \
  mjx/export_isaac_contract.py
```

golden contract 검사는 다음 명령으로 실행한다.

```bash
JAX_PLATFORMS=cpu /home/huro/.venvs/hexapod-mjx/bin/python \
  -m unittest mjx.tests.test_isaac_contract_artifacts -v
```

현재 결과는 5개 테스트 모두 통과다.

## 6. USD 생성과 Isaac 실행

USD를 다시 생성한다.

```bash
cd /path/to/Hexapod-Robot
ISAACLAB_ROOT=/home/huro/IsaacLab \
MJX_PYTHON=/home/huro/.venvs/hexapod-mjx/bin/python \
./isaaclab_hexapod/scripts/build_asset.sh
```

성공하면 다음 파일이 갱신된다.

- `isaaclab_hexapod/data/usd/hexapod_mjx_parity.usd`
- `isaaclab_hexapod/data/usd/configuration/*.usd`
- `isaaclab_hexapod/data/usd/config.yaml`

headless replay smoke는 다음과 같이 실행한다.

```bash
cd /path/to/Hexapod-Robot
ISAACLAB_ROOT=/home/huro/IsaacLab \
HEXAPOD_SMOKE_STEPS=500 \
./isaaclab_hexapod/scripts/zero_action.sh
```

완료 시 `HEXAPOD_SMOKE_OK`, step 수, `(1, 146)` observation shape, 평균 reward를
출력하도록 되어 있다. 현재 커밋 시점에는 USD 중복 articulation root 수정 후 이
표식까지 재확인하지 않았다.

## 7. Isaac task 내용

- task id: `Hexapod-Firmware-Flat-Direct-v0`
- action space: 18
- observation space: 146
- physics dt: 0.0025 s
- decimation: 8
- environment/policy dt: 0.02 s
- actuator target: golden trace의 `q_des`
- reward: 현재 joint target tracking용 단순 reward
- termination: episode timeout 또는 non-finite joint state

현재 observation은 146-D shape를 보존하지만 command, joint position/velocity,
last action slice만 채우고 나머지는 0이다. 이는 interface와 timing을 먼저 고정하기
위한 것이며 학습에 사용하면 안 된다.

## 8. 다음 작업과 최종 검토 항목

최종 목표와 이후 구현 순서는
[`docs/HEXAPOD_PERCEPTIVE_RESIDUAL_ISAACLAB_PLAN.md`](../docs/HEXAPOD_PERCEPTIVE_RESIDUAL_ISAACLAB_PLAN.md)를
기준으로 한다. 기존 146-D/15-point 지형 입력은 parity와 privileged teacher를 위한
중간 계약이며, 배포 actor에서는 LiDAR + Depth + IMU 기반 elevation map으로 교체한다.

1. `build_asset.sh`를 다시 실행해 USD articulation root가 하나인지 확인한다.
2. `zero_action.sh`의 `HEXAPOD_SMOKE_OK`까지 확인한다.
3. USD joint axis/sign, mass/COM/inertia, primitive collision을 manifest와 비교한다.
4. MJX firmware controller를 pure Torch state machine으로 이식한다.
5. 146-D observation 전체, reward, termination, terrain/RayCaster를 연결한다.
6. 동일 state/action을 넣은 MJX ↔ Isaac closed-loop diff를 통과시킨다.
7. 그 이후 PPO 학습과 checkpoint 변환/재학습을 활성화한다.

최종 검토는 1~3의 자산/관절 계약부터 하고, 시각적으로 걷는 모습은 그 이후에
확인하는 것이 안전하다.
