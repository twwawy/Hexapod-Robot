# SPIDER MuJoCo 변환/학습 스택 공부 문서

이 문서는 **다른 AI나 다른 사람한테 그대로 넘겨도 맥락이 유지되도록** 만든 해설 문서다. 목적은 두 가지다.

1. 지금 만들어진 HEXAPEDAL URDF -> MJCF -> MuJoCo 학습 경로를 빠르게 재실행하기
2. 이 변환이 어떻게 구성됐는지 구조적으로 이해하기

---

## 0. 한 줄 요약

이번 작업의 본질은:

- Isaac 쪽에서 쓰던 **HEXAPEDAL URDF 기반 로봇**을
- MuJoCo에서 바로 읽을 수 있는 **MJCF XML**로 변환하고
- 그 위에 **standalone Gymnasium + SB3 PPO 학습 스택**을 붙여서
- Isaac/Omniverse import 없이도 **독립 실행**되게 만든 것이다.

중요한 점은, 이건 단순 변환 스크립트 하나가 아니라:

- asset recovery
- MJCF generation
- env registration
- train/eval/check scripts
- packaging / editable install

까지 한 번에 묶인 runnable stack이라는 점이다.

---

## 1. 소스 오브 트루스

이 `mujoco_tuto` 폴더는 설명용이고, 실제 코드의 소스 오브 트루스는 아래다.

- code repo: `/home/huro/spider_ws/spider_rl`
- MuJoCo package root: `/home/huro/spider_ws/spider_rl/source/spider_rl/spider_mujoco`
- training scripts: `/home/huro/spider_ws/spider_rl/scripts/mujoco`
- default URDF: `/home/huro/spider_ws/HEXAPEDAL_URDF_description/urdf/HEXAPEDAL_URDF_fixed.urdf`

이 문서를 보는 다른 AI는 **`mujoco_tuto`를 수정 대상으로 보면 안 되고**, 실제 수정 대상은 거의 항상 `spider_rl` repo라고 보면 된다.

---

## 2. 결과물 구조

### 2.1 새 standalone package

`source/spider_rl/spider_mujoco/`

역할:
- IsaacLab/Omniverse와 분리된 MuJoCo 전용 entrypoint
- `import spider_mujoco`만으로 Gym env 등록
- 다른 환경에서 import side effect 최소화

핵심 파일:
- `spider_mujoco/__init__.py`
- `hexapedal_direct/env.py`
- `hexapedal_direct/env_cfg.py`
- `hexapedal_direct/model_builder.py`
- `hexapedal_direct/assets/hexapedal.xml`
- `hexapedal_direct/assets/source_map.yaml`
- `hexapedal_direct/agents/sb3_ppo_cfg.yaml`

### 2.2 실행 스크립트

- `scripts/mujoco/check_model.py`
- `scripts/mujoco/train_sb3.py`
- `scripts/mujoco/eval_sb3.py`

### 2.3 테스트

- `source/spider_rl/tests/test_spider_mujoco_model.py`
- `source/spider_rl/tests/test_spider_mujoco_env.py`

---

## 3. 변환 파이프라인

### 단계 A. source asset 찾기

`model_builder.py`는 URDF와 mesh를 아래 우선순위로 찾는다.

1. `SPIDER_HEXAPEDAL_URDF_PATH`
2. sibling workspace fallback 탐색
3. 기본 경로 `/home/huro/spider_ws/HEXAPEDAL_URDF_description/urdf/HEXAPEDAL_URDF_fixed.urdf`

mesh도 `SPIDER_HEXAPEDAL_MESH_DIR`로 override 가능하다.

즉 다른 AI가 이걸 다른 머신에서 다시 붙일 때 가장 먼저 볼 건:
- URDF 실제 위치
- mesh 폴더 실제 위치
- env var override가 필요한지

### 단계 B. URDF 파싱

`model_builder.py`는 URDF에서:
- 링크
- 조인트
- inertial
- visual/collision geometry
- transform chain

를 읽는다.

이 과정은 `yourdfpy` 같은 상위 변환기에 전부 맡기지 않고, 필요한 구조를 직접 조립하는 쪽에 가깝다.

### 단계 C. reduced-body MJCF 생성

이 모델은 원본 URDF 링크를 1:1 복사한 게 아니라, 학습/시뮬레이션에 필요한 구조만 남긴 **reduced-body MJCF**를 만든다.

핵심 특징:
- 6개 다리 순서 고정: `LF, LM, LB, RF, RM, RB`
- retained stage:
  - `motor_horn_1_1`
  - `DS51150_270_2_1`
  - `motor_horn_3_1`
- actuator 수: 18
- observation 차원: 48
- command schema: `[vx, vy, wz]`
- 현재 v1 계약에서는 `vy == 0.0`

### 단계 D. MJCF에 MuJoCo 쪽 안전장치 추가

변환은 기계적으로만 하지 않고, MuJoCo에서 굴리기 위한 설정이 추가됐다.

예:
- plane ground 추가
- robot self-collision 억제
- robot-ground collision 유지
- passive actuator 구조
- 각 foot 말단에 stance/contact geom 추가
- contact site 이름을 학습/검증 코드에서 바로 쓸 수 있게 고정

### 단계 E. 생성물 저장

생성 결과는 package asset으로 저장된다.

- `assets/hexapedal.xml`
- `assets/source_map.yaml`

`source_map.yaml`은 “이 body/joint가 원본 URDF의 어디서 왔는지” 추적하는 provenance 문서다.

---

## 4. Env 구조

`env.py`는 standalone MuJoCo direct env다.

핵심 역할:
- `gym.make("Hexapedal-MuJoCo-Direct-v0")` 등록 대상
- reset/step API 제공
- 관측/보상/종료 로직 제공
- 모델 파일이 없거나 stale이면 재생성 경로로 복구

`env_cfg.py`는 다음을 고정한다.

- 시뮬레이션 dt: `1/120`
- action dim: `18`
- observation dim: `48`
- default joint positions
- reward scales
- reset noise
- termination threshold
- undesired contact body names

다른 AI가 봐야 할 핵심은:
- 실제 command contract는 어디서 정의되는가?
- reward/termination이 어디서 정의되는가?
- 학습 성능이 안 나오면 어떤 상수를 먼저 볼 것인가?

답은 거의 항상 `env_cfg.py`와 `env.py`다.

---

## 5. 학습/평가 계약

### 5.1 train

`scripts/mujoco/train_sb3.py`

주요 역할:
- `spider_mujoco` bootstrap
- PPO config 로드
- env 생성
- eval callback 수행
- `latest.zip`, `best.zip` 관리
- eval metric log 저장

지원 프로토콜:
- `smoke`: 10k
- `baseline`: 500k
- `extension`: 1M

### 5.2 best checkpoint 선택 규칙

stage-06 기준으로 더 엄격한 tie-breaker가 들어가 있다.

1. mean eval return 최대
2. tie-breaker 1: `tracking_error_scalar` 더 낮은 것
3. tie-breaker 2: `fall_rate` 더 낮은 것

### 5.3 tracking_error_scalar

정의:

`mean_over_eval_episodes(mean_over_steps(0.5*abs(vx_err)/0.10 + 0.5*abs(wz_err)/0.25))`

### 5.4 fall_rate

정의:

`terminated_before_timeout_due_to_fall_or_height_violation / total_eval_episodes`

### 5.5 eval scorecard에서 봐야 할 것

- return
- forward distance
- mean `vx` tracking error
- mean `wz` tracking error
- `tracking_error_scalar`
- `fall_rate`
- undesired contact count
- termination reason

사용자가 지금은 “검증 성능보다 일단 runnable이면 된다”고 낮춘 상태였지만, 이 메트릭 정의 자체는 남아 있다. 나중에 다시 성능 튜닝할 때 그대로 재사용하면 된다.

---

## 6. 실제 실행 커맨드

### 설치
```bash
cd /home/huro/spider_ws/spider_rl
python -m pip install -e source/spider_rl[mujoco]
```

### 모델 체크
```bash
python scripts/mujoco/check_model.py --task Hexapedal-MuJoCo-Direct-v0
```

### smoke 학습
```bash
python scripts/mujoco/train_sb3.py \
  --task Hexapedal-MuJoCo-Direct-v0 \
  --protocol smoke \
  --device cpu \
  --run-dir /tmp/mujoco_goal_smoke
```

### eval
```bash
python scripts/mujoco/eval_sb3.py \
  --task Hexapedal-MuJoCo-Direct-v0 \
  --checkpoint /tmp/mujoco_goal_smoke/best.zip \
  --episodes 5 \
  --seed 42 \
  --device cpu \
  --output /tmp/mujoco_goal_eval.json
```

### URDF 경로 override
```bash
export SPIDER_HEXAPEDAL_URDF_PATH=/absolute/path/to/HEXAPEDAL_URDF_fixed.urdf
export SPIDER_HEXAPEDAL_MESH_DIR=/absolute/path/to/meshes
python scripts/mujoco/check_model.py --task Hexapedal-MuJoCo-Direct-v0
```

---

## 7. 실제로 확인된 성공 증거

실행 성공 증거 파일:
- `/tmp/mujoco_check_model.json`
- `/tmp/mujoco_goal_smoke/best.zip`
- `/tmp/mujoco_goal_smoke/latest.zip`
- `/tmp/mujoco_goal_eval.json`
- `/tmp/mujoco_runnable_receipt.json`

확인된 사실:
- editable install 성공
- `import spider_mujoco` 성공
- `gym.make("Hexapedal-MuJoCo-Direct-v0")` 성공
- env `reset()/step()` 성공
- `check_model.py` 성공
- smoke train 성공
- eval 성공
- Isaac import 강제 의존 제거

즉, 다른 AI가 이 상태를 이해할 때 “아직 성능 튜닝은 남아 있을 수 있어도, 변환 + 실행 경로는 이미 붙었다”로 이해하면 된다.

---

## 8. Isaac Sim 5.1 관점에서 왜 중요한가

이번 작업에서 중요한 건 “MuJoCo 스택이 Isaac Sim 5.1 환경에서도 쓸 수 있게 import/packaging을 분리했다”는 점이다.

의미:
- Isaac Sim python 환경을 쓰더라도 MuJoCo 전용 import가 IsaacLab heavy import를 타지 않음
- tutorial/runtime 코드가 MuJoCo만 필요할 때 Omniverse 모듈 부재로 바로 죽지 않음
- `spider_mujoco`가 독립 entrypoint 역할 수행

다른 AI가 packaging 문제를 다시 잡아야 할 때는 아래를 먼저 확인하면 된다.

- `source/spider_rl/setup.py`
- `source/spider_rl/pyproject.toml`
- `source/spider_rl/spider_mujoco/__init__.py`

---

## 9. 다른 AI가 이 작업을 다시 이어받을 때 보는 순서

### 9.1 runnable 상태만 확인하려면

1. `README.md` 읽기
2. `/home/huro/spider_ws/spider_rl/source/spider_rl/spider_mujoco/` 보기
3. `/home/huro/spider_ws/spider_rl/scripts/mujoco/` 보기
4. `python scripts/mujoco/check_model.py --task Hexapedal-MuJoCo-Direct-v0`
5. 필요하면 smoke train/eval 실행

### 9.2 변환 로직을 이해하려면

1. `model_builder.py` 전체 읽기
2. `source_map.yaml` 확인
3. `env_cfg.py`에서 dim/termination/reward 확인
4. `env.py`에서 reset/step/runtime metric 확인
5. `train_sb3.py`, `eval_sb3.py`에서 metric aggregation 확인

### 9.3 성능을 올리고 싶다면

우선순위:
1. `env_cfg.py` reward/termination
2. foot contact geometry
3. default joint pose
4. PD gains / torque scaling
5. command sampling / curriculum
6. PPO config

---

## 10. 다른 AI에게 바로 붙여넣을 설명 템플릿

아래 텍스트를 통째로 넘기면 된다.

```text
Context:
- Actual repo is /home/huro/spider_ws/spider_rl
- MuJoCo package is source/spider_rl/spider_mujoco
- Target env id is Hexapedal-MuJoCo-Direct-v0
- Default URDF is /home/huro/spider_ws/HEXAPEDAL_URDF_description/urdf/HEXAPEDAL_URDF_fixed.urdf
- Generated MJCF asset is source/spider_rl/spider_mujoco/hexapedal_direct/assets/hexapedal.xml
- Provenance map is source/spider_rl/spider_mujoco/hexapedal_direct/assets/source_map.yaml
- Scripts are scripts/mujoco/check_model.py, train_sb3.py, eval_sb3.py
- Package import should be spider_mujoco only; do not reintroduce Isaac/Omniverse import coupling

Current known-good runnable path:
1. cd /home/huro/spider_ws/spider_rl
2. python -m pip install -e source/spider_rl[mujoco]
3. python scripts/mujoco/check_model.py --task Hexapedal-MuJoCo-Direct-v0
4. python scripts/mujoco/train_sb3.py --task Hexapedal-MuJoCo-Direct-v0 --protocol smoke --device cpu --run-dir /tmp/mujoco_goal_smoke
5. python scripts/mujoco/eval_sb3.py --task Hexapedal-MuJoCo-Direct-v0 --checkpoint /tmp/mujoco_goal_smoke/best.zip --episodes 5 --seed 42 --device cpu --output /tmp/mujoco_goal_eval.json

Constraints:
- Keep spider_mujoco standalone
- Preserve command schema [vx, vy, wz] with vy==0 for current v1 contract
- Keep URDF path override support via SPIDER_HEXAPEDAL_URDF_PATH and SPIDER_HEXAPEDAL_MESH_DIR
- Do not assume mujoco_tuto is the source-of-truth repo
```

---

## 11. 자주 헷갈리는 포인트

### Q1. `mujoco_tuto` 안의 파일이 실제 실행 코드인가?
아니다. 설명/노트용이다. 실제 코드는 `spider_rl` repo에 있다.

### Q2. URDF가 없는데 USD만 있으면 되나?
이번 최종 runnable 경로는 **URDF source recovery가 된 상태**를 기준으로 붙었다. USD-only 경로를 완성한 건 아니다.

### Q3. Isaac Sim 5.1 자체 안에서 학습하는 건가?
아니다. 학습은 MuJoCo standalone 쪽이다. 다만 Isaac Sim 5.1 환경에서도 import 충돌 없이 설치/실행되도록 packaging을 정리한 것이다.

### Q4. 지금 성능까지 보장되나?
아니다. 지금 문서의 핵심은 **변환 + 실행 경로가 정상 동작**한다는 것이다.

---

## 12. 가장 짧은 재실행 체크리스트

```bash
cd /home/huro/spider_ws/spider_rl
python -m pip install -e source/spider_rl[mujoco]
python scripts/mujoco/check_model.py --task Hexapedal-MuJoCo-Direct-v0
python scripts/mujoco/train_sb3.py --task Hexapedal-MuJoCo-Direct-v0 --protocol smoke --device cpu --run-dir /tmp/mujoco_goal_smoke
python scripts/mujoco/eval_sb3.py --task Hexapedal-MuJoCo-Direct-v0 --checkpoint /tmp/mujoco_goal_smoke/best.zip --episodes 5 --seed 42 --device cpu --output /tmp/mujoco_goal_eval.json
```

이 네 줄이 다시 돌면, 현재 목표였던 runnable MuJoCo conversion stack은 살아 있다고 보면 된다.
