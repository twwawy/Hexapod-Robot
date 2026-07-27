# mujoco_tuto

MuJoCo/MJCF 공부용 메모 저장소다. 이번에 실제로 돌아가게 만든 HEXAPEDAL 변환 스택의 소스 오브 트루스는 이 폴더가 아니라 아래 repo다.

- 실제 코드 repo: `/home/huro/spider_ws/spider_rl`
- MuJoCo 패키지: `/home/huro/spider_ws/spider_rl/source/spider_rl/spider_mujoco`
- 학습/검증 스크립트: `/home/huro/spider_ws/spider_rl/scripts/mujoco`
- 원본 URDF 기본 경로: `/home/huro/spider_ws/HEXAPEDAL_URDF_description/urdf/HEXAPEDAL_URDF_fixed.urdf`
- 이 repo 안에도 업로드용 슬림 코드 사본을 `reference/spider_rl/` 아래 넣어뒀다. 공부/열람용이고, 수정 소스 오브 트루스는 여전히 위 `spider_rl` repo다.

## 지금 바로 쓰는 순서

### 1) 환경 준비
```bash
cd /home/huro/spider_ws/spider_rl
python -m pip install -e source/spider_rl[mujoco]
```

Isaac Sim 5.1 환경에서 써도 되게 packaging/import 경로를 정리해 둔 상태라, `spider_mujoco` import가 Isaac/Omniverse import를 강제로 타지 않는다.

### 2) 모델 생성/로드 확인
```bash
python scripts/mujoco/check_model.py --task Hexapedal-MuJoCo-Direct-v0
```

URDF 경로를 직접 바꾸고 싶으면:
```bash
export SPIDER_HEXAPEDAL_URDF_PATH=/path/to/HEXAPEDAL_URDF_fixed.urdf
export SPIDER_HEXAPEDAL_MESH_DIR=/path/to/meshes
python scripts/mujoco/check_model.py --task Hexapedal-MuJoCo-Direct-v0
```

### 3) smoke 학습
```bash
python scripts/mujoco/train_sb3.py \
  --task Hexapedal-MuJoCo-Direct-v0 \
  --protocol smoke \
  --device cpu \
  --run-dir /tmp/mujoco_goal_smoke
```

### 4) 체크포인트 평가
```bash
python scripts/mujoco/eval_sb3.py \
  --task Hexapedal-MuJoCo-Direct-v0 \
  --checkpoint /tmp/mujoco_goal_smoke/best.zip \
  --episodes 5 \
  --seed 42 \
  --device cpu \
  --output /tmp/mujoco_goal_eval.json
```

## 이 repo에 같이 넣어둔 것

- `reference/spider_rl/scripts/mujoco/`
  - 실제 MuJoCo check/train/eval 스크립트 사본
- `reference/spider_rl/source/spider_rl/spider_mujoco/`
  - MJCF 생성기, env, config, asset, PPO 설정 사본
- `reference/spider_rl/source/spider_rl/tests/`
  - focused test 사본

## 핵심 파일

- `source/spider_rl/spider_mujoco/__init__.py`
  - 독립 bootstrap. Isaac 쪽 import 없이 Gym env 등록.
- `source/spider_rl/spider_mujoco/hexapedal_direct/model_builder.py`
  - URDF를 읽어서 MJCF XML + `source_map.yaml` 생성.
- `source/spider_rl/spider_mujoco/hexapedal_direct/env.py`
  - MuJoCo direct env.
- `source/spider_rl/spider_mujoco/hexapedal_direct/env_cfg.py`
  - 관측/행동/보상/종료/기본 joint 세팅.
- `scripts/mujoco/check_model.py`
  - 변환/로드 계약 확인.
- `scripts/mujoco/train_sb3.py`
  - SB3 PPO 학습.
- `scripts/mujoco/eval_sb3.py`
  - best/latest 체크포인트 평가.

## 실제로 확인된 실행 경로

아래는 이미 한 번 성공한 경로다.

```bash
python -m pip install -e source/spider_rl[mujoco]
python scripts/mujoco/check_model.py --task Hexapedal-MuJoCo-Direct-v0
python scripts/mujoco/train_sb3.py --task Hexapedal-MuJoCo-Direct-v0 --protocol smoke --device cpu --run-dir /tmp/mujoco_goal_smoke
python scripts/mujoco/eval_sb3.py --task Hexapedal-MuJoCo-Direct-v0 --checkpoint /tmp/mujoco_goal_smoke/best.zip --episodes 5 --seed 42 --device cpu --output /tmp/mujoco_goal_eval.json
```

실행 산출물 예시:
- `/tmp/mujoco_check_model.json`
- `/tmp/mujoco_goal_smoke/best.zip`
- `/tmp/mujoco_goal_smoke/latest.zip`
- `/tmp/mujoco_goal_eval.json`
- `/tmp/mujoco_runnable_receipt.json`

## 공부용 문서

자세한 해설은 아래 문서를 보면 된다.

- `SPIDER_MUJOCO_STUDY_GUIDE.md`
