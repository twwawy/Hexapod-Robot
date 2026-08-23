---
title: Hexapod MJX Residual RL Study Guide
aliases:
  - Hexapod MJX Vault
  - Hexapod Locomotion Residual RL
tags:
  - robotics
  - hexapod
  - reinforcement-learning
  - residual-rl
  - mujoco
  - mjx
  - brax
  - wandb
status: active
updated: 2026-08-23
canonical_source: /home/huro/Downloads/mjx
---

# Hexapod MJX Residual RL Study Guide

> [!abstract]
> 이 노트는 `~/Downloads/mjx` 업데이트를 `SW/mjx/`에 반영한 현재 canonical MJX 경로를 설명한다. 상세 최신 계약은 [[RESIDUAL_RL]]이며, **평지 보행+회전**과 **mixed-terrain competence curriculum**을 분리한다.

> [!warning] 이전 6-D 실험과 호환되지 않음
> 이전 custom 경로는 `6-D Δz` action과 `62-D observation`을 사용했다. 현재 경로는 `22-D action`, `110-D observation`, mesh-free flat/mixed scene 및 Brax PPO를 사용한다. 현재 observation semantic contract가 없는 옛 checkpoint는 transfer하지 않는다.

## 이 노트를 보는 방법

처음에는 [[#30초 요약]] → [[#실행 경로와 명령어]] → [[#mesh가 제거된 RL scene]] → [[#제어 구조]] 순서로 읽는다. 코드를 공부할 때는 [[#코드 읽기 순서]], 실행이 안 되면 [[#디버깅 체크리스트]]를 사용한다.

## 30초 요약

| 항목 | 현재 값 |
| --- | --- |
| 기준 원본 | `~/Downloads/mjx` |
| CAD viewer scene | `SW/mjx/generated/hexapod_scene.xml`, CAD mesh 유지 |
| 평지 command scene | `SW/mjx/generated/hexapod_flat_rl.xml`, mesh 0개 |
| mixed terrain scene | `SW/mjx/generated/hexapod_mixed_rl.xml`, mesh 0개 |
| RL collider | torso box, leg capsule, foot sphere, terrain box |
| Nominal controller | tripod + quintic timing + analytical 3-DOF IK + position actuator |
| Policy | 18 raw foot actions → swing XYZ / stance Z-only + 4 bounded gait residual |
| Action contract | `cartesian_gait_residual_v2`; old v1 checkpoint resume 금지 |
| Observation contract | `body_state_coarse9_touchdown6_v1` |
| Action / observation | `22 / 110` |
| Physics | MuJoCo 3.12, MJX, `sim_dt=0.0025`, `ctrl_dt=0.02` |
| PPO | Brax PPO + `mujoco_playground` |
| 평지 학습 | `train_command_curriculum.py --wandb` |
| mixed terrain 학습 | `train_rough_terrain.py --terrain-layout mixed --wandb` |
| 실시간 최고 점수 | `SW/mjx/runs/<task>/<run-id>/monitor/best_score.txt` |
| 최고 policy 영상 | command는 `runs/.../videos/` 4종, terrain은 `videos/best_policy.gif` |

핵심 식:

```text
p_cmd,i = Π_workspace(contact_adapter(p_nominal,i + M_phase/contact Δp_RL,i))
```

RL은 관절 토크를 직접 만들지 않는다. classical tripod가 phase·stance/swing·IK를 소유하고, RL은 swing XYZ 또는 stance의 작은 Z만 보정한다. 안전 우선순위는 `joint/rate safety > contact adaptation > RL > nominal`이다.

## 레포 지도

```text
Hexapod-Robot/
├── HW/urdf/                         # CAD mesh와 원본 Xacro/URDF
├── SW/mjx/
│   ├── prepare_urdf.py              # standalone URDF 생성
│   ├── prepare_scene.py             # CAD mesh/home pose/actuator scene
│   ├── prepare_rl_scene.py          # primitive collider + multi-terrain lanes
│   ├── domain_randomization.py      # per-env dynamics; force cap 고정
│   ├── tripod_controller.py         # nominal tripod/IK
│   ├── tripod_core.py               # pure-JAX nominal/residual/contact/workspace/IK contract
│   ├── view_robot.py                # CAD scene viewer
│   ├── run_controller.py            # 평지 nominal gait demo
│   ├── view_rl_scene.py             # 계단 zero-residual viewer/GIF
│   ├── rough_terrain_env.py         # 공용 22-D residual environment
│   ├── command_curriculum_env.py    # flat 보행+회전 curriculum task
│   ├── train_command_curriculum.py  # flat curriculum entry point
│   ├── train_rough_terrain.py       # mixed terrain entry point + Brax PPO/W&B
│   ├── train_competence_curriculum.py # success 기반 level 조절
│   └── best_policy_video.py          # new-best policy의 deterministic GIF renderer
├── docs/Hexapod_MJX_Obsidian_Study_Vault.md
└── Hexapod-MJX-가이드/               # 이전 6-D custom PPO wrapper 참고용
```

## mesh가 제거된 RL scene

`prepare_rl_scene.py`는 CAD scene을 읽고 다음 순서로 training scene을 만든다.

```text
hexapod_scene.xml
  → inertia 값을 compiled model 값으로 확정
  → 모든 <mesh> asset 삭제
  → 모든 type="mesh" geom 삭제
  → torso/leg/foot primitive collider 추가
  → mixed면 flat/curb/ramp/blocks/stairs/rough lane 추가
  → hexapod_flat_rl.xml / hexapod_rl.xml / hexapod_mixed_rl.xml
```

따라서 RL scene에서 `nmesh=0`, mesh geom=0은 **정상**이다. visual fidelity가 아니라 MJX batch contact 속도·안정성·재현성을 위한 의도된 분리다.

| 부위 | RL collider | 목적 |
| --- | --- | --- |
| torso | box | body-ground 충돌과 낙상 판정 |
| coxa/femur/tibia | capsule | 다리 segment 충돌 |
| foot | sphere | 계단 접촉·clearance |
| flat task | 원래 plane | 보행·회전 명령 추종 |
| stairs task | 7개 box | 높이 5 cm, 폭 25 cm 계단 |
| mixed task | 6개 lane | reset마다 flat/curb/ramp/blocks/stairs/rough 선택 |

scene parameter는 다음과 같다.

```text
STEP_START_X = 0.55 m
STEP_DEPTH   = 0.25 m
STEP_HEIGHT  = 0.05 m
STEP_COUNT   = 7
```

> [!important]
> 학습용 `hexapod_rl.xml`에 CAD mesh를 다시 넣지 않는다. 외관을 보고 싶을 때만 `hexapod_scene.xml`을 사용한다.

## 제어 구조

강화학습 정책은 기존 Tripod 제어기를 대체하지 않고 다음과 같이 보정한다.

```text
command → Classical Tripod Controller → nominal local foot target
                                           ↓
                         RL: swing XYZ / stance Z-only + bounded gait residual
                                           ↓
                   Contact adaptation → workspace projection → analytical IK
                                           ↓
                                joint/rate limit → position actuator
```

우선순위는 `Safety > Contact adaptation > RL > Nominal gait`다. Swing contact는
해당 leg가 실제로 no-contact를 경험해 `airborne=true`가 된 뒤에만 early landing으로
판정한다. stance lost contact는 작은 downward search만 한다.

```text
p_cmd,i = Π_workspace(C_contact(p_nominal,i + M_i(phase, contact) Δp_RL,i))
```

`M_i`는 stance XY를 정확히 0으로 하는 authority mask다. Flat→terrain transfer에서
action 의미가 바뀌지 않도록 두 task 모두 swing XY ±25/±15 mm, Z −10/+50 mm,
stance Z ±8 mm를 사용한다. Flat에서는 residual penalty만 더 강하다.
기존 v1 `(±40, ±30, ±90 mm)` checkpoint는 재개하지 않는다.

## Action 22차원

| 범위 | 의미 |
|---|---|
| 0:18 | RF, RM, RB, LF, LM, LB raw local foot action → swing XYZ / stance Z-only |
| 18 | 보폭 scale: raw 0.8~1.2, 최종 horizontal stroke ≤120 mm |
| 19 | gait frequency scale: 0.85~1.15 |
| 20 | global swing height: 50~110 mm |
| 21 | radial offset: 5~25 mm |

정책 행동은 `[-1, 1]`로 제한한다. 위상 자체를 매 주기 직접 출력하게 하지 않고
주파수만 제한적으로 바꾸므로 Tripod A/B 순서와 기본 안정성은 유지된다. `a=0`은
nominal gait와 동일한 target을 만든다.

## Observation

- 목표 전진 속도와 목표 Yaw rate
- 몸체 선속도·각속도와 중력 방향
- 18개 관절 위치·속도
- 6개 발 위치와 접촉 추정값
- heading 기준 3×3 coarse terrain + 6개 nominal touchdown 높이
- gait phase의 sin/cos
- 이전 action

## Reward와 종료

주 보상은 목표 선속도·Yaw rate 추종이다. Upright, 몸체 높이 및 전진 진행을
보상하고 swing/stance/gait residual과 각각의 action-rate, torque, torque saturation,
slip, body/self contact,
workspace projection을 분리된 penalty로 둔다. 몸체가 지면에 너무 가까워지거나 크게 기울면 episode를
종료한다.

## 학습 분기와 curriculum

보행과 회전을 서로 다른 policy로 나누지 않는다. 평지에서 하나의 policy가 command
추종 난이도를 순서대로 경험하도록 만들고, 지형 적응은 별도 terrain run으로 분리한다.
두 run은 action/observation/controller가 같아도 scene·checkpoint·W&B project를 공유하지 않는다.

### A. 평지 보행 + 회전: 하나의 command curriculum

`HexapodCommandCurriculumEnv`는 `hexapod_flat_rl.xml`을 사용한다. 각 1,000-step
episode stage는 다음 command 범위를 제공한다.

| stage | 제어 step | forward speed | yaw rate | 학습 목적 |
| --- | ---: | --- | --- | --- |
| `0` | `0–249` | `0.03–0.10 m/s` | `0` | 안정적인 직진 tripod gait |
| `1` | `250–499` | `0.05–0.18 m/s` | 최대 `±0.15 rad/s` | 중속 완만한 곡선 보행 |
| `2` | `500–999` | `0.03–0.27 m/s` | 최대 `±0.35 rad/s` | 전체 보행·회전 명령 추종 |

실제 command는 stage의 고정 순서를 암기하지 못하도록 1.5–4.0초마다 범위 안에서
무작위로 다시 sample한다. policy는 항상 현재 command를
observation의 첫 2개 값으로 받으며, 22D residual action과 nominal tripod/IK는 바뀌지 않는다.
따라서 “보행 policy”와 “회전 policy”를 이어 붙이는 구조가 아니다.

120 mm stride와 최대 `1.15×` frequency에서 fastest-foot 속도 한계는 약
`0.276 m/s`다. `0.27 m/s`와 큰 yaw를 동시에 요구하지 않도록 각 forward speed에서
가능한 yaw 범위를 자동 계산한다. 따라서 최고속도는 사실상 직진으로 평가하고,
`±0.30 rad/s` 회전은 `0.14 m/s` 고정 평가에서 좌우 각각 확인한다.

### B. Mixed terrain: 별도 competence task

`HexapodRoughTerrainEnv`는 한 모델 안의 flat/curb/ramp/blocks/stairs/rough lane을
reset마다 선택하고, 9D coarse grid와 6D nominal touchdown height를 사용한다.
`train_competence_curriculum.py`는 evaluation success가 0.8보다 높으면 level을 올리고
0.5보다 낮으면 내린 뒤 직전 checkpoint를 다음 stage에 전달한다.

1. 20~50 mm 랜덤 단차
2. 계단 높이·폭·마찰과 로봇 질량 domain randomization
3. 외란, 센서 노이즈, actuator 지연

---

## 실행 경로와 명령어

### 가상환경

```bash
source /home/huro/.venvs/hexapod-mjx/bin/activate
cd /home/huro/Hexapod-Robot
```

기본 scene/viewer 의존성:

```bash
python -m pip install -r SW/mjx/requirements.txt
```

새 Brax PPO environment까지 포함한 전체 의존성:

```bash
python -m pip install -r SW/mjx/requirements-train.txt
```

현재 학습 조합은 다음을 사용한다.

```text
jax == 0.6.2
mujoco-mjx == 3.12.0
playground == 0.1.0       # Python import: mujoco_playground
brax == 0.14.1
ml-collections
```

장치와 import 확인:

```bash
unset LD_LIBRARY_PATH
PYTHONPATH=SW/mjx python - <<'PY'
import jax, mujoco, mujoco_playground, brax, ml_collections
print('JAX devices:', jax.devices())
print('MuJoCo:', mujoco.__version__)
print('mujoco_playground / Brax / ml_collections: OK')
PY
```

- `GpuDevice`가 보이면 GPU PPO를 실행할 수 있다.
- `CpuDevice`만 보이면 scene viewer와 smoke test는 가능하지만 장시간 PPO는 권장하지 않는다.

### CAD mesh와 home pose 확인

```bash
python SW/mjx/view_robot.py
```

headless PNG:

```bash
MUJOCO_GL=egl python SW/mjx/view_robot.py --headless \
  --output SW/mjx/generated/cad_preview.png
```

### RL mesh-stripped scene 생성과 확인

```bash
# flat command curriculum scene
PYTHONPATH=SW/mjx python - <<'PY'
from prepare_rl_scene import prepare_flat_rl_scene
prepare_flat_rl_scene()
PY

# stairs + mixed terrain scene
python - <<'PY'
import mujoco
from pathlib import Path
import sys
sys.path.insert(0, 'SW/mjx')
from prepare_rl_scene import prepare_rl_scene, MIXED_RL_SCENE_OUTPUT
prepare_rl_scene()
prepare_rl_scene(MIXED_RL_SCENE_OUTPUT, terrain='mixed')
mesh_type = mujoco.mjtGeom.mjGEOM_MESH
for path in ('SW/mjx/generated/hexapod_flat_rl.xml',
             'SW/mjx/generated/hexapod_rl.xml',
             'SW/mjx/generated/hexapod_mixed_rl.xml'):
    model = mujoco.MjModel.from_xml_path(path)
    mesh_geoms = sum(model.geom_type[i] == mesh_type for i in range(model.ngeom))
    print(f'{path}: nmesh={model.nmesh}, mesh_geoms={mesh_geoms}, ngeom={model.ngeom}, nu={model.nu}')
    assert model.nmesh == 0 and mesh_geoms == 0 and model.nu == 18
PY
```

### zero-residual 계단 baseline

학습 전에 꼭 실행한다. 이는 policy 성능이 아니라 nominal tripod가 계단에서 어디까지 가는지 보는 비교군이다.

```bash
python SW/mjx/view_rl_scene.py \
  --speed 0.10 \
  --phase-time 0.5 \
  --swing-height 0.07 \
  --radial-offset 0.01 \
  --duration 12
```

GUI가 없으면 GIF를 만든다.

```bash
MUJOCO_GL=egl python SW/mjx/view_rl_scene.py \
  --headless \
  --duration 12 \
  --output SW/mjx/generated/rough_terrain_baseline.gif
```

### A. 평지 보행 + 회전 curriculum

이 명령 하나가 직진 → 완만한 회전 → 전체 보행+회전 순서의 flat curriculum을
학습한다. `train_rough_terrain.py --task command`와 동등하지만, 아래 entry point를
사용하면 checkpoint와 W&B project 기본값이 평지 task로 자동 분리된다.

```bash
python SW/mjx/train_command_curriculum.py --smoke

python SW/mjx/train_command_curriculum.py \
  --timesteps 50000000 \
  --num-envs 2048 \
  --num-evals 50 \
  --seed 0 \
  --run-name command-v2-seed0 \
  --best-video \
  --wandb \
  --wandb-project hexapod-command-curriculum \
  --wandb-name flat-walk-turn-seed0
```

W&B에서 이름에 `curriculum_stage`가 들어간 metric이 `0 → 1 → 2`로 진행하는지와 `reward/velocity`,
`reward/yaw`, `reward/swing_residual`, `reward/stance_residual`, `projection_cost`를 함께 본다.
추가로 매 evaluation마다 독립 reset과 고정 command script로 계산되는
`eval/stage0|1|2/reward_mean`, `velocity_error_mps`, `yaw_error_rps`,
`survival_fraction`, `torque_rms_nm`, `torque_saturation_mean`,
`self_collision_rate`, `effective_stride_mean_m/max_m`을 비교한다. 학습 command
자체는 계속 1.5–4초 random이다.

### B. Mixed terrain transfer

평지 checkpoint의 policy/normalizer를 mixed terrain 초기값으로 사용한다.

```bash
python SW/mjx/train_rough_terrain.py \
  --timesteps 50000000 \
  --num-envs 2048 \
  --num-evals 100 \
  --seed 0 \
  --run-name terrain-transfer-level0 \
  --terrain-layout mixed \
  --terrain-level 0 \
  --init-checkpoint SW/mjx/runs/command/<flat-run>/checkpoints \
  --terrain-randomize \
  --best-video \
  --wandb \
  --wandb-project hexapod-rough-terrain \
  --wandb-name mixed-22d-seed0
```

Success 기반 전체 curriculum은 다음 한 명령으로 실행한다.

```bash
python SW/mjx/train_competence_curriculum.py \
  --run-name mixed-competence --stages 8 --stage-timesteps 5000000 \
  --init-checkpoint SW/mjx/runs/command/<flat-run>/checkpoints --wandb \
  -- --num-envs 2048 --num-evals 20 --terrain-randomize
```

### MJX smoke test

PPO 없이 zero action 100 step과 bounded random action 100 step을 각각 실행한다.

```bash
# flat walk + turn curriculum
python SW/mjx/train_command_curriculum.py --smoke --smoke-steps 100

# mixed terrain
python SW/mjx/train_rough_terrain.py --smoke --smoke-steps 100 --terrain-layout mixed
```

GPU가 없는 현재 환경처럼 CPU에서 검증할 때는:

```bash
JAX_PLATFORMS=cpu python SW/mjx/train_command_curriculum.py --smoke --smoke-steps 100
JAX_PLATFORMS=cpu python SW/mjx/train_rough_terrain.py --smoke --smoke-steps 100 --terrain-layout mixed
```

정상 출력의 핵심은 `obs=110 action=22 ... done=0`이다.

작은 CPU debugging run도 자동 run directory를 만들도록 이름만 분리한다.

```bash
python SW/mjx/train_command_curriculum.py \
  --allow-cpu \
  --timesteps 20000 \
  --num-envs 32 \
  --run-name command-v2-cpu-debug \
  --wandb --wandb-mode offline
```

> [!warning]
> 기본 학습은 GPU backend가 아니면 종료된다. `--allow-cpu`는 작은 debug run에만 사용한다.

### Run directory, checkpoint, 최고 policy 영상

trainer는 `--run-name`을 prefix로 사용하고 항상
`SW/mjx/runs/<task>/<name>_<UTC timestamp>_seed<seed>/`를 새로 만든다. 기존 run은
절대 재사용하지 않는다.

```text
runs/command/command-v2-seed0/
├── checkpoints/
├── monitor/
├── videos/
│   ├── best_stage0_forward.gif
│   ├── best_stage1_limited_yaw.gif
│   ├── best_stage2_full_command.gif
│   └── best_curriculum_full.gif
├── config.json
└── run_metadata.json     # git SHA, v2 action contract, config, PPO, version
```

`monitor/` 안에서 trainer는 evaluation마다 아래 파일을 갱신한다.

| 파일 | 갱신 시점 | 용도 |
| --- | --- | --- |
| `latest_metrics.json` | 매 evaluation | 가장 최근의 모든 scalar metric |
| `metrics_history.jsonl` | 매 evaluation | 시간 순서 전체 metric 이력 |
| `best_score.json` | `--score-key`가 최고 기록일 때 | 프로그램/분석용 best record |
| `best_score.txt` | `--score-key`가 최고 기록일 때 | terminal에서 바로 확인할 요약 |
| `stage_metrics_latest.json` | command 매 evaluation | Stage 0/1/2 독립 reward·tracking·survival 최신값 |
| `stage_metrics_history.jsonl` | command 매 evaluation | Stage별 scalar 전체 이력 |
| `../videos/*.gif` | `--score-key`가 최고 기록일 때 | Stage 0/1/2/full deterministic 영상; 새 best로 자동 교체 |
| `best_video.json` | 영상 생성 시 | step, score, 네 영상 경로, 렌더 오류 |

기본 최고 점수 기준은 `eval/episode_reward`다. 학습 중 다른 terminal에서 다음을 실행한다.

```bash
# 평지 curriculum의 현재 최고 점수만 계속 표시
watch -n 2 cat SW/mjx/runs/command/command-v2-seed0/monitor/best_score.txt

# 각 evaluation의 전체 metric을 실시간으로 본다.
tail -f SW/mjx/runs/command/command-v2-seed0/monitor/metrics_history.jsonl
```

`--num-evals 50`이면 5천만 step run에서 약 100만 step마다 평가·monitor가 갱신된다.
더 자주 보고 싶으면 `--num-evals 100`으로 올릴 수 있지만 평가 비용도 증가한다. W&B를 켠
run에서는 새 최고점이 `best/score`, `best/step`, `best/score_key` summary에도 기록된다.

기본값으로 `--best-video`가 켜져 있다. Command NEW_BEST에서는 서로 독립 reset된
Stage 0 10초, Stage 1 10초, Stage 2 12초 영상과 0→1→2 전환을 포함한 full 22초
영상을 생성한다. Stage 영상은 checkpoint마다 같은 command script를 사용하므로 영상 간
비교가 가능하다. 좌측 상단 overlay에는 stage, 시간, `v_cmd/v`, `yaw_cmd/yaw`가 표시되고,
full 영상의 5초·10초 경계에는 0.8초 전환 banner가 나온다. W&B Media key는
`best/video_stage0_forward`, `best/video_stage1_limited_yaw`,
`best/video_stage2_full_command`, `best/video_curriculum_full`이다. Terrain은 기존처럼
단일 `videos/best_policy.gif`와 `best/video`를 쓴다. Pillow GIF writer를 사용하며,
`best_video_error`가 발생해도 PPO 학습이나 checkpoint는 중단되지 않는다.

### 자주 조정하는 CLI 옵션

모든 옵션은 두 trainer에 공통이다. 현재 값은 언제든 `python SW/mjx/train_command_curriculum.py --help`로 확인한다.

| 목적 | 옵션 | 권장 시작값 / 주의점 |
| --- | --- | --- |
| 병렬 규모 | `--num-envs` | 3090은 `2048`부터, OOM이면 `1024 → 512` |
| 학습 시간 | `--timesteps` | 본 run `50000000`, 빠른 검증 `20000` |
| 평가/monitor 빈도 | `--num-evals`, `--num-eval-envs`, `--stage-eval-envs` | `50`, `64`, `8`; command stage 평가는 독립 reset 8개 |
| PPO rollout | `--unroll-length` | command `20`, terrain `32`; ablation `20/32/50` |
| PPO update | `--batch-size --num-minibatches --num-updates-per-batch` | 기본 `256 8 4`; 한 번에 하나만 바꿈 |
| optimizer | `--learning-rate --entropy-cost --discounting` | command γ `0.97`, terrain γ `0.99` |
| network | `--network-layers` | 기본 `256 256 128`; OOM이면 `192 128` |
| nominal gait | `--phase-time --base-swing-height --base-radial-offset` | controller 자체가 바뀌므로 기존 run과 직접 비교하지 않음 |
| RL foot 권한 | `--swing-x --swing-y --swing-z-low --swing-z-high --stance-z` | active task만 override; stance XY는 항상 0 |
| RL gait 권한 | `--stride-half-range --frequency-half-range --swing-height-min --swing-height-max --radial-min --radial-max --gait-filter-time-constant` | 기본 stride `0.8…1.2×`, frequency `0.85…1.15×`, filter `0.15 s` |
| terrain command 범위 | `--terrain-speed-min --terrain-speed-max --terrain-yaw-limit` | stairs/mixed terrain task에서 사용 |
| terrain 난이도 | `--terrain-layout --terrain-level --terrain-randomize` | mixed patch 확률 + level 4 per-env dynamics |
| policy transfer | `--init-checkpoint` | 동일 22/110 semantic contract의 flat policy로 terrain 초기화 |
| run 관리 | `--run-name --run-root` | checkpoint/monitor/video/config/metadata를 한 run directory에 저장 |
| 평지 curriculum 길이 | `--curriculum-forward-only-steps --curriculum-limited-yaw-steps` | 기본 `250 250`, 마지막 stage는 episode 끝까지 |
| 평지 curriculum 범위 | `--curriculum-speed-min`, `--curriculum-speed-max`, `--curriculum-yaw-limit` | 각 stage 순서의 값 3개 |
| 평지 마찰 | `--flat-friction` | 건조 아스팔트 nominal `0.8` |
| 최고점 기준/경로 | `--score-key --monitor-dir` | 기본 score key는 `eval/episode_reward` |
| 최고 정책 영상 | `--best-video --best-video-path` | command는 `videos/` 4종, terrain은 단일 GIF; NEW_BEST에서만 교체 |
| 영상 길이 | `--best-video-stage0-duration` 등 | command 기본 `10/10/12/22 s`; terrain은 `--best-video-duration=10` |
| 영상 품질/비용 | `--best-video-fps --best-video-width --best-video-height` | 기본 `20 fps, 640×360`; NEW_BEST 직후만 렌더링 |
| W&B | `--wandb-*` | project·name·group을 실험 단위로 분리 |

예: 평지 curriculum에서 회전 도입을 늦추고 policy 크기를 줄이는 debug run:

```bash
python SW/mjx/train_command_curriculum.py \
  --timesteps 2000000 --num-envs 1024 --num-evals 20 \
  --curriculum-forward-only-steps 400 \
  --curriculum-limited-yaw-steps 300 \
  --network-layers 192 128 \
  --monitor-dir SW/mjx/artifacts/command/late-yaw-debug \
  --wandb --wandb-name late-yaw-debug
```

---

## Action 22차원: index와 단위

정책 action은 먼저 `[-1, 1]`로 clip된다.

| index | 수량 | 의미 | 변환 후 범위 |
| --- | ---: | --- | --- |
| `0:18` | 6×3 | `RF, RM, RB, LF, LM, LB` local foot `(x,y,z)` | swing XYZ; stance `(0,0,Δz)` only |
| `18` | 1 | stride scale | raw `0.8 … 1.2`; forward+yaw fastest-leg stroke `≤120 mm` |
| `19` | 1 | gait frequency scale | `0.85 … 1.15` |
| `20` | 1 | global swing height | `50…110 mm`, 두 task 동일 |
| `21` | 1 | radial offset | `5…25 mm`, 두 task 동일 |

발 residual의 실제 계산은 다음과 같다.

```text
raw_i = clip(action[3i:3i+3], -1, 1)
Δp_i = swing ? [Δx, Δy, Δz_asymmetric] : [0, 0, Δz_stance]
p_cmd = workspace_projection(contact_adapter(p_nominal + blend × Δp_i))
```

`blend`는 reset 후 약 0.75초 동안 0에서 1로 증가한다. `a=0`이면 residual/gait correction도 0이므로 nominal controller와 일치한다. early landing hold, lost-contact search, workspace projection, joint/rate limit은 RL보다 우선한다.

---

## Observation 110차원

`rough_terrain_env.py`의 `_get_obs()`는 아래를 순서대로 concatenate한다. 평지
curriculum도 동일한 110D semantic contract를 쓰며 terrain feature 15D는 0이다.

| 항목 | 차원 | 내용 |
| --- | ---: | --- |
| command | 2 | target forward speed, yaw rate |
| local linear velocity | 3 | body local frame 속도 |
| angular velocity | 3 | base angular velocity |
| local gravity | 3 | IMU attitude 대체값 |
| joint position error | 18 | home pose 대비 관절각 |
| scaled joint velocity | 18 | `0.1 × qvel` |
| foot body position | 18 | `R_WBᵀ(p_foot^W-p_base^W)`; world yaw 불변 |
| contact | 6 | collision + 35/45 mm clearance hysteresis |
| coarse terrain | 9 | body heading 기준 3×3, support height 상대값 |
| touchdown terrain | 6 | classical nominal landing 위치의 support-relative 높이 |
| phase | 2 | `sin/cos(2πphase)` |
| last action | 22 | 이전 policy action |
| **합계** | **110** | policy input |

Touchdown feature는 grid·body pose·phase를 network가 다시 조합하지 않아도 각 leg의
다음 nominal landing 높이를 직접 알게 한다. Observation 크기는 같아도 의미가 바뀌었으므로
metadata의 `body_state_coarse9_touchdown6_v1`을 반드시 확인한다.

---

## Nominal controller와 analytical IK

`tripod_controller.py`가 nominal policy다. 기본 값은 다음과 같다.

| 파라미터 | 값 | 의미 |
| --- | ---: | --- |
| control_dt | 0.005 s | target refresh, 200 Hz |
| phase_time | 0.5 s | tripod half-cycle |
| stand_time | 1.0 s | 시작 pose hold |
| speed | 0.06 m/s | 평지 기준 속도 |
| swing_height | 0.06 m | 기본 foot lift |
| radial_offset | 0.01 m | swing outward clearance |
| actuator | kp=120, kv=3, ±8 Nm | native position actuator |

### quintic tripod trajectory

```text
s(τ) = 10τ³ - 15τ⁴ + 6τ⁵

swing offset = step_length × (s(τ) - 0.5)
stance offset = step_length × (0.5 - τ)
lift          = 4 × swing_height × s(τ) × (1-s(τ))
radial        = 4 × radial_offset × s(τ) × (1-s(τ))
```

Tripod A는 `RF, RB, LM`, complement tripod은 `RM, LF, LB`이다. policy는 action 19로 frequency만 제한적으로 바꾸며 tripod grouping과 phase 순서는 직접 바꾸지 않는다.

### 3-DOF inverse kinematics

각 target은 hip의 outward/tangent/local-z 좌표로 바꾼 뒤 analytical IK를 푼다.

```text
L1 = 0.074 m
L2 = 0.121 m
L3 = 0.230 m

θ1 = atan2(y, x)
ρ  = sqrt(x²+y²) - L1
cos(θ3) = (ρ² + z² - L2² - L3²) / (2 L2 L3)
θ2 = atan2(z, ρ) - atan2(L3 sinθ3, L2 + L3 cosθ3)
```

마지막에 servo angle은 raw URDF joint axis로 변환된다. right leg와 left leg의 sign이 다르므로 `RIGHT_LEGS` 변환은 절대 생략하지 않는다.

---

## Reward와 termination

| 항목 | 방향 | 역할 |
| --- | --- | --- |
| velocity | 보상 | forward speed command tracking |
| yaw | 보상 | yaw-rate tracking |
| upright | 보상 | body up direction 유지 |
| height | 보상 | terrain 위 body clearance 유지 |
| progress | 보상 | 실제 전진 진행 |
| swing_residual | penalty | swing에서만 필요한 Cartesian 보정 사용 |
| stance_residual | penalty | nominal stance를 덮는 Z 보정; swing보다 더 비쌈 |
| gait_residual | penalty | stride/frequency/height/radial deviation 분리 |
| foot/gait_action_rate | penalty | foot 보정 및 gait parameter의 50 Hz 흔들림 억제 |
| vertical/lateral velocity | penalty | bounce, sideways drift 억제 |
| joint_velocity | penalty | 불필요한 관절 고속 운동 억제 |
| torque | penalty | `mean((actuator_force / 8 Nm)^2)` |
| torque_saturation | penalty | 6.8 Nm(85%) 초과분을 8 Nm까지 0…1로 정규화 |
| slip | penalty | contact foot의 normalized XY velocity |
| projection | penalty | requested target이 safe workspace에서 밀려난 거리² |
| body_contact | penalty | torso collision; recovery를 위해 즉시 terminate하지 않음 |
| self_collision | penalty | active contact의 양쪽 geom이 모두 robot body인 경우 |
| termination | penalty | 낙상 종료 비용 |

episode는 아래 조건에서 끝난다.

```text
up_z < 0.35
or base clearance above terrain < 0.14 m
or qpos/qvel contains NaN
```

reward만 높다고 좋은 정책은 아니다. `swing_residual`, `stance_residual`,
`gait_residual`, `slip`, `torque`, `torque_saturation`, `self_collision`,
`projection_cost`, torso contact와 termination 빈도를 zero-residual baseline 동영상과 함께 비교한다.

---

## PPO, rollout, W&B

`train_command_curriculum.py`와 `train_rough_terrain.py`는 같은 Brax PPO 설정을
사용한다. 전자는 flat command curriculum, 후자는 mixed terrain task를 선택하는
별도 실행 진입점이다.

| 설정 | 기본값 | 뜻 |
| --- | ---: | --- |
| num_timesteps | 50,000,000 | 전체 environment step budget |
| num_envs | 2048 | GPU 병렬 환경 수 |
| episode_length | 1000 | 한 episode 길이 |
| action_repeat | 1 | 매 control step action 갱신 |
| unroll_length | command 20 / terrain 32 | PPO rollout chunk |
| batch_size | 256 | optimizer batch 구성 |
| num_minibatches | 8 | batch 분할 수 |
| num_updates_per_batch | 4 | PPO update 반복 |
| learning_rate | 3e-4 | optimizer step size |
| entropy_cost | 0.01 | exploration 강도 |
| discounting / GAE | command 0.97, terrain 0.99 / 0.95 | gait-cycle credit assignment |
| network | 256, 256, 128 | actor/critic hidden layers |

`--wandb`를 켜면 Brax evaluation progress callback의 metrics와 `train/global_step`이 현재 로그인된 W&B account로 전송된다. `--best-video`가 켜진 상태에서 score가 새 최고점이면 terrain은 `best/video`, command는 `best/video_stage0_forward`, `best/video_stage1_limited_yaw`, `best/video_stage2_full_command`, `best/video_curriculum_full`이 갱신된다.

```bash
python SW/mjx/train_command_curriculum.py \
  --wandb \
  --wandb-project hexapod-command-curriculum \
  --wandb-group walk-turn-ablation \
  --wandb-name seed0
```

| W&B 인수 | 역할 |
| --- | --- |
| `--wandb-project` | 프로젝트 이름 |
| `--wandb-entity` | team/user entity가 필요할 때만 지정 |
| `--wandb-name` | run 표시 이름 |
| `--wandb-group` | ablation/run 묶기 |
| `--wandb-mode offline` | 네트워크 없이 local queue 기록 |
| `--wandb-mode disabled` | W&B 완전 비활성화 |

영상은 W&B의 기본 설정이 아니라 trainer가 직접 기록하는 artifact다. 따라서 `--wandb`만 켜면 추가 login이나 별도 video 인수 없이 자동 업로드된다. local 저장만 원하면 `--wandb`를 빼면 되고, 영상까지 끄려면 `--no-best-video`를 준다.

W&B dashboard에서 최소한 `eval/episode_reward`, `train/global_step`,
`reward/velocity`, `reward/upright`, `reward/swing_residual`, `reward/stance_residual`,
`reward/gait_residual`, `reward/slip`, `reward/torque`, `reward/torque_saturation`,
`self_collision`, `effective_stride_m`, `projection_cost`를 함께 본다.

---

## 코드 읽기 순서

1. `prepare_urdf.py`: ROS/Xacro 의존성을 제거하고 mesh path를 standalone으로 바꾸는 과정
2. `prepare_scene.py`: home pose와 18 position actuator 생성
3. `tripod_controller.py`: viewer용 documented gait timing, local foot target, IK, 좌우 sign
4. `tripod_core.py`: pure-JAX nominal target, phase mask, contact adapter, workspace projection, IK
5. `prepare_rl_scene.py`: `_strip_cad_meshes`, primitive collider, parameterized stairs
6. `rough_terrain_env.py`: body-frame observation, heading scan, authority/reward/safety integration
7. `command_curriculum_env.py`: flat scene과 walk→turn stage를 선택하는 얇은 task wrapper
8. `best_policy_video.py`: new-best policy를 MJX에서 deterministic rollout하고 GIF로 저장
9. `train_command_curriculum.py`, `train_rough_terrain.py`: run directory/metadata + PPO/W&B entry point

읽으며 답해야 할 질문:

- CAD mesh는 왜 viewer에는 남고 RL scene에서는 없어지는가?
- policy action 0이 들어오면 exactly 어떤 nominal gait가 실행되는가?
- local foot residual은 어느 시점에 IK 전에 더해지는가?
- action 19는 phase 자체가 아니라 phase increment에 어떻게 반영되는가?
- reward가 좋지만 residual/action-rate가 큰 경우 왜 재검토해야 하는가?

---

## 디버깅 체크리스트

| 증상 | 확인 | 조치 |
| --- | --- | --- |
| `ModuleNotFoundError: mujoco_playground` | requirements-train 설치 | `pip install -r SW/mjx/requirements-train.txt` |
| `DISPLAY=` | GUI session 부재 | `--headless` GIF/PNG 사용 또는 GUI terminal 사용 |
| `nmesh > 0` | 잘못된 XML load | `prepare_rl_scene.py` 재실행, `hexapod_rl.xml` 사용 |
| `obs != 110` 또는 `action != 22` | 이전 trainer 사용 | 해당 task의 `--smoke` 실행 |
| 시작 직후 낙상 | home pose / actuator / RL scene | zero-residual `view_rl_scene.py`부터 확인 |
| GPU가 안 보임 | `jax.devices()` | CUDA JAX, `unset LD_LIBRARY_PATH` 점검 |
| W&B 로그 없음 | login/mode/project | `wandb status`, `--wandb-mode online` 확인 |

가장 안전한 진단 순서:

```text
CAD home pose → flat/stairs/mixed nmesh=0 assertion → zero-residual baseline
→ flat/terrain 각각 MJX smoke (110/22) → 작은 CPU debug → GPU PPO
```

---

## 이전 6-D custom 경로

아래는 연구 기록 보존용 legacy 경로다.

```text
SW/mjx/train_residual_ppo.py
SW/mjx/evaluate_residual_policy.py
SW/mjx/visualize_residual_policy.py
SW/mjx/hexapod_mjx/
Hexapod-MJX-가이드/residual_rl_run.sh
```

새 학습에서 이 경로를 실행 진입점으로 사용하지 않는다. canonical 명령은 다음 세 개다.

```bash
python SW/mjx/view_rl_scene.py
python SW/mjx/train_command_curriculum.py --smoke
python SW/mjx/train_rough_terrain.py --smoke
```

## 실험 노트 템플릿

```markdown
---
tags: [hexapod, rough-terrain, experiment]
status: planned
---

# Experiment: <name>

## 질문
- 예: terrain scan이 계단 상승 성공률을 올리는가?

## 고정 조건
- task / scene: `command / hexapod_flat_rl.xml` 또는 `terrain / hexapod_rl.xml`, `nmesh=0`
- action / observation: `22 / 110`
- nominal: documented tripod analytical IK
- seed:
- commit:

## 실행 명령
```bash
# 실제 명령 전문
```

## 결과
| Metric | zero residual | learned residual | 해석 |
| --- | ---: | ---: | --- |
| episode reward | | | |
| forward velocity | | | |
| termination rate | | | |
| residual/action rate | | | |
| workspace error | | | |
```

## 문서 유지 규칙

scene, action, observation, dependency, 실행 명령을 바꾸면 [[#30초 요약]], [[#실행 경로와 명령어]], [[#Action 22차원]], [[#Observation 110차원]], [[#PPO, rollout, W&B]]를 같이 갱신한다.
