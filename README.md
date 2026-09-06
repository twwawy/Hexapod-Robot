# Hexapod-Robot

6족 로봇의 기구·전자회로·펌웨어와 MuJoCo MJX 보행 학습을 관리하는 저장소다.
현재 작업은 **LiDAR 기반 지형 인식 + 제어기 기반 보행 + RL residual**을 MJX에 구성하는 것이다.
MJX에서 보행을 확인한 뒤 Isaac Lab으로 sim-to-sim 이전한다.

| 항목 | 현재 기준 |
|---|---|
| 주 작업 브랜치 | `codex/cartesian-residual-rl` |
| 하드웨어 보행 참고 | `main`의 Wave gait·접촉 처리·전환 조건 |
| 새 보행 모드 | adaptive 24-D, Tripod → short-step → Wave → HOLD |
| 기존 비교 모드 | stage31 v3 checkpoint, 18-D residual |
| 구현 상태 | planner·supervisor·controller·viewer·PPO 진입점 구현 |
| 검증 상태 | 최소 정적 확인만 진행; 보행·JIT rollout·학습 검증은 사용자 진행 |
| 새 학습 가중치 | 아직 없음; checkpoint 없이 실행하면 action=0 baseline |

## 목차

- [1. 프로젝트 구성](#project)
- [2. 어떻게 구현했는가](#implementation)
- [3. 실행 방법](#run)
- [4. 뷰어 조작과 확인 항목](#viewer)
- [5. 모델·센서·시뮬레이션 환경](#environment)
- [6. 학습 단계와 checkpoint](#training)
- [7. 기존 모드와 레거시](#legacy)
- [8. 문서 안내와 남은 작업](#docs)

<a id="project"></a>

## 1. 프로젝트 구성

인하대학교 로봇연구회가 2024년부터 개발하는 6족 보행 로봇 프로젝트다.
Jetson Orin Nano Super가 인지·상위 제어를, STM32 NUCLEO-F446RE가 센서 수집과
200 Hz 실시간 보행 제어를 담당하는 구조를 목표로 한다.

| 경로 | 역할 |
|---|---|
| [HW/](HW/README.md) | CAD, URDF, PCB, 부품과 제작 파일 |
| [SW/Controller/](SW/Controller/Controller_Architecture.md) | MATLAB/Simulink 제어기 설계·좌표계 |
| [SW/STM32/](SW/STM32/) | 하드웨어 보행 펌웨어 |
| [SW/Jetson/](SW/Jetson/README.md) | Jetson 상위 제어 소프트웨어 |
| [mjx/](mjx/) | MJX 제어기·LiDAR·planner·학습·viewer |
| [scripts/](scripts/) | viewer·학습 실행 스크립트 |
| [isaaclab_hexapod/](isaaclab_hexapod/README.md) | Isaac Lab 모델·센서·학습 scaffold |
| [docs/](docs/README.md) | 설계, 실행 안내, 업데이트 기록 |
| [SW/mjx/](SW/mjx/) | 과거 whole-body residual 학습 실험 |

<a id="implementation"></a>

## 2. 어떻게 구현했는가

### 2.1 설계 원칙과 데이터 흐름

기존 adaptive 코드의 action·observation·후보 거절 조건을 먼저 분석하고,
`main`의 Wave gait를 대조했다. 이후 geometry, residual, gait 선택, 접촉 제어를 분리했다.
변경 전 구성은 [구현 전 분석](docs/HEXAPOD_HYBRID_GAIT_ANALYSIS.md)에 기록했다.

```text
LiDAR + body state
        ↓
height map → surface normal / edge / confidence
        ↓
command → nominal landing → safe reference / feasible stride
                                   ↓
                            reference + RL residual
                                   ↓
                          Safety / IK / path projection
                                   ↓
                      swing 진입 시 목표·궤적 parameter 고정
                                   ↓
                         classical controller → Motor

Geometry feasibility → Tripod normal → short-step → Wave → HOLD
```

| 담당 | 결정하는 값 |
|---|---|
| LiDAR / geometry | terrain Z, normal, slope, roughness, edge, 관측률, support 크기 |
| Foothold planner | safe reference, 경로 장애물 높이, 도달 가능 후보, feasible stride |
| RL residual | reference 기준 XY, apex 여유, 몸체 자세·높이, 보폭, 궤적 timing |
| Supervisor | Tripod 유지·잔발·Wave·HOLD와 복귀 hysteresis |
| Controller / safety | swing latch, contact, 궤적 생성, IK·workspace·관절 제한 |

### 2.2 착지점과 지형 검사

기존 nominal 주변 3×3 후보를 **5×5 = 다리당 25개**로 확대했다.
검색 offset은 XY 각각 `-8, -4, 0, +4, +8 cm`이며, RL residual 범위 ±4 cm와 분리한다.

후보는 중심 관측, 최소 60% support coverage, 비공선 local plane fit을 요구한다.
관측된 높이의 plane residual·경사·단차·edge 여유를 검사하고, 이후 IK와 sampled swing path를 검사한다.
unknown을 평지로 채우지 않으며 **SAFE / UNKNOWN / UNSAFE**를 구분한다.

```text
p_final_xy = safe_reference_xy + RL_XY → 안전 후보로 projection
p_final_z  = terrain_surface_z
apex       = path_required_clearance + baseline_margin + RL_residual
body_pose  = terrain_normal_baseline + RL_residual
```

발 구의 반지름은 terrain surface Z를 IK endpoint로 변환할 때 더한다.
선택한 world landing과 clearance/timing은 swing 진입 시 고정한다.
진행 중 map 갱신으로 현재 landing을 바꾸지 않는다.

### 2.3 Tripod → 잔발 → Wave → HOLD

Tripod는 다리별 top-3 후보의 **27개 조합**으로 foothold와 support margin을 검사한다.
보폭 scale `[1.3, 1, .75, .5, .25, .125]`의 고정 검색으로 가능한 범위를 구한다.
normal이 불가능하면 작은 보폭을 먼저 시도하고, 관측된 Tripod 실패가 확인되면 Wave를 검토한다.

Wave는 `main`을 참고해 **RF → LB → RM → LF → RB → LM** 순서로 한 발씩 움직인다.
Early/Late Landing, support contact recovery, all-contact 전환 조건을 MJX scheduler에 포팅했다.
실제 모드 변경은 swing 완료 후 모든 발의 contact를 확인한 경계에서만 적용한다.
Wave→Tripod 복귀에는 다음 두 Tripod phase의 feasibility와 시간 hysteresis를 요구한다.

**UNKNOWN은 기본 hybrid에서 HOLD한다.** LiDAR 사각지대 때문에 첫걸음을 관측하지 못하면
자동으로 미관측 지형을 밟지 않는다. 무관측 기본 보행은 별도 `blind` debug 모드로 비교한다.

### 2.4 Action·observation과 코드 위치

| 24-D action | 내용 |
|---|---|
| 0:12 | 6다리 landing XY residual, ±4 cm |
| 12:18 | 6다리 apex/clearance residual, ±4 cm |
| 18 / 19 / 20 | body roll / pitch / height residual |
| 21 | 요청 step scale; planner의 feasible 범위로 제한 |
| 22 / 23 | apex plateau phase / XY transfer timing |

landing Z와 gait mode는 action에 없다. phase duration도 제어기가 보폭과 연결해 결정한다.
한 policy가 모든 다리의 action을 출력하고 controller가 active swing에만 적용한다.
actor는 **4410-D**, privileged critic은 **4725-D**다.

| 파일 (`mjx/`) | 구현 내용 |
|---|---|
| [adaptive_gait_perception.py](mjx/adaptive_gait_perception.py) | JAX LiDAR ray와 rolling height map |
| [adaptive_foothold_estimator.py](mjx/adaptive_foothold_estimator.py) | 후보·plane·edge·confidence·IK/path 검사 |
| [foothold_feasibility.py](mjx/foothold_feasibility.py) | top-K 조합과 support margin |
| [hybrid_gait_supervisor.py](mjx/hybrid_gait_supervisor.py) | normal / short / Wave / HOLD 결정 |
| [wave_gait_scheduler.py](mjx/wave_gait_scheduler.py) | Wave 순서·contact·안전 전환 |
| [adaptive_gait_controller.py](mjx/adaptive_gait_controller.py) | 24-D 적용·latch·궤적·기존 IK 연결 |
| [adaptive_gait_env.py](mjx/adaptive_gait_env.py) | MJX 통합·관측·보상·oracle 비교 |
| [adaptive_gait_policy.py](mjx/adaptive_gait_policy.py) | network와 checkpoint 계약 |
| [view_adaptive_gait.py](mjx/view_adaptive_gait.py) / [train_adaptive_gait.py](mjx/train_adaptive_gait.py) | 화면 진단 / 단계별 PPO |

세부 threshold, frame, 정규화, main과의 차이는 [adaptive 사용 가이드](docs/HEXAPOD_MJX_ADAPTIVE_GAIT_USAGE.md)를 따른다.

<a id="run"></a>

## 3. 실행 방법

### 3.1 작업 경로와 가상환경

```bash
cd /home/huro/Hexapod-Robot
source /home/huro/.venvs/hexapod-mjx/bin/activate
```

기존 MuJoCo·MJX·JAX·mujoco-playground·Brax/Orbax 환경을 사용한다.
실행 스크립트는 이 가상환경의 Python을 직접 선택하며 `HEXAPOD_PYTHON`으로 변경할 수 있다.
첫 화면까지 JAX 컴파일 시간이 필요하다.

### 3.2 새 adaptive 모드: oracle → LiDAR 순서

```bash
# 평지에서 action=0 baseline
bash scripts/view_foothold_planner.sh --controller adaptive --terrain flat --perception oracle

# 계단: GT 100% known으로 planner/controller 확인
bash scripts/view_foothold_planner.sh --controller adaptive --terrain steps --perception oracle

# 계단: LiDAR 입력 + 같은 자세의 GT 비교 지표
bash scripts/view_foothold_planner.sh --controller adaptive --terrain steps --perception lidar --stage0
```

| 옵션 | 용도 |
|---|---|
| `--perception oracle` | GT 높이·100% known, policy 없는 planner 진단 |
| `--perception lidar` | LiDAR actor/planner; GT는 critic·평가용 |
| `--perception teacher` | GT 높이 + LiDAR known mask; oracle과 다름 |
| `--perception blind` | 무관측 nominal 보행 debug; hybrid 안전 fallback 아님 |
| `--gait-mode tripod` / `wave` / `hybrid` | 고정 gait 또는 supervisor 비교 |
| `--stage0` | 같은 pose의 oracle 대비 후보·edge·오차 metric 추가 |
| `--terrain flat` / `ramp` / `steps` | 지형 선택 |

**`--controller adaptive`를 생략하면 기존 stage31 비교 viewer가 실행된다.**

<a id="viewer"></a>

## 4. 뷰어 조작과 확인 항목

### 4.1 Adaptive 조작

| 키 | 동작 |
|---|---|
| ↑ / ↓, ← / → | 전진 속도 / yaw 증감; 키를 놓아도 계속 보행 |
| Space / Enter | 속도 0 / 일시정지·재개 |
| H / C | 환경 초기화 / 지도 초기화 |
| M / G / B | 지도 / LiDAR FOV / 거절 후보 표시 |
| P | trace·map·후보 진단·모델·contract 저장 |

| 표시 | 의미 |
|---|---|
| 반투명 청록 | LiDAR height map |
| 회색 / 노랑 | unknown / support 또는 path coverage 부족 |
| 주황 / 파랑 / 보라 | rough·edge / IK / path 거절 |
| 초록 / 흰색 | SAFE 후보 / safe reference |
| 빨강 | selected / 실제 swing에 latch된 목표; label로 구분 |

FOV는 기본 표시하며 주황 하단 -7°, 파랑 상단 +52° 경계를 그린다.
`--fov-display-radius 1.2`는 안내선 반경이며 센서 측정 범위를 바꾸지 않는다.

### 4.2 사용자가 확인할 순서

| 확인 상황 | 볼 내용 |
|---|---|
| flat oracle | 연속 Tripod, normal stride, 안정적인 목표 |
| 계단 접근 | edge·거리 감지, 잔발, terrain Z·apex·pitch 변화 |
| Tripod 불가 | 작은 보폭도 실패한 뒤 all-contact 경계에서 Wave 전환 |
| 평탄 지형 복귀 | 두 phase feasible + hysteresis 후 Tripod 복귀 |
| 후보 없음 | UNKNOWN/UNSAFE 사유와 HOLD |
| oracle에서도 실패 | planner·workspace·IK·path·support·scheduler 확인 |
| oracle만 성공 | LiDAR TF·sampling·coverage·age·filter 확인 |

콘솔에 다리별 `known / coverage_pass / surface_safe / ik_safe / path_safe`,
reference·selected·active index, stride bank feasibility와 gait 결정이 출력된다.
초록 후보가 있어도 support 조합이나 contact 조건이 실패하면 phase가 시작되지 않을 수 있다.

P 저장 위치는 `mjx/generated/adaptive_gait/`다.
`foothold_diagnostics.json`, `foothold_plan.npz`, `trace.npz`, `map.npz`로 원인을 비교한다.
GT 오차는 관측된 샘플만 비교하며 비교 개수가 0인 지표를 성공으로 해석하지 않는다.

<a id="environment"></a>

## 5. 모델·센서·시뮬레이션 환경

두 viewer는 서로 다른 지도·지형 설정을 사용한다.

| 항목 | 새 adaptive | 기존 stage31 비교 viewer |
|---|---|---|
| 로봇 | 학습 link skeleton, distal 230 mm, foot sphere 반경 32 mm | 학습 box/capsule/sphere skeleton |
| LiDAR TF 기준 | 높이 215 mm, 전방 13.529 mm, 위를 보는 45° | 동일 measured TF 기준 |
| MID-360 FOV / range | H360°, V[-7°, +52°], 0.1–8 m | 동일 |
| angular ray | 90×8, 10 Hz, dropout 5%, noise 5 mm | 720×64 proxy |
| 지도 | 64×64 cell, 5 cm 해상도, 60초 유지 | 8×8 m, 4 cm 해상도, 60초 유지 |
| steps | 5 cm 계단 7단, tread 25 cm | 12×12 m 코스의 4 cm 계단 6단 등 |
| state estimation | simulator pose·velocity·contact | simulator body state |
| policy / controller 주기 | 20 ms / 5 ms | 기록된 v3 제어 계약 |

measured TF는 CAD URDF chain과 구분하며 Livox 비반복 스캔 패턴의 정확한 재현은 아니다.
루트 MJX의 물리 모델은 10 kg 질량 정규화, DS51150-270 12.6 V servo 기준이다.
토크 14.709975 Nm, 관절 속도 315.8 deg/s, `kp=500`, `kv=10`, armature 0.02,
damping 0.15, friction loss 0.8은 실측 식별 전의 모델 prior다.

<a id="training"></a>

## 6. 학습 단계와 checkpoint

### 6.1 학습 순서

| 단계 | 구성 | 목적 |
|---|---|---|
| Stage 0 | RL 없음, oracle vs LiDAR | planner·map·후보 검증 |
| Stage 1 | `--stage 1`, Tripod only | flat → ramp → 작은 계단 |
| Stage 2 | `--stage 2`, Wave only | 한 발씩 contact·지형 적응 학습 |
| Stage 3 | `--stage 3`, deterministic hybrid | 같은 policy로 잔발·Wave fallback 통합 |
| 이후 | Isaac Lab sim-to-sim | 동일 action·frame·contact 계약 이전 |

Stage 0에서 planner가 실패하면 PPO를 먼저 시작하지 않는다.
학습·보행 검증은 사용자가 진행하며 새 가중치는 아직 학습하지 않았다.

```bash
# Stage 0 확인 후 Stage 1 시작
bash scripts/train_adaptive_gait.sh --stage 1 --perception lidar --terrain-level 0 \
  --num-envs 64 --timesteps 10000000 --output mjx/runs/adaptive-lidar-flat

# 생성된 새 adaptive checkpoint 재생
bash scripts/view_foothold_planner.sh --controller adaptive --terrain flat \
  --checkpoint mjx/runs/adaptive-lidar-flat
```

단계별 이관 명령과 reward 설명은 [학습 설계](docs/HEXAPOD_MJX_ADAPTIVE_GAIT_LEARNING_PLAN.md)와
[실행 가이드](docs/HEXAPOD_MJX_ADAPTIVE_GAIT_USAGE.md#사용자-확인-후-학습)에 있다.
`--restore`는 가중치·normalizer 초기화이며 optimizer는 재시작한다.
`--wandb`를 붙이면 로그인된 계정의 `hexapod-adaptive-gait` 프로젝트에 기록한다.

### 6.2 호환성

| 경로 | Action / observation | 상태 |
|---|---|---|
| 새 adaptive | 24-D / actor 4410-D, critic 4725-D | 새 학습 필요 |
| stage31 v3 재생 | 18-D / 146-D | 기존 checkpoint 사용 |
| 루트 MJX v4 residual | 18-D / 146-D | v3와 residual scale·소스 계약 다름 |
| 과거 `SW/mjx/` | 24-D / 113-D | 별도 whole-body residual 계약 |

차원이 같아도 checkpoint를 서로 공유하지 않는다.
새 adaptive는 이전 23-D adaptive 및 stage31 18-D checkpoint를 거절한다.
가중치와 함께 action/observation 계약·source SHA·센서·모델 설정을 기록한다.

<a id="legacy"></a>

## 7. 기존 모드와 레거시

### 7.1 Stage31 비교 viewer

기본 가중치는 `progress-v2-stage31-level6_20260828-111825_seed40`의
checkpoint `000001703936`이다. GT 높이로 학습된 정책에 LiDAR 입력을 연결한 비교 실험이며,
LiDAR 공백/오차에 적응하도록 새로 학습한 가중치는 아니다.

```bash
# stage31 + LiDAR 입력
bash scripts/view_foothold_planner.sh --terrain steps

# 같은 동역학에서 기본 제어기만 비교
bash scripts/view_foothold_planner.sh --terrain steps --residual-scale 0

# 저장된 설정과 GT 입력으로 기존 정책 재생
bash scripts/view_trained_policy.sh

# 이전 기구학 preview / CAD 비교
bash scripts/view_foothold_planner.sh --terrain steps --controller nominal --robot-model mesh
```

이 모드는 미관측 첫걸음에서 nominal로 걷고, 관측이 생기면 residual gain을 완만하게 적용한다.
LiDAR 후보는 표시용이며 실제 제어 목표를 뜻하지 않는다. 새 adaptive의 HOLD·landing latch와 구분한다.
P 저장 위치는 `mjx/generated/foothold_preview/`다.
조작키·gain·GT 비교·저장 파일은 [기존 viewer 사용법](docs/HEXAPOD_FOOTHOLD_PREVIEW_USAGE.md)에 있다.

v3 재생은 `0805164` 코드/URDF를 격리해 사용한다. 당시 미커밋 소스가 저장되지 않아
W&B 영상과의 정확한 재현은 미확인이다. [가중치 출처](mjx/policies/progress-v2-stage31-level6/README.md)를 참고한다.

### 7.2 V4·SW/mjx·Isaac Lab

| 구분 | 참고 문서 |
|---|---|
| 루트 v4/18-D, curriculum 0~16 | [MJX 학습 계약](mjx/RL_DESIGN.md), [기본 펌웨어](mjx/FIRMWARE_BASE.md) |
| `SW/mjx/`의 foot18 + body6 residual, 과거 학습·W&B 명령 | [레거시 Residual RL](docs/RESIDUAL_RL.md) |
| Isaac Lab v4 Torch controller·USD·센서 scaffold | [실행 안내](isaaclab_hexapod/README.md), [이식 기록](isaaclab_hexapod/PORT_RESULT_AND_USAGE.md) |

과거 `SW/mjx/`의 24-D는 새 adaptive 24-D와 의미가 다르다.
이전 학습 명령·설치 절차는 해당 문서를 따르며 현재 adaptive 실행 명령과 섞지 않는다.

<a id="docs"></a>

## 8. 문서 안내와 남은 작업

### 설계·실행·업데이트

| 문서 | 찾을 내용 |
|---|---|
| [전체 문서 목차](docs/README.md) | 문서별 적용 모드와 상태 |
| [Hybrid 구현 전 분석](docs/HEXAPOD_HYBRID_GAIT_ANALYSIS.md) | 기존 데이터 흐름·gate·main Wave 비교·변경 계획 |
| [Adaptive 사용 가이드](docs/HEXAPOD_MJX_ADAPTIVE_GAIT_USAGE.md) | 상세 환경·threshold·action/observation·진단·학습 명령 |
| [Adaptive 학습 설계](docs/HEXAPOD_MJX_ADAPTIVE_GAIT_LEARNING_PLAN.md) | 역할 분리·Stage 0–3·sim-to-sim 순서 |
| [기존 LiDAR/residual 계획](docs/HEXAPOD_LIDAR_FOOTHOLD_RESIDUAL_PLAN.md) | stage31 통합 실험과 후속 학습 배경 |
| [2026-09-06 업데이트 기록](docs/HEXAPOD_UPDATE_2026-09-06.md) | 당시 viewer·소스·문서·모델 변경 기록 |

### 하드웨어·펌웨어

| 문서 | 내용 |
|---|---|
| [제어기 Architecture](SW/Controller/Controller_Architecture.md) / [상세 설계](SW/Controller/Controller_detail.md) | 제어 계층과 설계 |
| [좌표계·관절 정의](SW/Controller/좌표축/README.md) | frame과 joint 기준 |
| [조종기 입력](SW/Controller/드론%20조종기%20입력/README.md) | 사용자 명령 입력 |
| [STM32 설정](SW/STM32/STM32F446RE%20설정%20정리본.md) | 보드·개발환경 |
| [부품 목록](HW/parts.md) | 하드웨어 구성 |

### 현재 제한과 다음 단계

현재 safety는 sampled foot path와 quasi-static support 검사다.
전신 swept collision, 동적 안정성, 새 장애물에 대한 mid-swing abort는 아직 포함하지 않는다.
실제 LIO·센서 사각지대 대응·대규모 JIT/batch 성능·계단 등판은 확인이 남아 있다.

사용자 보행 검증 → Stage 1–3 학습 → Isaac Lab 이전 순서로 진행한다.
D435 RGB semantic/traversability를 LiDAR map 위에 projection하는 기능과 learned gait selector는 이후 확장이다.
추가된 [hybrid 단위 테스트](mjx/tests/test_adaptive_hybrid.py)는 사용자가 필요할 때 실행하며,
정적 확인만으로 최종 보행 성공 조건을 달성했다고 간주하지 않는다.
