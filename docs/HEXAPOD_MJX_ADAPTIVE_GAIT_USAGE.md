> 통합 브랜치의 실행/SPI 계약과 검증 범위는 [v4 통합 문서](ADAPTIVE_INTEGRATION_V4.md)가 기준입니다.

# MJX hybrid residual 보행 사용 가이드

업데이트: 2026-09-06. 브랜치 `codex/cartesian-residual-rl`.
구현과 정적 확인까지 진행했다. GUI 보행, JIT rollout, PPO, 아래 테스트의 실행 검증은 사용자에게 맡겼다.
새 24-D 학습 가중치는 없다. 계단 등판이나 안전한 gait 전환이 검증됐다는 뜻은 아니다.

## 경로와 실행

```bash
cd /home/huro/Hexapod-Robot
source /home/huro/.venvs/hexapod-mjx/bin/activate

# Stage 0: RL 없이 GT 100% known으로 planner/controller 분리 확인
bash scripts/view_foothold_planner.sh --controller adaptive --terrain flat --perception oracle
bash scripts/view_foothold_planner.sh --controller adaptive --terrain steps --perception oracle

# 동일 코드에서 LiDAR 입력으로 실행. 같은 자세의 GT 비교 지표도 출력
bash scripts/view_foothold_planner.sh --controller adaptive --terrain steps --perception lidar --stage0
```

`--controller adaptive`가 새 구조를 선택한다. 생략하면 기존 stage31 비교 viewer다.
스크립트가 가상환경 Python을 직접 선택하며 `HEXAPOD_PYTHON`으로 바꿀 수 있다.
MuJoCo, MJX, JAX, mujoco-playground, Brax/Orbax가 있는 기존 `hexapod-mjx` 환경을 사용한다.
새 의존성을 설치하지 않았다. 첫 화면까지 JAX 컴파일 시간이 필요하며 새 검색의 성능은 아직 측정하지 않았다.

| 키 | 동작 |
|---|---|
| ↑ / ↓ | 전진 속도 증가 / 감소; 입력 후 계속 보행 |
| ← / → | yaw 명령 증가 / 감소 |
| Space | 속도·yaw를 0으로; 진행 중 swing은 접촉까지 처리 |
| Enter / H | 일시정지 / 환경 초기화 |
| C / M | 지도 초기화 / 지도 표시 |
| G / B | LiDAR FOV / 거절 후보 표시 |
| P | 최근 trace, map, 후보 진단, 모델, contract 저장 |

저장 위치: `mjx/generated/adaptive_gait/`. `foothold_diagnostics.json`에는 후보별 gate,
선택 index, 실제 latch index, 궤적 parameter가 들어간다. trace는 최근 500 policy tick이다.
`--gait-mode tripod`, `--gait-mode wave`, `--gait-mode hybrid`로 고정/혼합 비교한다. 기본은 hybrid다.

## 환경 구성과 데이터 흐름

학습에 쓰던 링크 collision skeleton을 사용한다. distal link 230 mm, foot sphere 반경 32 mm이며
IK endpoint와 collision/site는 구의 중심으로 맞췄다. 메시 기반 CAD 표시와 구분한다.
LiDAR는 기존 측정 TF를 유지한다: 기준 높이 215 mm, 전방 offset 13.529 mm, 위를 보는 45° mount.
FOV는 MID-360 H360°, V[-7°, +52°], range 0.1–8 m. G로 표시하는 선은 시야각 경계다.
policy 20 ms, firmware tick 5 ms, LiDAR 100 ms, 90×8 ray, dropout 5%, noise 5 mm,
지도 cell 5 cm다. 시뮬레이터 pose/velocity/contact를 이상적인 state estimator로 사용하며 실제 LIO는 연결하지 않았다.

```text
LiDAR ray → rolling height map → support plane / edge / confidence
command → nominal foothold → 25-candidate search → safe reference
safe reference + RL XY → 같은 safe set으로 projection → terrain Z
path maximum height + margin + RL clearance → sampled path / IK gate
Tripod normal → short-step Tripod → known-infeasible이면 Wave → HOLD
phase boundary + all contacts → 목표/높이/궤적 parameter latch → 단일 controller / IK
```

`lidar`: actor와 planner에는 LiDAR만 사용한다. GT는 critic, reward/종료 판정, 비교 metric에만 사용한다.
`oracle`: GT height와 100% known/age 0으로 같은 planner를 실행한다. debug 전용이며 checkpoint/PPO에 쓰지 않는다.
`teacher`: GT 높이를 쓰되 LiDAR known mask는 유지한다. oracle과 다르다.
`blind`: 관측 안전 gate를 우회하는 명시적 nominal controller 진단이다. 지형 적응·hybrid 안전 fallback이 아니다.

최신 설계에 맞춰 **UNKNOWN은 기본 hybrid에서 HOLD**한다. 첫걸음 주변이 관측되지 않으면
자동으로 미관측 지형을 밟지 않는다. 초기 사각지대가 지속되면 그대로 정지할 수 있다.
이전 요청의 무관측 첫걸음 자체만 비교하려면 아래 debug 명령을 쓴다.

```bash
bash scripts/view_foothold_planner.sh --controller adaptive --terrain flat --perception blind
```

## 후보·workspace·안전 조건

각 다리의 nominal 주변 controller XY `[-8,-4,0,4,8] cm`의 25개 후보를 검사한다.
search ±8 cm는 허용 residual ±4 cm와 별개다. IK 링크 길이는 74/121/230 mm,
기본 발은 leg root에서 수평 218.728 mm, Z -287.006 mm다. ±8 cm 전부가 도달 가능하다는 뜻은 아니다.
각 후보의 endpoint와 21개 궤적 sample에서 실제 link IK, reach 여유 1 mm, joint 여유 1°를 확인한다.

support는 중심 known 필수 + coverage≥0.6 + 비공선 plane fit이 가능해야 한다.
중심과 일직선 이웃 2개만 있는 3/5 관측은 거절한다. 관측된 cell만으로 plane을 맞추고,
RMS 8 mm, slope 25°, 이웃 단차/cell spread 25 mm, foot-radius+5 mm edge 여유를 검사한다.
주변 5×5 cell의 관측된 단차로 edge distance를 계산하며, 미관측 edge가 없다고 보증하지 않는다.
normal/slope, coverage/confidence/age, 관측 support 폭·길이, path max/known fraction,
IK/joint margin을 고정 shape로 계산한다. support 폭·길이는 world grid 축상의 연속 관측 길이다.

경로는 21개 sample과 구형 발의 5점 footprint로 관측 장애물을 검사한다.
path known fraction≥0.6이어야 SAFE다. full mesh/다리 링크/몸통 swept collision 검사는 아니다.
RL XY는 reference 주변 ±4 cm 이내의 안전 후보로 projection하므로 결과는 4 cm grid에 양자화된다.
요청 clearance/timing이 실패하면 neutral reference 궤적으로 복구하며, 그 reference도 없으면 HOLD다.

Tripod는 다리별 top-3, 조합 27개에 대해 현재 stance와 다음 landing support polygon의
CoM 여유 12 mm를 검사한다. Wave는 현재 한 발 후보와 5발 stance를 검사한다.
각 phase의 stance IK를 5개 미래 body pose에서도 검사한다. 정적 CoM 근사이며 동적 안정성 증명은 아니다.
`Lmax_scale`은 `[1.3,1,.75,.5,.25,.125]` 중 통과한 최대 scale이다. 연속 최댓값이 아니다.
이동 성분의 최대 phase 이동 거리는 `abs(v_command)*1.0*Lmax_scale` m이다.

## action / observation 계약

`adaptive_hybrid_geometry_residual_24_v4`. 다리 순서 RF RM RB LF LM LB.

| index | semantic |
|---|---|
| 0:12 | 6다리 safe reference 기준 XY residual, X ±6 cm / Y ±4 cm; 별도 local25 후보, Z action 없음 |
| 12:18 | path-required clearance 위의 여유 residual ±4 cm |
| 18 / 19 / 20 | body roll ±5° / pitch ±10° / height ±3 cm |
| 21 | 요청 stride scale 0.5–1.3; supervisor는 필요시 0.125까지 축소 |
| 22 | apex plateau 중심 phase residual ±0.15 |
| 23 | XY transfer 중심 phase residual ±0.15 |

Z는 terrain surface + foot radius로 IK에 전달한다. 높이 기준은
`required=max(path_max-max(start_surface_z,landing_surface_z),0)`;
`clearance=clip(max(required+0.06+RL, required+0.02, 0.04),0.04,0.18)`이다.
18 cm 제한으로 required+margin을 못 만족하면 거절한다.
body normal 기반 roll/pitch baseline(각 ±12°) 위에 residual을 더하고 기존 자세/IK gate가 제한한다.
action=0도 planner baseline을 사용한다. 비활성 leg action은 현재 swing에 적용하지 않는다.

기존 quintic lift/transfer/lower를 유지했다. apex는 단일 Bézier 점이 아니라
Z plateau 중심이며 최종 범위 0.3–0.7, plateau 폭 0.4다.
상승 시 앞당기고 하강 시 늦추는 geometry baseline을 사용한다.
transfer 중심은 0.35–0.65, 이동 구간 폭 0.5다. 해당 변화도 같은 path/IK 검사를 통과해야 한다.

phase duration action은 제거했다. Tripod `clip(1.0*scale,0.5,1.4)`초,
Wave `clip(1.0*scale,0.6,1.4)`초, speed scale 0.2, stance 5 phase다.
속도에도 `scale*baseline_period/period`를 적용한다. Adaptive 상한은 전후0.10 m/s, yaw18°/s로 main과 맞춘다.

actor **4434-D** = proprio 157 + global 23 + reference 54 + 6×25×28 candidate.
critic **4749-D** = actor + GT/error 300 + terrain 15.
proprio는 기존 155 필드를 유지하고 accepted/previous action 증가로 157이 됐다.
global에는 vy command(현재 0), roll/pitch, contact 기반 body clearance, slip 6,
current gait, active swing mask 6, supervisor decision, maximum stride, Tripod/Wave feasibility,
mean confidence, support margin이 들어간다. reference는 다리별 body XYZ·nominal까지 거리·valid 및 local residual dx/dy extrema4개다.
candidate 28개 필드의 순서/정규화는 `adaptive_foothold_estimator.CANDIDATE_FEATURES`와 feature 조립 코드가 기준이다.
normal/slope는 world 축, candidate offset은 controller 축이다.
새 shape/source SHA 계약은 이전 adaptive24-D v3, 23-D 및 stage31 18-D checkpoint 로드를 거절한다.
기존 18-D controller/checkpoint 경로는 유지한다.

## main Wave 포팅과 전환

참고 revision `origin/main`의 `9752c760`; 상세 파일은 [구현 전 분석](HEXAPOD_HYBRID_GAIT_ANALYSIS.md)에 있다.
순서는 **RF → LB → RM → LF → RB → LM**이다.
첫 6 Wave phase는 stance 속도 절반, phase 시작 전 all-contact 100 ms 대기,
airborne 확인 후 progress≥0.5의 Early Landing, raw contact 후보 동결,
Late Landing 0.12 m/s 아래·0.096 m/s 안쪽, 최대 10 cm 검색 후 fault HOLD를 포팅했다.
support contact를 잃으면 phase를 멈추고 빠진 support만 검색한다. Late 동안 stance 적분도 멈춘다.
fault는 명령 해제 후 재무장한다. 실제 접촉은 20 ms policy sample로 확인하므로 하드웨어 10 ms FSR와 시간 해상도가 다르다.

모드 전환은 swing 완료 후 다음 all-contact phase 경계에서만 적용한다.
Wave→Tripod는 다음 두 Tripod phase가 feasible인 상태를 0.5초 유지하고 Wave 체류 2초를 요구한다.
두 번째 preview는 첫 착지의 가상 contact/발 위치와 root 직선 이동을 사용하며 yaw 연속 예측은 근사다.
main의 Tripod 두 phase 명령 묶음 대신 안전 supervisor는 매 phase에서 재검토한다.
현재 swing의 world landing Z/XY와 clearance/timing은 latch된다. 새 map이 목표를 이동시키지 않는다.
새 장애물에 대한 mid-swing 재계획/abort는 아직 없고 contact recovery와 매 tick IK 제한을 사용한다.

## 화면에서 확인할 것

| 상황 | 기대 / 확인 |
|---|---|
| flat oracle | action 0, normal Tripod, 후보/목표 안정, 연속 swing |
| steps 접근 | edge/거리 → short stride; terrain Z / apex / pitch 변화 |
| Tripod 불가·Wave 가능 | known-bad + short 실패 로그 → all-contact 경계에서 한 발씩 |
| 평지 복귀 | 두 phase feasibility와 hysteresis → Tripod |
| 후보 없음 / map 지우기 | UNKNOWN 또는 unsafe 사유와 HOLD; 진행 중 swing은 기존 latch 유지 |
| oracle도 실패 | workspace / IK / path / support / scheduler 로그 확인 |
| oracle만 성공 | LiDAR density, TF, age, coverage, edge filter 확인 |

회색 UNKNOWN, 노랑 LOW_COVERAGE, 주황 EDGE/ROUGH, 파랑 IK, 보라 PATH, 초록 SAFE,
흰색 reference, 빨강 selected/latched. 청록 map은 alpha 0.14다.
콘솔은 다리별 candidate_total/known/coverage_pass/surface_safe/ik_safe/path_safe,
ref/selected/active index와 residual rejection을 출력한다.
초록이 있어도 support 조합·stance IK·contact·hysteresis가 실패하면 phase는 시작하지 않을 수 있다.

`--stage0`는 동일 pose·명령의 oracle 대비 safe recall, false-safe rate,
unknown fraction, reference XYZ error, edge precision/recall을 추가 계산한다.
reference error는 양쪽 reference가 있는 다리만 비교한다(`oracle_compared`).
map MAE도 관측 cell만 비교한다(`map_compared_count`). 분모 0 지표를 성능 성공으로 해석하지 않는다.
LiDAR 스캔 미실행인 oracle 화면의 map MAE는 n/a다. 이 지표들은 현재 sample 기반 oracle 기준이며 실제 접촉 성공률은 별도 확인해야 한다.

## 사용자 확인 후 학습

Stage 0에서 planner가 실패하면 PPO를 시작하지 않는다. 학습은 자동 실행하지 않았다.

```bash
# Stage 1: Tripod only / flat
bash scripts/train_adaptive_gait.sh --stage 1 --perception lidar --terrain-level 0 \
  --num-envs 64 --timesteps 10000000 --output mjx/runs/hybrid-s1-flat

# 같은 24-D checkpoint로 ramp → 5cm stairs 순차 이관 (각 실행은 별도 output)
bash scripts/train_adaptive_gait.sh --stage 1 --terrain-level 3 \
  --restore mjx/runs/hybrid-s1-flat --output mjx/runs/hybrid-s1-ramp
bash scripts/train_adaptive_gait.sh --stage 1 --terrain-level 5 \
  --restore mjx/runs/hybrid-s1-ramp --output mjx/runs/hybrid-s1-steps

# Stage 2: Wave only, 먼저 flat에서 contact/sequence 학습
bash scripts/train_adaptive_gait.sh --stage 2 --terrain-level 0 \
  --restore mjx/runs/hybrid-s1-steps --output mjx/runs/hybrid-s2-flat
# 이후 Stage 2도 level 3 → 5 순차 학습. 완료한 run으로 Stage 3 시작
bash scripts/train_adaptive_gait.sh --stage 3 --terrain-level 5 \
  --restore mjx/runs/hybrid-s2-steps --output mjx/runs/hybrid-s3-steps

bash scripts/view_foothold_planner.sh --controller adaptive --terrain steps \
  --checkpoint mjx/runs/hybrid-s3-steps
```

`--restore`는 같은 source 계약의 가중치/normalizer 초기화이고 optimizer 재시작이다.
terrain/gait stage는 변경 가능하다. teacher는 `--perception teacher`, LiDAR 이관은
`--init-teacher RUN`; 이는 weight transfer 후 PPO이며 별도 imitation trainer는 아니다.
level 0 flat, 3/4 ramp 8°/15°, 5부터 stairs, tread 25 cm. hfield rough level 1/2는 ray 지원 때문에 거절한다.
`--wandb`로 기존 계정의 새 run을 기록한다. 대규모 env 수/메모리/throughput은 측정 전이다.

reward는 기존 진행/속도/자세/높이, residual/action rate, 충돌/IK/관절 제한 항목에
landing error·slip·projection·residual rejection을 더한다. 속도 목표는 수락한 gait 속도에 맞춘다.
Wave cost 0.005/s, mode switch cost 0.01로 안전 penalty보다 작게 둔다.
불필요한 apex는 기존 swing-height cost, body tilt는 자세 목표 오차와 residual cost로 억제한다.
별도 learned gait selector는 구현하지 않았다.

추가한 단위 테스트는 사용자가 필요할 때 실행한다. 에이전트가 실행하지 않았다.

```bash
python -m unittest discover -s mjx/tests -p test_adaptive_hybrid.py
# legacy 회귀가 필요할 때
python -m unittest discover -s mjx/tests -p test_firmware_mjx_controller.py
```

현재 확인 범위는 문법/CLI/정적 계약뿐이다. 전체 형상 collision, 센서 사각지대 대응,
동적 support 안정성, JIT/batch 메모리, 실제 등판과 sim-to-sim은 사용자 검증 후 이어갈 항목이다.
