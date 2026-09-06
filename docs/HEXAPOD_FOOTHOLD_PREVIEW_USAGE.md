# 학습 정책을 사용하는 MuJoCo 착지점 뷰어

기존 `view_foothold_planner.sh` 화면에 stage31 학습 정책을 연결했다.
사용자 요청에 따라 이번 통합본의 정책 로딩·추론·렌더링·동역학 보행은 실행 검증하지 않았다.

## 시작

```bash
cd /home/huro/Hexapod-Robot
source /home/huro/.venvs/hexapod-mjx/bin/activate
bash scripts/view_foothold_planner.sh --terrain steps
```

기존 창을 닫고 새로 실행한다. 기본값은 `--controller trained`다.
`progress-v2-stage31-level6_20260828-111825_seed40`의 checkpoint `000001703936`을 사용하며,
화면에 **PPO stage31 / DYNAMICS**가 표시된다. 키 입력 전에는 이동 command가 0이다.
가상환경은 `/home/huro/.venvs/hexapod-mjx`이고, 스크립트가 직접 사용하므로 activate는 선택 사항이다.
MuJoCo만 설치한 뷰어 환경이 아니라 기존 JAX/MJX/Brax 학습 환경이 필요하다.

처음에는 JAX 컴파일이 필요하다. 준비 상태는 터미널과 HUD에 표시한다.
정책 계산은 별도 프로세스에서 실행해 창 조작과 LiDAR 표시가 GPU 계산을 기다리며 멈추지 않도록 했다.
실시간보다 느릴 수 있으며, HUD의 simulation 시간은 실제 진행한 물리 시간이다.

## 같은 화면에서 무엇이 동작하는가

- **로봇 움직임:** 146-D 관측과 저장된 정규화 → stage31 PPO 18-D action → 펌웨어 기반 gait/IK → MJX 동역학.
- **LiDAR:** 움직인 로봇의 자세에서 MID-360 angular raycast → odom 높이 지도 누적.
- **착지 후보:** LiDAR 지도에서 기존 geometric 후보·nominal fallback·IK 도달 가능성을 계산해 표시.
- **흰 점:** 학습 제어기의 현재 발 목표. 최종 착지점 예측이 아니라 해당 제어 시점의 목표다.
- **큰 색상 점/경로:** LiDAR 후보. 이 모드에서는 후보가 정책을 직접 제어하지 않는다.

정책의 지형 관측은 기존 학습 계약을 유지하기 위해 시뮬레이터 지형에서 제공한다.
따라서 아직 LiDAR만으로 보행하는 정책은 아니며, LiDAR 착지점 보정을 PPO 앞단에 넣은 것도 아니다.
지도 표시와 학습 보행을 같은 공간에서 비교하는 통합 실행 모드다.

로봇은 학습에 쓰인 박스 몸체·캡슐 링크·32 mm 구형 발 모델이다.
학습 가중치와 함께 기록된 `0805164` 코드/URDF를 별도로 추출하며, 작업 폴더의 v4 학습 코드는
불러오지 않는다. 당시 미커밋 코드 전체는 저장되지 않아 영상과 동일한 재현은 보장하지 않는다.

## 조작법

**방향키는 목표 속도를 증감한다. 키를 놓아도 계속 움직인다.** MuJoCo 창에 포커스를 둔다.

| 키 | trained 기본 모드 |
|---|---|
| ↑ / ↓, W / S | 전후진 command ±0.04 m/s; 최대 ±0.12 m/s |
| ← / →, Q / E | 회전 command ±0.15 rad/s; 최대 ±0.3 rad/s |
| Space | 이동 command 0; 물리·자세 제어는 계속 진행 |
| Enter | 동역학 일시정지 / 재개 |
| PageUp / PageDown | 몸체 높이 보정 ±2 cm; 초기 기준 -5~+10 cm |
| H | 정책·물리 상태와 지도 초기화 |
| R | 인식 계산 재시도; 동역학 종료 상태에서는 로봇 reset |
| 1~6 | RF/RM/RB/LF/LM/LB의 LiDAR 후보 상세 선택 |
| M | 높이 셀 표시 전환 |
| L | LiDAR 반환점 표시 전환 |
| G | 센서에 부착된 MID-360 FOV 경계선 전환 |
| K | scan on/off; 기존 관측은 60초 이후 만료 |
| C | 지도 초기화와 이동 command 0 |
| P | 지도·후보·정책 action과 목표를 저장 |
| F / T / V | 추적 카메라 / 위에서 보기 / 전체 코스 |

이 정책의 command에는 횡이동이 없어 A/D는 동작시키지 않는다.
원본 run은 전진 0.06~0.12 m/s, yaw=0에서 학습했으므로 후진·회전 성능은 별도로 확인해야 한다.
PageUp/PageDown은 절대 world 높이가 아니라 펌웨어가 계산하는 지지 높이에 대한 보정이다.
정책의 pitch feedforward와 보행 제어가 지형 높이를 반영한다.

동역학의 종료 조건이나 코스 경계에 도달하면 마지막 상태를 표시한다. H로 다시 시작한다.
Space는 순간적으로 자세를 고정하는 키가 아니다. 상태를 그대로 멈춰 관찰하려면 Enter를 사용한다.

## 코스·지도·센서

코스는 기존 배치의 12×12 m 공간이며 초기 전방은 world +X다.

| 구역 | 구성 |
|---|---|
| 정면 | x≈1.4~3.9, y≈0: 4 cm 계단 6단과 상단 landing |
| 좌측 | y≈2: 4/8/12 cm 플랫폼 |
| 좌측 뒤 | 징검다리 |
| 우측 | 8° 경사로 |
| 우측 먼 쪽 | 2.5/5/7.5/10 cm 돌출물 |
| 뒤쪽 | 요철 타일과 둥근 바위 |

trained 모드에서는 같은 장애물 배치를 **2 cm heightfield**로 변환한다.
넓은 코스를 하나의 접촉 표면으로 만들어 MJX 접촉 그래프 크기를 줄이는 구성이다.
원래 상자 모서리와 좁은 돌출물은 이 해상도로 근사한다. 화면·접촉·LiDAR는 동일한 heightfield를
사용하고, 정책 지형 관측은 해당 grid에서 보간한다. 이는 LiDAR 높이 지도와 별도인 시뮬레이터 지형이다.
별도 `view_trained_policy.sh`가 사용하는 6.5 cm 계단 7단과는 다른 탐색 코스다.

LiDAR TF는 몸체 밑면 중심 기준 높이 **215 mm**, 전방 **13.529 mm**, 위를 향해 전방으로 **45°**다.
`base_link` 위치는 `(0, -0.013529, 0.1642) m`, RPY는 `(0, 45°, -90°)`다.
MID-360 FOV는 수평 360°, 수직 -7°~+52°, raycast 거리는 0.1~8 m다.
기본 720×64 angular rays를 사용하며 Livox 비반복 패턴의 재현은 아니다.
센서 FOV 선은 측정 거리와 무관한 1.2 m 길이의 각도 경계이며 G로 숨긴다.

지도는 odom 축에 정렬된 8×8 m rolling grid, 4 cm 셀, 60초 관측 유지다.
몸체에 가린 지면은 관측으로 채우지 않는다. M/L/G/K 조작은 표시·인식 경로에만 적용하며,
이 모드의 정책 관측을 LiDAR 관측으로 전환하지 않는다.
지도 불투명도는 16%, 점군은 22%, 후보 마커는 불투명하게 표시한다.

| 후보 색상 | 의미 |
|---|---|
| 주황 | 미관측 nominal 후보 |
| 파랑 | 관측 기반 geometric 후보 |
| 청록 | 착지 patch는 관측, 경로 일부는 미관측 |
| 빨간 X | 후보 계획 실패; trained 보행의 정지 명령은 아님 |
| 흰 점 | 학습 제어기의 현재 발 목표 |

## 내가 확인할 것

1. HUD가 `PPO stage31 / DYNAMICS`인지, 컴파일 후 simulation 시간이 증가하는지 본다.
2. ↑를 2번 눌러 0.08 m/s 전진 command를 주고, 정면 계단까지 접근하는지 확인한다.
3. 실제 발과 흰 목표점, 색상 LiDAR 후보를 비교한다. 후보에 발이 반드시 착지하는 구성은 아니다.
4. Space로 감속·정지하고 Enter로 물리를 멈춰 지도와 발 위치를 살펴본다.
5. 계단에서 몸체 상승·접촉·넘어짐과 HUD 종료 사유를 확인한다. H로 초기화한다.
6. 문제가 보이는 상태는 P로 저장한다. 실행 파일은 변경했지만 실제 등반 성공은 아직 확인하지 않았다.

## 저장과 비교

`mjx/generated/foothold_preview/` 아래에 저장한다.

- `scene_manifest.json`: 정책 run·checkpoint·코드 버전·센서 TF·지형 해상도.
- `latest_plan.json`: 실제 qpos, 정책 action, 흰 목표 좌표, LiDAR 후보, 제어 모드.
- `latest_map.npz`: 지도 높이·관측 여부·시각·odom 중심·반환점.
- `policy/`: 격리된 v3 소스, 실제 GUI/물리 모델인 `explorer_policy.mjb`, JAX cache.

```bash
# 기존 LiDAR 후보 기반 기구학 제어
bash scripts/view_foothold_planner.sh --terrain steps --controller nominal

# nominal 모드에서 최신 CAD 외형 확인
bash scripts/view_foothold_planner.sh --terrain steps --controller nominal --robot-model mesh

# 학습 정책 모드에서 지도 표시를 더 옅게
bash scripts/view_foothold_planner.sh --terrain steps --map-alpha 0.10 --lidar-point-alpha 0.15
```

`--controller nominal`은 이전 continuous tripod + SiteIK 방식이다. 동역학과 PPO를 쓰지 않으며,
이 모드에서만 LiDAR 후보/nominal fallback이 swing 경로를 결정한다. Enter는 제자리 반복 swing,
A/D는 횡이동, `--gait-swing-duration`은 swing 시간이다. 미관측 nominal 시작을 금지하려면
`--require-observed`를 사용한다. 지지 다리와 swing 다리의 전체 경로를 미리 검사하고,
도달 범위가 부족하면 보폭을 줄이며 알려진 위험이나 IK 실패에서는 hold한다.

`--revision`은 nominal 모델의 URDF snapshot을 지정한다. 기본은 로컬 `origin/main`이며,
새 URDF를 올린 뒤에는 `git fetch origin main`을 먼저 실행한다.
trained 모드는 가중치의 로봇/제어 계약을 유지하므로 이 옵션을 사용하지 않는다.
