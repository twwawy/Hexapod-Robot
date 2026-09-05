# MuJoCo 지형 탐색·높이 지도·여섯 다리 착지점 안내

업데이트: 2026-09-06. 이번 연속 tripod swing/반투명 지도/skeleton 확장본은 사용자 요청에 따라
실행·렌더링·동작 테스트를 수행하지 않았다. 이전 고정 화면의 조작법은 이 문서로 대체한다.

## 시작 명령어

기존 창을 닫고 아래 명령으로 새 환경을 연다.

```bash
cd /home/huro/Hexapod-Robot
bash scripts/view_foothold_planner.sh --terrain steps
```

- 작업 폴더: `/home/huro/Hexapod-Robot`
- Python 가상환경: `/home/huro/.venvs/hexapod-mjx`
- 진입점: `mjx/view_foothold_planner.py`
- 탐색 화면 구현: `mjx/view_foothold_explorer.py`
- 스크립트가 가상환경 Python을 직접 사용하므로 activate는 선택 사항이다.
- 장애물 없는 비교 환경은 `--terrain flat`으로 실행한다.

모델은 로컬 `origin/main`의 Git 커밋에서 URDF와 메시를 추출한다. 기준 커밋을 고정하려면
`--revision d67abc1`을 붙인다. GitHub에 새 URDF를 올린 경우 먼저 `git fetch origin main`을
실행한다. 기존 로컬 URDF 변경사항을 덮어쓰지 않고 별도 asset snapshot을 만든다.

기본 `--robot-model skeleton`은 **예전 MJX 학습용 링크 모델**이다.
`mjx/prepare_rl_scene.py@3a817c4`의 `_add_robot_colliders`와 같은 형상을
`mjx/foothold_link_model.py`에서 재사용한다. 박스 몸체(반치수 0.17/0.15/0.045 m),
coxa/femur/tibia 캡슐(반지름 28/26/23 mm), tibia 끝까지 길이 230 mm,
반지름 32 mm인 구형 발로 구성한다. 관절 위치·축·범위는 지정한 URDF에서 가져온다.

URDF 변환 중 CAD 메시를 읽어 관절·관성을 확보한 뒤 최종 scene에서는 메시를 제거한다.
화면과 LiDAR의 몸체 가림 계산에 같은 primitive 형상을 사용한다. 발 목표와 지지 발
좌표는 구 중심에서 world Z로 32 mm 아래인 접지점이다. 따라서 구 중심을 바닥에 놓거나
다리 회전으로 가상의 sole offset이 돌아가는 문제가 없도록 구성했다.
LiDAR TF/FOV·지도·지형·착지 마커도 함께 사용한다.

CAD 메시 모델과 비교하려면 다음처럼 실행한다. `--robot-display`는 `--robot-model`의
호환 별칭이며, 이전 선·점 overlay 모드는 제거했다.

```bash
bash scripts/view_foothold_planner.sh --terrain steps --robot-model mesh
```

## 이동과 화면 조작

**방향키는 누르는 동안만 이동하는 방식이 아니라 목표 속도를 증감하는 방식이다.
키를 놓아도 움직이며 Space가 정지 키다.** MuJoCo 창에 키보드 포커스를 둔다.

| 키 | 동작 |
|---|---|
| ↑ / ↓ 또는 W / S | 전후진 속도를 누를 때마다 ±0.04 m/s 변경; 최대 ±0.16 m/s |
| ← / → 또는 Q / E | 좌우 회전 속도를 ±0.15 rad/s 변경; 최대 ±0.6 rad/s |
| A / D | 좌우 평행이동 속도 변경; 최대 ±0.12 m/s |
| **Space** | 몸체 이동 명령 0; 진행 중 swing은 착지까지 마치고 다음 swing을 시작하지 않음 |
| PageUp / PageDown | 몸체 높이 ±2 cm; 장애물 위에서 계획 높이 조절 |
| 1 / 2 / 3 | RF / RM / RB를 상세 표시할 다리로 선택 |
| 4 / 5 / 6 | LF / LM / LB 선택 |
| **Enter** | 제자리 연속 tripod swing on/off; 숫자 선택 불필요 |
| R | 계산 갱신 요청; IK/지형 변경으로 hold된 swing 재시도 |
| P | 다음 결과에서 여섯 다리 계획과 지도 snapshot 저장 |
| H | 출발 pose로 복귀하고 지도 초기화 |
| C | 현 위치를 유지하며 지도 초기화 |
| M | 높이 셀 표시 on/off |
| L | 실제 raycast 점군 표시 on/off |
| G | LiDAR에 부착된 MID-360 FOV 경계선 on/off (기본 켜짐) |
| K | LiDAR scan on/off; 이전 관측은 유지시간 이후 만료 |
| F | 로봇을 따라가는 카메라 on/off |
| T | 위에서 보기 / 사선 보기 |
| V | 전체 코스 보기 |
| 마우스 | MuJoCo 기본 회전·확대·이동 |

방향키에 속도를 주면 자동으로 반복 swing한다. 숫자 키는 상세 표시만 바꾸며 동작을 끊지 않는다.
**Enter는 제자리 연속 swing, K는 LiDAR dropout**이다.

## 환경 구성

`steps`는 약 12×12 m의 지형 탐색 공간이다. 출발점은 (0, 0)이며 초기 전방은 world +X다.

| 구역 | 위치와 구성 |
|---|---|
| 정면 | x≈1.4~3.9 m, y≈0: 단당 4 cm인 6단 계단과 24 cm 높이 상단 |
| 좌측 | y≈2 m: 높이 4/8/12 cm의 넓은 플랫폼 |
| 좌측 뒤 | x≈-2.8~-1.5 m, y≈1.4~2.2 m: 징검다리 |
| 우측 | x≈1.8 m, y≈-2 m: 8° 경사로 |
| 우측 먼 쪽 | y≈-3.7 m: 높이 2.5/5/7.5/10 cm의 좁은 돌출물 |
| 뒤쪽 | x≈-3.2~-2 m, y≈-1.7~-0.8 m: 높이가 다른 타일 |
| 뒤쪽 먼 쪽 | y≈-3.4 m: 둥근 바위 모양 장애물 |

방향키는 **RF/RB/LM → RM/LF/LB를 번갈아 swing하는 기구학 제어**를 구동한다.
각 tripod의 기본 swing 시간은 0.8초이며, 지지 중인 발은 odom 위치를 유지한다.
다음 swing 시작 시 nominal 주변의 지형 후보를 선택하고, 진행 중인 착지점은 고정한다.
nominal은 현재 발 위치에 offset을 계속 더하지 않고, 몸체의 기준 stance와 예측 이동량에서 만든다.
착지 목표를 다음 지지 구간의 절반만큼 앞에 두어, 지지 발이 몸체 뒤로만 밀리지 않게 한다.

각 tripod를 들기 전에 예측한 몸체 경로와 함께 **여섯 다리의 전체 41개 경로 샘플**을
IK로 검사한다. 지지 다리는 고정된 접지점을, swing 다리는 lift/transfer/lower 경로를 검사한다.
회전으로 인한 발 주변 이동까지 포함해 한 tripod당 최대 4 cm로 명령을 제한하고,
필요하면 1/2, 1/4, 1/8, 제자리 순서로 줄여 가능한 경로를 찾는다. 알려진 지형 위험을
무시해서 통과시키지는 않는다. 제자리 경로도 불가능하면 발을 들기 전에 원인을 표시한다.

속도와 높이는 계획을 통과한 값을 해당 swing 동안 유지한다. 방향키/PageUp/PageDown
변경은 다음 tripod에서 적용하고, Space는 현재 몸체 위치에서 즉시 정지한 뒤 착지를 마친다.
HUD의 입력 `vx/vy/wz`와 실제 `Applied`는 보폭·관절 속도 제한 때문에 다를 수 있다.
`completed swings`가 착지 때마다 증가하고 tripod가 번갈아 바뀌는지 확인한다.

발 IK와 관절 속도 제한을 통과한 경우에만 몸체와 여섯 다리 목표를 함께 반영한다.
실행 가능한 swing 계획이 없으면 몸체를 계속 끌고 가지 않고 기다린다. 진행 중 IK 또는
지형 검사에 실패하면 현재 pose를 유지하고 HUD에 원인을 표시한다. R로 재시도하거나 H로 초기화한다.
Space로 정상 정지를 요청하면 몸체 이동은 멈추고 현재 swing은 착지까지 마친다.
다만 지형/IK 오류로 이미 hold된 경우에는 강제로 착지시키지 않는다.

이것은 실제 STM32 firmware gait 또는 동역학 시뮬레이션은 아니다. `mj_step`은 호출하지 않으며,
힘·실제 접촉·미끄러짐·몸체 충돌·다중 다리 안정성 검증은 포함하지 않는다.
계단에서 몸체 높이는 자동으로 올라가지 않는다. PageUp/PageDown으로 목표 높이를 바꾸면
지지 발의 IK가 가능한 범위에서 서서히 조정한다.

실제 LIO/IMU fusion, 접촉 적응, 다중 다리 지지 안정성, 동역학 보행, RGB 점수와 RL은
아직 연결되지 않았다. 움직이면서 센서 시야·지도 누적·다리 도달 가능성을 살펴보는 단계다.

## height map 변경점

| 항목 | 새 구성 |
|---|---|
| 범위 | 로봇을 따라가는 8×8 m local ROI |
| 좌표계 | world/odom 축 정렬, 이동 시 정수 셀만큼 grid를 옮겨 겹치는 관측 보존 |
| 해상도 | 기본 4 cm |
| 저장 채널 | 관측 높이, 셀 내 수직 범위, 마지막 관측 시각 |
| 보존 시간 | 기본 60초; static terrain 탐색용 값 |
| LiDAR 거리 | 0.1~8 m |
| LiDAR 축 | 사용자 수정값 유지: +Z 위쪽, 수직에서 전방으로 45° |
| FOV | 수평 360°, 수직 -7°~52° 유지 |
| sampling | 720×64 rays; 수직 ray의 약 75%를 -7°~8°에 배치 |
| 계산 | 하나의 background worker에서 scan과 여섯 다리 계획을 계산 |
| 주기 | 최대 3 Hz 요청, 실제 완료 속도는 raycast/IK 비용에 따라 달라짐 |
| 표시 | 높이 색상 셀 불투명도 16%, LiDAR 점 22%; 착지 목표는 불투명 표시 |

지면 방향 ray를 늘리기 위해 기존 FOV 안에서 하단 각도의 sampling을 조밀하게 했다.
이는 관측 가능한 지면의 점밀도를 위한 시뮬레이션 proxy이며 실제 Livox 패턴 재현은 아니다.
sensor TF를 아래로 뒤집거나 가상의 지면 높이를 주입하지 않는다.
몸체에 맞은 ray는 지면까지 통과시키지 않는다. 기본 링크 모델에서는 primitive가 ray를 가린다.
CAD 비교 모드에서는 센서 housing/FOV helper를 self-ray 대상에서 제외한다.

높이 지도는 완전한 terrain 정답이 아니다. 위를 향한 센서에서는 가까운 발밑이 보이지 않을 수
있으며, 처음에 주황색 nominal만 보이는 상황이 가능하다. 멀리 보인 셀은 전진 후에도 odom
위치에 유지되어 발 주변의 계획에 사용된다. local ROI 밖으로 나간 셀은 버리고, 관측
유지시간을 넘긴 셀도 invalid로 취급한다. global 영구 map이나 pose covariance/deskew는 미구현이다.

화면 부담을 줄이기 위해 먼 지도 셀은 일부만 그린다. 실제 map 데이터는 모두 보존한다.
기본 표시 예산은 2,500 cells이고, 발 목표·궤적을 먼저 그려 지도 때문에 가려지거나
시각화 용량을 다 차지하는 문제를 줄였다.

## 착지점 표시와 모드

여섯 다리의 계획을 매 계산 주기마다 갱신한다. 정지 상태에서는 nominal을 현재 기준 발 위치에서
전방 8 cm에 놓고, 이동 중에는 입력한 병진/회전 속도의 0.6초 예측 offset을 사용한다.
계산 결과는 **계산을 시작한 pose 기준 odom 좌표**에 표시하며, 기다리는 동안 로봇이 움직여도
마커를 임의로 로봇 쪽으로 끌어오지 않는다. HUD의 snapshot age로 지연을 확인한다.

| 표시 | 의미 |
|---|---|
| 노란 작은 점 | nominal 착지 목표 |
| 주황색 큰 점·경로 | 미관측 영역의 nominal fallback |
| 파란 큰 점 | 관측된 geometric 착지 목표 |
| 청록 큰 점 | 착지 patch는 관측됨, swing 통로 일부는 미관측 |
| 보라색 경로 | geometric 목표를 향한 궤적 |
| 작은 파란 점 | geometric 후보가 있지만 실제 적용 목표/궤적과 다른 경우 별도 표시 |
| 빨간 X | 현재 실행 가능한 목표 없음 |
| 선택한 다리의 초록/빨강 작은 점 | 후보 patch 통과/거부 |
| 하얀 점 (L) | 실제 LiDAR raycast 반환점 |

큰 마커는 착지점 위 **11 cm**에 세워서 표시한다. 실제 착지점은 세로선의 아래 끝이며,
높이 11 cm가 제어 목표에 더해지는 것은 아니다. 다리별 상세 mode/reason은 HUD에도 나온다.

- `geometric / ready`: 관측된 patch, 관측된 통로와 샘플 IK를 통과.
- `geometric_partial / endpoint_observed_path_partial`: 착지 patch는 관측됐지만 swing 통로
  일부가 unknown. 관측된 장애물로 clearance를 계산하고 IK를 검사하지만 전체 통로가 안전하다는 뜻은 아니다.
- `nominal / nominal_unknown_terrain`: 유효한 관측 후보가 없어 nominal 사용.
- `hold_known_hazard`: nominal 경로에서 이미 관측한 충돌/높이 불일치.
- `hold_nominal_ik`, `endpoint_ik`, `path_ik`: 다리의 도달 범위 또는 IK 문제.
- `candidate_budget_exhausted`: 제한된 후보 검사 수 안에서 실행 경로를 찾지 못함.

후보 검색은 nominal 반경 14 cm이고, 발 지지 원반과 교차하는 grid 셀을 검사한다.
기존의 과도한 사각 patch 대신 발 반경 3.2 cm를 반영한다. patch 자체는 여전히 관측되어야
geometric 후보로 통과한다. patch 경사·높이 차·수직 범위·착지 높이 변화와 수치 IK를 검사한다.

기본은 사용자가 요청한 미관측 nominal 시작을 허용한다. 비교를 위해 unknown 통로와
nominal fallback을 모두 차단하려면 다음처럼 실행한다.

```bash
bash scripts/view_foothold_planner.sh --terrain steps --require-observed
```

진행 중 tripod 다리에는 실제 고정한 착지 목표와 경로를 표시한다. 나머지 다리는 background
계산에서 갱신한 다음 후보를 표시한다. 지도 갱신이 진행 중인 목표를 바꾸지는 않는다.
새로 관측된 충돌이나 필요한 endpoint 관측 만료가 있으면 hold한다.
번호 선택 없이 연속 swing하며, 번호는 후보 상세 표시 대상을 바꾸는 데만 사용한다.

## LiDAR TF

몸체 밑면 중심 기준으로 전방 13.529 mm, 위로 215 mm이다.
현재 base_link STL 밑면은 base_link 원점의 z=-50.8 mm에 있다.

```text
robot_bottom_center: +X 전방, +Y 좌측, +Z 상방
bottom → lidar: xyz = (0.013529, 0, 0.215) m
                rpy = (0°, +45°, 0°)

CAD base_link: 전방=-Y, 좌측=+X
base_link → lidar: xyz = (0, -0.013529, 0.1642) m
                   rpy = (0°, +45°, -90°)
```

센서 +Z는 위를 향하고 수직에서 전방으로 45° 기울어진다. 상수는
`mjx/lidar_extrinsics.py`에 있다. CAD 시각 메시를 이동시키지 않고 센서 측정 frame을
적용한다. CAD 연결 기준과 비교할 때만 `--lidar-tf-source urdf`를 사용한다.

## 사용자가 확인할 순서

1. `steps`로 실행하고 **V**로 장애물 배치를 본다. **F**로 추적 시점으로 돌아온다.
2. **L**을 켜서 실제 LiDAR 반환점이 어디에 생기는지 본다. HUD의 `Ground hits`와
   `body hits`를 확인한다. 지면 점이 있지만 발밑 map이 비어 있다면 시야 사각/관측 이력 문제일 수 있다.
3. **↑를 한 번** 눌러 0.04 m/s로 정면 계단에 접근한다. 멀리 관측된 셀이 가까워져도
   같은 지형 위치에 남는지 본다. 멈출 때는 **Space**.
4. **T**로 위에서 보며 주황색 nominal과 파란색/청록색 목표를 비교한다.
   1~6으로 각 다리의 거부 사유와 후보 분포를 본다.
5. 장애물 위로 기준 pose를 옮길 때 **PageUp**으로 몸체 높이를 조정한다.
   이 환경은 자동으로 지형 높이를 읽어 몸을 들지 않는다.
6. Enter로 제자리에서 `completed swings`가 1, 2, 3… 증가하고 두 tripod가 번갈아 드는지 본다.
   이어서 ↑를 한 번 눌러 이동하며 반복되는지, 지지 발이 지면 좌표를 유지하는지 본다.
   `Applied`가 0이면 보폭을 제자리로 줄였거나 hold 중인지 HUD 상태를 확인한다.
   **Space**로 현재 swing을 마치며 멈추는지, **Enter**로 제자리에서 반복 swing하는지 확인한다.
7. map이 이상하면 **P**로 저장한다. **C**로 관측 이력을 지우면 누적 오차인지 현재 raycast
   위치 문제인지 구분할 수 있다. **K**는 scan을 끄므로 map은 유지시간 뒤 만료된다.

## 결과 파일과 설정

`mjx/generated/foothold_preview/` 아래에 저장한다.

- `scene_manifest.json`: 모델 커밋, TF, 장애물 배치와 환경의 대체 입력 설명.
- `latest_plan.json`: 약 2초마다 갱신하는 여섯 다리 목표·mode·거부 사유·계산 시간.
- `latest_map.npz`: P를 누른 후 다음 계산 결과의 높이, validity, 관측 시간, odom 중심, 점군.
- `scene.xml`, `robot.urdf`, `asset-<commit>/`: 실행에 사용한 모델 snapshot.

느리면 ray 수를 낮출 수 있지만 관측 밀도도 줄어든다.

```bash
bash scripts/view_foothold_planner.sh --terrain steps --azimuth-samples 360 --elevation-samples 48 --scan-hz 2
```

기존 2.5 cm grid와 비교하거나 관측 보존 시간을 바꾸려면:

```bash
bash scripts/view_foothold_planner.sh --terrain steps --map-resolution 0.025 --map-max-age 120
```

실행 오류는 terminal traceback, 착지점 문제는 위치/다리 번호와 `latest_plan.json`,
지도 문제는 P로 저장한 `latest_map.npz`를 기준으로 확인한다.

## MID-360 FOV 표시 (높이 215 mm / 전방 기울기 45° 수정)

[Livox 공식 사양](https://www.livoxtech.com/mid-360/specs)의 수평 360°, 수직 -7°~+52°
범위를 raycast와 FOV 표시가 함께 사용한다. 설정은 `mjx/mid360_profile.py`에 있다.
FOV는 센서 고유 좌표에서 정의되며, 변경한 45° 장착 회전과 센서 원점 위치가 그대로 적용된다.

- 주황색 경계: 센서 기준 하단 -7°.
- 파란색 경계: 센서 기준 상단 +52°.
- 연결선: 수평 360°를 둘러싼 각도 범위.
- 표시 반경: 기본 1.2 m. 실제 raycast 거리 기본 8 m와 별개다.
- G: FOV 표시 on/off. 지도나 다리 후보를 가리면 끈다.

이 선은 가림이 없는 경우의 시야각 경계이며 관측된 점군이나 장애물 충돌 경계가 아니다.
관측 반환점은 L로 표시한다. FOV 선은 시각화 장면에만 있으므로 LiDAR ray를 가리거나
height map에 기록되지 않는다. 기존 CAD FOV helper mesh를 재배치하지 않고,
실제 센서 TF에서 공식 각도 경계를 생성한다.

FOV를 더 크게 보고 싶으면:

```bash
cd /home/huro/Hexapod-Robot
bash scripts/view_foothold_planner.sh --terrain steps --fov-display-radius 2
```

이번 수정도 실행·렌더링 검증은 수행하지 않았다. 기존 창을 닫고 재실행해야 반영된다.

## 연속 swing과 표시 농도 설정

```bash
cd /home/huro/Hexapod-Robot
bash scripts/view_foothold_planner.sh --terrain steps
```

- 방향키 이동 시 자동 연속 swing: 번호 선택/Enter가 필요하지 않다.
- 기본 swing 시간: tripod당 0.8초. `--gait-swing-duration 1.2`로 늦출 수 있다.
- 높이 지도: 기본 `--map-alpha 0.16` (16% 불투명도).
- LiDAR 점: 기본 `--lidar-point-alpha 0.22` (22% 불투명도), L로 표시 전환.
- 착지점: 실제 목표 위치에 불투명 마커, 그 위에 큰 마커와 두꺼운 세로선.
- 실행 중 경로: 더 굵고 불투명하게 표시. 단순 후보 점은 옅게 유지.

```bash
bash scripts/view_foothold_planner.sh --terrain steps --map-alpha 0.10 --lidar-point-alpha 0.15
```

실행·렌더링·보행 검증은 사용자에게 맡겼다. 이 버전의 정상 동작을 확인했다는 의미는 아니다.
