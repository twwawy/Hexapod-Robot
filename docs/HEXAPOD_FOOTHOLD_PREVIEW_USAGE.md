> 이 문서는 이전 개발 단계의 설계/사용 이력을 포함합니다. 현재 v4 contract, timing, 실행 경로는 [통합 문서](ADAPTIVE_INTEGRATION_V4.md)를 따릅니다.

# MuJoCo 보행·LiDAR 뷰어 사용법

최종 갱신: 2026-09-06. 기본 모드는 **펌웨어 제어기 연속 보행 + stage31 RL residual + MJX 동역학**이다.
사용자 요청에 따라 이번 수정의 GUI·추론·동역학 보행 검증은 실행하지 않았다.

## 시작과 환경

```bash
cd /home/huro/Hexapod-Robot
source /home/huro/.venvs/hexapod-mjx/bin/activate
bash scripts/view_foothold_planner.sh --terrain steps
```

기존 창을 닫고 다시 실행한다. 스크립트가 가상환경 Python을 직접 사용하므로 activate는 선택 사항이다.
MuJoCo/MJX, JAX, Brax, Orbax가 설치된 학습 환경을 사용하며 첫 실행에는 JAX 컴파일 시간이 필요하다.
GPU 정책/물리 계산은 별도 프로세스, LiDAR/map 계산은 CPU 스레드에서 처리한다.
컴파일 중 창은 조작할 수 있지만 보행 시작은 준비가 끝난 뒤다. 물리 시간은 HUD의 simulation을 본다.

| 구성 | 현재 구현 |
|---|---|
| 로봇 | 과거 학습의 box 몸체·capsule 링크·반지름 32 mm 구형 발 skeleton |
| 가중치 | `progress-v2-stage31-level6_20260828-111825_seed40`, checkpoint `000001703936` |
| 제어기 | 가중치에 대응하는 격리된 `0805164` v3 펌웨어 step |
| 주기 | policy 20 ms, firmware 5 ms, physics 2.5 ms |
| 몸체 상태/odom | MuJoCo 상태 사용; 실제 LiDAR+IMU odometry 구현은 후속 작업 |
| 정책 지형 입력 | LiDAR 높이 지도에서 15개 위치를 샘플링 |
| 물리/시각화 지형 | 같은 12×12 m 코스의 2 cm heightfield |
| GT 접근 | 환경의 물리·평가 및 별도 LiDAR 비교용; actor 지형 입력과 보정에는 사용하지 않음 |

현재 루트 `mjx/` 학습 코드는 v4/100 mm 계약이다. 뷰어의 stage31 v3/18-D 가중치에 이를 섞지 않는다.
당시 미커밋 코드 전체가 보관되지 않았으므로 기록 영상의 정확한 재현을 보장하지 않는다.

## 제어 구조와 이번 수정 이유

```text
속도·높이 명령 + 몸체/접촉 상태 → 펌웨어 nominal gait ─┐
                                                     ├→ Safety/IK → servo → MJX
LiDAR 높이 지도 + proprioception → stage31 residual ─┘

LiDAR 지도 → geometric 착지 후보 → 화면의 참고 마커
LiDAR 높이 샘플 ↔ 동일 XY의 simulator GT → 비교 지표/저장 전용
```

이전 구현은 관측된 경로를 월드 좌표 발 목표로 덮어쓰고, 착지 후 그 위치를 펌웨어 foot memory에
되돌려 넣었다. 이 경로와 펌웨어의 접촉·posture·지지 궤적이 충돌할 수 있었다. 사용자가 관측된
착지점 진입 때 IK 문제를 보고했으며, 실행 재현으로 직접 원인을 확정한 것은 아니다.

현재는 외부 경로/stance anchor 주입과 다리별 제어 모드 전환을 제거했다. 원래 펌웨어 step이
nominal 궤적, 스윙/지지/late landing, foot memory, posture, residual 해석, IK, 관절 속도 제한을 맡는다.
추가 어댑터는 policy action의 크기만 조절한다. 후보를 못 찾아도 기본 gait는 계속 진행한다.

- 정책이 읽는 15개 지형 샘플이 모두 미관측이면 action과 residual filter를 0으로 만든다.
- 관측이 생기면 목표 gain은 `--residual-scale × 유효 샘플 수/15`다. 0.5초 시정수로 증가/변화를 완화한다.
- 관측이 모두 만료되거나 scale이 0이면 residual을 즉시 0으로 지운다. 기본 제어기는 계속 동작한다.
- gain은 임시 관측 가용성 기준이며, 후보 patch 유효성이나 학습된 confidence를 뜻하지 않는다.
- v3의 원래 swing XY, swing Z 높이, stance Z, late-landing residual 규칙을 그대로 사용한다.
- GT 기반 pitch feedforward와 swing boost는 끈다. 지지 높이는 접촉 발 운동학으로 추정한다.

미관측 지형 입력의 0은 checkpoint 차원을 유지하기 위한 중립값이다. 지도에는 관측으로 기록하지 않는다.
기존 146-D actor는 유효 마스크/관측 나이를 입력받지 않아 공백과 실제 평지를 구분하지 못한다.
그 정보는 비교 파일에 저장하고, 향후 sensor student 학습 때 입력에 추가해야 한다.

**착지 후보 마커가 실제 발 목표는 아니다.** 후보 실패의 빨간 X도 기본 모드의 정지 명령이 아니다.
실제 명령은 흰 점이며, 알려진 장애물에 대한 별도 경로 보장 기능을 이 모드에 구현한 것은 아니다.
기존 환경의 IK·workspace 제한과 물리 실패 종료 조건은 남아 있다.

## 조작법

방향키는 목표 속도를 증감한다. 키를 놓아도 계속 움직이므로 창에 포커스를 두고 사용한다.

| 키 | 기본 `--controller trained` 모드 |
|---|---|
| ↑ / ↓, W / S | 전후진 command ±0.04 m/s; 범위 ±0.12 m/s |
| ← / →, Q / E | yaw command ±0.15 rad/s; 범위 ±0.3 rad/s |
| Space | 이동 command 0; 물리와 자세 제어는 계속 진행 |
| Enter | 물리 일시정지 / 재개 |
| PageUp / PageDown | 초기 높이 기준 trim ±2 cm; 범위 -5~+10 cm |
| H | 정책·물리·지도 초기화 |
| R | 인식 재시도; 물리 종료 상태이면 reset |
| 1~6 | RF/RM/RB/LF/LM/LB 후보 상세 보기 |
| M / L / G | 높이 셀 / 반환점 / MID-360 FOV 표시 |
| K | scan 켜기/끄기; 기존 관측은 60초 뒤 만료 |
| C | 지도 초기화 + 이동 command 0 |
| P | 계획·지도·제어 진단·LiDAR/GT 샘플 쌍 저장 |
| F / T / V | 추적 카메라 / 위에서 보기 / 전체 코스 |

A/D 횡이동은 checkpoint command에 없어 지원하지 않는다. 원본 run은 전진 0.06~0.12 m/s,
yaw=0으로 학습했으므로 후진·회전과 sensor 입력에서의 성능은 사용자가 별도로 확인한다.

## 사용자 확인 순서

먼저 같은 동역학 환경에서 residual 없이 비교한다.

```bash
bash scripts/view_foothold_planner.sh --terrain steps --residual-scale 0
```

1. ↑를 2번 눌러 전진한다. 관측 유무와 무관하게 `NOMINAL / RL=0`, gain 0으로 스윙이 이어지는지 본다.
2. 다시 기본 명령 또는 `--residual-scale 0.25`로 실행한다. 관측 샘플이 생기면 `NOMINAL + RL RESIDUAL`과 gain이 표시되는지 본다. `FOOTHOLD + RL`로 바뀌는 이전 구조는 제거했다.
3. K로 scan을 끄고 C로 지도를 지운 뒤 다시 전진한다. residual 0으로 기본 보행이 이어지는지 확인한다.
4. `IK valid`, `residual IK valid`, `reach limited`, 흰 목표와 실제 발을 비교한다. 첫 항목은 최종 IK, 두 번째는 residual 적용 전후 후보의 IK 수락 여부다.
5. 문제가 생긴 시점에 Enter로 멈추고 P로 저장한다. scale 0에서도 문제가 있으면 기본 제어기/물리 쪽, residual 적용 때만 생기면 정책·관측·보정 크기 쪽을 구분해 볼 수 있다.
6. `LiDAR policy samples n/15`와 `GT raster RMSE`를 확인한다. 관측 샘플이 없으면 RMSE는 `n/a`이며 오차 0으로 취급하지 않는다.

## 코스·지도·센서

초기 전방은 world +X다. 정면 4 cm 계단 6단, 좌측 4/8/12 cm 플랫폼과 징검다리,
우측 8° 경사와 돌출물, 뒤쪽 요철·둥근 바위가 있다. 독립 재생기의 6.5 cm 계단 7단과 다른 코스다.
2 cm heightfield는 원래 상자 모서리와 좁은 돌출물을 근사한다.

기본 measured TF는 밑면 중심에서 높이 **215 mm**, 전방 **13.529 mm**, 전방 기울기 **45°**다.
38° 변경은 사용자 요청으로 롤백했다.
base 기준 XYZ `(0, -0.013529, 0.1642) m`, RPY `(0, 45°, -90°)`다.
MID-360 FOV는 수평 360°, 수직 -7°~+52°, 거리 0.1~8 m, 기본 720×64 angular rays다.
실제 Livox 비반복 스캔 패턴의 재현은 아니다. URDF CAD 장착 chain과 measured override는 별도다.

별도의 새 23-D 파라미터 제어/학습 모드는 같은 스크립트에 `--controller adaptive`를 붙인다.
그 모드는 학습용 MJX 계단과 JAX 센서 지도를 사용하므로 이 문서의 stage31 코스·키·지도 설정과 구분한다.
실행/학습/확인 항목은 [adaptive 사용 가이드](HEXAPOD_MJX_ADAPTIVE_GAIT_USAGE.md)를 따른다.

지도는 odom 정렬 8×8 m rolling grid, 4 cm 셀, 60초 유지다. 몸체에 가린 지면은 채우지 않는다.
지도 불투명도 16%, 점군 22%, 후보 마커는 불투명하다. M/L/G는 표시만, K/C는 정책의 센서 입력에도 영향을 준다.
주황은 미관측 nominal 후보, 파랑/청록은 관측 기반 후보, 빨간 X는 후보 실패, 흰 점은 실제 제어 목표다.

## 저장 파일과 비교 모드

출력 경로는 `mjx/generated/foothold_preview/`다.

| 파일 | 내용 |
|---|---|
| `scene_manifest.json` | checkpoint·격리된 소스·센서 TF·residual 설정·GT 용도 |
| `latest_plan.json` | qpos, 후보, 실제 발 목표, scaled action/gain, IK 진단, LiDAR/GT 비교 지표 |
| `latest_map.npz` | 지도 높이·관측 여부·시각·odom 중심·점군 |
| `latest_lidar_gt_pair.npz` | 동일 15개 XY의 LiDAR/GT 높이, valid/age, support height, 입력 pose와 시각 |
| `policy/` | 격리된 v3 소스·MJB 모델·JAX cache |

GT 비교는 action 계산 후 수행하며 결과를 actor에 넣지 않는다. GT는 2 cm 코스 raster의 bilinear
기준이므로 MuJoCo 삼각형 표면과 모서리에서 차이가 날 수 있다. 오차는 관측된 유효 샘플만 계산한다.
P는 최근 한 쌍을 덮어쓴다. 이것은 비교 자료이며 대규모 데이터셋 수집기나 재학습 완료 모델이 아니다.
센서 정책의 후속 학습 구조는 [설계 문서](HEXAPOD_LIDAR_FOOTHOLD_RESIDUAL_PLAN.md)를 따른다.

```bash
# 예전 착지점 경로를 따라가는 기구학 모드: 물리/학습 residual 비교와 구분
bash scripts/view_foothold_planner.sh --terrain steps --controller nominal

# 기구학 모드에서 CAD mesh 확인
bash scripts/view_foothold_planner.sh --terrain steps --controller nominal --robot-model mesh

# 별도 v3 정책 재생기: 기록된 지형/GT 입력 사용
bash scripts/view_trained_policy.sh
```

기구학 모드만 A/D, Enter 제자리 swing, `--gait-swing-duration`, `--require-observed`를 지원한다.
`--revision`은 기구학 모델의 URDF snapshot에만 적용하며 기본은 로컬 `origin/main`이다.
