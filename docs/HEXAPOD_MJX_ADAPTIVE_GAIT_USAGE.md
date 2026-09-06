# MJX LiDAR 지형 적응 보행: 구현과 사용자 확인

업데이트: 2026-09-06. LiDAR 기울기는 사용자 요청으로 **45°**로 복원했다.
높이 215 mm, 전방 13.529 mm, MID-360 수평 360°/수직 -7°~+52°는 그대로다.

새 **23-D 파라미터 제어기·MJX 환경·PPO 학습 진입점·키보드 재생기**를 연결했다.
이번 작업에서 학습, 추론, GUI 또는 동역학 보행 검증을 실행하지 않았다. 새 학습 가중치는 없다.
아래 명령으로 사용자가 먼저 기본 보행과 관측 전환을 확인한 뒤 학습을 시작한다.

## 경로와 가장 먼저 실행할 명령

```bash
cd /home/huro/Hexapod-Robot
source /home/huro/.venvs/hexapod-mjx/bin/activate

# 1. 순수 기본 제어기: 미관측 조건에서 반복 스윙 확인
bash scripts/view_foothold_planner.sh --controller adaptive --terrain flat --perception blind

# 2. action 0 + LiDAR 착지/단차 기준값: 계단 앞 관측 전환 확인
bash scripts/view_foothold_planner.sh --controller adaptive --terrain steps --perception lidar
```

가상환경은 `/home/huro/.venvs/hexapod-mjx`이며 JAX/MJX/Brax는 기존 설치를 사용한다.
`--controller adaptive`를 생략하면 기존 stage31 비교 뷰어다. 새 모드는 학습 환경과 같은
MJX 지형에서 실행한다. `steps`는 **5 cm × 7단(level 5)**, `ramp`는 8°(level 3),
`flat`은 level 0이다. 기존 stage31 뷰어의 12×12 m 코스와 구분한다.
`--terrain-level 6`으로 6.5 cm × 7단 등을 선택할 수 있다. `--terrain`과 동시에 지정하지 않는다.

초기 속도는 0이다. 첫 reset/step 때 JAX 컴파일 대기가 발생할 수 있다.
체크포인트가 없으면 콘솔에 `ZERO ACTION, NO TRAINED POLICY`가 표시된다.

| 키/표시 | 기능 |
|---|---|
| ↑ / ↓ | 전진속도 ±0.02 m/s, 범위 ±0.12 m/s; 키를 놓아도 명령 유지 |
| ← / → | yaw 속도 ±0.1 rad/s, 범위 ±0.3 rad/s |
| Space | 속도와 회전 명령 0; 제어기가 현재 보행을 마무리 |
| Enter | 물리 일시정지/재개 |
| H | 로봇·지도·보행 상태 reset, 속도 0 |
| C | 지도만 지움; 다음 센서 주기에 다시 관측 |
| M | 반투명 지도 표시 전환 |
| P | 최근 최대 500 step trace, 지도, 모델, 계약 저장 |
| 흐린 청록색 점 | 실제 LiDAR 반환점으로 누적한 지도, alpha 0.14 |
| 진한 녹색 점 | 현재 관측으로 지지 patch 조건을 만족한 후보 |
| 진한 빨간 점 | 제어기가 스윙 시작에 수락한 발 중심 목표 |

빨간 점은 지면에서 발 반지름 32 mm 위에 표시한다. 녹색 후보와 지면 높이 기준이 다르다.
종료 조건을 만나면 자동으로 새 에피소드로 넘기지 않고 일시정지한다. 콘솔의 종료 원인을 보고 H로 초기화한다.
역주행/회전은 사용자가 조사할 수 있도록 열었으며 기본 PPO 명령 분포는 전진 0.06~0.12 m/s, yaw 0이다.
따라서 그 외 명령에서 학습 성능을 보장하는 설정은 아니다.

## 환경 구성

```text
simulator 몸체·관절·접촉 상태 → 기본 보행 제어기
                                    ↑
MID-360 angular raycast → rolling map → 다리별 9개 후보
                                    ↓
LiDAR actor → 23개 보행 파라미터 요청 → 제어기 수락/궤적/접촉 처리 → IK → MJX

지형 GT → privileged critic / 보상 / 관측된 높이 오차 비교
```

로봇은 기존 학습용 box/capsule/sphere skeleton이며 distal link 230 mm와 발 구 반지름 32 mm를
사용한다. IK 끝점과 foot site를 발 구 중심에 맞췄다. 높이 지도는 접촉 표면의 Z이며 제어기가
발 반지름을 더한다. 현재 CAD 발끝에서 구 반지름을 빼던 루트 v4 형상과 새 adaptive 형상을 섞지 않는다.
질량/관절/servo 물리는 루트 MJX 설정을 계승한다.

- 제어기 5 ms, 정책 20 ms, 물리 2.5 ms. 정책 step에서 4개 제어기 tick 후 8개 물리 substep을 실행한다.
  이 구조에서 접촉·몸체 입력은 해당 20 ms 동안 유지된다.
- 새 GPU 지도: 64×64 셀, 5 cm 해상도, 3.2×3.2 m, 최대 60초 유지.
  저속 보행에서 전방에 관측한 지면이 뒤 다리까지 이동할 시간을 확보한다.
- 기본 센서: 100 ms마다 90×8=720개 ray, 거리 0.1~8 m, dropout 5%, 거리 noise 표준편차 5 mm.
  하단 FOV에 샘플을 더 배정하며 스캔마다 각도를 옮긴다. 실제 Livox 비반복 패턴의 재현은 아니다.
- ray는 로봇과 지형의 첫 교차를 계산한다. 로봇에 가려진 ray는 뒤쪽 지형을 지도에 쓰지 않는다.
  지형 GT에 noise를 더해 LiDAR로 대체한 구현이 아니다.
- 지도는 관측 시간/높이 분포를 저장하며 미관측 셀을 지면 0으로 채우지 않는다.
  actor의 미관측 높이 입력만 중립값 0으로 마스킹하고 valid/age를 함께 준다.
- 기존 뷰어의 CPU 지도(8×8 m, 4 cm, 720×64 ray)와 새 학습용 JAX 지도는 별도 설정이다.
- 이 버전의 MJX ray는 heightfield 교차를 지원하지 않아 rough level 1·2는 명시적으로 거부한다.
  평지·box 계단·box 경사(level 0, 3, 4, 5~16)를 사용한다.

몸체 pose/velocity와 접촉은 아직 MuJoCo 상태를 이상적인 추정값으로 사용한다.
실제 LiDAR+IMU odometry, odom/TF drift, 시간 지연 센서 모델은 후속 작업이다.

## 정책과 제어기의 역할

| action | 범위와 의미 |
|---|---|
| `0:12` | RF/RM/RB/LF/LM/LB 순서 XY, 각 축 ±4 cm 요청 |
| `12:18` | 다리별 스윙 여유 높이 residual ±4 cm; 최종 여유 4~18 cm |
| `18:21` | pitch ±10°, roll ±5°, 몸체 높이 ±3 cm |
| `21` | 전진 보폭 배율 0.5~1.3 |
| `22` | tripod 위상 시간 0.3~0.7초, 중립 0.5초 |

action은 [-1,1], 0이 중립이다. **action 0 + LiDAR 모드도 관측 높이에 맞춘 착지 Z와
경로 clearance, 센서 기반 pitch 기준값을 사용한다.** 학습 보정과 센서 기준값을 모두 끈
기본 제어기 비교는 `--perception blind`로 실행한다.

다리별 nominal 주변 3×3 후보에서 policy XY 요청에 가까운 지지 후보를 선택한다.
지지 patch는 중심과 ±3.5 cm의 네 이웃 모두 관측되고 높이 분포가 2.5 cm 이내여야 한다.
이는 edge/거칠기 판정의 초기 근사이며 전체 지지 다각형이나 연속 충돌 검사를 대신하지 않는다.
후보 선택은 이산 projection이므로 매끄러운 XY 이동 전체를 표현하지 않는다.

관측한 경로의 최대 높이를 스윙 기준에 반영하고 남는 clearance가 2 cm 미만이면 거부한다.
스윙 경로 21개 표본에서 IK를 검사하고, 매 tick 전체 다리 자세/높이 gate와 관절 속도 제한을 적용한다.
숨은 장애물과 표본 사이의 충돌까지 보장하는 검사는 아니다.

해당 다리의 관측 후보가 없으면 기본 궤적으로 걷는다. 관측된 높이 불연속이 있는데 모든 후보가
위험하면 단순 미관측으로 바꾸지 않고 해당 tripod의 새 시작을 보류한다. 요청이 바뀌면 재시도한다.
관측된 후보가 모두 IK 불가능한 경우도 보류하고 거부 정보를 정책/로그에 제공한다.

착지 XY/Z·clearance는 스윙 시작에 고정한다. 지도 갱신으로 진행 중 목표를 바꾸지 않는다.
고정된 world 목표를 제어기 내부 궤적 좌표로 변환하며 foot memory는 제어기가 한 번만 갱신한다.
접촉이 먼저 발생하면 접촉 위치에서 stance를 시작하고, 늦게 착지하면 제어기 접촉 탐색을 유지한다.

몸체 자세·보폭·주기는 모든 다리에 영향을 주므로 **6개 다리 모두 후보가 있을 때** 활성화한다.
tripod 경계에서 목표를 수락하고 자세에는 15°/s, 높이에는 4 cm/s 제한을 적용한다.
전진속도는 `명령 속도 × stride / (phase_time / 0.5)`로 보폭과 주기를 함께 반영한다.
자세 기준 pitch는 센서 지면 추정으로 최대 ±12°이고 여기에 residual이 더해진다.

actor observation은 **641-D**, critic observation은 **764-D**다.
actor는 155-D 상태·명령·이전 action/수락 파라미터와 486-D 후보 특징을 받는다.
critic만 후보 지형 GT, 유효 LiDAR-GT 차이, 15개 지형 샘플을 추가로 받는다.
높이 MAE는 관측된 위치만 비교한다. 콘솔에서 비교 표본이 없으면 `n/a`다.

보상은 기존 명령 추종/progress/넘어짐/관절/충격 등을 계승하고, 수락한 자세 목표에 대한 안정성,
착지 순간 목표 오차·stance slip·projection 비용을 추가한다. 몸체를 많이 기울이거나 발을 높게
드는 행위 자체를 보상하지 않는다. 기본 제어기의 GT pitch feedforward와 GT swing boost는 0이다.

## 사용자 확인 순서

1. `blind + flat`에서 ↑로 전진한다. tripod가 여러 차례 교대하고 모든 다리의 IK가 유지되는지 본다.
2. `lidar + flat`에서 `known` 증가 전후를 본다. 관측되는 순간 발이 튀거나 stance가 미끄러지는지 본다.
3. `lidar + steps`에서 첫걸음의 미관측 fallback, 후보 높이와 계단 높이, 실제 접촉을 비교한다.
4. 빨간 수락 목표가 스윙 중 고정되는지, 접촉 뒤 다리를 다시 억지로 목표로 끌지 않는지 본다.
5. `reject`, IK 배열, stride/phase를 확인한다. 반복 거부나 관절 꼬임이 있으면 P로 trace를 저장한다.
6. 이 확인 후 새 정책을 학습하고 action 0과 동일 조건에서 비교한다.

P 산출물은 `mjx/generated/adaptive_gait/`에 저장한다.
`trace.npz`는 step 전 observation/qpos/qvel/command, 요청 action, step 후 수락 action/관절 목표/
착지 목표/contact/phase/done/next_qpos를 담는다. `accepted_action`은 경계에서 latch한 요청이며,
자세 rate limit과 IK gate가 반영한 최종 모터 명령은 `targets`다.
`map.npz`, `model.xml`, `adaptive_contract.json`도 저장한다.

## 학습 시작

사용자가 위 동작을 확인한 다음 실행한다. 출력 디렉터리는 매번 새 경로여야 한다.

```bash
# LiDAR actor + GT critic으로 직접 학습 (새 23-D 정책)
bash scripts/train_adaptive_gait.sh --perception lidar --terrain-level 0 \
  --num-envs 64 --timesteps 10000000 --output mjx/runs/adaptive-lidar-flat

# 평지 정책으로 5 cm 계단 학습 시작; 가중치/정규화 복원, optimizer는 새로 시작
bash scripts/train_adaptive_gait.sh --perception lidar --terrain-level 5 \
  --restore mjx/runs/adaptive-lidar-flat \
  --num-envs 64 --timesteps 20000000 --output mjx/runs/adaptive-lidar-steps

# 새 정책을 같은 MJX 환경에서 방향키로 재생
bash scripts/view_foothold_planner.sh --controller adaptive --terrain steps \
  --checkpoint mjx/runs/adaptive-lidar-steps
```

`--wandb`를 추가하면 `hexapod-adaptive-gait` 프로젝트에 로그를 기록한다.
64개 환경은 시작 설정이며 처리속도/메모리를 실측한 값이 아니다.
`batch-size × num-minibatches`는 `num-envs`로 나누어떨어져야 한다.

별도의 같은 action 공간 GT teacher를 먼저 만들 수도 있다.

```bash
bash scripts/train_adaptive_gait.sh --perception teacher --terrain-level 0 \
  --output mjx/runs/adaptive-teacher-flat
bash scripts/train_adaptive_gait.sh --perception teacher --terrain-level 5 \
  --restore mjx/runs/adaptive-teacher-flat --output mjx/runs/adaptive-teacher-steps
bash scripts/train_adaptive_gait.sh --perception lidar --terrain-level 5 \
  --init-teacher mjx/runs/adaptive-teacher-steps --output mjx/runs/adaptive-student-steps
```

teacher 모드는 LiDAR 관측 mask/age를 유지하고 관측 높이를 GT로 치환한다. 미관측 첫걸음의
기본 보행 규칙은 동일하다. teacher checkpoint 자체를 LiDAR 배포 정책으로 간주하지 않는다.
`--init-teacher`는 동일 23-D 네트워크/정규화 가중치 이관 후 asymmetric PPO이며,
별도 행동 imitation loss나 높이 복원 auxiliary loss는 아직 구현하지 않았다.
현재 GT 비교는 critic 입력·보상·진단에 사용한다. 기존 stage31 18-D checkpoint는 거부한다.

각 checkpoint에는 `adaptive_contract.json`과 `ppo_network_config.json`, 네트워크/정규화가 저장된다.
checkpoint 경로 대신 run 경로를 주면 가장 큰 step 번호를 선택하며 **best score 자동 선택은 아니다**.
제어/관측/센서 소스 hash가 달라지면 계약 검사에서 중단한다. 변경 전 정책은 기록된 소스로 재생하거나
명시적인 버전 이관을 구현해야 한다. 가중치가 있다고 계단 통과가 검증됐다는 뜻은 아니다.

## 다음 단계와 Isaac Lab 이전

이번 구현으로 MJX 기본 제어기→23-D 파라미터→LiDAR 관측→PPO→재생 경로를 마련했다.
사용자 동작 확인과 실제 학습 결과를 바탕으로 보상·범위·지지 판정을 조정한다.
held-out 지형, 센서/odom 지연·오차, teacher imitation/높이 복원 loss, 넓은 rough 코스 지원은 후속이다.

Isaac Lab에는 아직 이 23-D 제어기/정책을 이식하지 않았다. 우선 MJX에서 사용자가 검증한
checkpoint와 contract, 모델, 정규화, trace를 고정한 다음 같은 관절 순서·부호·단위·주기·TF·접촉 기준으로
Torch 제어기를 구현하고 sim-to-sim 재생한다. 기존 Isaac v4/18-D 모드에 이 가중치를 직접 넣지 않는다.
