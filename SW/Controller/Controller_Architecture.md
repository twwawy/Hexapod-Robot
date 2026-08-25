# 6족 로봇 보행 제어기 Architecture

6족 로봇의 수동 조종, 자세·위치 피드백, Tripod 보행, 접촉 적응, 기구학과 Safety의 전체 구조를 정의한다. 세부 수식은 [Controller_detail.md](Controller_detail.md), 입력 채널은 [드론 조종기 입력 README](드론%20조종기%20입력/README.md), 좌표 부호는 [좌표축 README](좌표축/README.md)를 따른다.

## 1. 설계 기준

| 항목 | 현재 값 |
|---|---:|
| 제어 주기 | 5 ms, 200 Hz |
| 기본 보행 | Tripod Gait |
| Tripod A | Leg 1, 3, 5 |
| Tripod B | Leg 2, 4, 6 |
| 한 Phase 시간 | 0.5 s |
| 최대 x·y 선속도 | 각 ±0.28 m/s |
| 최대 Yaw 각속도 | ±45 deg/s |
| 최대 Roll·Pitch 목표각 | ±45 deg |
| 보정 모드 이동 속도 | 각 축 ±0.05 m/s |
| 보정 모드 Yaw 목표각 | ±10 deg |
| 기본 Swing Height | 0.20 m |
| Swing Height 범위 | 0.15~0.25 m |
| Swing 방사 오프셋 | 0.07 m |
| IK 작업공간 여유 | 0.0001 m |
| 관절 범위 | -135~135 deg |
| 관절 명령 속도 제한 | 315.8 deg/s |
| 5 ms당 관절 최대 변화 | 1.579 deg |

몸체 축은 +X 전진, +Y 왼쪽, +Z 위쪽을 사용한다.

## 2. 전체 제어 흐름

~~~text
센서와 사용자 명령
        ↓
상위 상태와 동작 허가 결정
        ↓
입력 정규화·Dead Zone·LPF
        ↓
적용 가능한 속도·자세 명령 결정
        ↓
몸체 위치 추정과 Position PI
        ↓
Heading PI와 몸체 자세 PI
        ↓
최종 Body Twist
        ↓
Tripod 위상과 Stance/Swing 궤적
        ↓
Early/Late Landing
        ↓
몸체 자세 오버레이
        ↓
다리 좌표 변환과 3DOF IK
        ↓
Safety와 관절 출력 제한
        ↓
18개 서보 PWM 또는 릴레이 차단
~~~

위 흐름은 5 ms마다 한 번 실행한다. GPS, LoRa와 같이 200 Hz가 필요하지 않은 통신은 실시간 제어 흐름과 분리한다.

## 3. 상위 상태와 모드

상위 상태는 다음 우선순위를 사용한다.

~~~text
KILL
  > ROLLOVER_FAULT
  > CONTROLLER_FAULT
  > LANDING
  > STANDING
  > READY
~~~

READY 안에서 현재 구현된 사용자 동작은 조종 모드와 보정 모드이다. 암 모드와 자율주행 모드는 조종기 채널 배치를 유지하지만 제어 기능은 추후 구현한다.

| 상태 | 주요 동작 |
|---|---|
| LANDED | 착지 자세를 유지한다. |
| STANDING | 5.6 s 동안 기본 자세로 일어서고 0.5 s 안정 시간을 확인한다. |
| READY | 동작 명령을 받을 수 있는 서기 완료 상태이다. |
| MANUAL | 보행 속도와 Roll·Pitch 자세를 조종한다. |
| CORRECTION | 보행 없이 몸체 x·y·z 위치와 Yaw를 미세 조정한다. |
| LANDING | Swing 발을 정리한 뒤 5.6 s 동안 착지 자세로 내려간다. |
| FAULT | Safety Fault를 유지하고 서보 전원 릴레이를 끈다. |
| KILL | 사용자 Kill 요청에 따라 서보 전원 릴레이를 끈다. |

READY에 들어간 뒤 Throttle·Yaw·Roll·Pitch가 0.2 s 동안 중립이어야 동작을 허가한다. 서기·착지 중에는 짐벌 명령을 전달하지 않는다.

모드가 바뀌면 새 모드의 출력을 계산하기 전에 네 짐벌 필터 상태를 0으로 초기화한다. 조종 모드에서 S1이 중앙의 반대편으로 넘어가면 Yaw 회전 명령과 y축 이동 명령이 서로 이어지지 않도록 Yaw 필터만 초기화한다.

## 4. 사용자 명령

조종 모드의 주요 명령은 다음과 같다.

| 입력 | 명령 |
|---|---|
| Throttle | x축 전진·후진 속도 |
| Yaw, S1 왼쪽 | Yaw 회전 속도 |
| Yaw, S1 오른쪽 | Heading을 유지한 y축 좌·우 속도 |
| Roll | Roll 목표각 |
| Pitch | Pitch 목표각 |

보정 모드는 Pitch·Roll·Throttle로 몸체 x·y·z 이동 속도를 명령하고 Yaw로 상대 Yaw 목표각을 명령한다. 보정 Yaw는 각속도가 아니라 짐벌 위치에 직접 대응하며 중립에서 0 deg로 돌아온다.

실제 RC, 임시 시험 입력과 강화학습 출력은 모두 연속 명령을 사용한다. 불연속 Action은 RC 짐벌과 같은 변화율 제한을 통과시킨 뒤 제어기에 전달한다. 발끝 위치나 관절각을 사용자 명령에서 직접 출력하지 않는다.

## 5. 몸체 위치 추정

STANCE이면서 발 접촉이 확인된 다리만 위치 추정에 사용한다. 각 다리는 관절 측정각의 FK, Stance 시작 때 저장한 지면 Anchor와 IMU 자세를 이용해 절대좌표계의 몸체 위치 후보를 계산한다.

다리별 후보 중 다른 정상 후보가 0.05 m 안에 없는 고립 후보는 해당 주기부터 평균에서 제외한다. 이 상태가 5회 연속, 즉 25 ms 유지되면 Stance Foot Slip으로 Latch한다. 해당 다리가 STANCE 또는 CONTACT 조건을 벗어나면 Latch를 해제하고 다음 Stance에서 Anchor를 다시 만든다.

정상 후보는 압력 크기로 가중하지 않고 단순 평균한다. 추정 결과는 절대좌표계 x·y·z 위치와 유효 다리 수, 6비트 Slip Leg Mask이다. 현재 Slip Mask는 진단값이며 Safety Fault를 직접 발생시키지 않는다.

## 6. 위치와 자세 피드백

### 6.1 Position PI

조종 모드의 적용 x·y 속도를 현재 Heading으로 절대좌표계에 회전하고 적분하여 목표 위치를 만든다. FK 기반 몸체 위치 추정과의 오차를 x·y Position PI에 입력한다.

Position PI는 조종 모드에서만 사용한다. z 위치는 추정하지만 Position PI에는 사용하지 않는다. 보정 모드의 x·y·z 명령과 암 모드에는 Position PI를 적용하지 않는다.

조종 모드 진입 시 목표 위치를 현재 추정 위치에 맞추고 적분항을 초기화한다. 출력 포화 또는 작업공간 제한 중에는 적분항이 바깥 방향으로 누적되지 않게 한다.

### 6.2 Heading PI

S1 왼쪽에서는 사용자 Yaw 각속도를 적분해 목표 Heading을 만들고, IMU Yaw와의 오차를 Heading PI로 보정한다. S1 오른쪽에서는 사용자 Yaw 각속도를 0으로 두고 전환 시점의 Heading을 유지한다.

S1 위치가 어느 방향으로 바뀌든 현재 IMU Yaw를 새 목표 Heading으로 사용하여 전환 순간의 회전 Jump를 막는다.

### 6.3 몸체 자세 PI

조종 모드 Roll·Pitch와 보정 모드 Yaw는 하나의 몸체 자세 PI를 사용한다. 자세 PI 출력은 보정 각속도이며 200 Hz로 적분해 몸체 자세 오버레이를 만든다. 별도의 내부 각속도 PI는 두지 않는다.

자세 오버레이는 Tripod 위상, 보폭과 Swing 착지점을 변경하지 않는다. 선택된 6개 발끝 목표를 몸체 원점 기준으로 역회전하여 몸체만 기울이거나 회전시킨다.

## 7. 최종 Body Twist

조종 모드의 보행 명령은 다음 세 성분을 합한다.

~~~text
사용자 x·y 속도와 Yaw 각속도
  + x·y Position PI 보정
  + Heading Yaw PI 보정
  = 최종 보행 Body Twist
~~~

Roll·Pitch 자세 PI와 보정 모드 Yaw 자세 PI는 보행 Body Twist에 합하지 않는다. 보정 모드의 x·y·z 속도는 PI 없이 몸체 이동 경로에 직접 사용한다.

최종 Body Twist에는 축별 크기 제한과 변화율 제한을 적용한다. Position과 Heading Reference에는 사용자가 요청한 값이 아니라 작업공간 검사를 통과한 실제 적용 속도를 적분한다.

## 8. Tripod Gait

Tripod 그룹은 다음과 같다.

~~~text
Tripod A = Leg 1, 3, 5
Tripod B = Leg 2, 4, 6
~~~

두 그룹은 0.5 s마다 STANCE와 SWING을 교대한다. 실제 x·y 이동 또는 Yaw 회전 명령이 임계값을 넘을 때만 보행 위상을 진행한다.

보행 명령이 중립으로 돌아오면 진행 중인 Swing 다리를 먼저 착지시킨다. 착지 완료 시점의 6개 STANCE 발끝 목표를 그대로 유지하며, 발끝 목표를 0이나 기본 위치로 순간 변경하지 않는다.

보정 모드에서는 6개 다리를 모두 STANCE로 유지한 채 몸체 위치와 자세만 변경한다.

## 9. 발끝 궤적

### 9.1 Stance

STANCE 발은 지면에 고정된 것처럼 보이도록 몸체 선속도와 각속도의 반대 방향으로 발끝 목표를 갱신한다.

\[
\dot p_i=-v_B-\omega_B\times p_i
\]

### 9.2 Swing

Swing 착지점은 다음 Stance 동안 예상되는 몸체 이동과 회전을 절반 선행 보상해 계산한다. 시작점과 착지점을 연결하는 3차 Bezier Curve에 Quintic Time Scaling을 적용한다.

Swing 중간에는 다리 장착 방향 바깥쪽으로 최대 0.07 m의 방사 오프셋을 추가한다. 오프셋은 시작과 착지에서 0이므로 착지점은 바뀌지 않는다.

Swing Height 기본값은 0.20 m이고 몸체 z Offset에 따라 0.15~0.25 m 범위에서 변한다.

## 10. 접촉 적응

압력센서는 발별 CONTACT/NO CONTACT 판정에 사용한다. 접촉과 해제는 서로 다른 임계값과 연속 Sample 조건을 사용하며 실제 값은 센서 캘리브레이션으로 정한다.

Early Landing은 Swing 진행률 50% 이후의 하강 구간에서 접촉이 검출될 때 판단한다. 해당 다리는 현재 발끝 위치를 저장하고 즉시 STANCE로 전환한다.

정상 Swing 시간이 끝났는데 접촉하지 못하면 Late Landing으로 전환한다. 발을 -Z 방향으로 0.20 m/s로 내리면서 다리 안쪽으로 0.16 m/s 이동시킨다. 모든 Swing 다리가 접촉하기 전에는 다음 Tripod Phase로 넘어가지 않는다.

Late Landing의 최대 탐색 거리와 최대 탐색 시간은 아직 정하지 않았다. 탐색 중 IK가 유효하지 않으면 Controller Fault를 발생시킨다.

## 11. 작업공간 제한과 연속성

사용자 출력 범위는 Roll·Pitch ±45 deg, x·y 속도 ±0.28 m/s와 Yaw ±45 deg/s로 유지한다. 실제 적용값은 현재 발 배치에서 6개 다리가 모두 IK 가능한 범위로 동적으로 제한한다.

한 주기의 자세 후보가 가능하면 Roll·Pitch·Yaw 증분을 함께 적용한다. 한 다리라도 불가능하면 세 축을 모두 직전 유효 명령으로 유지한다. 작은 연속 입력을 전제로 하므로 최대 가능값을 찾는 이분 탐색은 사용하지 않는다.

x·y 속도와 Yaw 회전은 다음 보행 궤적 전체를 검사한 뒤 같은 비율과 공통 적용 여부를 사용한다. 합성 이동 방향과 회전 반경을 보존하고 보행 Phase 중간에는 적용값을 바꾸지 않는다.

보정 x·y·z 이동은 다음 위치 후보가 불가능하면 바깥 방향 속도를 0으로 만들고 위치 적분 상태도 갱신하지 않는다. 안쪽 복귀 명령은 즉시 허용한다.

최종 IK 앞에는 0.0001 m 여유의 발끝 작업공간 제한을 둔다. 정상 동작에서 이 마지막 제한이 큰 위치 보정을 수행하면 상위 동적 제한 오류로 판단한다.

## 12. 기구학과 관절 출력

각 발끝 목표는 몸체 좌표계에서 다리 로컬 좌표계로 변환한 뒤 Yaw-Pitch-Pitch 3DOF IK로 관절각을 계산한다. IK는 해 존재 여부와 -135~135 deg 관절 범위를 함께 검사한다.

18개 관절 명령은 각각 독립적으로 315.8 deg/s, 즉 5 ms당 1.579 deg로 제한한다. 이후 관절별 방향, 중립점과 Pulse 보정을 적용하여 500~2500 us 범위의 200 Hz PWM으로 출력한다.

현재 속도 제한은 DS51150-270의 12.6 V 최고 무부하 속도를 기준으로 한다. 실기에서는 관절 ADC로 부하 상태 추종 속도를 측정한 뒤 필요하면 낮춘다.

## 13. Safety

현재 Safety는 다음 조건을 평가한다.

| Fault | 조건 |
|---|---|
| Rollover Fault | 유효한 IMU Roll 또는 Pitch의 절댓값이 80 deg 이상 |
| Controller Fault | IMU 값이 유한하지 않거나 6개 다리 중 하나라도 IK Invalid |

두 Fault는 한 번 발생하면 Latch되며 Reset 입력을 두지 않는다. Fault가 발생하면 상위 상태가 FAULT로 전환되고 Kill 동작으로 6개 서보 전원 릴레이를 모두 끈다. 자동 Recovery는 수행하지 않는다.

다음 항목은 설계 후보이지만 현재 Safety 판정에는 아직 포함하지 않는다.

- Safe Support Polygon
- 비정상 Joint Jump
- 센서별 Timeout
- Stance Foot Slip Fault 승격

관절 범위, 발끝 작업공간과 관절 명령 속도 제한은 Safety Latch와 별개로 항상 적용하는 출력 보호이다.

## 14. 시험 원칙

시험은 센서와 기구학부터 시작해 실제 서보 출력으로 범위를 넓힌다.

1. 좌표축, 관절 방향과 FK/IK를 검증한다.
2. 압력센서 CONTACT/RELEASE를 튜닝한다.
3. Stance와 Swing 궤적을 서보 전원 없이 확인한다.
4. Position PI와 Heading PI를 낮은 Gain으로 조정한다.
5. Roll·Pitch와 보정 Yaw 자세 PI를 조정한다.
6. 작업공간 제한과 모드 전환 연속성을 확인한다.
7. IK Invalid와 전복 입력에서 Safety Latch와 릴레이 차단을 확인한다.
8. 낮은 속도에서 시작해 최대 0.28 m/s까지 단계적으로 실기 검증한다.
