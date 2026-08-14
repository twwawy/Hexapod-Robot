# 6족 로봇 보행 제어기 Architecture

6족 로봇의 수동 보행 및 몸체 안정화 제어 구조를 정의한다.

좌표계와 관절 방향은 좌표축 README의 정의를 따른다.

---

## 전체 제어 구조

```text
조종기 입력
    │
    ▼
Dead Zone + LPF
    │
    ▼
사용자 Body Command
    │
    ├──────────────────────────────┐
    │                              │
    │                      몸체 상태 Feedback
    │                              │
    │                 ┌────────────┴────────────┐
    │                 │                         │
    │           Body Position             Body Attitude
    │                PI                        PI
    │                 │                         │
    │     Stance Leg + FK + Contact            IMU
    │                 │                         │
    └─────────────────┴─────────────┬───────────┘
                                    │
                                    ▼
                             Final Body Twist
                           [vx vy vz wx wy wz]
                                    │
                                    ▼
                           Tripod Gait Manager
                         ┌──────────┴──────────┐
                         │                     │
                       Stance                Swing
                         │                     │
                  Stance Trajectory      Target Position
                         │                     │
                         │     Quintic + Cubic Bezier + Radial Offset
                         │                     │
                         └──────────┬──────────┘
                                    │
                            Contact Adaptation
                           Early / Late Landing
                                    │
                                    ▼
                             6개 발끝 목표 위치
                                    │
                                    ▼
                    몸체 좌표계 {B} → 각 다리 좌표계 {L_i}
                                    │
                                    ▼
                                 3DOF IK
                                    │
                                    ▼
                            18개 관절 목표 각도
                                    │
                                    ▼
                                 안전장치
                                    │
                                    ▼
                              Servo Output
```

---

## 블록별 설명

### 1. 조종기 입력

조종기 입력으로 로봇의 전진·후진 속도와 Roll·Pitch·Yaw 각속도를 명령한다.  
입력에는 **Dead Zone과 Low Pass Filter**를 적용한다.

| 입력 | 명령 |
|---|---|
| Throttle | 전진·후진 선속도 |
| Roll | Roll 각속도 |
| Pitch | Pitch 각속도 |
| Yaw | Yaw 각속도 |

Yaw는 **Heading Hold**를 사용한다.

---

### 2. Body Position Feedback

STANCE 상태이면서 실제로 CONTACT가 확인된 다리의 관절각과 Forward Kinematics를 이용하여 몸체 위치를 추정한다.

여러 유효 Stance 다리의 위치 추정값은 **단순 평균**하며, 압력센서 값은 접촉 여부 판단에만 사용한다.

---

### 3. Body Attitude Feedback

9축 IMU에서 얻은 Roll, Pitch, Yaw를 이용하여 현재 몸체 자세를 추정한다.

목표 자세와 현재 자세의 차이는 **PI Controller**로 보정한다.

---

### 4. Final Body Twist

사용자 명령과 Body Position / Attitude Feedback을 합쳐 최종 몸체 선속도와 각속도를 만든다.

```text
User Command
    +
Position Feedback
    +
Attitude Feedback
    ↓
Final Body Twist
```

---

### 5. Tripod Gait Manager

기본 보행은 **Tripod Gait**를 사용한다.

```text
Tripod A = 1, 3, 5
Tripod B = 2, 4, 6
```

두 그룹은 Swing과 Stance를 번갈아 수행한다.

---

### 6. Stance Trajectory

STANCE 발은 지면에 고정된 상태를 유지하도록 몸체 이동과 회전의 반대 방향으로 발끝 목표 위치를 갱신한다.

---

### 7. Swing Trajectory

SWING 다리는 다음 착지 목표점을 계산한 뒤 **3차 Bezier Curve와 방사 방향 오프셋**으로 발끝 경로를 생성한다.

시간 진행에는 **Quintic Time Scaling**을 사용하여 Swing 시작과 종료를 부드럽게 만든다.

Swing Height는 몸체 기준점에 대한 몸체 원점의 z방향 상대 위치에 따라 보정한다.

방사 방향 오프셋은 Swing 시작점과 착지점에서는 0이고 최고점에서 최대가 되며, 발끝을 각 다리의 장착 방향 바깥쪽으로 이동시켜 IK 관절각을 줄인다. 현재 Simulink 검증값은 Swing Height `0.25 m`, 방사 방향 오프셋 `0.07 m`이다.

---

### 8. Early / Late Landing

**Early Landing**이 발생하면 해당 다리는 즉시 STANCE로 전환한다.

**Late Landing**이 발생하면 미접촉 발만 `-z_B` 방향으로 Search Down하며, 모든 Swing 다리가 접촉하기 전까지 다음 Tripod Phase로 넘어가지 않는다.

---

### 9. 좌표변환 및 IK

몸체 좌표계에서 계산한 6개 발끝 목표 위치를 각 다리 좌표계로 변환한 뒤 **3DOF Inverse Kinematics**를 이용해 18개 관절 목표각을 계산한다.

모든 관절의 사용 범위는 **-135° ~ +135°**이다.

---

### 10. 안전장치

Servo 출력 전에 다음 항목을 검사한다.

- ROLLOVER FAULT
- IK Workspace
- 관절 각도 범위
- Safe Support Polygon
- 비정상 목표각 Jump
- 관절 최대 각속도

Safe Support Polygon은 실제 지지다각형을 중심 기준으로 **10% 축소한 영역**을 사용하며, 몸체 원점이 이 영역을 벗어나면 보행을 중단한다.

비정상적인 Joint Jump가 검출되면 새 명령을 적용하지 않고 **직전 관절각을 HOLD**한다.

Roll 또는 Pitch가 **80° 이상**이면 ROLLOVER FAULT로 판단하고 즉시 정지한다.

---

### 11. Servo Output

사용 Servo는 **DS51150-270**이다.

| 항목 | 값 |
|---|---:|
| PWM 주파수 | 200 Hz |
| PWM 주기 | 5 ms |
| 중립 Pulse | 1500 us |
| Pulse 범위 | 500 ~ 2500 us |
| 관절 사용 범위 | -135° ~ +135° |

관절의 0° 방향은 좌표축 README의 정의를 따른다.

---

## 필수 설정

| 항목 | 설정 |
|---|---|
| 전체 제어 주파수 | 200 Hz |
| Sampling Time | 5 ms |
| 기본 보행 | Tripod Gait |
| Tripod A | 1, 3, 5 |
| Tripod B | 2, 4, 6 |
| Position Controller | PI |
| Attitude Controller | PI |
| Yaw 제어 | Heading Hold |
| Swing Path | Cubic Bezier + Radial Offset |
| Swing Time Scaling | Quintic |
| Safe Support Polygon | 실제 지지다각형의 90% |
| Rollover 기준 | Roll 또는 Pitch 80° |
| Joint Jump | 직전 각도 HOLD |
| Servo | DS51150-270 |
