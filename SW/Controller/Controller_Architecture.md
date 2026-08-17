# 6족 로봇 보행 제어기 Architecture

6족 로봇의 수동 보행 및 몸체 안정화 제어 구조를 정의한다.

좌표계와 관절 방향은 좌표축 README의 정의를 따른다.

절대 좌표계 $\{W\}$는 지면에 고정하며, 모든 관절각이 0°인 최초 착지 상태의 몸체 원점을 절대 원점으로 사용한다. 그 자리에서 기본 자세로 일어서면 절대좌표계 기준 몸체 위치는 $[0,0,h_{stand}]^T$가 된다. 몸체 좌표계 $\{B\}$와 몸체 기준 좌표계 $\{R\}$는 로봇을 따라 이동하고, Body Position Estimator와 Position PI는 절대좌표계의 위치 ${}^W\hat p_B$와 목표 ${}^Wp_{B,ref}$를 사용한다.

---

## 전체 제어 구조

```text
조종기 입력
    │
    ▼
Dead Zone + LPF
    │
    ▼
사용자 Body Command [v_user, φ_ref, θ_ref, ω_z,user]
    │
    ├──▶ Position Reference Generator
    │    p_ref[k] = p_ref[k-1] + R·v_user·Ts
    │                 │
    │                 ▼
    │          Body Position PI ◀─ Body Position Estimate
    │                 │                ▲
    │          Position Feedback     Stance Leg + FK + Contact
    │                 │
    ├──▶ Body Attitude PI ◀─ IMU
    │                 │
    │          Attitude Feedback
    │                 │
    └─────────────────┬───────────┘
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

조종기 입력으로 로봇의 전진·후진 속도, Roll·Pitch 목표각과 Yaw 각속도를 명령한다.
입력에는 **Dead Zone과 Low Pass Filter**를 적용한다.

| 입력 | 명령 |
|---|---|
| Throttle | 전진·후진 선속도 |
| Roll | Roll 목표각 |
| Pitch | Pitch 목표각 |
| Yaw | Yaw 각속도 |

Yaw는 **Heading Hold**를 사용한다.

#### Control Priority Manager와 Drone Controller

조종기 입력은 `ControlPriorityManager → DroneController` 순서로 처리한다. `ControlPriorityManager`가 스위치, 동작 완료 신호와 Fault 신호를 이용하여 허용 모드를 결정한 뒤, `DroneController`가 해당 모드의 입력 필터와 기능별 Enable 및 명령을 생성한다. 따라서 우선순위에서 차단된 기능의 진행률은 내부에서 미리 진행하지 않는다.

상위 제어 상태는 다음 우선순위를 따른다.

$$
\boxed{
KILL
>
ROLLOVER\_FAULT
>
CONTROLLER\_FAULT
>
LANDING
>
STANDING
>
READY(MANUAL/CORRECTION)
}
$$

READY 진입 후에는 조종기 네 축이 0.2초 동안 중립일 때 `Motion_Armed`를 활성화한다. 활성화 전에는 수동 조종 및 보정 명령을 전달하지 않는다.

Tripod 보행은 수동 조종 모드에서 다음 이동 명령 조건 중 하나를 만족할 때만 활성화한다.

$$
|v_{x,user}|\ge V_{gait,th}
\quad\text{or}\quad
|\omega_{z,user}|\ge \Omega_{gait,th}
$$

READY 상태만으로는 `Tripod_Enable`을 활성화하지 않는다.

---

### 2. Body Position Feedback

STANCE 상태이면서 실제로 CONTACT가 확인된 다리의 관절각과 Forward Kinematics를 이용하여 몸체 위치를 추정한다.

여러 유효 Stance 다리의 위치 추정값은 **단순 평균**하며, 압력센서 값은 접촉 여부 판단에만 사용한다.

보행 중의 목표 몸체 위치는 사용자 선속도 명령을 200 Hz로 적분하여 생성한다.

$$
p_{B,ref}[k]
=
p_{B,ref}[k-1]
+
{}^W\hat R_B[k]v_{user}[k]T_s
$$

Position Reference Generator 활성화 시에는 $p_{B,ref}[k_0]={}^W\hat p_B[k_0]$로 초기화한다. 목표 위치와 FK·Stance Anchor로 계산한 Body Position Estimate의 차이를 Position PI에 입력한다.

$p_{B,ref}=0$을 고정하는 방식은 참조 원점의 정지 위치를 유지하는 특수한 경우에만 사용한다.

---

### 3. Body Attitude Feedback

9축 IMU에서 얻은 Roll, Pitch, Yaw를 이용하여 현재 몸체 자세를 추정한다.

목표 자세와 현재 자세의 차이는 **PI Controller**로 보정한다.

Roll·Pitch 조종기 입력은 각속도로 적분하지 않고 각 축의 목표각 $\phi_{ref}$, $\theta_{ref}$로 바로 변환한다. Yaw 입력만 각속도 명령으로 사용하여 Heading Reference를 적분한다.

---

### 4. Final Body Twist

사용자 명령과 Body Position / Attitude Feedback을 합쳐 최종 몸체 선속도와 각속도를 만든다.

사용자 선속도는 Position Reference를 적분하는 목표 속도이자 Final Body Twist의 Feedforward 성분이다. Position Feedback은 목표 위치와 FK 기반 추정 위치 사이의 오차만 보정한다.

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

**Late Landing**이 발생하면 미접촉 발을 `-z_B` 방향으로 내리는 동시에 각 다리 장착점 방향으로 당겨 Search Down한다. 현재 Simulink 검증값은 하강 속도 `0.20 m/s`, 안쪽 이동 비율 `0.8`, 안쪽 이동 속도 `0.16 m/s`이다. 모든 Swing 다리가 접촉하기 전까지 다음 Tripod Phase로 넘어가지 않는다.

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
