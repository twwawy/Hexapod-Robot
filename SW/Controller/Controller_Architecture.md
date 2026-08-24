# 6족 로봇 보행 제어기 Architecture

6족 로봇의 수동 보행 및 몸체 안정화 제어 구조를 정의한다.

좌표계와 관절 방향은 좌표축 README의 정의를 따른다.

절대 좌표계 $\{W\}$는 지면에 고정하며, 모든 관절각이 0°인 최초 착지 상태의 몸체 원점을 절대 원점으로 사용한다. 그 자리에서 기본 자세로 일어서면 절대좌표계 기준 몸체 위치는 $[0,0,h_{stand}]^T$가 된다. 몸체 좌표계 $\{B\}$와 몸체 기준 좌표계 $\{R\}$는 로봇을 따라 이동한다. Body Position Estimator는 절대좌표계의 위치 ${}^W\hat p_B$를 추정하고, Position PI는 그중 x·y 위치 ${}^W\hat p_B^{xy}$와 목표 $p_{B,ref}^{xy}$를 사용한다.

---

## 전체 제어 구조

```text
조종기 입력
    │
    ▼
Dead Zone + LPF
    │
    ▼
사용자 Body Command 개별 신호
(v_x,user / φ_ref / θ_ref / ω_z,user / ψ_corr,ref)
    │
    ▼
연속 명령 및 동적 Motion Workspace 제한
    │
    ├──▶ 조종 모드 X·Y 속도 Pose Reference 적분
    │            ├──▶ x·y Position PI ◀─ FK + Contact
    │            └──▶ Gait Heading Yaw PI ◀─ IMU Yaw
    │                          │
    │                          ▼
    │                  Final Gait Body Twist
    │                          │
    │                          ▼
    │                 Tripod Gait Manager
    │                    ┌─────┴─────┐
    │                  Stance       Swing
    │                    └─────┬─────┘
    │                          ▼
    │                  Contact Adaptation
    │                          │
    │                          ▼
    │       STANCE Hold / Tripod 기준 발끝 목표 선택
    │                          │
    └──▶ 조종 Roll·Pitch / 보정 Yaw 단일 자세 PI ◀─ IMU 자세
                               │
                      자세 보정 각속도 적분
                               │
                               ▼
                  몸체 원점 기준 자세 역회전 적용
                               │
                               ▼
                몸체 좌표계 {B} → 다리 좌표계 {L_i}
                               │
                               ▼
                  다리별 Foot Workspace Limiter
                               │
                               ▼
                            3DOF IK
                               │
                               ▼
                  18개 관절 목표각 → Joint Rate Limiter
                               │
                               ▼
                         Servo Output
```

---

## 블록별 설명

### 1. 조종기 입력

조종기 입력으로 로봇의 x·y 이동 속도, Roll·Pitch 목표각과 Yaw 각속도를 명령한다. 스로틀 raw 범위는 중립을 0으로 한 `-1000~1000`으로 사용한다.
입력에는 **Dead Zone과 Low Pass Filter**를 적용한다.

| 입력 | 명령 |
|---|---|
| Throttle | 부호가 있는 x축 전진·후진 선속도 |
| Roll | Roll 목표각 |
| Pitch | Pitch 목표각 |
| Yaw, SA OFF | Yaw 각속도 |
| Yaw, SA ON | y축 좌·우 선속도 |
| SA | Yaw 회전 모드와 Yaw 고정 x·y 이동 모드 선택 |

현재 최대 명령은 x·y 이동 속도 각각 `0.28 m/s`, Roll·Pitch 목표각 `±45°`, Yaw 각속도 `±45°/s`로 설정한다.

실제 RC 짐벌 입력은 시간에 따라 연속적으로 변하는 신호로 취급한다. Simulink `USER` 시험 신호는 raw 입력에 `5000/s` Ramp 제한을 적용하여 `-1000→1000`이 `0.4 s`, 중립→최대가 `0.2 s`에 변하게 한다. 강화학습 제어기도 RC와 동일한 Throttle, Roll, Pitch, Yaw 스칼라 입력 포트를 사용하며 Step Action을 발끝 위치나 관절각으로 직접 전달하지 않는다. 강화학습 출력은 **반드시 RC 짐벌과 동일한 Ramp 변화율 제한을 먼저 통과**시킨 뒤 LPF에 입력한다. 학습 정책의 Step 출력을 제어기에 직접 연결하지 않는다.

Simulink 구현에서는 여러 명령, 6개 발 위치 또는 18개 관절각을 행렬·벡터 입출력으로 묶지 않는다. 각 축, 각 다리의 x·y·z와 각 관절을 독립된 스칼라 포트로 연결한다.

Yaw는 **Heading Hold**를 사용한다.

위 정의는 조종 모드에 적용한다. 보정 모드에서는 Yaw 입력을 각속도로 사용하지 않고 yaw 보정 목표각에 직접 대응시킨다. 보정 모드의 Yaw 입력은 적분하지 않으며 조이스틱이 중앙으로 복귀하면 보정 목표각도 0°로 복귀한다.

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
|v_{y,user}|\ge V_{gait,th}
\quad\text{or}\quad
|\omega_{z,user}|\ge \Omega_{gait,th}
$$

READY 상태만으로는 `Tripod_Enable`을 활성화하지 않는다.

`Body_Control_Enable`과 `Tripod_Enable`은 서로 다른 역할을 가진다. `Body_Control_Enable`은 Final Gait Body Twist, Stance 궤적과 몸체 자세 오버레이 경로를 허용하고, `Tripod_Enable`은 Tripod 위상 진행과 Swing 궤적만 허용한다. READY에서 `Motion_Armed=1`이면 조종 모드와 보정 모드의 `Body_Control_Enable`을 활성화한다. 실제 x·y 이동 또는 Yaw 회전 명령이 없으면 `Tripod_Enable=0`으로 두고 6개 다리를 모두 STANCE로 유지한다.

---

### 2. Body Position Feedback

STANCE 상태이면서 실제로 CONTACT가 확인된 다리의 관절각과 Forward Kinematics를 이용하여 몸체 위치를 추정한다.

각 유효 Stance 다리가 계산한 몸체 위치 후보끼리 3차원 거리를 비교한다. 다른 정상 후보가 `0.05 m` 이내에 없는 고립 후보는 Stance Foot Slip 의심 다리로 보고 즉시 평균에서 제외한다.

Slip 의심이 200 Hz 제어 주기에서 5회 연속 발생하면 해당 다리의 Slip 상태를 확정한다. 확정 상태는 다리가 STANCE 또는 CONTACT 조건에서 벗어날 때 해제하며, 확정된 다리는 다음 Stance Anchor가 생성될 때까지 위치 추정에 사용하지 않는다.

Foot Slip Reject를 통과한 Stance 다리의 위치 추정값만 **단순 평균**한다. 압력센서 값은 접촉 여부 판단에만 사용하며, Slip 다리 정보는 Leg 1~6에 대응하는 6비트 `Slip_Leg_Mask`로 출력한다.

Body Position Estimator는 절대좌표계의 x·y·z 위치를 추정하지만, 현재 Position PI에는 x·y 위치만 사용한다. z 위치는 추정과 검증에만 사용하고 Position PI에는 입력하지 않는다.

조종 모드에서 몸체 좌표계의 x·y 선속도를 현재 Yaw Reference로 절대좌표계에 회전한 뒤 200 Hz로 적분하여 x·y 목표 위치를 생성한다. SA OFF에서는 $v_{y,user}=0$이고, SA ON에서는 Yaw 짐벌이 $v_{y,user}$를 만든다.

$$
p_{B,ref}^{xy}[k]
=
p_{B,ref}^{xy}[k-1]
+
\begin{bmatrix}
\cos\psi_{ref}[k] & -\sin\psi_{ref}[k]\\
\sin\psi_{ref}[k] & \cos\psi_{ref}[k]
\end{bmatrix}
\begin{bmatrix}
v_{x,user}[k]\\
v_{y,user}[k]
\end{bmatrix}T_s
$$

Position Reference Generator 활성화 시에는 $p_{B,ref}^{xy}[k_0]={}^W\hat p_B^{xy}[k_0]$로 초기화한다. x·y 목표 위치와 FK·Stance Anchor로 계산한 x·y Body Position Estimate의 차이를 Position PI에 입력한다.

Position PI는 조종 모드에서만 사용한다. 보정 모드의 x·y·z 이동속도와 암 모드 입력에는 Position PI를 적용하지 않는다.

LANDED, STANDING, LANDING, CORRECTION, ARM, FAULT, KILL 상태에서는 Position PI 출력을 0으로 만들고 적분항을 누적하지 않는다. 조종 모드 진입 시 x·y 목표 위치를 현재 추정 위치로 맞추고 적분항을 초기화한다.

---

### 3. Gait Heading 및 Body Posture Feedback

9축 IMU에서 얻은 Roll, Pitch, Yaw를 이용하여 현재 몸체 자세를 추정한다.

조종 모드 Yaw는 SA OFF에서 보행 방향 제어에 사용한다. Yaw 각속도 명령을 적분하여 $\psi_{ref}$를 만들고, IMU Yaw와 비교한 Gait Heading Yaw PI 출력 $\omega_{z,heading,fb}$를 사용자 Yaw 각속도에 더한다. SA ON이 되면 현재 IMU Yaw를 $\psi_{ref}$로 다시 잡고, 사용자 Yaw 각속도를 0으로 두며 Yaw 짐벌을 y축 속도로 사용한다. 따라서 헤딩 PI의 보정 회전만 허용하여 방향을 유지한다.

조종 모드의 Roll·Pitch와 보정 모드의 Yaw는 보행 경로와 분리된 몸체 자세 제어에 사용한다. 목표 자세와 IMU 자세의 오차를 하나의 **단일 자세 PI**에 입력하여 자세 보정 각속도 $\omega_{posture}$를 생성한다. 내부 각속도 PI를 추가한 이중 PI 구조는 사용하지 않는다.

자세 보정 각속도는 200 Hz로 적분하고 Saturation과 Rate Limiter를 적용하여 몸체 원점 기준 자세 명령 $\eta_{posture,cmd}$를 만든다. 이 자세 명령은 Final Gait Body Twist, Tripod 위상, 보폭 또는 Swing 착지점 계산에 넣지 않는다.

먼저 `Tripod_Enable`에 따라 기준 좌표계 $\{R\}$의 발끝 목표를 선택한다. `Tripod_Enable=1`이면 Tripod와 Contact Adaptation이 만든 Stance/Swing 목표를 사용하고, `Tripod_Enable=0`이면 6개 다리의 기본 또는 직전 안전 STANCE 목표를 계속 출력한다. 이후 선택된 기준 발끝 목표에 몸체 자세의 역회전을 적용한다.

$$
{}^Bp_{F_i}^{cmd}
=
({}^RR_B(\eta_{posture,cmd}))^T
{}^Rp_{F_i}^{base}
$$

두 좌표계는 같은 몸체 원점을 사용하므로 평행이동 없이 원점 기준 회전만 적용된다. STANCE 발은 지면에 고정되고, SWING 발의 명목 보행 궤적과 착지점도 그대로 유지된다.

LANDED, STANDING, LANDING, FAULT, KILL 상태에서는 자세 PI 출력과 자세 오버레이를 0으로 만들고 적분항을 누적하지 않는다. 각 PI가 다시 활성화될 때는 목표 자세를 해당 모드의 진입 기준에 맞추고 적분항을 초기화한다.

---

### 4. Final Gait Body Twist

조종 모드에서는 사용자 x·y 선속도·Yaw 각속도 Feedforward와 x·y Position PI 및 Gait Heading Yaw PI Feedback을 합쳐 보행용 Body Twist를 만든다.

사용자 선속도는 Position Reference를 적분하는 목표 속도이자 Final Gait Body Twist의 Feedforward 성분이다. Position Feedback은 목표 위치와 FK 기반 추정 위치 사이의 오차만 보정한다.

보정 모드에서는 x·y·z 이동속도 명령을 PI 없이 직접 사용한다. 보정 모드 Yaw는 Final Gait Body Twist에 더하지 않고 몸체 자세 오버레이에만 사용한다. 암 모드에는 Position PI와 자세 PI를 적용하지 않는다.

```text
User Command
    +
Position Feedback
    +
Gait Heading Yaw Feedback
    ↓
Final Gait Body Twist
```

---

### 5. Tripod Gait Manager

기본 보행은 **Tripod Gait**를 사용한다.

`Tripod_Enable=1`일 때만 Tripod 위상과 Swing 진행률을 갱신한다. `Tripod_Enable=0`은 발끝 목표 출력을 0으로 만드는 조건이 아니다. `Body_Control_Enable=1`, `Tripod_Enable=0`이면 Tripod 위상을 정지하고 6개 다리의 기본 또는 직전 안전 STANCE 목표를 계속 출력한다. 따라서 정지 상태에서도 Position PI, 자세 PI 또는 보정 모드의 Body 명령이 작동한다. 자세 오버레이는 `Tripod_Enable`과 독립적으로 동작하므로 보행 중에도 Roll·Pitch 또는 보정 Yaw 명령이 Tripod 궤적을 바꾸지 않는다. 두 Enable이 모두 0이면 보행용 Body Twist와 발끝 궤적 갱신을 정지한다.

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

방사 방향 오프셋은 Swing 시작점과 착지점에서는 0이고 최고점에서 최대가 되며, 발끝을 각 다리의 장착 방향 바깥쪽으로 이동시켜 IK 관절각을 줄인다. 현재 Swing Height 기본값은 `0.20 m`, 제한 범위는 `0.15~0.25 m`이며 방사 방향 오프셋은 `0.07 m`이다.

---

### 8. Early / Late Landing

**Early Landing**이 발생하면 해당 다리는 즉시 STANCE로 전환한다.

**Late Landing**이 발생하면 미접촉 발을 기준 좌표계의 `-z_R` 방향으로 내리는 동시에 각 다리 장착점 방향으로 당겨 Search Down한다. 현재 Simulink 검증값은 하강 속도 `0.20 m/s`, 안쪽 이동 비율 `0.8`, 안쪽 이동 속도 `0.16 m/s`이다. 모든 Swing 다리가 접촉하기 전까지 다음 Tripod Phase로 넘어가지 않는다.

---

### 9. 좌표변환 및 IK

`Tripod_Enable`에 따라 선택한 6개 STANCE Hold 또는 Tripod 기준 발끝 목표에 몸체 자세 오버레이의 역회전을 적용하여 몸체 좌표계 $\{B\}$로 변환한다. 이후 각 다리 좌표계로 변환한 뒤 **3DOF Inverse Kinematics**를 이용해 18개 관절 목표각을 계산한다.

Roll·Pitch 목표각은 `±45°` 범위를 유지하되, 매 주기 생성한 후보 자세에서 6개 다리가 모두 IK 가능한 경우에만 실제 자세 명령을 갱신한다. Roll, Pitch, Yaw는 결합된 후보 자세에 공통 적용 여부를 사용하므로 후보가 작업공간 밖이면 이번 주기의 세 축 증분을 모두 0으로 만들고 직전 가능 자세를 유지한다. 결합된 후보가 다시 작업공간 안쪽을 향하면 세 축 증분을 즉시 허용한다. RC 입력과 자세 Rate Limiter가 만드는 작은 주기별 증분을 사용하므로 최대 가능각을 찾는 이분 탐색은 사용하지 않는다.

확인용 출력은 `Posture_Command_Accepted`로 명명한다. `1`은 이번 자세 명령 후보 채택, `0`은 후보 거부와 직전 채택 명령 유지를 뜻한다.

자세 후보 생성과 6개 다리 작업공간 검사는 `BodyPosturePIOverlay` MATLAB Function 내부에서 함께 수행한다. 별도의 자세 제한 블록이나 자세 출력 피드백 선은 추가하지 않는다.

x·y 이동 속도와 Yaw 회전 속도는 다음 보행 주기의 전체 발끝 궤적을 미리 검사하여 가능한 보폭과 회전량까지만 증가시킨다. 세 명령이 동시에 들어오면 동일한 적용 여부를 사용하여 이동 방향과 회전 반경을 유지한다. 한 걸음 `0.5 s`에서 한 축 최대 `0.28 m/s` 명령은 보폭 `0.14 m`에 해당한다. 현재 자세나 회전 명령 때문에 마지막 가능 보폭이 `0.12 m`이면 적용 속도는 `0.24 m/s`가 된다.

Position Reference와 Heading Reference에는 요청값이 아니라 작업공간 검사를 통과한 실제 적용 x·y 선속도와 Yaw 각속도를 적분한다. 자세 후보가 거부된 동안에는 자세 명령 상태와 자세 PI 적분항이 바깥 방향으로 누적되지 않게 한다.

보정 모드의 x·y·z 속도는 다음 위치 후보가 작업공간 밖이면 바깥 방향 적용 속도를 0으로 만들고 위치 적분 상태도 갱신하지 않는다. 작업공간 안쪽으로 복귀시키는 명령은 즉시 허용한다.

각 `Body2Leg` 출력과 `LegIK` 입력 사이에는 작업공간 여유 `0.0001 m`를 사용하는 최종 `FootWorkspaceLimiter`를 둔다. 이 블록은 상위 동적 명령 제한 뒤에 배치하며 수치 오차만 보정한다.

시간 기반 모드 전환 보간은 사용하지 않는다. 모드 전환 시 새 제어 경로의 상태를 직전 출력에 맞추고, 보행 정지 시 진행 중인 Swing 다리를 착지시킨 뒤 직전 안전 STANCE 목표를 유지하여 출력 연속성을 확보한다.

모든 관절의 사용 범위는 **-135° ~ +135°**이다.

---

### 10. 안전장치

이 절의 Safety 판단과 출력 보호는 후속 구현 대상으로 둔다. 최초 STM32 구현에서는 두 Fault를 0으로 유지하고 관절 명령 상태만 `0°`로 초기화한다.

Servo 출력 전에 다음 항목을 검사한다.

- ROLLOVER FAULT
- IK Workspace
- 관절 각도 범위
- Safe Support Polygon
- 비정상 목표각 Jump
- 관절 최대 각속도 초과 여부

Safe Support Polygon은 실제 지지다각형을 중심 기준으로 **10% 축소한 영역**을 사용하며, 몸체 원점이 이 영역을 벗어나면 보행을 중단한다.

비정상적인 Joint Jump가 검출되면 새 명령을 적용하지 않고 **직전 관절각을 HOLD**한다.

IK 해가 없다는 검출은 기구학 계산에서 수행하고, 검출 즉시 로봇을 정지시키는 결정은 Safety가 `Controller_Fault`를 발생시키는 방식으로 나중에 구현한다.

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
| 관절 명령 각속도 제한 | 315.8°/s |
| 5 ms당 최대 관절 명령 변화량 | 1.579° |

관절의 0° 방향은 좌표축 README의 정의를 따른다.

Joint Rate Limiter는 Safety Fault와 별개의 상시 출력 보호이다. 18개 IK 출력 각각에 독립된 Discrete Rate Limiter를 적용한 뒤 서보 방향·중립점 보정과 PWM 변환을 수행한다.

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
| IK Workspace Margin | 0.0001 m |
| Joint Rate Limit | 315.8°/s, 5 ms당 1.579° |
| Servo | DS51150-270 |
