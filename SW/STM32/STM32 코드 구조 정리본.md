# STM32 코드 구조

## 1. 목적

STM32 사용자 코드를 기능별 폴더로 나누고 `main.c`에는 초기화, 반복 처리와 HAL 콜백 전달을 위한 함수 호출만 남긴다.

제어 로직은 [Controller Architecture](../Controller/Controller_Architecture.md)와 [Controller detail](../Controller/Controller_detail.md)의 현재 동작을 기준으로 옮겼다. 이 문서는 현재 구현된 파일의 책임과 참조 관계를 정리한다.

## 2. 기본 원칙

- 모든 사용자 헤더와 소스는 기능별 폴더 안에 둔다.
- `Core/Inc`와 `Core/Src`의 폴더 구조를 동일하게 유지한다.
- 하나의 파일에는 서로 강하게 연결된 한 책임 영역을 넣는다. 함수마다 파일을 만들지는 않는다.
- 파일 내부 보조 함수는 해당 `.c` 파일의 `static` 함수로 둔다.
- 여러 제어기를 다시 합친 `controller.h/.c`는 만들지 않는다.
- 공통 자료형은 `robot_types.h`, 공통 설정값은 `robot_config.h`에서 관리한다.
- CubeMX 생성 파일과 HAL 파일은 기존 위치를 유지한다.
- 5 ms 제어 계산은 인터럽트 안에서 직접 수행하지 않는다. TIM6 콜백은 실행 요청만 남긴다.
- UART 인터럽트는 수신 바이트 저장과 다음 수신 시작만 담당한다.
- 실제 운용 빌드에는 `test` 폴더의 입력 생성 코드를 포함하지 않는다.

## 3. 최종 폴더 구조

```text
Core
├─ Inc
│  ├─ common
│  │  ├─ robot_config.h
│  │  └─ robot_types.h
│  ├─ app
│  │  └─ hexapod_app.h
│  ├─ sensor
│  │  ├─ gps.h
│  │  ├─ imu.h
│  │  ├─ mcp3008.h
│  │  ├─ joint_feedback.h
│  │  ├─ foot_pressure.h
│  │  └─ sensor_manager.h
│  ├─ user_command
│  │  ├─ crsf_receiver.h
│  │  ├─ crsf_protocol.h
│  │  └─ user_command.h
│  ├─ high_control
│  │  ├─ control_priority.h
│  │  ├─ drone_controller.h
│  │  ├─ stand_landing.h
│  │  ├─ body_position_estimator.h
│  │  ├─ gait_pose_controller.h
│  │  ├─ body_posture_controller.h
│  │  ├─ workspace_limiter.h
│  │  ├─ gait_manager.h
│  │  ├─ foot_trajectory.h
│  │  ├─ stance_trajectory.h
│  │  ├─ swing_trajectory.h
│  │  ├─ contact_adaptation.h
│  │  ├─ leg_kinematics.h
│  │  └─ safety.h
│  ├─ low_control
│  │  ├─ servo_pwm.h
│  │  └─ relay.h
│  ├─ communication
│  │  ├─ lora.h
│  │  ├─ robot_telemetry.h
│  │  └─ jetson_spi.h
│  └─ test
│     ├─ test_runner.h
│     ├─ sensor_test.h
│     ├─ imu_calibration_test.h
│     ├─ low_control_test.h
│     ├─ joint_sensor_calibration_test.h
│     ├─ foot_pressure_calibration_test.h
│     ├─ servo_relay_calibration_test.h
│     ├─ crsf_calibration_test.h
│     ├─ rc_command_generator.h
│     ├─ user_command_test.h
│     ├─ leg6_test.h
│     ├─ kinematics_test.h
│     ├─ workspace_test.h
│     ├─ gait_test.h
│     ├─ mode_transition_test.h
│     ├─ controller_test.h
│     ├─ safety_test.h
│     └─ communication_test.h
└─ Src
   ├─ app
   │  └─ hexapod_app.c
   ├─ sensor
   │  ├─ gps.c
   │  ├─ imu.c
   │  ├─ mcp3008.c
   │  ├─ joint_feedback.c
   │  ├─ foot_pressure.c
   │  └─ sensor_manager.c
   ├─ user_command
   │  ├─ crsf_receiver.c
   │  ├─ crsf_protocol.c
   │  └─ user_command.c
   ├─ high_control
   │  ├─ control_priority.c
   │  ├─ drone_controller.c
   │  ├─ stand_landing.c
   │  ├─ body_position_estimator.c
   │  ├─ gait_pose_controller.c
   │  ├─ body_posture_controller.c
   │  ├─ workspace_limiter.c
   │  ├─ gait_manager.c
   │  ├─ foot_trajectory.c
   │  ├─ stance_trajectory.c
   │  ├─ swing_trajectory.c
   │  ├─ contact_adaptation.c
   │  ├─ leg_kinematics.c
   │  └─ safety.c
   ├─ low_control
   │  ├─ servo_pwm.c
   │  └─ relay.c
   ├─ communication
   │  ├─ lora.c
   │  ├─ robot_telemetry.c
   │  └─ jetson_spi.c
   └─ test
      ├─ test_runner.c
      ├─ sensor_test.c
      ├─ imu_calibration_test.c
      ├─ low_control_test.c
      ├─ joint_sensor_calibration_test.c
      ├─ foot_pressure_calibration_test.c
      ├─ servo_relay_calibration_test.c
      ├─ crsf_calibration_test.c
      ├─ rc_command_generator.c
      ├─ user_command_test.c
      ├─ leg6_test.c
      ├─ kinematics_test.c
      ├─ workspace_test.c
      ├─ gait_test.c
      ├─ mode_transition_test.c
      ├─ controller_test.c
      ├─ safety_test.c
      └─ communication_test.c
```

## 4. Simulink 코드 반영 범위

### 4.1 STM32에 옮기는 기능

| Simulink 기능 | STM32 파일 |
|---|---|
| `ControlPriorityManager` | `high_control/control_priority.*` |
| `DroneController` | `high_control/drone_controller.*` |
| `StandLandingFootDelta` | `high_control/stand_landing.*` |
| `BodyPositionEstimator`와 Stance Foot Slip | `high_control/body_position_estimator.*` |
| `GaitPosePI` | `high_control/gait_pose_controller.*` |
| `BodyPosturePIOverlay` | `high_control/body_posture_controller.*` |
| `GaitWorkspaceGuard`와 자세·보정 명령 작업공간 검사 | `high_control/workspace_limiter.*` |
| `TripodGaitManager` | `high_control/gait_manager.*` |
| `TripodFootTrajectory` 조합 | `high_control/foot_trajectory.*` |
| Stance 계산 | `high_control/stance_trajectory.*` |
| Swing과 Bezier 계산 | `high_control/swing_trajectory.*` |
| Early/Late Landing | `high_control/contact_adaptation.*` |
| `BaseFootPosition`, FK, Body↔Leg 변환과 IK | `high_control/leg_kinematics.*` |
| `SafetyEvaluator` | `high_control/safety.*` |
| `JointRateLimiter` | `low_control/servo_pwm.*` |
| `RelaySafetyGate` | `low_control/relay.*` |

### 4.2 STM32에 옮기지 않는 기능

| Simulink 시험 기능 | STM32 처리 |
|---|---|
| `Plant/VirtualIMU` | 옮기지 않는다. `sensor/imu.*`가 WT931 실측값을 제공한다. |
| `Plant/TestContact` | 옮기지 않는다. `sensor/foot_pressure.*`가 실제 압력센서 접촉값을 제공한다. |
| `USER` 시험 시퀀스 | 운용 코드에 옮기지 않는다. 실제 입력은 CRSF이며 임시 입력은 `test/rc_command_generator.*`에만 둔다. |
| Simscape Plant와 관절별 시뮬레이션 PID | 옮기지 않는다. 실제 관절 출력은 서보 PWM으로 수행한다. |
| Step, Ramp, Constant 등 모델 시험 신호원 | 옮기지 않는다. 필요한 시험값은 각 테스트 함수의 명시적인 입력으로 전달한다. |

가상 IMU와 가상 접촉센서를 별도 `.h/.c`로 만들지 않는다. 테스트 코드에서도 센서처럼 지속적으로 가짜 값을 발행하지 않으며, 순수 계산 함수 검증이 필요할 때만 테스트 함수가 고정된 입력 구조체를 직접 전달한다.

## 5. 헤더 참조 구조

### 5.1 폴더 간 참조 방향

```text
main.c ───────────────▶ app/hexapod_app.h

app/hexapod_app.c ────┬▶ sensor
                      ├▶ user_command
                      ├▶ high_control
                      ├▶ low_control
                      └▶ communication

sensor ───────────────┐
user_command ─────────┤
high_control ─────────┼▶ common
low_control ──────────┤
communication ────────┘

test/*.c ─────────────▶ 시험 대상 모듈
```

`common`은 다른 프로젝트 폴더를 참조하지 않는다. `main.c`는 `app/hexapod_app.h`만 직접 포함하며, 개별 모듈의 호출 순서와 전체 조합은 `hexapod_app.c`에서 수행한다.

### 5.2 high_control 내부 참조

공개 헤더는 가능한 한 `common/robot_types.h`와 `common/robot_config.h`만 포함한다. 아래 의존성은 주로 `.c` 파일에서 사용한다.

```text
control_priority ────────────────▶ common
drone_controller ────────────────▶ control_priority
stand_landing ───────────────────▶ common
body_position_estimator ─────────▶ leg_kinematics
gait_pose_controller ────────────▶ common
body_posture_controller ─────────▶ workspace_limiter
gait_manager ────────────────────▶ common
foot_trajectory ────────┬────────▶ stance_trajectory
                        ├────────▶ swing_trajectory
                        ├────────▶ contact_adaptation
                        └────────▶ workspace_limiter
workspace_limiter ──────┬────────▶ leg_kinematics
                        ├────────▶ stance_trajectory
                        └────────▶ swing_trajectory
safety ──────────────────────────▶ common
```

`workspace_limiter`는 보행 속도, 몸체 자세와 보정 위치 명령이 6개 다리의 IK 범위를 벗어나는지 미리 검사한다. `leg_kinematics`는 한 발의 좌표변환과 최종 IK 유효성만 판단하며 상위 명령 유지 여부를 결정하지 않는다.

`safety`와 `control_priority`는 서로 직접 포함하지 않는다. `hexapod_app.c`가 Safety 결과를 Priority 입력으로 전달하여 순환 참조를 막는다.

## 6. 공통과 애플리케이션 파일

### `common/robot_config.h`

다리 수, 관절 수, 5 ms 주기, 기구 치수, 작업공간 여유, 제어 Gain과 출력 한계 같은 공통 설정값을 정의한다. 핀과 HAL Handle은 CubeMX 생성 파일의 정의를 사용한다.

관절 ADC, 압력센서, 서보와 CRSF의 채널별 보정 테이블은 해당 기능 모듈에 둔다. 아직 측정하지 않은 값을 공통 설정에 임의로 넣지 않는다.

### `common/robot_types.h`

3차원 벡터, 자세, Body Twist, 센서 스냅샷, 사용자 명령, 여섯 발 상태, 18개 관절각, 제어 모드와 Fault 결과처럼 여러 모듈이 공유하는 가벼운 자료형을 정의한다.

### `app/hexapod_app.h`

전체 사용자 코드의 진입점만 공개한다. `main.c`에서는 다음 수준의 함수만 호출한다.

```text
HexapodApp_Init(...)
HexapodApp_Process()
HexapodApp_UartRxCallback(...)
HexapodApp_UartErrorCallback(...)
HexapodApp_TimerCallback(...)
```

센서, 제어기, 출력과 통신 모듈의 실제 호출 순서는 `hexapod_app.c`에 둔다.

## 7. 센서 파일

### `sensor/gps.h`

USART2에서 NEO-M8N 프레임을 수신하고 위치와 GPS 유효 상태를 제공한다.

### `sensor/imu.h`

USART3, 115200 baud로 WT931 프레임을 수신하고 가속도, 각속도와 Roll·Pitch·Yaw를 제공한다. 좌표축은 +X 전진, +Y 왼쪽, +Z 위쪽으로 변환한다.

### `sensor/mcp3008.h`

SPI1에 연결된 MCP3008 세 개의 24개 raw 채널을 읽는다. 값의 물리 단위 변환은 담당하지 않는다.

### `sensor/joint_feedback.h`

관절센서 raw 값 18개를 관절각으로 변환한다. 채널, ADC 최소·영점·최대, 각도 범위와 방향 테이블을 관리한다.

### `sensor/foot_pressure.h`

압력센서 raw 값 6개를 접촉 상태로 변환한다. 채널, 접촉 진입 임계값과 접촉 해제 임계값 테이블을 관리한다.

### `sensor/sensor_manager.h`

각 센서의 최신 실제 측정값을 모아 하나의 `SensorSnapshot`을 만든다. 센서값을 임의로 생성하거나 누락된 값을 가상값으로 대체하지 않는다.

## 8. 조종기 파일

### `user_command/crsf_receiver.h`

USART6 수신 바이트와 링 버퍼를 관리한다. 프레임 파싱은 담당하지 않는다.

### `user_command/crsf_protocol.h`

CRSF 프레임 길이, CRC와 16개 채널 unpack을 처리한다. UART와 핀에는 의존하지 않는다.

### `user_command/user_command.h`

CRSF 채널을 프로젝트 명령으로 변환한다. 네 짐벌 축은 보정 테이블을 거쳐 중립 0, `-1000~1000`으로 정규화하고 SA~SE를 스위치 상태로 변환한다.

마지막 정상 CRSF 프레임 이후 100 ms가 지나면 이동 명령을 0으로 만들고 보행을 정지한다. 재연결 후에는 네 축이 0.2 s 동안 중립일 때 다시 동작을 허가한다. 상세 채널과 시험 입력은 [드론 조종기 입력](../Controller/드론%20조종기%20입력/README.md)을 따른다.

## 9. 상위 제어 파일

### `high_control/control_priority.h`

현재 Fault와 사용자 스위치를 받아 KILL, FAULT, LANDING, STANDING, READY, MANUAL과 CORRECTION의 우선순위를 결정한다. ARM과 AUTONOMOUS 자리는 후속 구현을 위해 모드 자료형에 유지한다.

### `high_control/drone_controller.h`

선택된 모드를 실제 Enable과 사용자 명령으로 변환한다. 모드 전환 시 짐벌 필터 상태를 초기화하고 Motion Armed, Heading 고정, 서기·착지 진행 상태를 관리한다.

### `high_control/stand_landing.h`

5.6 s 서기와 착지 동안 기본 발 위치에 적용할 높이 변화를 계산한다. 완료 상태는 `drone_controller`가 진행률과 안정 시간을 기준으로 관리한다.

### `high_control/body_position_estimator.h`

접촉 중인 STANCE 다리의 관절각과 FK로 몸체 X/Y/Z 위치를 추정한다. 0.05 m 고립 후보와 5회 연속 조건을 사용하는 Stance Foot Slip 검출도 이 파일에서 처리한다. Slip은 진단 상태이며 현재 Safety Fault로 직접 변환하지 않는다.

### `high_control/gait_pose_controller.h`

Position PI, Heading PI와 사용자 이동 명령을 결합하여 Body Twist 후보를 만든다. 최대 x·y 속도는 각 0.28 m/s, 최대 Yaw 속도는 45 deg/s이다.

### `high_control/body_posture_controller.h`

Roll·Pitch와 보정 Yaw의 자세 PI를 계산하고 발끝 목표에 몸체 자세 역회전을 적용한다. 자세 명령의 작업공간 채택 여부는 `workspace_limiter`에 요청한다.

### `high_control/workspace_limiter.h`

현재 발 배치에서 6개 다리가 모두 IK 가능한지 미리 확인하고 다음 세 명령을 동적으로 제한한다.

- 보행 x·y 속도와 Yaw 속도
- Roll·Pitch와 보정 Yaw 자세
- 보정 모드 X/Y/Z 이동

후보가 유효하면 새 명령을 채택하고 유효하지 않으면 직전에 채택한 명령을 유지한다. 최종 IK 앞의 0.0001 m 수치 여유 검사는 `leg_kinematics`가 한 번 더 수행한다.

### `high_control/gait_manager.h`

Tripod A와 B의 STANCE/SWING 상태, 진행률과 0.5 s Phase를 관리한다. 모드 전환과 접촉 적응 중 발생하는 한 주기 Enable 변화에도 발 상태가 초기화되지 않도록 한다.

### `high_control/foot_trajectory.h`

여섯 다리의 발끝 궤적 계산 순서와 최종 결과 조합을 담당한다. Stance, Swing과 접촉 적응의 세부 수식은 하위 파일에 맡긴다.

### `high_control/stance_trajectory.h`

STANCE 발이 지면의 같은 위치를 유지하도록 몸체 이동과 회전의 반대 방향으로 발끝 목표를 갱신한다.

### `high_control/swing_trajectory.h`

착지점, Quintic 시간 스케일링, Bezier 궤적과 방사 오프셋을 계산한다. Swing Height 기본값은 0.20 m이며 0.15~0.25 m 범위에서 사용한다.

### `high_control/contact_adaptation.h`

Swing 진행률 50% 이후의 Early Landing과 정상 Swing 종료 후 Late Landing을 처리한다. Late Landing 하강 속도는 0.20 m/s이다.

### `high_control/leg_kinematics.h`

기본 발 위치, Body↔Leg 좌표변환, 공통 3DOF FK/IK와 한 발의 작업공간 유효성 검사를 제공한다. 여섯 다리를 개별 파일로 만들지 않고 다리 번호와 기구 파라미터를 입력으로 받는다.

### `high_control/safety.h`

현재 Simulink `SafetyEvaluator`와 동일하게 다음 Fault를 Latch한다.

- `Rollover_Fault`: 유효한 Roll 또는 Pitch 절댓값이 80 deg 이상이다.
- `Controller_Fault`: IMU 값이 유한수가 아니거나 6개 다리 중 하나라도 IK Invalid이다.

한 번 Fault가 발생하면 Reset 없이 유지한다. 결과는 `control_priority`와 `drone_controller`를 거쳐 Kill을 활성화하고 `relay`가 서보 전원 릴레이를 모두 끄게 한다. CRSF 연결 끊김은 Safety Fault가 아니라 `user_command`의 입력 Failsafe로 처리한다.

## 10. 하위 제어 파일

### `low_control/servo_pwm.h`

18개 관절 목표각을 DS51150-270 PWM으로 변환한다. 채널, 방향, 중립 Pulse와 최소·최대 Pulse 테이블을 관리한다.

관절각 범위 제한 후 315.8 deg/s, 즉 5 ms당 1.579 deg의 관절 명령 속도 제한을 적용한다. 그 다음 서보 보정과 500~2500 us Pulse 제한을 적용한다.

### `low_control/relay.h`

INA1~INC2 여섯 Active High 출력을 관리한다. 초기 상태는 OFF이며 Kill이 활성화되면 모든 릴레이를 즉시 끈다. 릴레이-다리 대응은 `servo_relay_calibration_test`에서 기록하며 현재 운용 Kill은 여섯 채널을 동시에 차단하므로 개별 다리 매핑에 의존하지 않는다.

## 11. 통신 파일

### `communication/lora.h`

UART5 115200 baud에 연결된 RYLR998의 AT 명령, 주소·Network 설정, 송수신 파싱과 전송을 담당한다.

### `communication/robot_telemetry.h`

관제탑에 필요한 STATUS, JOINT와 GPS 패킷을 구성한다. 디버그용 ADC raw와 내부 제어값은 포함하지 않는다.

### `communication/jetson_spi.h`

SPI2 Slave의 최소 인터페이스 자리만 유지한다. 명령, 프레임, CRC, 타임아웃과 DRDY 동작은 프로토콜이 확정된 뒤 설계한다.

## 12. 테스트 파일

테스트 파일은 실제 운용 경로와 분리한다. 한 번에 하나의 테스트만 실행하고 액추에이터 테스트는 기본적으로 릴레이 OFF에서 시작한다.

### `test/test_runner.h`

선택한 테스트 하나의 초기화와 반복 실행만 담당한다. 정상 운용 빌드에서는 테스트 모듈을 호출하지 않는다.

### `test/sensor_test.h`

GPS, WT931과 MCP3008 24채널 raw 값, 수신 주기와 오류 횟수를 확인한다. 가상 센서값은 만들지 않는다.

### `test/imu_calibration_test.h`

로봇을 수평으로 정지한 상태에서 WT931 Roll·Pitch·Yaw 평균 Offset을 기록한다. +X 전진, +Y 왼쪽, +Z 위쪽이 되도록 확인한 가속도·각속도·자세 축별 부호와 함께 `imu` 보정값을 만든다.

### `test/low_control_test.h`

릴레이 여섯 채널과 PWM 18채널의 전기적 출력만 확인한다. 서보 전원과 기구를 분리하고 오실로스코프로 먼저 검사한다.

### `test/joint_sensor_calibration_test.h`

18개 관절센서의 ADC 최소·영점·최대, 각도 범위와 방향을 측정하여 `joint_feedback` 설정 테이블을 완성한다.

### `test/foot_pressure_calibration_test.h`

6개 압력센서의 무부하와 접촉 raw 값을 반복 측정하여 접촉·해제 임계값을 정하고 `foot_pressure` 설정 테이블을 완성한다.

### `test/servo_relay_calibration_test.h`

한 번에 서보 한 채널과 릴레이 한 채널만 동작시킨다. 서보 방향·중립점·Pulse 범위와 릴레이-다리 대응 관계를 확인하여 두 설정 테이블을 완성한다.

### `test/crsf_calibration_test.h`

실제 조종기가 준비되면 CH1~CH10 raw 최소·중립·최대, 방향과 스위치 위치를 기록하여 CRSF 보정 테이블을 완성한다.

### `test/rc_command_generator.h`

조종기가 준비되기 전까지만 프로젝트 명령 범위의 임시 입력을 만든다. CRSF 프레임이나 센서값은 흉내 내지 않는다.

네 짐벌 축은 `-1000~1000`, 변화율은 5000 raw/s로 제한한다. 최대 반대편에서 반대편까지 0.4 s가 걸리며 Step 입력은 사용하지 않는다. 강화학습 Action도 같은 Ramp 제한을 거친다는 시험 기준을 유지한다.

### `test/user_command_test.h`

정규화, Dead Zone, SA OFF 회전, SA ON 횡이동, 모드 전환 필터 초기화, 100 ms 연결 끊김과 0.2 s 재활성 조건을 검사한다.

### `test/leg6_test.h`

6번 다리 세 관절만 작은 범위로 움직여 ADC, PWM, 방향과 중립점의 전체 경로를 먼저 확인한다.

### `test/kinematics_test.h`

기본 발 위치, Body↔Leg 좌표변환, FK와 IK를 여섯 다리에 대해 수치 비교한다.

### `test/workspace_test.h`

Roll·Pitch ±45 deg, x·y 0.28 m/s, Yaw 45 deg/s와 보정 명령의 경계값을 넣어 동적 제한을 검사한다. 여러 축 동시 입력, 직전 명령 유지와 최종 0.0001 m 여유도 확인한다.

### `test/gait_test.h`

Tripod 위상, STANCE/SWING, 궤적 연속성, Swing Height, Early Landing과 Late Landing을 검사한다. 접촉 입력은 테스트 함수의 인자로 직접 전달하며 가상 압력센서 모듈은 사용하지 않는다.

### `test/mode_transition_test.h`

READY↔MANUAL, MANUAL↔CORRECTION, SA 전환, 정지와 재시작에서 필터 상태와 발끝·관절 명령의 연속성을 검사한다. `Tripod_Enable`이 한 주기 변할 때 위상이 초기화되지 않는지도 확인한다.

### `test/controller_test.h`

Simulink와 같은 명시적 입력 벡터를 제어 함수에 전달하고 상위 제어부터 IK 직전까지 결과를 비교한다. VirtualIMU나 TestContact 코드는 사용하지 않는다.

### `test/safety_test.h`

Roll/Pitch 80 deg 경계, NaN·Inf IMU, 각 다리 IK Invalid, Fault Latch와 릴레이 차단을 각각 검사한다. Safety Reset 시험은 만들지 않는다.

### `test/communication_test.h`

LoRa와 Robot Telemetry 패킷을 검사한다. Jetson SPI는 인터페이스 빌드만 확인하고 프로토콜 시험은 규격 확정 후 추가한다.

## 13. 단계별 구현과 시험 순서

1. `robot_config`, `robot_types`와 App 인터페이스를 정의한다.
2. `sensor_test`로 GPS, WT931과 MCP3008 실제 raw 값을 확인한다.
3. `imu_calibration_test`로 WT931 축 부호와 수평 Offset을 확정한다.
4. `low_control_test`로 릴레이와 PWM의 전기적 출력을 확인한다.
5. 관절센서, 압력센서, 서보·릴레이 보정 테스트로 설정 테이블을 완성한다.
6. 임시 RC 명령으로 상위 입력을 시험하고 실제 조종기가 오면 CRSF 테이블을 보정한다.
7. `kinematics_test`로 좌표변환, FK와 IK를 확인한다.
8. Priority, Drone Controller, Position·Heading·Posture 제어기를 Simulink 결과와 비교한다.
9. `workspace_test`로 보행·자세·보정 명령의 동적 제한을 확인한다.
10. Gait Manager와 발끝 궤적을 연결하고 접촉 적응을 시험한다.
11. `mode_transition_test`로 모드 사이의 순간이동과 한 주기 Enable 변화를 확인한다.
12. `safety_test`로 Fault Latch와 릴레이 차단을 확인한다.
13. 낮은 속도에서 시작하여 최대 0.28 m/s까지 실제 보행을 검증한다.
14. LoRa를 추가하고 Jetson SPI는 프로토콜 확정 전까지 비워둔다.
15. 모든 시험이 끝난 뒤 마지막으로 `main.c`를 `hexapod_app` 함수 호출 중심으로 정리한다.

## 14. 5 ms 제어 흐름

```text
TIM6 5 ms 실행 요청
    ↓
실제 Sensor Snapshot 생성
    ↓
CRSF User Command와 입력 Failsafe
    ↓
최신 IMU·IK 결과로 Safety 평가
    ↓
Control Priority와 Drone Controller
    ↓
Body Position Estimator
    ↓
Gait Pose → 보행 명령 Workspace 검사
    ↓
Tripod Manager와 Foot Trajectory → 보정 명령 Workspace 검사
    ↓
Stand/Landing Offset → Body Posture Overlay와 자세 명령 Workspace 검사
    ↓
Body↔Leg 변환과 IK
    ↓
Joint Rate Limit와 Servo PWM
    ↓
Kill 상태를 반영한 Relay 출력
```

세 Workspace 검사는 모두 `workspace_limiter`의 공통 유효성 함수를 사용한다. GPS와 LoRa 파싱, Robot Telemetry와 Jetson SPI처럼 매 주기 완료할 필요가 없는 작업은 `HexapodApp_Process()`의 백그라운드 경로에서 처리한다. 제어 계산이 끝나면 다음 5 ms Tick까지 기다린다.

## 15. 현재 확정값

| 항목 | 값 |
|---|---|
| 제어 주기 | 5 ms, 200 Hz |
| WT931 | USART3, 115200 baud |
| CRSF | USART6, 420000 baud, NVIC 사용 |
| 최대 x·y 속도 | 각 ±0.28 m/s |
| 최대 Yaw 속도 | ±45 deg/s |
| 최대 Roll·Pitch | ±45 deg |
| Swing Height | 기본 0.20 m, 범위 0.15~0.25 m |
| Early Landing 시작 | Swing 진행률 50% |
| Late Landing 하강 | 0.20 m/s |
| 작업공간 여유 | 0.0001 m |
| Stance Foot Slip | 0.05 m, 5회 연속 |
| 관절 명령 속도 | 315.8 deg/s, 5 ms당 1.579 deg |
| Rollover Fault | Roll 또는 Pitch 절댓값 80 deg 이상 |
| Controller Fault | 비유한 IMU 또는 IK Invalid |

관절 ADC, 압력센서 임계값, 서보 방향·중립점, 릴레이-다리 대응과 CRSF raw 보정값은 단계별 실측 테스트에서 채운다.
