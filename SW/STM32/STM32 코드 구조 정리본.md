# STM32 코드 구조 정리본

## 1. 목적

STM32 사용자 코드를 기능별 폴더로 분리하고 `main.c`에는 초기화, 반복 실행, 인터럽트 전달을 위한 함수 호출만 남긴다.

제어 코드는 `SW/Controller/Simulink/plant.slx`의 MATLAB Function 블록을 기준으로 옮긴다. Simscape Plant 내부의 다리별 PID 블록은 시뮬레이션용이므로 STM32 코드로 옮기지 않는다.

현재 실제 IMU는 `WT931`이며 USART3, 115200 baud를 사용한다. CRSF는 USART6, 420000 baud를 사용한다.

---

## 2. 기본 원칙

- 모든 사용자 헤더와 소스는 기능별 폴더 안에 둔다.
- `Core/Inc`와 `Core/Src`의 폴더 구조를 동일하게 유지한다.
- 헤더 하나에는 하나의 책임 영역에 속한 자료형, 상수와 관련 함수들을 함께 넣는다. 함수 하나마다 헤더를 만드는 뜻은 아니다.
- 파일 내부에서만 쓰는 보조 함수는 해당 `.c` 파일의 `static` 함수로 둔다.
- 모든 기능을 다시 합친 무거운 `controller.h`는 만들지 않는다.
- `robot_config.h`, `robot_types.h`는 선언 전용이므로 대응하는 `.c` 파일을 만들지 않는다.
- CubeMX가 생성하는 `main.h`, `main.c`, `stm32f4xx_it.*`, HAL 파일은 기존 위치를 유지한다.
- 제어 계산은 TIM6 기준 200 Hz, 5 ms 주기로 실행한다.
- UART 인터럽트에서는 수신 바이트 저장만 하고 파싱과 제어 계산은 인터럽트 밖에서 실행한다.

---

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
│  │  ├─ gait_pose_controller.h
│  │  ├─ body_posture_controller.h
│  │  ├─ body_position_estimator.h
│  │  ├─ gait_manager.h
│  │  ├─ foot_trajectory.h
│  │  ├─ stance_trajectory.h
│  │  ├─ swing_trajectory.h
│  │  ├─ contact_adaptation.h
│  │  ├─ leg_kinematics.h
│  │  ├─ stand_landing.h
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
│     ├─ calibration_test.h
│     ├─ rc_command_generator.h
│     ├─ user_command_test.h
│     ├─ low_control_test.h
│     ├─ leg6_test.h
│     ├─ kinematics_test.h
│     ├─ gait_test.h
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
   │  ├─ gait_pose_controller.c
   │  ├─ body_posture_controller.c
   │  ├─ body_position_estimator.c
   │  ├─ gait_manager.c
   │  ├─ foot_trajectory.c
   │  ├─ stance_trajectory.c
   │  ├─ swing_trajectory.c
   │  ├─ contact_adaptation.c
   │  ├─ leg_kinematics.c
   │  ├─ stand_landing.c
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
      ├─ calibration_test.c
      ├─ rc_command_generator.c
      ├─ user_command_test.c
      ├─ low_control_test.c
      ├─ leg6_test.c
      ├─ kinematics_test.c
      ├─ gait_test.c
      ├─ controller_test.c
      ├─ safety_test.c
      └─ communication_test.c
```

---

## 4. 헤더 참조 구조

### 4.1 참조 방향

헤더 참조는 아래 방향만 허용하여 순환 참조를 막는다.

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

화살표의 시작점이 끝점의 헤더를 참조한다. 예를 들어 `high_control`은 `common`을 참조할 수 있지만 `common`은 `high_control`을 참조하지 않는다.

`app/hexapod_app.h`가 모든 모듈 헤더를 포함하지 않도록 한다. 전체 모듈 조합은 외부에 공개되지 않는 `hexapod_app.c`에서만 수행한다.

### 4.2 헤더별 직접 참조

표에는 각 헤더가 직접 `#include`할 프로젝트 헤더만 적는다. `<stdint.h>`, `<stdbool.h>`, `stm32f4xx_hal.h` 같은 표준·HAL 헤더는 생략한다.

헤더는 공개 함수의 매개변수와 반환형을 선언하는 데 꼭 필요한 헤더만 포함한다. 계산 과정에서만 사용하는 모듈은 헤더가 아니라 `.c` 파일에서 포함한다.

| 헤더 | 직접 참조할 프로젝트 헤더 |
|---|---|
| `common/robot_config.h` | 없음 |
| `common/robot_types.h` | `common/robot_config.h` |
| `app/hexapod_app.h` | 없음 |
| `sensor/gps.h` | 없음 |
| `sensor/imu.h` | 없음 |
| `sensor/mcp3008.h` | 없음 |
| `sensor/joint_feedback.h` | `common/robot_config.h`, `common/robot_types.h`, `sensor/mcp3008.h` |
| `sensor/foot_pressure.h` | `common/robot_config.h`, `common/robot_types.h`, `sensor/mcp3008.h` |
| `sensor/sensor_manager.h` | `common/robot_types.h`, `sensor/gps.h`, `sensor/imu.h`, `sensor/mcp3008.h`, `sensor/joint_feedback.h`, `sensor/foot_pressure.h` |
| `user_command/crsf_receiver.h` | 없음 |
| `user_command/crsf_protocol.h` | 없음 |
| `user_command/user_command.h` | `common/robot_types.h`, `user_command/crsf_protocol.h` |
| `high_control/control_priority.h` | `common/robot_types.h` |
| `high_control/drone_controller.h` | `common/robot_config.h`, `common/robot_types.h`, `high_control/control_priority.h` |
| `high_control/gait_pose_controller.h` | `common/robot_config.h`, `common/robot_types.h` |
| `high_control/body_posture_controller.h` | `common/robot_config.h`, `common/robot_types.h` |
| `high_control/body_position_estimator.h` | `common/robot_config.h`, `common/robot_types.h` |
| `high_control/gait_manager.h` | `common/robot_config.h`, `common/robot_types.h` |
| `high_control/foot_trajectory.h` | `common/robot_types.h`, `high_control/gait_manager.h` |
| `high_control/stance_trajectory.h` | `common/robot_config.h`, `common/robot_types.h` |
| `high_control/swing_trajectory.h` | `common/robot_config.h`, `common/robot_types.h` |
| `high_control/contact_adaptation.h` | `common/robot_config.h`, `common/robot_types.h` |
| `high_control/leg_kinematics.h` | `common/robot_config.h`, `common/robot_types.h` |
| `high_control/stand_landing.h` | `common/robot_config.h`, `common/robot_types.h` |
| `high_control/safety.h` | 없음. 현재는 두 Fault 자료형과 평가 함수만 공개한다. |
| `low_control/servo_pwm.h` | `common/robot_config.h`, `common/robot_types.h` |
| `low_control/relay.h` | 없음 |
| `communication/lora.h` | 없음 |
| `communication/robot_telemetry.h` | `common/robot_config.h`, `common/robot_types.h` |
| `communication/jetson_spi.h` | `common/robot_config.h`, `common/robot_types.h` |
| `test/test_runner.h` | 없음 |
| 나머지 `test/*.h` | 없음. 시험 대상 헤더는 각 `test/*.c`에서 참조한다. |

### 4.3 상위 모듈 연결 규칙

`safety.h`가 `control_priority.h`를 포함하거나 반대로 참조하지 않는다. `hexapod_app.c`가 두 헤더를 각각 포함하고 아래처럼 결과만 전달한다.

```text
safety.c
   ↓ SafetyOutput_t
hexapod_app.c
   ↓ rollover_fault, controller_fault
control_priority.c
```

`foot_trajectory.c`가 `stance_trajectory.h`, `swing_trajectory.h`, `contact_adaptation.h`를 포함하여 결과를 조합한다. `foot_trajectory.h`는 외부 API에 필요한 `gait_manager.h`와 공통 자료형만 참조한다. 세 하위 궤적 헤더는 서로 참조하지 않는다.

`body_position_estimator.c`는 FK 계산을 위해 `leg_kinematics.h`를 참조한다. 이 구현 의존성을 공개 헤더까지 전달하지 않으며 `leg_kinematics.h`가 Estimator를 역으로 참조하면 안 된다.

`robot_telemetry.h`는 LoRa 드라이버를 참조하지 않는다. `hexapod_app.c`가 Telemetry 패킷 생성과 `LoRa_SendText()` 호출을 연결하여 데이터 구성과 전송 장치를 분리한다.

각 `.c` 파일은 자신의 헤더를 가장 먼저 포함하고, 실제로 사용하는 헤더만 추가한다. 다른 헤더가 우연히 포함해 준 간접 헤더에 의존하지 않는다.

---

## 5. 공통 파일

### `common/robot_config.h`

다리 수, 관절 수, 제어 주기, 서보 펄스 범위, 기구 치수와 제어기 게인 등 프로젝트 공통 상수를 정의한다. CubeMX 핀 매크로는 중복 정의하지 않고 `main.h`의 정의를 사용한다.

### `common/robot_types.h`

3차원 벡터, 자세, Body Twist, 6개 발끝 위치, 18개 관절각처럼 여러 모듈이 함께 사용하는 가벼운 자료형만 정의한다. 센서 드라이버의 내부 Handle이나 UART 버퍼는 넣지 않는다.

---

## 6. 애플리케이션 파일

### `app/hexapod_app.h`

전체 사용자 코드의 진입점을 제공한다. 장치 초기화, 백그라운드 처리, 5 ms 제어 실행, HAL 콜백 전달을 담당하며 실제 계산은 각 기능 모듈에 맡긴다.

최종적으로 `main.c`의 사용자 영역에서는 다음 수준의 함수만 호출한다.

```text
HexapodApp_Init(...)
HexapodApp_Process()
HexapodApp_UartRxCallback(...)
HexapodApp_UartErrorCallback(...)
HexapodApp_TimerCallback(...)
```

`main.c` 변경은 모든 개별 모듈과 단계별 테스트가 끝난 뒤 마지막 통합 단계에서 진행한다.

---

## 7. 센서 파일

### `sensor/gps.h`

NEO-M8N의 USART2 수신, NMEA/UBX 파싱과 최신 위치·속도 값을 제공한다. 현재 작성된 코드를 유지한다.

### `sensor/imu.h`

WT931의 USART3 수신과 WIT 프레임 파싱을 담당하고 가속도, 각속도, Roll, Pitch, Yaw를 제공한다. USART3은 115200 baud를 사용한다.

### `sensor/mcp3008.h`

SPI1에 연결된 MCP3008 3개의 24개 raw 채널을 읽고, 6개 다리의 관절 3개와 압력센서 1개 배열로 배치한다.

### `sensor/joint_feedback.h`

MCP3008 관절 raw 값 18개를 실제 관절각으로 변환한다. 영점, 방향, 최소·최대 ADC 값은 실제 측정 후 보정값으로 추가한다.

관절별 설정 테이블에는 MCP3008 장치·채널, ADC 최소값·영점·최대값, 각도 범위와 방향을 둔다. 우선 테이블 구조를 만든 뒤 `calibration_test`로 측정한 값을 바로 채운다.

### `sensor/foot_pressure.h`

MCP3008 압력 raw 값 6개를 접촉 여부로 변환한다. 임계값과 히스테리시스는 실제 센서 시험 후 결정한다.

센서별 설정 테이블에는 MCP3008 장치·채널, 접촉 진입 임계값과 접촉 해제 임계값을 둔다. 우선 테이블 구조를 만든 뒤 `calibration_test`로 무부하·접촉 측정값을 확인하여 바로 채운다.

### `sensor/sensor_manager.h`

GPS, IMU, MCP3008, 관절각과 발 접촉 값을 한 번에 갱신하고 제어기가 읽을 센서 스냅샷을 만든다. 개별 센서 파싱 로직은 포함하지 않는다.

---

## 8. 조종기 파일

### `user_command/crsf_receiver.h`

USART6 인터럽트 수신과 링 버퍼만 담당한다. USART6은 420000 baud와 NVIC 인터럽트를 사용하며 프레임 해석은 하지 않는다.

### `user_command/crsf_protocol.h`

CRSF 프레임 길이 확인, CRC 검사, RC 채널 unpack을 담당한다. UART Handle이나 STM32 핀에는 의존하지 않도록 작성한다.

### `user_command/user_command.h`

CRSF 채널을 Throttle, Yaw, Roll, Pitch와 SA~SE 스위치 입력으로 매핑한다. 네 축은 중립 0, 범위 `-1000~1000`으로 정규화하고, 스로틀 부호로 전진·후진을 직접 구분한다. 조종 모드에서 SA OFF는 Throttle x이동+Yaw 회전, SA ON은 Throttle x이동+Yaw 짐벌 y이동으로 변환하고 전환 시점의 Heading을 유지한다. Simulink의 `USER` 블록은 시험 신호원이므로 그대로 옮기지 않고 이 파일의 실제 수신값으로 대체한다.

마지막 정상 CRSF 프레임 이후 `100 ms`가 지나면 연결 끊김으로 판단한다. 이때 이동 명령을 모두 0으로 만들고 `Tripod_Enable`과 `Motion_Armed`를 해제하여 현재 STANCE 위치를 유지한다. 재연결 후에는 네 축이 `0.2초` 동안 중립이어야 다시 `Motion_Armed`를 활성화한다. 이 처리는 Safety Fault가 아니라 조종 입력 Failsafe로 구현한다.

CRSF 처리 흐름은 다음과 같다.

```text
USART6 IRQ
    ↓
crsf_receiver
    ↓
crsf_protocol
    ↓
user_command
    ↓
control_priority
```

---

## 9. 상위 제어 파일

### `high_control/control_priority.h`

Simulink `ControlPriorityManager`를 옮긴다. 스위치, 서기·착지 완료와 두 Fault를 입력받아 KILL, FAULT, LANDING, STANDING, READY 모드의 우선순위를 결정한다.

### `high_control/drone_controller.h`

Simulink `DroneController`를 옮긴다. 선택된 모드에 따라 입력 필터, Motion Armed, 기능별 Enable, 사용자 명령과 서기·착지 진행률을 생성한다.

### `high_control/gait_pose_controller.h`

Simulink `GaitPosePI`를 옮긴다. 위치 PI, Heading Yaw PI와 사용자 명령을 합쳐 최종 보행 Body Twist를 만든다.

### `high_control/body_posture_controller.h`

Simulink `BodyPosturePIOverlay`를 옮긴다. Roll, Pitch와 보정 모드 Yaw의 자세 PI를 계산하고 발끝 목표에 몸체 자세 역회전을 적용한다.

### `high_control/body_position_estimator.h`

Simulink `BodyPositionEstimator`를 옮긴다. 접촉 중인 STANCE 다리의 관절각과 FK를 이용해 `Body_X_Est`, `Body_Y_Est`, `Body_Z_Est`와 `Valid_Leg_Count`를 계산한다.

### `high_control/gait_manager.h`

Simulink `TripodGaitManager`를 옮긴다. Tripod A와 B의 위상, STANCE/SWING 상태와 진행률을 관리한다.

### `high_control/foot_trajectory.h`

여섯 다리의 발끝 궤적 계산 순서와 결과 조합만 담당한다. 무거운 수식은 Stance, Swing, Contact Adaptation 파일에 맡긴다.

### `high_control/stance_trajectory.h`

Simulink `TripodFootTrajectory`의 Stance 계산을 분리한다. 몸체 이동과 회전의 반대 방향으로 지면에 고정된 발끝 목표를 갱신한다.

### `high_control/swing_trajectory.h`

Simulink `TripodFootTrajectory`의 Swing 계산을 분리한다. 착지점, 3차 Bezier 궤적, Quintic 시간 스케일링과 방사 방향 오프셋을 계산한다.

### `high_control/contact_adaptation.h`

Simulink `TripodFootTrajectory`의 접촉 적응 계산을 분리한다. Early Landing, Late Landing과 Search Down을 처리한다.

### `high_control/leg_kinematics.h`

Simulink `BaseFootPosition`과 다리별 FK/IK 계산을 담당한다. 여섯 다리마다 파일을 만들지 않고 다리 번호와 기구 파라미터를 받는 공통 함수로 작성한다.

IK 계산은 해의 유효 여부를 함께 출력한다. 동적 Motion 제한은 제한되지 않은 후보 명령으로 6개 다리의 IK 가능 여부를 먼저 검사하며, 각 다리의 최종 IK 입력에는 `0.0001 m` 여유의 발끝 작업공간 제한을 적용한다. IK 실패를 검출하는 책임은 이 파일에 두지만 로봇을 즉시 정지시키는 결정은 Safety가 담당한다.

### `high_control/stand_landing.h`

Simulink `StandLandingFootDelta`를 옮긴다. 서기와 착지 과정의 발끝 높이 변화와 완료 신호를 계산한다.

### `high_control/safety.h`

Safety 판단 결과인 `rollover_fault`와 `controller_fault`를 제공한다.

현재 Simulink에서는 두 입력 모두 `Constant1 = 0`에 연결되어 있으며 실제 Safety 판정은 구현되어 있지 않다. 따라서 최초 STM32 구현에서도 `Safety_Evaluate()`가 두 Fault를 항상 `false`로 반환한다. 나중에 IMU 자세, 통신 상태, IK 결과와 관절 명령 검사를 이 함수 내부에 추가한다.

후속 Safety 구현에서는 `IK_Valid=0`을 `Controller_Fault`로 변환하여 즉시 정지한다. 정책만 먼저 확정하며 최초 구현에는 이 정지 판단을 넣지 않는다.

현재 데이터 흐름은 다음과 같다.

```text
Safety_Evaluate()
    ├─ rollover_fault = false
    └─ controller_fault = false
                 ↓
        ControlPriorityManager
```

`0`은 Safety가 거짓이라는 뜻이 아니라 Fault가 없다는 뜻이다.

---

## 10. 하위 제어 파일

### `low_control/servo_pwm.h`

18개 관절의 각도를 DS51150-270 PWM 펄스로 변환하고 각 TIM 채널에 출력한다. 타이머·채널 매핑, 중립값, 방향과 관절별 보정값을 한곳에서 관리한다.

서보별 설정 테이블에는 타이머·채널, 중립 Pulse, 방향과 최소·최대 Pulse를 둔다. 우선 테이블 구조를 만든 뒤 `calibration_test`에서 한 채널씩 측정한 값을 바로 채운다.

18개 관절 명령에는 DS51150-270을 고려한 독립적인 각속도 제한을 항상 적용한다. 현재 최대 명령 각속도는 12.6 V 최고 무부하 속도 기준인 `315.8°/s`이며 5 ms 제어 주기당 최대 변화량은 `1.579°`이다. Rate Limiter는 관절각 범위 제한 뒤, 서보 방향·중립점 보정과 PWM 변환 전에 적용한다. Joint Jump Fault 기준은 Safety 항목으로 별도 관리한다.

실제 서보 전원을 켤 때 직전 명령 상태는 가능하면 보정된 관절 ADC의 현재 측정각으로 초기화한다. ADC 보정 전 시험에서는 릴레이가 꺼진 상태에서 첫 유효 관절 명령으로 초기화한 뒤 출력한다.

각도 제한과 펄스 범위 제한은 하드웨어 보호를 위한 최종 출력 제한으로 항상 유지한다. 이것은 아직 미구현인 상위 Safety 판단과 별개의 보호 장치이다.

### `low_control/relay.h`

INA1, INB1, INC1, INA2, INB2, INC2의 6개 릴레이를 제어한다. Active High이며 초기 상태는 반드시 모두 OFF로 설정한다.

서보 PWM 18개가 안전한 중립 펄스를 출력하는 것이 확인되기 전에는 `Relay_AllOn()`을 호출하지 않는다.

---

## 11. 통신 파일

### `communication/lora.h`

UART5에 연결된 RYLR998의 AT 명령, 송수신 파싱과 메시지 전송만 담당한다. 모니터링 항목 선택과 문자열 구성은 포함하지 않는다.

### `communication/robot_telemetry.h`

관제탑에 필요한 로봇의 자세, 위치, 관절과 동작 상태를 모니터링 패킷으로 만든다. 전송 주기, 패킷 종류, 순번과 타임스탬프를 관리하지만 UART와 RYLR998 AT 명령은 직접 다루지 않는다.

`hexapod_app.c`가 `RobotTelemetry`에서 만든 패킷을 `LoRa_SendText()`에 전달한다.

### `communication/jetson_spi.h`

SPI2 Slave 통신을 위한 자리만 유지한다. 헤더에는 상위 코드가 참조할 최소 인터페이스만 두고 소스는 거의 비워둔다. 패킷 길이, 필드, CRC, DRDY 동작과 실제 명령은 Jetson 통신 요구사항이 정해진 뒤 함께 설계한다.

### 모니터링 정보 구성

LoRa는 속도가 느리므로 관제탑에서 로봇 운용 상태를 확인하는 데 필요한 정보만 전송한다. 디버깅용 raw 값, 내부 계산값과 오류 카운터는 포함하지 않고 유선 연결로 확인한다.

| 그룹 | 관제탑 전송 정보 | 데이터 출처 |
|---|---|---|
| 기본 상태 | 패킷 순번, STM32 동작 시간, 현재 `Active_Mode` | App, Control Priority |
| 고장 상태 | `Rollover_Fault`, `Controller_Fault` | Safety |
| 로봇 자세 | Roll, Pitch, Yaw | WT931 |
| 전역 위치 | GPS 위도, 경도, 고도와 `GPS_Valid` | NEO-M8N |
| 관절 상태 | 18개 측정 관절각 | Joint Feedback |
| 다리 상태 | 6개 발 접촉 비트 | Foot Pressure |
| 서보 전원 상태 | INA1~INC2의 6개 Relay ON/OFF 비트 | Relay |

DS51150-270은 자체 각도 Telemetry를 보내는 스마트 서보가 아니다. 따라서 `측정 관절각`은 서보 내부값이 아니라 MCP3008에 연결된 관절 위치센서로 계산한 실제 관절각을 의미한다.

`Active_Mode`는 새로 만든 모니터링 값이 아니라 Simulink `ControlPriorityManager`의 실제 출력이다. 값은 `LANDING`, `STAND`, `READY`, `MANUAL`, `CORRECTION`, `FAULT`, `KILL` 중 하나이며 관제탑에서는 로봇이 현재 어떤 동작 상태인지 표시하는 데 사용한다.

`Rollover_Fault`와 `Controller_Fault`는 Safety의 두 고장 상태이다. 현재는 Safety가 미구현이므로 둘 다 항상 0이지만 나중에 Safety 로직을 구현하면 같은 `STATUS` 필드에 실제 값을 넣는다.

`GPS_Valid`는 현재 GPS 좌표를 사용해도 되는지만 나타내는 1 bit 값이다.

### LoRa 패킷 분리

RYLR998 한 패킷에 모든 값을 넣지 않고 다음 세 종류만 사용한다.

| 패킷 | 주요 내용 |
|---|---|
| `STATUS` | Active Mode, 두 Fault, Roll/Pitch/Yaw, 발 접촉과 Relay 상태 |
| `JOINT` | 18개 측정 관절각 |
| `GPS` | GPS 위도, 경도, 고도와 GPS Valid |

각 패킷에는 프로토콜 버전, 패킷 종류, 순번과 STM32 시간값을 넣는다. 문자열은 고정된 필드 순서와 정수 스케일을 사용하고 한 패킷을 180 byte 이하로 제한한다.

| 값 | 전송 단위 |
|---|---|
| 측정 관절각, Roll, Pitch, Yaw | `0.01 deg` 단위의 signed 정수 |
| 위도, 경도 | `1e-7 deg` 단위의 signed 정수 |
| GPS 고도 | `mm` 단위의 signed 정수 |
| 시간 | `ms` 단위의 unsigned 정수 |
| 접촉, Relay와 Fault 상태 | 비트 마스크 |
| GPS Valid | 1 bit 값 |

기본 전송 주기는 `STATUS` 1 Hz, `JOINT` 0.5 Hz, `GPS` 0.5 Hz로 시작하여 전체 전송을 초당 2개로 제한한다.

관제탑은 패킷 순번과 마지막 수신 시각으로 LoRa 연결 끊김을 판단한다. 로봇은 LoRa 연결 여부에 의존하지 않고 독립적으로 안전 동작을 수행한다.

목표 관절각, PWM 펄스, 각속도, 가속도, ADC raw, 제어 루프 실행시간과 오류 카운터는 LoRa 모니터링에 넣지 않는다. 이러한 디버그 정보는 유선 연결로 확인한다.

---

## 12. 테스트 파일

### `test/test_runner.h`

한 번에 하나의 시험만 선택하여 초기화하고 반복 실행한다. 시험 모드는 enum 또는 하나의 빌드 설정으로 선택하며 여러 액추에이터 시험이 동시에 실행되지 않게 한다.

### `test/sensor_test.h`

1번 테스트이다. GPS, WT931, MCP3008 raw 값과 갱신 주기·오류 횟수를 확인하여 센서 통신과 채널 배선이 정상인지 먼저 검사한다.

### `test/calibration_test.h`

3번 테스트이다. `sensor_test`의 raw 입력 확인과 `low_control_test`의 릴레이·PWM 파형 확인이 끝난 뒤 실행한다. 다음 세 시험을 한 파일에서 선택하여 한 번에 하나만 실행한다.

| 순서 | 시험 | 테이블에 채울 값 |
|---:|---|---|
| 3-1 | 관절을 기준 위치와 가동 끝 위치로 움직여 ADC 값을 기록한다. | 관절별 ADC 최소값·영점·최대값, 각도 범위, 방향 |
| 3-2 | 발을 무부하와 접촉 상태로 반복 측정한다. | 압력센서별 접촉·해제 임계값 |
| 3-3 | 서보 한 채널만 저범위 Pulse로 움직인다. | 서보별 방향, 중립 Pulse, 최소·최대 Pulse |

측정 결과는 디버거 또는 유선 로그로 확인하고 각 설정 테이블에 바로 옮긴다. 미보정 상태의 별도 운용 규칙은 두지 않으며 전체 제어 시험 전에 세 테이블을 완성한다.

### `test/rc_command_generator.h`

RadioMaster Pocket과 ELRS 수신기가 준비되기 전까지 임시 조종기 입력을 만든다. CRSF 프레임을 흉내 내지 않고 실제 `user_command`가 제어기에 전달할 `Throttle`, `Yaw`, `Roll`, `Pitch`, `SA`~`SE` 값을 직접 생성한다. 네 축 값은 실제 명령과 같은 `-1000~1000`을 사용하고 `5000 raw/s` Ramp를 적용한다. 최대 반대편→최대 반대편은 `0.4 s`, 중립→최대는 `0.2 s`로 생성한다.

중립 입력, 서기 요청, 수동 조종, 보정, 착지와 Kill 같은 고정 입력 조합 및 시간 순서 입력을 선택할 수 있게 구성한다. 수동 조종 시험에는 SA OFF 전진+회전, SA OFF 후진+반대 회전, SA ON +x·+y 횡이동, SA ON −x·−y 횡이동과 SA 전환 순간을 포함한다. 시험 입력도 Step으로 바꾸지 않고 위 Ramp를 반드시 거친다. 후속 강화학습 Action에도 동일한 변화율 제한을 적용하고 Step Action 직결 경로를 두지 않는다. `user_command_test`, `controller_test`, `gait_test`에서 공통으로 사용하며 실제 운용 코드에서는 사용하지 않는다.

### `test/user_command_test.h`

조종기가 없는 동안에는 `rc_command_generator`의 임시 입력이 제어 명령으로 올바르게 전달되는지 확인한다. Throttle 양·음 최대값, SA OFF의 Yaw 양·음 회전, SA ON의 y축 양·음 이동, SA 전환 전·후 Heading 명령 연속성을 각각 검사한다. 이후 저장된 CRSF 예제 프레임으로 CRC와 채널 unpack을 검사하고, 실제 조종기가 준비되면 USART6 수신값과 채널 매핑을 검사한다.

### `test/low_control_test.h`

2번 테스트이다. 릴레이 6개를 서보 전원과 분리한 상태에서 하나씩 시험하고, 18개 PWM 채널의 출력 여부와 안전한 시험 범위를 오실로스코프로 확인한다. 실제 중립점과 방향 측정은 3번 `calibration_test`에서 수행한다.

### `test/leg6_test.h`

6번 다리 세 관절만 작은 펄스 범위로 움직여 배선, 방향과 중립점을 확인한다.

### `test/kinematics_test.h`

알고 있는 관절각의 FK 결과와 알고 있는 발끝 위치의 IK 결과를 검사한다. 여섯 다리의 좌우 방향과 좌표 변환 부호를 각각 확인한다.

### `test/gait_test.h`

서보 전원을 끈 상태에서 Tripod 위상, Stance/Swing 전환, 궤적 연속성과 Early/Late Landing 입력을 검사한다.

### `test/controller_test.h`

Simulink와 같은 입력 시퀀스를 넣고 Priority, Drone Controller, Position PI와 Posture PI 출력을 비교한다. Safety 모듈과 별개로 Fault 입력을 강제로 1로 넣어 FAULT/KILL 반응도 검사한다.

### `test/safety_test.h`

현재 단계에서는 두 Fault가 항상 0인지 검사한다. 실제 Safety 로직을 추가할 때 전복 각도, 통신 끊김, IK 실패와 관절 Jump 시험을 이 파일에 추가한다.

### `test/communication_test.h`

LoRa 송수신과 Robot Telemetry의 세 패킷 길이·필드 순서·순환 주기를 검사한다. Jetson SPI는 인터페이스가 빌드되는지만 확인하고, 프로토콜 시험은 통신 요구사항이 정해질 때 추가한다.

---

## 13. 단계별 구현 및 시험 순서

### 1단계: 공통 구조 준비

`robot_config`, `robot_types`와 각 폴더의 공개 인터페이스를 먼저 정의하고 개별 모듈이 독립적으로 빌드되는지 확인한다.

### 2단계: 센서 입력

GPS, WT931, MCP3008 raw 수신을 각각 확인하고 1번 `sensor_test`로 MCP3008 채널 배선을 검사한다.

### 3단계: 릴레이와 PWM 파형

2번 `low_control_test`로 릴레이를 서보 전원과 분리하여 확인하고, 18개 PWM 채널의 파형과 안전한 시험 범위를 측정한다.

### 4단계: 센서·서보 캘리브레이션

3번 `calibration_test`에서 관절 ADC, 압력센서, 서보 순서로 한 채널씩 측정한다. 측정 결과를 `joint_feedback`, `foot_pressure`, `servo_pwm`의 채널별 설정 테이블에 바로 채우고 전체 테이블을 완성한다.

### 5단계: CRSF

조종기가 없는 동안에는 `rc_command_generator`로 `-1000~1000` 스로틀, SA OFF 회전과 SA ON 횡이동을 먼저 검증하고 이후 제어 시험을 진행한다. CRSF는 저장된 프레임으로 `crsf_protocol`을 먼저 시험하고, 실제 조종기가 준비되면 USART6 인터럽트 수신과 채널 매핑을 확인한다.

### 6단계: 좌표변환과 기구학

기본 발 위치, 몸체 좌표계에서 다리 좌표계 변환, FK, IK 순서로 수치 결과를 검증한다. 이 단계까지는 서보 전원을 끈다.

### 7단계: 상위 제어

`control_priority`, `drone_controller`, `stand_landing`, `body_position_estimator`, `gait_pose_controller`, `body_posture_controller` 순서로 Simulink 출력과 비교한다.

### 8단계: 보행 궤적

`gait_manager`, `stance_trajectory`, `swing_trajectory`, `contact_adaptation`, `foot_trajectory` 순서로 연결한다. 먼저 배열과 로그로 확인한 뒤 실제 서보에 연결한다.

### 9단계: Safety

초기에는 `safety`가 Fault 두 개를 0으로 반환하는지만 확인한다. 이후 Safety를 구현할 때 IK 실패를 `Controller_Fault`로 변환하여 즉시 정지하고, 전복과 관절 출력 보호를 단계적으로 추가한다.

### 10단계: 외부 통신

LoRa를 제어 루프와 분리하여 시험하고 `robot_telemetry`의 패킷 종류를 하나씩 추가한다. Jetson SPI는 최소 인터페이스만 유지하며 프로토콜 요구사항이 정해질 때 별도로 구현한다.

### 11단계: 최종 통합

모든 단위 시험이 끝난 뒤 `hexapod_app`에서 센서, 조종기, Safety, 상위 제어, IK와 서보 출력을 연결한다. 마지막에만 `main.c`를 함수 호출 중심으로 정리한다.

---

## 14. 5 ms 제어 흐름

```text
TIM6 5 ms Tick
    ↓
Sensor Snapshot
    ↓
CRSF User Command
    ↓
Safety Fault 출력
    ↓
Control Priority
    ↓
Drone Controller
    ↓
Body Position / Gait Pose / Body Posture
    ↓
Tripod Manager
    ↓
Stance + Swing + Contact Adaptation
    ↓
Foot Trajectory
    ↓
Leg Coordinate Transform + IK
    ↓
Servo PWM 제한 및 출력
```

GPS, LoRa, Robot Telemetry, Jetson SPI와 UART 프레임 파싱처럼 매 주기 완료할 필요가 없는 작업은 `HexapodApp_Process()`의 백그라운드 처리로 실행한다.

---

## 15. 현재 확정된 하드웨어 연결

| 기능 | 주변장치 | 설정 |
|---|---|---|
| GPS | USART2 | 9600 baud |
| WT931 IMU | USART3 | 115200 baud |
| LoRa RYLR998 | UART5 | 115200 baud |
| ELRS CRSF | USART6 | 420000 baud, NVIC 사용 |
| MCP3008 3개 | SPI1 Master | 24개 raw 채널 |
| Jetson | SPI2 Slave | DRDY는 PC9 |
| 제어 주기 | TIM6 | 200 Hz, 5 ms |
| 서보 | TIM1/2/3/4/5/8 | 18개 PWM, 200 Hz |
| 릴레이 | GPIO 6개 | Active High, 초기 OFF |
