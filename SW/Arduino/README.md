# Hexapod 매니퓰레이터 Arduino 제어

이 디렉터리는 Hexapod-Robot에 장착된 6축 매니퓰레이터를 Arduino Uno로 제어하는 펌웨어를 관리한다. Uno는 STM32가 보내는 조종기 패킷을 받아 6개 PWM 서보를 구동하며, 조종기 연결·Motion Armed·ARM 모드가 모두 유효할 때만 목표 자세를 갱신한다.

현재 소스는 Arduino Uno 컴파일까지 확인했다. 실제 로봇에서의 전원 인가, 전 관절 동작 범위, 링크 길이와 충돌 한계는 별도 실기 검증이 필요하다.

## 디렉터리 구성

| 경로 | 용도 |
|---|---|
| [`ManipulatorController/ManipulatorController.ino`](ManipulatorController/ManipulatorController.ino) | STM32 패킷 수신, 안전 조건 검사, 부팅 자세 정렬, 역기구학 및 6축 서보 제어 |
| [`ManipulatorController/README.md`](ManipulatorController/README.md) | 제어기 내부 설정과 패킷 처리 상세 |
| [`ManipulatorCalibration/ManipulatorCalibration.ino`](ManipulatorCalibration/ManipulatorCalibration.ino) | J1~J6 소프트웨어 영점 조정 및 Uno EEPROM 저장 |

관련 STM32 문서는 다음과 같다.

- [매니퓰레이터 UART 패킷 명세](../STM32/Manipulator_UART_Protocol.md)
- [드론 조종기 입력 및 ARM 모드](../Controller/드론%20조종기%20입력/README.md)

## 하드웨어 구성

현재 핀과 관절 배치는 다음과 같다.

| 관절 | 기능 | 서보 | Uno 신호 핀 | 기본 영점 |
|---|---|---|---:|---:|
| J1 | 베이스 회전 | DS51150-270 | D5 | 1290 µs |
| J2 | 어깨 | DS51150-270 | D6 | 1600 µs |
| J3 | 팔꿈치 | DS51150-270 | D7 | 1480 µs |
| J4 | 손목 Pitch | DS51150-270 | D8 | 1510 µs |
| J5 | 손목 Roll | SPT5435LV-180 | D9 | 1610 µs |
| J6 | 그리퍼 | SPT5435LV-180 | D10 | 1540 µs |

위 영점은 현재 조립 상태에서 얻은 소프트웨어 기준값이다. 기구 분해, 혼 재체결 또는 서보 교체 후에는 다시 캘리브레이션해야 한다.

## 전원 및 신호 배선

서보 전류를 Arduino Uno에서 공급하면 안 된다. 현재 구성은 서보 종류별 외부 전원과 공통 신호 접지를 사용한다.

```text
STM32 PC12 / UART5_TX ─────────────> Uno D0 / RX
STM32 GND ───────────────┬─────────> Uno GND
                        ├─────────> DS 전원 GND
                        └─────────> SPT 전원 GND

외부 DS 전원 + ───────────────────> J1~J4 전원 +
외부 SPT 전원 + ──────────────────> J5~J6 전원 +
Uno D5~D10 ───────────────────────> J1~J6 PWM 신호
```

전원 연결 시 다음 원칙을 지킨다.

1. J1~J4와 J5~J6의 양극 전원은 서로 분리한다. 현재 사용 중인 정확한 서보 모델의 라벨과 데이터시트에서 허용 전압을 다시 확인한다.
2. STM32, Uno와 두 서보 전원의 GND는 PWM 기준 전압을 위해 공통으로 연결한다.
3. 서보 부하 전류는 Uno GND 핀이나 브레드보드 전원 레일을 경유하지 않고 전원 분배점으로 직접 귀환시킨다.
4. Uno의 5 V 핀과 디지털 핀에서는 서보 전원을 공급하지 않는다.
5. 코드 업로드와 캘리브레이션 중에는 STM32 TX선을 Uno D0에서 분리한다. Uno의 USB-UART와 STM32 TX가 동시에 D0을 구동하지 않게 한다.

## 전체 동작 흐름

```text
전원 인가
  → EEPROM 영점 확인
  → J1~J6 순차 활성화
  → 직립 영점으로 저속 이동
  → 영점에서 2초 유지
  → 접힌 READY 자세로 이동
  → 마지막 자세 유지
  → 정상 패킷 + 안전 Flags 0x07 + SC=1
  → 조종기 입력으로 매니퓰레이터 제어
```

부팅 자세 정렬은 통신 상태와 관계없이 한 번 실행된다. 이후 ARM 조건이 해제되거나 패킷이 100 ms 이상 끊기면 원점 복귀나 PWM 해제를 하지 않고 마지막 목표 자세를 유지한다.

일반 PWM 서보에는 현재각 피드백이 없으므로, Uno는 전원 인가 당시의 실제 관절 위치를 알 수 없다. 전원이 꺼진 상태에서 팔을 임의로 움직였다면 첫 PWM 출력에서 예상보다 빠른 움직임이 생길 수 있다. 첫 시험에서는 팔을 받침대로 지지하고 비상 전원 차단이 가능한 상태를 유지한다.

## 캘리브레이션

캘리브레이션 스케치는 J1(D5)부터 J6(D10)까지 한 축씩 조정한다. 각 관절은 기존 실측 영점에서 시작하며 `1000~2000 µs` 범위에서 `10 µs`씩 조절한다.

### 실행 순서

1. STM32 TX선을 Uno D0에서 분리한다.
2. [`ManipulatorCalibration.ino`](ManipulatorCalibration/ManipulatorCalibration.ino)를 Arduino Uno에 업로드한다.
3. 시리얼 모니터를 `115200 baud`, `Newline` 또는 `Both NL & CR`로 연다.
4. 현재 관절을 `up` 또는 `down`으로 움직인다.
5. 기구학적 영점에 도달하면 `ok`를 입력한다.
6. J6까지 같은 절차를 완료한다.
7. 마지막에 출력되는 `HOME_US={...}` 값을 기록한다.
8. [`ManipulatorController.ino`](ManipulatorController/ManipulatorController.ino)를 다시 업로드한다.
9. 제어기 부팅 시 출력되는 `HOME_US={...}`가 캘리브레이션 결과와 같은지 확인한다.

### 명령

| 명령 | 동작 |
|---|---|
| `up` | 현재 PWM을 10 µs 증가 |
| `down` | 현재 PWM을 10 µs 감소 |
| `ok` | 현재 관절값을 확정하고 다음 관절로 이동 |

EEPROM에는 Magic, Version과 J1~J6 값이 함께 저장된다. J6까지 완료한 뒤 유효 표시를 마지막으로 기록하므로 중간에 전원이 꺼진 불완전한 보정값은 제어기가 사용하지 않는다. 유효한 기록이 없으면 소스에 포함된 기본 영점 `{1290,1600,1480,1510,1610,1540}`를 사용한다.

캘리브레이션 중 관절이 기계적 끝단에 닿거나, 링크가 비틀리거나, 서보가 지속적으로 진동하거나, 배선이 뜨거워지면 즉시 서보 전원을 차단한다.

## STM32 통신

STM32 `UART5_TX/PC12`에서 Uno `D0/RX`로 `115200 baud, 8-N-1` 단방향 통신을 사용한다. STM32는 5 ms 주기로 16바이트 패킷을 전송한다.

| Byte | 필드 | 형식 |
|---:|---|---|
| 0 | Sync 0 | `0xA5` |
| 1 | Sync 1 | `0x5A` |
| 2 | Version | `0x01` |
| 3 | Sequence | `uint8_t` |
| 4 | Flags | `uint8_t` |
| 5 | Switches | `uint8_t` |
| 6~7 | Roll | `int16_t`, little-endian |
| 8~9 | Pitch | `int16_t`, little-endian |
| 10~11 | Throttle | `int16_t`, little-endian |
| 12~13 | Yaw | `int16_t`, little-endian |
| 14~15 | CRC-16/CCITT-FALSE | little-endian |

CRC는 Byte 0~13에 대해 Polynomial `0x1021`, Initial value `0xFFFF`, XorOut `0x0000`으로 계산한다. Uno는 Sync, Version, CRC와 중복 Sequence를 검사한 정상 패킷만 적용한다.

## ARM 허가 조건

다음 조건이 모두 참이어야 조종 명령이 적용된다.

- Flags bit 0: Controller connected
- Flags bit 1: Motion armed
- Flags bit 2: STM32 ARM mode allowed
- Switches의 SC 값이 `1`
- 마지막 정상 패킷 수신 후 100 ms 이내
- 부팅 영점 및 READY 자세 이동 완료

즉, 정상 안전 Flags는 `(flags & 0x07) == 0x07`이며 SC도 별도로 확인한다.

## 조종기 매핑

| 입력 | 기본 동작 | SA 활성 상태 |
|---|---|---|
| Roll | 베이스를 이용한 말단 y축 이동 | 동일 |
| Pitch | 말단 x축 전진·후진 | 동일 |
| Throttle | 말단 z축 상승·하강 | 그리퍼 비례 개폐 |
| Yaw | 손목 Roll 회전 | 동일 |
| SA | 해제 시 일반 제어 | 누르는 동안 Throttle을 그리퍼로 전환 |
| SC | `1`일 때 ARM 모드 | 동일 |

SA를 누른 동안 Throttle 최저는 그리퍼 열림, 최고는 닫힘으로 매핑되며 z축 이동은 멈춘다. SA를 놓으면 마지막 그리퍼 각도를 유지하고 Throttle은 다시 z축 이동에 사용된다.

Pitch와 Throttle은 J2·J3의 평면 Jacobian 역기구학으로 x·z 속도로 변환한다. J4는 J2와 J3 움직임을 보정해 READY 자세의 말단 Pitch를 유지한다. 입력에는 데드밴드, 저역통과 필터와 관절별 PWM Slew Limit를 적용한다.

## 주요 제어 설정

| 설정 | 현재값 | 의미 |
|---|---:|---|
| `LINK1_MM` | 300 mm | 어깨-팔꿈치 축간거리 임시값 |
| `LINK2_MM` | 300 mm | 팔꿈치-손목 축간거리 임시값 |
| `MAX_XZ_SPEED_MM_S` | 90 mm/s | 말단 x·z 최대 명령 속도 |
| `MAX_Y_SPEED_MM_S` | 70 mm/s | 말단 y 최대 명령 속도 |
| `MAX_WRIST_SPEED_DEG_S` | 50 deg/s | 손목 최대 회전 속도 |
| `AXIS_DEADBAND` | 35 | 조종기 입력 데드밴드 |
| `PAYLOAD_TIMEOUT_MS` | 100 ms | 패킷 Freshness 제한 |
| `READY_DEG` | `{0,35,-100,-25,0,0}` | 현재 접힌 READY 자세 |

`LINK1_MM`, `LINK2_MM`, READY 자세와 관절 한계는 현재 실측 Fusion 값이 아닌 보수적 설정을 포함한다. 최종 운용 전에 실제 축간거리와 충돌 없는 관절 범위로 갱신해야 한다.

## 빌드 및 업로드

Arduino IDE에서는 보드를 `Arduino Uno`로 선택하고 각 스케치 폴더와 같은 이름의 `.ino` 파일을 연다. CLI 환경에서는 다음처럼 컴파일할 수 있다.

```powershell
arduino-cli compile --warnings all --fqbn arduino:avr:uno SW/Arduino/ManipulatorCalibration
arduino-cli compile --warnings all --fqbn arduino:avr:uno SW/Arduino/ManipulatorController
```

업로드 전에는 STM32 TX선을 Uno D0에서 분리한다. 제어기 업로드가 끝난 뒤 전원을 끈 상태에서 `STM32 PC12 → Uno D0`, `STM32 GND → Uno GND`를 다시 연결한다.

## 시리얼 진단

제어기는 0.5초마다 다음 형식의 상태를 출력한다.

```text
DBG bytes=0 ok=0 verErr=0 crcErr=0 dup=0 ageMs=NONE ready=1 flags=0x0 sw=0x0 SC=0 SA=0 safety=0 arm=0 allow=0 R=0 P=0 T=0 Y=0
```

| 항목 | 의미 |
|---|---|
| `bytes` | Uno D0에서 읽은 누적 바이트 수 |
| `ok` | Version과 CRC를 통과한 정상 패킷 수 |
| `verErr` | Version 오류 패킷 수 |
| `crcErr` | CRC 오류 패킷 수 |
| `dup` | 중복 Sequence 수 |
| `ageMs` | 마지막 정상 패킷 이후 경과 시간 |
| `ready` | 부팅 READY 자세 도달 여부 |
| `safety` | Flags bit 0~2 허가 여부 |
| `arm` | SC가 1인지 여부 |
| `allow` | 현재 조종 입력 적용 여부 |

`bytes=0`이면 CRC보다 먼저 물리 경로를 확인한다. STM32 PC12 TX와 Uno D0 RX의 교차 연결, 공통 GND, UART5 핀 설정, 115200 baud, STM32 펌웨어 다운로드 여부를 점검한다. `bytes`는 증가하지만 `ok=0`이면 Sync, Version, 패킷 길이, 바이트 순서와 CRC 계산 범위를 확인한다.

Uno의 D0/D1은 USB-UART와 공유된다. USB 시리얼 모니터를 열면 Uno가 리셋되어 부팅 자세 정렬이 다시 실행될 수 있으며, 시리얼 모니터에서 문자를 전송하면 STM32 수신 데이터와 섞일 수 있으므로 진단 중에는 입력하지 않는다.

## 최초 실기 시험 체크리스트

1. 배터리를 분리한 상태에서 전원 양극과 GND 단락 여부를 확인한다.
2. 서보를 연결하기 전에 J1~J4 전원과 J5~J6 전원 전압을 각각 측정한다.
3. STM32, Uno와 서보 전원 GND가 공통인지 확인한다.
4. 캘리브레이션 스케치로 J1부터 한 축씩 연결하고 방향과 영점을 확인한다.
5. 제어기 업로드 후 서보 전원을 넣기 전에 출력된 `HOME_US`를 확인한다.
6. 팔을 받친 상태에서 부팅 영점 이동과 READY 이동을 확인한다.
7. ARM 진입 전 `bytes`, `ok`, `ready`, `flags`, `SC`를 확인한다.
8. SC를 `1`로 바꾼 뒤 작은 조종 입력부터 시험한다.
9. 각 축의 방향, 충돌 범위, 전원 전압 강하와 커넥터 발열을 기록한다.

## 문제 해결

### 서보가 드드드 진동하고 움직이지 않음

즉시 서보 전원을 끈다. 링크 또는 혼을 분리할 수 있으면 무부하 상태로 한 축만 시험하고, 증상 발생 중 서보 단자 전압을 측정한다. 정상 서보를 같은 핀에 연결하고 의심 서보를 정상 핀에 연결해 전원·신호·서보 자체 문제를 분리한다. 전압 강하, 기계적 끝단 구속, GND 불량, 커넥터 발열을 확인하기 전에는 서보 고장으로 단정하지 않는다.

### 전원 인가 직후 빠르게 튐

현재 PWM 서보에서 실제 시작 각도를 읽을 수 없기 때문이다. 마지막 종료 자세를 READY로 유지하고 팔을 받친 상태에서 전원을 넣는다. 임의 자세에서도 폐루프 저속 이동이 필요하면 관절 위치 피드백을 추가해야 한다.

### ARM 모드인데 움직이지 않음

`ready=1`, `ok` 증가, `ageMs≤100`, `flags`의 하위 3비트가 모두 1, `SC=1`, `allow=1`인지 순서대로 확인한다. 하나라도 만족하지 않으면 제어기는 안전하게 마지막 자세를 유지한다.

### 방향이 반대임

해당 관절의 `SERVO_DIRECTION[]` 부호를 바꾼다. 현재 설정은 J3만 `-1`이고 나머지는 `+1`이다. 변경 후 반드시 한 축씩 낮은 이동량으로 재시험한다.

## 검증 상태

| 항목 | 상태 |
|---|---|
| Arduino Uno 컴파일 | 통과 |
| STM32 16바이트 패킷 형식 대조 | 소스 및 명세 대조 완료 |
| J1~J6 EEPROM 캘리브레이션 연동 | 코드 구현 및 컴파일 완료 |
| 실제 6축 동시 구동 | 미검증 |
| 실제 링크 길이와 작업공간 | 실측 필요 |
| 장시간 부하·발열 시험 | 미검증 |

소스 컴파일 성공은 실제 기구의 안전한 동작을 보장하지 않는다. 최종 운용 전에는 한 축 시험, 무부하 시험, 전압 강하 확인, 제한각 설정과 비상 차단 시험을 완료해야 한다.
