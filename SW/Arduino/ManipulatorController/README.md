# Arduino Uno 매니퓰레이터 제어기

`드론 조종기 입력/README.md`와 `Manipulator_UART_Protocol.md`를 기준으로 Arduino Uno에서 6개 PWM 서보를 제어한다. STM32 소스는 수정하지 않는다.

## 부팅 동작

부팅 자세 정렬은 통신 및 ARM 모드와 무관하게 한 번 실행한다.

1. Uno가 J1~J6을 `D5~D10`에서 0.25초 간격으로 활성화한다.
2. 정상적인 직전 종료 자세인 접힌 초기포즈 펄스에서 시작한다.
3. 직립 영점까지 `2us/20ms`로 천천히 이동한다. 최대 이동 관절 기준 약 7.4초다.
4. 직립 영점에서 2초 동안 토크를 유지한다.
5. 접힌 초기포즈까지 `5us/20ms`로 이동한다. 최대 이동 관절 기준 약 3초다.
6. 초기포즈를 토크로 유지하면서 ARM 조종 허가를 기다린다.

일반 3선 PWM 서보에는 현재각 피드백이 없다. 코드가 전원 인가 당시의 임의 자세를 자동 측정할 수는 없다. 부팅 시 같은 Uno의 EEPROM에서 J1~J6 영점값을 읽고, 그 영점과 `READY_DEG[]`로 접힌 초기포즈 펄스를 계산한다. 전원이 꺼진 동안 팔을 손으로 옮긴 경우 첫 PWM 명령에서 움직임이 발생할 수 있다. 임의 자세에서도 항상 저속 이동하려면 관절 엔코더, 별도 위치 피드백 또는 피드백 지원 서보가 필요하다.

## UART 연결과 패킷

- STM32 `UART5_TX/PC12` → Uno `D0/RX`
- STM32 GND와 Uno GND 공통
- `115200 baud, 8-N-1`, STM32→Uno 단방향
- 고정 16바이트, little-endian
- Sync `A5 5A`, Version `01`
- CRC-16/CCITT-FALSE: poly `0x1021`, init `0xFFFF`, xorout `0`
- CRC 입력: Byte 0~13, CRC 저장: Byte 14 Low, Byte 15 High

| Byte | 필드 |
|---:|---|
| 0 | `0xA5` |
| 1 | `0x5A` |
| 2 | Version `0x01` |
| 3 | Sequence |
| 4 | Flags |
| 5 | Switches |
| 6~7 | Roll `int16_t` |
| 8~9 | Pitch `int16_t` |
| 10~11 | Throttle `int16_t` |
| 12~13 | Yaw `int16_t` |
| 14~15 | CRC16 little-endian |

수신기는 Sync 검색, Version 검사, CRC 검사, Sequence 중복 검사를 수행한다. 손상된 프레임은 한 바이트씩 재동기화하며 정상 프레임만 명령 상태를 갱신한다.

## ARM 조종 조건

부팅 자세 정렬이 끝난 뒤 다음 조건이 모두 만족될 때만 조종값을 적용한다.

- Bit 0: Controller connected
- Bit 1: Motion armed
- Bit 2: STM32 ARM mode allowed
- Switches의 `SC == 1`: ARM mode

매핑은 다음과 같다.

- Roll: 말단 y축 이동
- Pitch: 말단 x축 이동
- Throttle: 평상시 말단 z축 이동, SA를 누르는 동안 그리퍼 비례 개폐
- Yaw: 손목 회전
- SA: Throttle의 그리퍼 제어 모드 선택

짐벌에는 ±35의 데드밴드와 약 0.1초의 저역통과 필터를 적용한다. Pitch·Throttle 입력은 J2·J3 역기구학으로 x·z 방향 속도로 변환하고 J4를 함께 보정하여 말단 피치를 유지한다. 최대 x·z 속도는 `90mm/s`, 베이스 방향 속도는 `70mm/s`, 손목 회전은 `50deg/s`이며 실제 출력은 관절별 PWM slew limit로 한 번 더 제한한다.

ARM 해제, 연결 해제, Motion armed 해제, CRC/Version 오류, 중복 Sequence 또는 정상 패킷 100ms timeout에서는 원점 복귀나 PWM detach를 하지 않고 마지막 관절 목표를 계속 유지한다.

ARM 모드는 SC 값으로 판정한다. SA를 누른 동안 Throttle 최저는 그리퍼 열림, 최고는 닫힘으로 비례 매핑되고 z축 이동은 정지한다. SA를 놓으면 마지막 그리퍼 각도를 유지하면서 Throttle이 다시 z축 이동을 제어한다.

## 시리얼 디버그

USB 시리얼 모니터를 `115200 baud`로 열면 0.5초마다 `DBG` 한 줄을 출력한다. `bytes`는 D0에서 읽은 바이트 수, `ok`는 정상 패킷 수, `verErr`와 `crcErr`는 형식 오류 수, `ageMs`는 마지막 정상 패킷 이후 시간이다. `SC=1`, `safety=1`, `arm=1`, `allow=1`이어야 조종 입력이 적용된다. 시리얼 모니터를 열면 Uno가 리셋되어 부팅 자세 정렬부터 다시 실행될 수 있다.

## 실기 보정값

코드 상단 `Config`에서 다음을 실측해야 한다.

1. 기본 `HOME_US[]`는 실측값 `{1290,1600,1480,1510,1610,1540}`이다.
2. `ManipulatorCalibration.ino`를 끝까지 실행하면 J1~J6 값이 EEPROM에 저장되고, 이후 제어기는 여섯 값을 모두 자동 적용한다. 보정이 중단되었거나 저장값이 유효하지 않으면 위 기본값을 사용한다.
3. 부팅 접힌 자세 펄스는 `HOME_US[]`, `READY_DEG[]`, 방향과 서보 각도당 펄스로 계산한다.
4. `SERVO_DIRECTION[]`: 현재 J3(D7)만 `-1`로 반전했다.
5. `READY_DEG[]`: 현재 `{0, +35, -100, -25, 0, 0}` 근사값이다.
6. `LINK1_MM`, `LINK2_MM`: 어깨-팔꿈치와 팔꿈치-손목 축간거리로 교체한다.
7. `JOINT_MIN_DEG[]`, `JOINT_MAX_DEG[]`: 실제 기구 충돌 한계로 좁힌다.

## 캘리브레이션 순서

1. STM32 TX선을 Uno D0에서 분리한다.
2. `../ManipulatorCalibration/ManipulatorCalibration.ino`를 Uno에 업로드한다.
3. 시리얼 모니터를 `115200 baud`, 줄바꿈으로 열고 J1(D5)부터 J6(D10)까지 `up`, `down`, `ok`로 맞춘다.
4. 마지막 J6에서 `ok`를 입력해 `HOME_US={...}`가 출력되어야 전체 EEPROM 기록이 유효해진다.
5. `ManipulatorController.ino`를 다시 업로드하고 부팅 때 출력되는 `HOME_US={...}`가 같은지 확인한다.

첫 통합 시험은 팔을 받침대로 지지하고 한 축씩 전원을 연결한다. 반대 방향, 드드드 진동, 링크 비틀림, 전압 강하 또는 발열이 보이면 즉시 서보 전원을 끈다.
