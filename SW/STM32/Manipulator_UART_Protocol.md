# 매니퓰레이터 유선 UART 패킷 명세

## 1. 연결 설정

- STM32 송신 포트: `UART5_TX`, `PC12`
- 매니퓰레이터 연결: `PC12`를 상대 MCU의 UART RX에 연결한다.
- 공통 접지: 두 보드의 GND를 반드시 연결한다.
- 신호 전압: STM32 기준 3.3 V TTL UART를 사용한다.
- 통신 설정: 115200 baud, 8 data bits, no parity, 1 stop bit, flow control 없음
- 송신 방향: 현재 구현은 STM32에서 매니퓰레이터로 보내는 단방향 통신이다.
- 송신 주기: 5 ms, 200 Hz

기존 LoRa 모듈은 사용하지 않는다. UART5 포트에 매니퓰레이터 보드를 유선으로 직접 연결한다.

## 2. 패킷 구조

모든 패킷은 고정 16바이트이며 다중 바이트 정수는 Little-endian이다.

| Byte | 크기 | 필드 | 설명 |
|---:|---:|---|---|
| 0 | 1 | Sync 0 | 항상 `0xA5` |
| 1 | 1 | Sync 1 | 항상 `0x5A` |
| 2 | 1 | Version | 현재 버전 `0x01` |
| 3 | 1 | Sequence | 패킷마다 1 증가하며 `255` 다음은 `0` |
| 4 | 1 | Flags | 연결 및 ARM 허가 상태 |
| 5 | 1 | Switches | SA, SB, SC, SD, SE, S1 상태 |
| 6~7 | 2 | Roll | 보정된 `int16_t`, 범위 `-1000~1000` |
| 8~9 | 2 | Pitch | 보정된 `int16_t`, 범위 `-1000~1000` |
| 10~11 | 2 | Throttle | 보정된 `int16_t`, 범위 `-1000~1000` |
| 12~13 | 2 | Yaw | 보정된 `int16_t`, 범위 `-1000~1000` |
| 14~15 | 2 | CRC16 | Byte 0~13의 CRC-16/CCITT-FALSE, Little-endian |

## 3. Flags 필드

| Bit | 이름 | `1`의 의미 |
|---:|---|---|
| 0 | Controller connected | STM32가 정상 CRSF 패킷을 받고 있다. |
| 1 | Motion armed | 조종기 연결 후 중립 재허가가 완료되었다. |
| 2 | ARM mode | SA 활성화와 조종 안전 조건이 모두 만족되어 조작이 허가되었다. |
| 3~7 | Reserved | 현재 사용하지 않으며 무시한다. |

매니퓰레이터는 Flags의 bit 0, bit 1, bit 2가 **모두 1일 때만** Roll, Pitch, Throttle, Yaw 명령을 적용한다.
SD Kill이 활성화되면 ARM mode flag는 즉시 0으로 전송한다.

## 4. Switches 필드

| Bit | 스위치 | 형식 |
|---:|---|---|
| 0 | SA | `0`: ARM 해제, `1`: ARM 활성화 |
| 1~2 | SB | `0~2`: 3단 위치 |
| 3~4 | SC | `0~2`: 3단 위치 |
| 5 | SD | `0`: 해제, `1`: 활성화 |
| 6 | SE | `0`: 해제, `1`: 활성화 |
| 7 | S1 | `0`: 첫 기능, `1`: 두 번째 기능 |

SB는 `(switches >> 1) & 0x03`, SC는 `(switches >> 3) & 0x03`으로 해제한다.

## 5. CRC 계산

- 알고리즘: CRC-16/CCITT-FALSE
- Polynomial: `0x1021`
- Initial value: `0xFFFF`
- RefIn/RefOut: false
- XorOut: `0x0000`
- 입력 범위: Byte 0부터 Byte 13까지 총 14바이트
- 저장 순서: Byte 14에 CRC 하위 바이트, Byte 15에 CRC 상위 바이트를 넣는다.

## 6. 수신 처리 순서

1. UART 스트림에서 `0xA5`, `0x5A`를 연속으로 찾는다.
2. 동기 바이트를 포함하여 총 16바이트를 모은다.
3. Version이 `0x01`인지 확인한다.
4. Byte 0~13으로 CRC16을 계산하여 Byte 14~15와 비교한다.
5. CRC가 맞으면 Sequence와 Flags를 확인한다.
6. Flags bit 0~2가 모두 1일 때만 조종값을 적용한다.
7. CRC가 틀리면 현재 시작 위치에서 한 바이트만 버리고 다시 동기 바이트를 찾는다.

## 7. 필수 안전 동작

다음 조건 중 하나라도 해당하면 매니퓰레이터는 **현재 자세를 유지**해야 한다.

- SA가 꺼져 ARM mode flag가 0이다.
- Controller connected flag가 0이다.
- Motion armed flag가 0이다.
- CRC가 맞지 않는다.
- 마지막 정상 패킷 이후 100 ms 이상 새 정상 패킷이 없다.
- Version이 지원되지 않는다.

현재 자세 유지는 마지막으로 적용한 관절 목표를 계속 유지하는 동작을 뜻한다. ARM 해제 시 원점 복귀, 영점 명령, 토크 해제를 자동 실행하지 않는다.

## 8. 6족 본체 동작

- SA가 켜지고 ARM 모드가 선택되면 6족 본체는 ARM 진입 직전의 18개 관절 명령을 유지한다.
- ARM 모드 중 Roll, Pitch, Throttle, Yaw 입력은 6족 보행에 적용하지 않고 매니퓰레이터 패킷으로만 전달한다.
- SA를 끄면 ARM 모드를 해제하고 기존 6족 모드 선택 로직으로 돌아간다.
- Kill 또는 Safety Fault는 ARM 모드보다 우선하며 기존 릴레이 차단 동작을 유지한다.

## 9. 구현 참고 코드

```c
static int16_t read_i16_le(const uint8_t *data)
{
    uint16_t raw = (uint16_t)data[0] |
                   ((uint16_t)data[1] << 8U);

    return (int16_t)raw;
}

static bool command_enabled(uint8_t flags)
{
    const uint8_t required = (1U << 0U) |
                             (1U << 1U) |
                             (1U << 2U);

    return (flags & required) == required;
}
```

수신 코드에서는 구조체를 UART 버퍼에 직접 캐스팅하지 않는다. 패딩과 정렬 차이를 피하기 위해 위 표의 바이트 위치를 기준으로 각 값을 해제한다.
