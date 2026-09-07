# STM32–Jetson SPI 64바이트 전이중 프로토콜 v3

## 1. 개요

Jetson Orin Nano Super를 SPI Master, STM32F446RE를 SPI2 Slave로 사용한다. Jetson이 약 5 ms마다 한 번씩 정확히 64바이트의 클록을 발생시키며, 같은 전이중 트랜잭션에서 다음 데이터를 교환한다.

```text
MISO, STM32 -> Jetson: [SENSOR 32B][GPS 32B]
MOSI, Jetson -> STM32: [COMMAND 64B]
```

SENSOR와 GPS는 각각 독립된 헤더와 CRC를 갖는 32바이트 Subframe이다. 따라서 GPS 한쪽이 손상돼도 SENSOR Subframe을 별도로 검증해 사용할 수 있다. COMMAND는 64바이트 전체를 하나의 프레임으로 검증한다.

발압 ADC raw 값은 보내지 않으며, 각 발의 접촉 여부 6개만 SENSOR의 1바이트 비트마스크로 전송한다.

## 2. SPI 설정과 배선

| 항목 | 설정 |
|---|---|
| Master | Jetson |
| Slave | STM32 SPI2 |
| SPI Mode | Mode 0, CPOL=0, CPHA=0 |
| 데이터 | 8 bit, MSB First |
| 멀티바이트 정수 | Little Endian |
| 전송 길이 | CS Low 한 번당 정확히 64바이트 |
| 기준 주기 | 5 ms, 200 Hz |
| 권장 초기 속도 | 1 MHz |
| 준비 신호 | STM32 PC9 `DRDY`, High이면 DMA 준비 완료 |

STM32F446RE SPI2 핀은 PB12=NSS, PB13=SCK, PB14=MISO, PB15=MOSI다. 두 장치의 GND를 반드시 공통으로 연결한다.

64바이트는 512비트이므로 100 kHz에서는 순수 클록 시간만 5.12 ms가 걸린다. 200 Hz 주기에는 여유가 없으므로 1 MHz 이상으로 시작하는 것이 적절하다. 1 MHz에서는 약 0.512 ms가 걸린다.

## 3. 전체 64바이트 트랜잭션

### 3.1 STM32에서 Jetson 방향, MISO

| 전체 오프셋 | 크기 | 내용 |
|---:|---:|---|
| 0~31 | 32 | SENSOR Subframe, 자체 CRC 포함 |
| 32~63 | 32 | GPS Subframe, 자체 CRC 포함 |

### 3.2 Jetson에서 STM32 방향, MOSI

| 전체 오프셋 | 크기 | 내용 |
|---:|---:|---|
| 0~63 | 64 | COMMAND Frame, Byte 0~61 CRC 적용 |

Jetson이 명령 없이 센서만 읽을 때는 MOSI로 64바이트 전부 `0x00`을 보낼 수 있다. STM32는 이 값을 오류 명령으로 집계하지 않는다.

## 4. 버전과 패킷 종류

Byte 1은 상위 4비트가 버전, 하위 4비트가 종류다.

```text
VERSION_TYPE = (version << 4) | type
Protocol version = 3
```

| 종류 | Type | Byte 1 |
|---|---:|---:|
| SENSOR | `0x1` | `0x31` |
| COMMAND | `0x2` | `0x32` |
| GPS | `0x5` | `0x35` |

이번 변경은 전송 길이와 COMMAND CRC 위치가 v2와 호환되지 않으므로 프로토콜 버전을 3으로 올렸다. Jetson의 기존 v2 파서는 그대로 사용할 수 없다.

## 5. SENSOR Subframe, MISO Byte 0~31

| Subframe 바이트 | 크기 | 자료형 | 필드 | 변환 |
|---:|---:|---|---|---|
| 0 | 1 | `uint8_t` | MAGIC | `0xA5` |
| 1 | 1 | `uint8_t` | VERSION/TYPE | `0x31` |
| 2~3 | 2 | `uint16_t` | Sequence | 패킷마다 증가 |
| 4 | 1 | `uint8_t` | Delta time | 100 us/LSB, 최대 25.5 ms |
| 5 | 1 | `uint8_t` | Foot contact | Bit 0~5 = Leg 1~6 |
| 6~23 | 18 | `uint8_t[18]` | Joint angle | -135~+135 deg를 0~255로 매핑 |
| 24~25 | 2 | `int16_t` | IMU roll | rad x 10000 |
| 26~27 | 2 | `int16_t` | IMU pitch | rad x 10000 |
| 28~29 | 2 | `int16_t` | IMU yaw | rad x 10000 |
| 30~31 | 2 | `uint16_t` | CRC16 | Byte 0~29, Low byte 먼저 |

관절 순서는 `leg * 3 + joint`다. 즉 Byte 6~8은 Leg 1의 Joint 1~3이고, Byte 21~23은 Leg 6의 Joint 1~3이다.

관절 복원식은 다음과 같다.

```text
angle_rad = -2.35619449 + encoded / 255 * 4.71238898
```

인코딩값 약 128이 0도이며 해상도는 약 1.059도/LSB다. STM32는 Jetson 표시 좌표계에 맞추기 위해 송신 시 선택 관절의 부호만 변환한다. 내부 센서값은 변경하지 않는다. 릴레이가 꺼져 있으면 관절 18개는 0도에 해당하는 값으로 전송한다.

## 6. GPS Subframe, MISO Byte 32~63

아래 바이트 번호는 GPS Subframe 내부 기준이다. 실제 64바이트 수신 버퍼에서는 모든 오프셋에 32를 더한다.

| Subframe 바이트 | 크기 | 자료형 | 필드 | 단위/설명 |
|---:|---:|---|---|---|
| 0 | 1 | `uint8_t` | MAGIC | `0xA5` |
| 1 | 1 | `uint8_t` | VERSION/TYPE | `0x35` |
| 2~3 | 2 | `uint16_t` | GPS Sequence | 새 GPS 측정 시 증가 |
| 4 | 1 | `uint8_t` | GPS age | 100 ms/LSB, 255=25.5초 이상 또는 미수신 |
| 5 | 1 | `uint8_t` | GPS flags | 아래 비트 정의 참조 |
| 6~9 | 4 | `int32_t` | Latitude | degree x 1e7 |
| 10~13 | 4 | `int32_t` | Longitude | degree x 1e7 |
| 14~17 | 4 | `int32_t` | Altitude | mm |
| 18~21 | 4 | `uint32_t` | iTOW | GPS week time, ms |
| 22~23 | 2 | `int16_t` | Velocity north | cm/s |
| 24~25 | 2 | `int16_t` | Velocity east | cm/s |
| 26~27 | 2 | `uint16_t` | Horizontal accuracy | cm, 포화 655.35 m |
| 28 | 1 | `uint8_t` | Satellites | 사용 위성 수 |
| 29 | 1 | `uint8_t` | Fix type | GPS 드라이버의 Fix 값 |
| 30~31 | 2 | `uint16_t` | CRC16 | Subframe Byte 0~29 |

GPS flags는 다음과 같다.

| 비트 | 의미 |
|---:|---|
| 0 | Fix 성공 |
| 1 | 위도·경도·고도 유효 |
| 2 | 속도 유효 |
| 3 | GPS 시각 유효 |
| 4 | UBX 데이터 |
| 5 | NMEA 데이터 |
| 6~7 | 예약 |

GPS는 보통 SPI 200 Hz보다 느리게 갱신된다. 같은 GPS 값이 여러 SPI 트랜잭션에 반복되는 것은 정상이다. 새 값 여부는 GPS Sequence로, 데이터의 오래된 정도는 GPS age로 판단한다.

## 7. COMMAND Frame, MOSI Byte 0~63

| 바이트 | 크기 | 자료형 | 필드 | 설명 |
|---:|---:|---|---|---|
| 0 | 1 | `uint8_t` | MAGIC | `0xA5` |
| 1 | 1 | `uint8_t` | VERSION/TYPE | `0x32` |
| 2~3 | 2 | `uint16_t` | Sequence | Jetson 명령 순번 |
| 4 | 1 | `uint8_t` | Delta time | 100 us/LSB |
| 5 | 1 | `uint8_t` | Flags | 현재 Raw 보관, 의미 미할당 |
| 6~61 | 56 | `uint8_t[56]` | Payload | 현재 Raw 보관, 의미 미할당 |
| 62~63 | 2 | `uint16_t` | CRC16 | Byte 0~61, Low byte 먼저 |

현재 STM32는 COMMAND의 헤더·버전·종류·CRC·순번을 검증하고 56바이트 Payload를 보관한다. 이 Payload를 보행이나 모터 제어에 적용하는 규격은 아직 정의하지 않았다.

## 8. CRC 규격

모든 CRC는 CRC-16/CCITT-FALSE를 사용한다.

| 항목 | 값 |
|---|---|
| Polynomial | `0x1021` |
| Initial value | `0xFFFF` |
| RefIn / RefOut | False / False |
| Final XOR | `0x0000` |
| 저장 순서 | CRC Low, CRC High |

검사 범위는 SENSOR와 GPS가 각각 해당 Subframe의 Byte 0~29이고, COMMAND는 64바이트 프레임의 Byte 0~61이다.

## 9. DRDY와 DMA 동작 순서

1. STM32가 `tx_frame[0:32]`에 SENSOR, `tx_frame[32:64]`에 GPS를 만든다.
2. STM32가 `HAL_SPI_TransmitReceive_DMA(..., 64)`로 SPI2 Slave DMA를 Arm한다.
3. DMA Arm 성공 후 STM32가 PC9 `DRDY`를 High로 만든다.
4. Jetson이 DRDY High를 확인하고 CS를 Low로 내린다.
5. Jetson이 정확히 64바이트를 전송해 MOSI COMMAND와 MISO SENSOR+GPS를 동시에 교환한다.
6. Jetson이 CS를 High로 올린다.
7. STM32 완료 콜백이 DRDY를 Low로 내린다.
8. STM32 메인 루프가 수신 COMMAND를 검증하고 다음 송신 프레임을 준비한다.

CS를 32바이트마다 올렸다 내리거나, 기존처럼 32바이트만 전송하면 DMA 완료가 발생하지 않으므로 반드시 한 CS 구간에서 64바이트를 교환해야 한다.

## 10. Jetson 파싱 골격

```python
import struct

TRANSFER_SIZE = 64
VERSION = 3


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = (((crc << 1) ^ 0x1021) if (crc & 0x8000)
                   else (crc << 1)) & 0xFFFF
    return crc


def validate_subframe(frame: bytes, expected_type: int) -> None:
    if len(frame) != 32:
        raise ValueError("subframe length")
    if frame[0] != 0xA5 or frame[1] != ((VERSION << 4) | expected_type):
        raise ValueError("header/version/type")
    received_crc = struct.unpack_from("<H", frame, 30)[0]
    if received_crc != crc16_ccitt_false(frame[:30]):
        raise ValueError("CRC")


rx = bytes(spi.xfer2([0x00] * TRANSFER_SIZE))
sensor = rx[0:32]
gps = rx[32:64]
validate_subframe(sensor, 0x1)
validate_subframe(gps, 0x5)

latitude_deg = struct.unpack_from("<i", gps, 6)[0] / 1e7
longitude_deg = struct.unpack_from("<i", gps, 10)[0] / 1e7
altitude_m = struct.unpack_from("<i", gps, 14)[0] / 1000.0
```

실제 Jetson 코드는 DRDY를 확인한 뒤 `xfer2()`를 호출해야 한다. 명령을 보낼 때는 `[0x00] * 64` 대신 규격에 맞춰 CRC까지 생성한 COMMAND 64바이트를 전달한다.

## 11. 현재 구현 파일과 남은 작업

STM32 구현은 다음 파일에 반영되어 있다.

- `workspace/Hexapod/Core/Inc/communication/jetson_spi.h`
- `workspace/Hexapod/Core/Src/communication/jetson_spi.c`
- `workspace/Hexapod/Core/Inc/common/robot_types.h`
- `workspace/Hexapod/Core/Src/sensor/sensor_manager.c`
- `workspace/Hexapod/Core/Src/test/communication_test.c`

남은 작업은 Jetson 실행 코드의 v3/64바이트 대응, COMMAND Payload 56바이트의 구체적인 제어 필드 정의, 명령 Timeout 및 Safety 우선순위 연결, STM32 재빌드·플래시 후 실제 하드웨어 검증이다.
