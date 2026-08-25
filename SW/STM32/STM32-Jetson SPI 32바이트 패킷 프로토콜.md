# STM32–Jetson SPI 32바이트 패킷 프로토콜

## 1. 문서 목적

STM32F446RE와 Jetson Orin Nano Super 사이에서 SPI로 전달할 32바이트 고정 길이 센서 패킷을 정의한다. 이 문서의 패킷은 STM32가 측정한 18개 관절각과 IMU 자세를 Jetson으로 전달하는 상태 패킷이다.

`workspace/Hexapod/Core/Inc/communication/jetson_spi.h`와 `workspace/Hexapod/Core/Src/communication/jetson_spi.c`에 아래 프로토콜의 패킹, CRC, 공통 수신 파싱과 SPI 송수신 코드가 구현되어 있다. Jetson 명령 Payload의 구체적인 의미와 제어 적용은 아직 정의하지 않는다.

Jetson에서 STM32로 보내는 패킷도 동일한 32바이트 헤더와 CRC 위치를 사용한다. `COMMAND` 패킷은 공통 검증 후 24바이트 Payload를 Raw 상태로 보관하며, Payload 내부 필드는 추후 명령 규격 확정 시 할당한다.

## 2. 통신 기본 조건

| 항목 | 설정 |
|---|---|
| Jetson | SPI Master |
| STM32 | SPI2 Slave |
| 프레임 길이 | 항상 32바이트 |
| SPI Mode | Mode 0, CPOL=0, CPHA=0 |
| 데이터 길이 | 8비트 |
| 비트 순서 | MSB First |
| 바이트 순서 | 멀티바이트 정수는 Little Endian |
| Chip Select | 하드웨어 NSS, 한 프레임 동안 Low |
| 준비 신호 | STM32 PC9 `DRDY`, High일 때 전송 준비 완료 |
| 기준 전송 주기 | 5 ms, 200 Hz |

## 3. 32바이트 패킷 구조

```text
[A5][VER/TYPE][SEQ 2B][DT 2B][JOINT x18][ROLL 2B][PITCH 2B][YAW 2B][CRC16 2B]
```

| 바이트 | 크기 | 필드 | 자료형 | 설명 |
|---:|---:|---|---|---|
| 0 | 1 | `MAGIC` | `uint8_t` | 패킷 시작 확인값 `0xA5` |
| 1 | 1 | `VERSION_TYPE` | `uint8_t` | 상위 4비트 프로토콜 버전, 하위 4비트 패킷 종류 |
| 2~3 | 2 | `SEQUENCE` | `uint16_t` | 패킷 순번, Little Endian |
| 4~5 | 2 | `DELTA_TIME` | `uint16_t` | 이전 패킷 생성 후 경과 시간, 10 us/LSB |
| 6~23 | 18 | `JOINT[18]` | `uint8_t[18]` | 18개 관절 측정각 |
| 24~25 | 2 | `IMU_ROLL` | `int16_t` | Roll rad x 10000 |
| 26~27 | 2 | `IMU_PITCH` | `int16_t` | Pitch rad x 10000 |
| 28~29 | 2 | `IMU_YAW` | `int16_t` | Yaw rad x 10000 |
| 30~31 | 2 | `CRC16` | `uint16_t` | Byte 0~29의 CRC-16/CCITT-FALSE |

전체 크기는 다음과 같다.

```text
1 + 1 + 2 + 2 + 18 + 6 + 2 = 32바이트
```

## 4. 헤더 정의

### 4.1 시작 확인값

```c
#define JETSON_SPI_MAGIC  0xA5U
```

SPI의 NSS 상승·하강으로 프레임 경계가 구분되지만, 수신 데이터가 올바른 프로토콜인지 확인하기 위해 첫 바이트를 검사한다.

### 4.2 버전 및 패킷 종류

`VERSION_TYPE` 한 바이트를 다음과 같이 나눈다.

```text
Bit 7~4: 프로토콜 버전
Bit 3~0: 패킷 종류
```

```c
#define JETSON_SPI_PROTOCOL_VERSION  1U

typedef enum
{
    JETSON_SPI_TYPE_NONE        = 0x0U,
    JETSON_SPI_TYPE_SENSOR      = 0x1U,
    JETSON_SPI_TYPE_COMMAND     = 0x2U,
    JETSON_SPI_TYPE_ACK         = 0x3U,
    JETSON_SPI_TYPE_ERROR       = 0x4U
} JetsonSpi_PacketType_t;

#define JETSON_SPI_MAKE_VERSION_TYPE(version, type) \
    (uint8_t)((((version) & 0x0FU) << 4U) | ((type) & 0x0FU))
```

버전 1의 센서 패킷은 `0x11`이다.

### 4.3 패킷 순번

`SEQUENCE`는 센서 패킷을 만들 때마다 1씩 증가한다. `uint16_t`이므로 200 Hz 전송 시 약 327.68초마다 `65535`에서 `0`으로 순환한다. 순환은 오류가 아니며 송신과 수신 모두 unsigned 뺄셈으로 처리한다.

```c
uint16_t sequence_gap = (uint16_t)(current_sequence - previous_sequence);
```

- `sequence_gap == 1`: 정상적으로 연속 수신함
- `sequence_gap > 1`: 중간에 `sequence_gap - 1`개가 유실됨
- `sequence_gap == 0`: 동일 패킷이 중복되었거나 새 패킷이 아님

### 4.4 Delta time

`DELTA_TIME`은 이번 센서 스냅샷 패킷을 만든 시각과 이전 패킷을 만든 시각의 차이다. 단위는 10 us로 정의한다.

```text
전송값 = 경과 시간[us] / 10
복원 시간[s] = 전송값 x 0.00001
```

정상적인 5 ms 주기에서는 `500`이 들어간다. 표현 가능한 최대 시간은 655.35 ms다. `HAL_GetTick()`만 사용하면 측정 분해능은 1 ms이므로 `delta_ms * 100`으로 저장한다. 실제 10 us 수준의 주기 측정이 필요하면 별도의 마이크로초 타이머를 사용한다.

관절 ADC와 IMU가 서로 다른 시각에 갱신될 수 있으므로 이 필드는 개별 센서의 내부 샘플 주기가 아니라 패킷 생성 주기를 의미한다.

## 5. 관절각 인코딩

현재 프로젝트의 관절 배열은 `leg * 3 + joint` 순서다.

| 패킷 바이트 | 배열 인덱스 | 데이터 |
|---:|---:|---|
| 6~8 | 0~2 | Leg 1의 Joint 1, 2, 3 |
| 9~11 | 3~5 | Leg 2의 Joint 1, 2, 3 |
| 12~14 | 6~8 | Leg 3의 Joint 1, 2, 3 |
| 15~17 | 9~11 | Leg 4의 Joint 1, 2, 3 |
| 18~20 | 12~14 | Leg 5의 Joint 1, 2, 3 |
| 21~23 | 15~17 | Leg 6의 Joint 1, 2, 3 |

프로젝트의 공통 관절 범위 `-135도~+135도`를 `0~255`로 선형 매핑한다.

```text
-135도 -> 0
   0도 -> 약 128
+135도 -> 255
```

해상도는 약 `1.059도/LSB`다. 이 정밀도는 상태 확인과 상위 판단용으로 사용할 수 있지만, Jetson에서 정밀한 저수준 관절 제어를 수행하는 용도로는 부족할 수 있다.

```c
#include <math.h>
#include <stdint.h>

#define JOINT_MIN_RAD  (-2.35619449f)
#define JOINT_MAX_RAD  ( 2.35619449f)

static uint8_t JetsonSpi_EncodeJoint(float angle_rad)
{
    float normalized;

    if (angle_rad < JOINT_MIN_RAD)
    {
        angle_rad = JOINT_MIN_RAD;
    }
    if (angle_rad > JOINT_MAX_RAD)
    {
        angle_rad = JOINT_MAX_RAD;
    }

    normalized = (angle_rad - JOINT_MIN_RAD) /
                 (JOINT_MAX_RAD - JOINT_MIN_RAD);
    return (uint8_t)lroundf(normalized * 255.0f);
}

static float JetsonSpi_DecodeJoint(uint8_t encoded)
{
    return JOINT_MIN_RAD +
           ((float)encoded / 255.0f) *
           (JOINT_MAX_RAD - JOINT_MIN_RAD);
}
```

## 6. IMU 인코딩

Roll, Pitch, Yaw는 각각 `int16_t`로 저장하며 `1 LSB = 0.0001 rad`로 정의한다.

```text
인코딩값 = round(각도[rad] x 10000)
각도[rad] = 인코딩값 / 10000
```

해상도는 약 `0.00573도/LSB`이며 `-pi~+pi rad` 범위를 표현할 수 있다.

```c
static int16_t JetsonSpi_EncodeImu(float angle_rad)
{
    float scaled = angle_rad * 10000.0f;

    if (scaled > 32767.0f)
    {
        scaled = 32767.0f;
    }
    if (scaled < -32768.0f)
    {
        scaled = -32768.0f;
    }

    return (int16_t)lroundf(scaled);
}

static float JetsonSpi_DecodeImu(int16_t encoded)
{
    return (float)encoded / 10000.0f;
}
```

## 7. CRC 정의

CRC는 `CRC-16/CCITT-FALSE`를 사용한다.

| 설정 | 값 |
|---|---|
| Polynomial | `0x1021` |
| Initial value | `0xFFFF` |
| RefIn/RefOut | False |
| Final XOR | `0x0000` |
| 검사 범위 | Byte 0~29 |
| 패킷 저장 순서 | CRC Low, CRC High |

```c
static uint16_t JetsonSpi_Crc16CcittFalse(const uint8_t *data,
                                          uint32_t length)
{
    uint16_t crc = 0xFFFFU;
    uint32_t index;
    uint8_t bit;

    for (index = 0U; index < length; ++index)
    {
        crc ^= (uint16_t)data[index] << 8U;

        for (bit = 0U; bit < 8U; ++bit)
        {
            if ((crc & 0x8000U) != 0U)
            {
                crc = (uint16_t)((crc << 1U) ^ 0x1021U);
            }
            else
            {
                crc <<= 1U;
            }
        }
    }

    return crc;
}
```

## 8. STM32 패킷 생성 예제

구조체를 통째로 전송하면 컴파일러 패딩과 엔디언에 의존할 수 있으므로, 반드시 `uint8_t frame[32]`에 명시적으로 넣는다.

```c
#define JETSON_SPI_FRAME_SIZE           32U
#define JETSON_SPI_JOINT_COUNT          18U
#define JETSON_SPI_CRC_INPUT_SIZE       30U

#define SPI_OFFSET_MAGIC                 0U
#define SPI_OFFSET_VERSION_TYPE          1U
#define SPI_OFFSET_SEQUENCE              2U
#define SPI_OFFSET_DELTA_TIME             4U
#define SPI_OFFSET_JOINTS                6U
#define SPI_OFFSET_IMU_ROLL             24U
#define SPI_OFFSET_IMU_PITCH            26U
#define SPI_OFFSET_IMU_YAW              28U
#define SPI_OFFSET_CRC                  30U

static void JetsonSpi_WriteU16Le(uint8_t *destination, uint16_t value)
{
    destination[0] = (uint8_t)(value & 0xFFU);
    destination[1] = (uint8_t)((value >> 8U) & 0xFFU);
}

static void JetsonSpi_WriteI16Le(uint8_t *destination, int16_t value)
{
    JetsonSpi_WriteU16Le(destination, (uint16_t)value);
}

static void JetsonSpi_BuildSensorFrame(uint8_t frame[JETSON_SPI_FRAME_SIZE],
                                       uint16_t sequence,
                                       uint16_t delta_time_10us,
                                       const float joint_angle_rad[18],
                                       float roll_rad,
                                       float pitch_rad,
                                       float yaw_rad)
{
    uint16_t crc;
    uint32_t joint;

    frame[SPI_OFFSET_MAGIC] = JETSON_SPI_MAGIC;
    frame[SPI_OFFSET_VERSION_TYPE] =
        JETSON_SPI_MAKE_VERSION_TYPE(JETSON_SPI_PROTOCOL_VERSION,
                                     JETSON_SPI_TYPE_SENSOR);

    JetsonSpi_WriteU16Le(&frame[SPI_OFFSET_SEQUENCE], sequence);
    JetsonSpi_WriteU16Le(&frame[SPI_OFFSET_DELTA_TIME], delta_time_10us);

    for (joint = 0U; joint < JETSON_SPI_JOINT_COUNT; ++joint)
    {
        frame[SPI_OFFSET_JOINTS + joint] =
            JetsonSpi_EncodeJoint(joint_angle_rad[joint]);
    }

    JetsonSpi_WriteI16Le(&frame[SPI_OFFSET_IMU_ROLL],
                         JetsonSpi_EncodeImu(roll_rad));
    JetsonSpi_WriteI16Le(&frame[SPI_OFFSET_IMU_PITCH],
                         JetsonSpi_EncodeImu(pitch_rad));
    JetsonSpi_WriteI16Le(&frame[SPI_OFFSET_IMU_YAW],
                         JetsonSpi_EncodeImu(yaw_rad));

    crc = JetsonSpi_Crc16CcittFalse(frame, JETSON_SPI_CRC_INPUT_SIZE);
    JetsonSpi_WriteU16Le(&frame[SPI_OFFSET_CRC], crc);
}
```

## 9. STM32 수신 패킷 검증 예제

향후 Jetson 명령 패킷도 같은 32바이트 외피를 사용한다. STM32는 데이터를 적용하기 전에 최소한 크기, 시작값, 버전, 패킷 종류와 CRC를 검사해야 한다.

```c
static uint16_t JetsonSpi_ReadU16Le(const uint8_t *source)
{
    return (uint16_t)source[0] |
           ((uint16_t)source[1] << 8U);
}

static int16_t JetsonSpi_ReadI16Le(const uint8_t *source)
{
    return (int16_t)JetsonSpi_ReadU16Le(source);
}

static bool JetsonSpi_ValidateFrame(const uint8_t frame[32],
                                    uint8_t expected_type)
{
    uint8_t version;
    uint8_t type;
    uint16_t received_crc;
    uint16_t calculated_crc;

    if (frame[SPI_OFFSET_MAGIC] != JETSON_SPI_MAGIC)
    {
        return false;
    }

    version = (uint8_t)(frame[SPI_OFFSET_VERSION_TYPE] >> 4U);
    type = (uint8_t)(frame[SPI_OFFSET_VERSION_TYPE] & 0x0FU);

    if ((version != JETSON_SPI_PROTOCOL_VERSION) ||
        (type != expected_type))
    {
        return false;
    }

    received_crc = JetsonSpi_ReadU16Le(&frame[SPI_OFFSET_CRC]);
    calculated_crc = JetsonSpi_Crc16CcittFalse(
        frame,
        JETSON_SPI_CRC_INPUT_SIZE);

    return received_crc == calculated_crc;
}
```

검증에 실패한 프레임은 부분적으로 사용하지 않고 통째로 폐기한다. 명령 패킷의 경우 마지막 정상 명령을 제한 시간 동안만 유지하고, 통신 Timeout이 발생하면 안전 상태로 전환해야 한다.

## 10. Jetson Python 파싱 예제

```python
import struct
from dataclasses import dataclass

FRAME_SIZE = 32
MAGIC = 0xA5
PROTOCOL_VERSION = 1
PACKET_TYPE_SENSOR = 1

JOINT_MIN_RAD = -2.35619449
JOINT_MAX_RAD = 2.35619449


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF

    for value in data:
        crc ^= value << 8

        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF

    return crc


def decode_joint(encoded: int) -> float:
    return JOINT_MIN_RAD + (encoded / 255.0) * (
        JOINT_MAX_RAD - JOINT_MIN_RAD
    )


@dataclass
class SensorPacket:
    sequence: int
    delta_time_s: float
    joint_angle_rad: list[float]
    roll_rad: float
    pitch_rad: float
    yaw_rad: float


def parse_sensor_packet(frame: bytes) -> SensorPacket:
    if len(frame) != FRAME_SIZE:
        raise ValueError(f"invalid frame size: {len(frame)}")

    if frame[0] != MAGIC:
        raise ValueError("invalid magic")

    version = frame[1] >> 4
    packet_type = frame[1] & 0x0F

    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {version}")

    if packet_type != PACKET_TYPE_SENSOR:
        raise ValueError(f"unexpected packet type: {packet_type}")

    received_crc = struct.unpack_from("<H", frame, 30)[0]
    calculated_crc = crc16_ccitt_false(frame[:30])

    if received_crc != calculated_crc:
        raise ValueError(
            f"CRC mismatch: rx=0x{received_crc:04X}, "
            f"calc=0x{calculated_crc:04X}"
        )

    sequence = struct.unpack_from("<H", frame, 2)[0]
    delta_time_10us = struct.unpack_from("<H", frame, 4)[0]
    joints = [decode_joint(value) for value in frame[6:24]]
    roll_raw, pitch_raw, yaw_raw = struct.unpack_from("<hhh", frame, 24)

    return SensorPacket(
        sequence=sequence,
        delta_time_s=delta_time_10us * 0.00001,
        joint_angle_rad=joints,
        roll_rad=roll_raw / 10000.0,
        pitch_rad=pitch_raw / 10000.0,
        yaw_rad=yaw_raw / 10000.0,
    )
```

## 11. Jetson SPI 수신 예제

Jetson은 `DRDY`가 High인 것을 확인한 후 NSS/CS를 Low로 만들고 정확히 32바이트를 교환한다. Python `spidev`를 사용하는 최소 형태는 다음과 같다.

```python
import spidev

spi = spidev.SpiDev()
spi.open(0, 0)
spi.mode = 0
spi.max_speed_hz = 1_000_000
spi.bits_per_word = 8


def read_stm32_sensor_packet() -> SensorPacket:
    # 실제 코드에서는 이 호출 전에 DRDY GPIO가 High인지 확인한다.
    received = bytes(spi.xfer2([0x00] * FRAME_SIZE))
    return parse_sensor_packet(received)
```

`spi.open(bus, chip_select)` 값과 `DRDY` GPIO 번호는 Jetson의 실제 핀 설정에 맞춰 확정해야 한다. SPI 속도는 처음에는 1 MHz로 검증하고 배선과 신호 무결성을 확인한 뒤 높인다.

## 12. SPI 및 DRDY 동작 순서

권장 트랜잭션 순서는 다음과 같다.

1. STM32가 최신 센서 스냅샷으로 다음 `tx_frame[32]`를 완성한다.
2. STM32가 `DRDY`를 High로 만든다.
3. Jetson이 `DRDY == High`를 확인한다.
4. Jetson이 NSS/CS를 Low로 만들고 정확히 32바이트의 SPI 클록을 발생시킨다.
5. Jetson은 MOSI로 명령 또는 NOP 패킷을 보내면서 동시에 MISO로 센서 패킷을 받는다.
6. 32바이트 교환 후 Jetson이 NSS/CS를 High로 만든다.
7. STM32가 전송 완료를 확인하고 `DRDY`를 Low로 만든다.
8. STM32가 수신 프레임을 검증하고 다음 송신 프레임을 준비한다.
9. 준비가 끝나면 STM32가 다시 `DRDY`를 High로 만든다.

SPI는 전이중 통신이므로 같은 트랜잭션에서 송신과 수신이 동시에 진행된다. 따라서 Jetson이 이번 트랜잭션에서 보낸 명령의 처리 결과는 일반적으로 다음 트랜잭션의 STM32 송신 패킷에 반영된다.

## 13. 수신 검사 순서

Jetson과 STM32 모두 다음 순서로 검사한다.

1. 수신 길이가 정확히 32바이트인지 확인한다.
2. `MAGIC == 0xA5`인지 확인한다.
3. 프로토콜 버전을 확인한다.
4. 예상한 패킷 종류인지 확인한다.
5. Byte 0~29로 CRC를 다시 계산한다.
6. 계산 CRC와 Byte 30~31의 수신 CRC를 비교한다.
7. 순번을 이전 정상 패킷과 비교한다.
8. 모든 검사가 통과한 경우에만 데이터를 사용한다.

CRC 오류 패킷의 순번은 마지막 정상 순번으로 갱신하지 않는다.

## 14. 현재 코드 구현 상태

현재 구현된 기능은 다음과 같다.

- 18개 관절각과 IMU 자세의 32바이트 센서 프레임 생성
- 순번 증가와 패킷 생성 Delta time 계산
- CRC-16/CCITT-FALSE 생성 및 검증
- 수신 패킷의 시작값, 버전, 순번, 종류와 Payload 파싱
- `COMMAND` 패킷의 24바이트 Raw Payload 보관 및 일회성 소비 API
- 수신 순번 유실, 유효 패킷, 잘못된 패킷과 HAL SPI 오류 횟수 기록
- Jetson이 센서 읽기용으로 보내는 32바이트 `0x00` Dummy 프레임 허용
- 센서 프레임 준비 시 `DRDY` High, 트랜잭션 완료 시 Low
- `HAL_SPI_TransmitReceive(..., HAL_MAX_DELAY)`를 사용한 블로킹 처리

추가로 필요한 작업은 다음과 같다.

1. Jetson에서 보낼 `COMMAND` Payload의 24바이트 세부 배치를 정의한다.
2. 정상 `COMMAND` 패킷을 프로젝트의 자율주행 명령과 안전 우선순위에 연결한다.
3. 메인 제어 처리 전체가 Jetson의 SPI 클록을 무한 대기하지 않도록 인터럽트 또는 DMA 방식과 통신 Timeout을 검토한다.
4. Jetson 명령이 일정 시간 들어오지 않으면 자율주행 명령을 해제하고 프로젝트의 안전 상태로 전환한다.
5. 현재 `main.c`는 `MeasurementStage3` 측정 모드로 실행되므로 실제 로봇 앱 운용 단계에서 `HexapodApp_Init()`, `HexapodApp_Process()`와 `HexapodApp_RunControlIfDue()` 실행 경로로 전환한다.

## 15. 향후 확장 원칙

- Jetson에서 STM32로 보내는 제어 패킷도 동일한 `MAGIC`, 버전/종류, 순번, Delta time, CRC 위치를 사용한다.
- Byte 6~29의 24바이트 데이터 영역만 패킷 종류에 따라 다르게 해석한다.
- 기존 필드의 의미나 배율을 바꾸면 프로토콜 버전을 증가시킨다.
- 패킷 종류를 추가할 때 기존 종류의 바이트 배치를 변경하지 않는다.
- 각 패킷 종류별로 정상 패킷, CRC 오류, 순번 순환, 최대·최소 센서값에 대한 테스트 벡터를 만든다.
