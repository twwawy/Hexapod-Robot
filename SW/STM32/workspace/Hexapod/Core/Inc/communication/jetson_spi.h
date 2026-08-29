#ifndef JETSON_SPI_H
#define JETSON_SPI_H

#include "common/robot_types.h"
#include "stm32f4xx_hal.h"

#include <stdbool.h>
#include <stdint.h>

#define JETSON_SPI_FRAME_SIZE              32U
#define JETSON_SPI_PAYLOAD_SIZE            24U
#define JETSON_SPI_CRC_INPUT_SIZE          30U
#define JETSON_SPI_MAGIC                   0xA5U
#define JETSON_SPI_PROTOCOL_VERSION        2U

#define JETSON_SPI_OFFSET_MAGIC             0U
#define JETSON_SPI_OFFSET_VERSION_TYPE      1U
#define JETSON_SPI_OFFSET_SEQUENCE          2U
#define JETSON_SPI_OFFSET_DELTA_TIME        4U
#define JETSON_SPI_OFFSET_FLAGS             5U
#define JETSON_SPI_OFFSET_PAYLOAD           6U
#define JETSON_SPI_OFFSET_JOINTS            6U
#define JETSON_SPI_OFFSET_IMU_ROLL         24U
#define JETSON_SPI_OFFSET_IMU_PITCH        26U
#define JETSON_SPI_OFFSET_IMU_YAW          28U
#define JETSON_SPI_OFFSET_CRC              30U

#define JETSON_SPI_SENSOR_FOOT_CONTACT_MASK 0x3FU

#define JETSON_SPI_MAKE_VERSION_TYPE(version, type) \
    ((uint8_t)((((uint8_t)(version) & 0x0FU) << 4U) | \
               ((uint8_t)(type) & 0x0FU)))

typedef enum
{
    JETSON_SPI_TYPE_NONE = 0x0U,
    JETSON_SPI_TYPE_SENSOR = 0x1U,
    JETSON_SPI_TYPE_COMMAND = 0x2U,
    JETSON_SPI_TYPE_ACK = 0x3U,
    JETSON_SPI_TYPE_ERROR = 0x4U
} JetsonSpi_PacketType_t;

typedef struct
{
    JetsonSpi_PacketType_t type;                  // 하위 4비트의 패킷 종류를 저장한다.
    uint16_t sequence;                            // 송신 측 패킷 순번을 저장한다.
    uint8_t delta_time_100us;                     // 이전 패킷과의 간격을 100 us 단위로 저장한다.
    uint8_t flags;                                // SENSOR에서는 Bit 0~5가 Leg 1~6 접촉 상태다.
    uint8_t payload[JETSON_SPI_PAYLOAD_SIZE];     // Byte 6~29의 해석 전 Payload를 저장한다.
} JetsonSpi_ParsedPacket_t;

typedef struct
{
    uint16_t sequence;                            // Jetson 명령 패킷 순번을 저장한다.
    uint8_t delta_time_100us;                     // Jetson 명령 생성 간격을 100 us 단위로 저장한다.
    uint8_t flags;                                // COMMAND에서는 예약값이며 현재 0으로 송신한다.
    uint8_t payload[JETSON_SPI_PAYLOAD_SIZE];     // 아직 할당하지 않은 24바이트 명령 Payload다.
} JetsonSpi_CommandFrame_t;

typedef struct
{
    SPI_HandleTypeDef *spi;                       // SPI2 Slave Handle을 저장한다.
    uint8_t tx_frame[JETSON_SPI_FRAME_SIZE];      // 다음 전송에서 Jetson으로 보낼 프레임이다.
    uint8_t rx_frame[JETSON_SPI_FRAME_SIZE];      // Jetson에서 마지막으로 받은 프레임이다.
    JetsonSpi_ParsedPacket_t rx_packet;           // 마지막 정상 수신 패킷을 저장한다.
    JetsonSpi_CommandFrame_t command;             // 마지막 정상 Jetson 명령 프레임을 저장한다.
    uint16_t tx_sequence;                         // 다음 센서 패킷에 넣을 순번이다.
    uint16_t last_rx_sequence;                    // 마지막 정상 수신 순번이다.
    uint32_t last_frame_ms;                       // 마지막 센서 패킷 생성 시각이다.
    uint32_t transfer_count;                      // 완료된 SPI 트랜잭션 수를 저장한다.
    uint32_t valid_rx_count;                      // 정상 수신 프로토콜 패킷 수를 저장한다.
    uint32_t command_count;                       // 정상 수신 COMMAND 패킷 수를 저장한다.
    uint32_t invalid_rx_count;                    // 헤더 또는 CRC 오류 수를 저장한다.
    uint32_t sequence_gap_count;                  // 수신 패킷 유실 추정 개수를 누적한다.
    uint32_t error_count;                         // HAL SPI 오류 수를 저장한다.
    bool protocol_ready;                          // SPI 프로토콜 사용 가능 여부다.
    bool tx_frame_ready;                          // 송신할 센서 프레임 준비 여부다.
    bool rx_packet_valid;                         // 마지막 트랜잭션의 수신 패킷 유효 여부다.
    bool command_pending;                         // 아직 상위 코드가 소비하지 않은 명령 존재 여부다.
    bool has_rx_sequence;                         // 수신 순번 기준값 존재 여부다.
    volatile bool transfer_active;                // 인터럽트 방식 SPI 송수신 진행 여부다.
    volatile bool transfer_complete;              // ISR에서 설정하는 송수신 완료 플래그다.
    volatile bool transfer_error;                 // ISR에서 설정하는 SPI 오류 플래그다.
} JetsonSpi_Handle_t;

void JetsonSpi_Init(JetsonSpi_Handle_t *handle,
                    SPI_HandleTypeDef *spi);  // SPI2 Slave와 프로토콜 상태를 초기화한다.

uint16_t JetsonSpi_Crc16CcittFalse(const uint8_t *data,
                                   uint32_t length);  // CRC-16/CCITT-FALSE를 계산한다.

bool JetsonSpi_ParseFrame(const uint8_t frame[JETSON_SPI_FRAME_SIZE],
                          JetsonSpi_ParsedPacket_t *packet);  // 공통 헤더와 CRC를 검사해 Payload를 분리한다.

bool JetsonSpi_ParseCommandFrame(const uint8_t frame[JETSON_SPI_FRAME_SIZE],
                                 JetsonSpi_CommandFrame_t *command);  // COMMAND 패킷을 검사하고 Raw Payload를 분리한다.

bool JetsonSpi_PrepareSensorFrame(JetsonSpi_Handle_t *handle,
                                  const RobotSensorSnapshot_t *snapshot,
                                  uint32_t now_ms);  // 관절각, 발 접촉 상태와 IMU 자세로 송신 프레임을 만든다.

/*
 * 준비된 32바이트를 SPI2 Slave 인터럽트 방식으로 동시 송수신한다.
 * SPI를 먼저 Arm한 뒤 DRDY를 올리고, 완료된 프레임의 파싱은 메인 루프에서 처리한다.
 */
bool JetsonSpi_Process(JetsonSpi_Handle_t *handle);

bool JetsonSpi_GetLastRxPacket(const JetsonSpi_Handle_t *handle,
                               JetsonSpi_ParsedPacket_t *packet);  // 마지막 정상 수신 패킷을 복사한다.

bool JetsonSpi_TakeCommand(JetsonSpi_Handle_t *handle,
                           JetsonSpi_CommandFrame_t *command);  // 대기 중인 명령을 한 번 꺼내고 소비 처리한다.

#endif
