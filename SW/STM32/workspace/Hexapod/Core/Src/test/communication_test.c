#include "test/communication_test.h"

#include "communication/jetson_spi.h"
#include "communication/robot_telemetry.h"

#include <string.h>

/* STATUS 관제 패킷 생성과 비초기화 Jetson SPI 거부를 검사한다. */
bool CommunicationTest_Run(void)
{
    RobotTelemetry_Handle_t telemetry;                 // 시험용 전송 주기를 저장한다.
    RobotSensorSnapshot_t sensor;                       // 명시적인 관제 센서값을 저장한다.
    RobotSafetyOutput_t safety;                         // 명시적인 정상 Fault를 저장한다.
    JetsonSpi_Handle_t jetson;                          // 비초기화 SPI 상태를 저장한다.
    JetsonSpi_ParsedPacket_t parsed;                    // SPI 파싱 결과를 저장한다.
    JetsonSpi_CommandFrame_t command;                   // 명령 전용 파싱 결과를 저장한다.
    uint8_t spi_frame[JETSON_SPI_FRAME_SIZE];           // CRC 시험 패킷을 저장한다.
    uint16_t crc;                                       // 시험 패킷 CRC를 저장한다.
    char text[ROBOT_TELEMETRY_MAX_TEXT + 1U];           // 생성한 패킷을 저장한다.

    memset(&sensor, 0, sizeof(sensor));  // 관제 센서 입력을 0으로 준비한다.
    memset(&safety, 0, sizeof(safety));  // 정상 Safety 입력을 준비한다.
    memset(&jetson, 0, sizeof(jetson));  // 비초기화 SPI Handle을 준비한다.
    memset(spi_frame, 0, sizeof(spi_frame));  // SPI 시험 패킷을 0으로 준비한다.
    RobotTelemetry_Init(&telemetry);     // 패킷 주기를 초기화한다.
    if (RobotTelemetry_BuildNext(&telemetry, 999U, ROBOT_MODE_READY,
                                 &safety, &sensor, 0U, text, sizeof(text)))
    {
        return false;
    }
    if (!RobotTelemetry_BuildNext(&telemetry, 1000U, ROBOT_MODE_READY,
                                  &safety, &sensor, 0U, text, sizeof(text)) ||
        (strncmp(text, "S,", 2U) != 0))
    {
        return false;
    }

    spi_frame[JETSON_SPI_OFFSET_MAGIC] = JETSON_SPI_MAGIC;
    spi_frame[JETSON_SPI_OFFSET_VERSION_TYPE] =
        JETSON_SPI_MAKE_VERSION_TYPE(JETSON_SPI_PROTOCOL_VERSION,
                                     JETSON_SPI_TYPE_COMMAND);
    spi_frame[JETSON_SPI_OFFSET_SEQUENCE] = 0x34U;
    spi_frame[JETSON_SPI_OFFSET_SEQUENCE + 1U] = 0x12U;
    spi_frame[JETSON_SPI_OFFSET_DELTA_TIME] = 0xF4U;
    spi_frame[JETSON_SPI_OFFSET_DELTA_TIME + 1U] = 0x01U;
    spi_frame[JETSON_SPI_OFFSET_PAYLOAD] = 0x5AU;
    crc = JetsonSpi_Crc16CcittFalse(spi_frame,
                                    JETSON_SPI_CRC_INPUT_SIZE);
    spi_frame[JETSON_SPI_OFFSET_CRC] = (uint8_t)(crc & 0xFFU);
    spi_frame[JETSON_SPI_OFFSET_CRC + 1U] = (uint8_t)(crc >> 8U);

    if (!JetsonSpi_ParseFrame(spi_frame, &parsed) ||
        (parsed.type != JETSON_SPI_TYPE_COMMAND) ||
        (parsed.sequence != 0x1234U) ||
        (parsed.delta_time_10us != 500U))
    {
        return false;
    }

    if (!JetsonSpi_ParseCommandFrame(spi_frame, &command) ||
        (command.sequence != 0x1234U) ||
        (command.delta_time_10us != 500U) ||
        (command.payload[0] != 0x5AU))
    {
        return false;
    }

    spi_frame[JETSON_SPI_OFFSET_PAYLOAD] ^= 0x01U;  // CRC가 데이터 변조를 검출하는지 확인한다.
    if (JetsonSpi_ParseFrame(spi_frame, &parsed))
    {
        return false;
    }

    return !JetsonSpi_Process(&jetson);  // 준비되지 않은 SPI 전송을 거부하는지 확인한다.
}
