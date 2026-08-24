#include "test/communication_test.h"

#include "communication/jetson_spi.h"
#include "communication/robot_telemetry.h"

#include <string.h>

/* STATUS 관제 패킷 생성과 미정 Jetson 프로토콜 비활성을 검사한다. */
bool CommunicationTest_Run(void)
{
    RobotTelemetry_Handle_t telemetry;                 // 시험용 전송 주기를 저장한다.
    RobotSensorSnapshot_t sensor;                       // 명시적인 관제 센서값을 저장한다.
    RobotSafetyOutput_t safety;                         // 명시적인 정상 Fault를 저장한다.
    JetsonSpi_Handle_t jetson;                          // 미정 SPI 상태를 저장한다.
    char text[ROBOT_TELEMETRY_MAX_TEXT + 1U];           // 생성한 패킷을 저장한다.

    memset(&sensor, 0, sizeof(sensor));  // 관제 센서 입력을 0으로 준비한다.
    memset(&safety, 0, sizeof(safety));  // 정상 Safety 입력을 준비한다.
    memset(&jetson, 0, sizeof(jetson));  // 미정 SPI Handle을 준비한다.
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

    return !JetsonSpi_Process(&jetson);  // 프로토콜 확정 전 전송이 없는지 확인한다.
}
