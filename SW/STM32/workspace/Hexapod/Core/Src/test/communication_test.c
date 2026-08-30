#include "test/communication_test.h"

#include "communication/jetson_spi.h"
#include "communication/manipulator_link.h"
#include "communication/robot_telemetry.h"

#include <string.h>

/* STATUS 관제 패킷 생성과 비초기화 Jetson SPI 거부를 검사한다. */
bool CommunicationTest_Run(void)
{
    RobotTelemetry_Handle_t telemetry;                 // 시험용 전송 주기를 저장한다.
    RobotSensorSnapshot_t sensor;                       // 명시적인 관제 센서값을 저장한다.
    RobotSafetyOutput_t safety;                         // 명시적인 정상 Fault를 저장한다.
    JetsonSpi_Handle_t jetson;                          // 비초기화 SPI 상태를 저장한다.
    JetsonSpi_Handle_t jetson_tx;                       // 관절각 송신 프레임을 저장한다.
    JetsonSpi_ParsedPacket_t parsed;                    // SPI 파싱 결과를 저장한다.
    JetsonSpi_CommandFrame_t command;                   // 명령 전용 파싱 결과를 저장한다.
    RobotUserCommand_t user;                            // 매니퓰레이터 패킷 조종값을 저장한다.
    uint8_t spi_frame[JETSON_SPI_FRAME_SIZE];           // CRC 시험 패킷을 저장한다.
    uint8_t manipulator_packet[MANIPULATOR_PACKET_SIZE];  // 유선 조종 패킷을 저장한다.
    uint8_t expected_joint;                             // 관절별 예상 인코딩값을 저장한다.
    uint16_t crc;                                       // 시험 패킷 CRC를 저장한다.
    uint32_t joint;                                     // 검사할 관절 번호를 저장한다.
    uint32_t leg;                                       // 검사할 다리 번호를 저장한다.
    uint32_t joint_in_leg;                              // 다리 안의 관절 번호를 저장한다.
    bool invert_expected;                               // 송신 부호 반전 대상 여부를 저장한다.
    char text[ROBOT_TELEMETRY_MAX_TEXT + 1U];           // 생성한 패킷을 저장한다.

    if ((JETSON_SPI_FRAME_SIZE != 32U) ||
        (JETSON_SPI_PAYLOAD_SIZE != 24U) ||
        (JETSON_SPI_OFFSET_DELTA_TIME != 4U) ||
        (JETSON_SPI_OFFSET_FLAGS != 5U) ||
        (JETSON_SPI_OFFSET_CRC != 30U))
    {
        return false;
    }

    memset(&sensor, 0, sizeof(sensor));  // 관제 센서 입력을 0으로 준비한다.
    memset(&safety, 0, sizeof(safety));  // 정상 Safety 입력을 준비한다.
    memset(&jetson, 0, sizeof(jetson));  // 비초기화 SPI Handle을 준비한다.
    memset(&jetson_tx, 0, sizeof(jetson_tx));  // 관절각 송신 시험 상태를 준비한다.
    memset(&user, 0, sizeof(user));      // 매니퓰레이터 조종값을 0으로 준비한다.
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
    spi_frame[JETSON_SPI_OFFSET_DELTA_TIME] = 50U;
    spi_frame[JETSON_SPI_OFFSET_FLAGS] = 0x2DU;
    spi_frame[JETSON_SPI_OFFSET_PAYLOAD] = 0x5AU;
    crc = JetsonSpi_Crc16CcittFalse(spi_frame,
                                    JETSON_SPI_CRC_INPUT_SIZE);
    spi_frame[JETSON_SPI_OFFSET_CRC] = (uint8_t)(crc & 0xFFU);
    spi_frame[JETSON_SPI_OFFSET_CRC + 1U] = (uint8_t)(crc >> 8U);

    if (!JetsonSpi_ParseFrame(spi_frame, &parsed) ||
        (parsed.type != JETSON_SPI_TYPE_COMMAND) ||
        (parsed.sequence != 0x1234U) ||
        (parsed.delta_time_100us != 50U) ||
        (parsed.flags != 0x2DU))
    {
        return false;
    }

    if (!JetsonSpi_ParseCommandFrame(spi_frame, &command) ||
        (command.sequence != 0x1234U) ||
        (command.delta_time_100us != 50U) ||
        (command.flags != 0x2DU) ||
        (command.payload[0] != 0x5AU))
    {
        return false;
    }

    spi_frame[JETSON_SPI_OFFSET_PAYLOAD] ^= 0x01U;  // CRC가 데이터 변조를 검출하는지 확인한다.
    if (JetsonSpi_ParseFrame(spi_frame, &parsed))
    {
        return false;
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        sensor.joint_angle_rad[joint] = 1.0f;  // 모든 관절에 같은 양의 각도를 입력한다.
    }
    jetson_tx.spi = (SPI_HandleTypeDef *)&jetson_tx;  // 프레임 생성 조건만 만족하는 시험 포인터를 지정한다.
    jetson_tx.protocol_ready = true;                  // 송신 프로토콜 준비 상태를 지정한다.
    if (!JetsonSpi_PrepareSensorFrame(&jetson_tx, &sensor, true, 0U))
    {
        return false;
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        leg = joint / ROBOT_JOINTS_PER_LEG;             // 관절이 속한 다리를 구한다.
        joint_in_leg = joint % ROBOT_JOINTS_PER_LEG;    // 다리 안의 관절 번호를 구한다.
        invert_expected = ((leg < 3U) && (joint_in_leg == 1U)) ||
                          ((leg >= 3U) && (joint_in_leg == 2U));  // 요청된 여섯 관절만 선택한다.
        expected_joint = invert_expected ? 73U : 182U;           // -1 rad와 +1 rad의 인코딩값을 선택한다.

        if ((jetson_tx.tx_frame[JETSON_SPI_OFFSET_JOINTS + joint] != expected_joint) ||
            (sensor.joint_angle_rad[joint] != 1.0f))
        {
            return false;  // 송신값만 반전되고 센서 원본은 유지되는지 확인한다.
        }
    }

    jetson_tx.tx_frame_ready = false;  // 릴레이 OFF 프레임 생성을 허가한다.
    if (!JetsonSpi_PrepareSensorFrame(&jetson_tx, &sensor, false, 1U))
    {
        return false;
    }
    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        if ((jetson_tx.tx_frame[JETSON_SPI_OFFSET_JOINTS + joint] != 128U) ||
            (sensor.joint_angle_rad[joint] != 1.0f))
        {
            return false;  // 릴레이 OFF 시 송신각만 0도로 바뀌는지 확인한다.
        }
    }

    user.roll = -1000;           // 음수 Roll의 Little-endian 저장을 검사한다.
    user.pitch = 1000;           // 양수 Pitch의 Little-endian 저장을 검사한다.
    user.throttle = -321;        // 음수 Throttle 저장을 검사한다.
    user.yaw = 456;              // 양수 Yaw 저장을 검사한다.
    user.sa = 1U;                // 그리퍼 놓기 명령을 선택한다.
    user.sb = 2U;                // SB 끝 위치를 선택한다.
    user.sc = 1U;                // SC ARM 모드를 선택한다.
    user.sd = 0U;                // SD Kill을 해제한다.
    user.s1 = 1U;                // S1 두 번째 기능을 선택한다.
    user.connected = true;       // 조종기 연결을 표시한다.
    user.motion_armed = true;    // 조종 입력 허가를 표시한다.

    if (!ManipulatorLink_BuildPacket(manipulator_packet,
                                     0x34U,
                                     &user,
                                     ROBOT_MODE_ARM) ||
        (manipulator_packet[0] != 0xA5U) ||
        (manipulator_packet[1] != 0x5AU) ||
        (manipulator_packet[4] != 0x07U) ||
        (manipulator_packet[5] != 0x8DU) ||
        (manipulator_packet[6] != 0x18U) ||
        (manipulator_packet[7] != 0xFCU) ||
        (manipulator_packet[14] != 0x6CU) ||
        (manipulator_packet[15] != 0x1CU))
    {
        return false;  // 고정 패킷 배치와 알려진 CRC 값을 확인한다.
    }

    user.sa = 0U;  // 그리퍼 잡기 명령을 선택한다.
    if (!ManipulatorLink_BuildPacket(manipulator_packet,
                                     0x35U,
                                     &user,
                                     ROBOT_MODE_ARM) ||
        (manipulator_packet[4] != 0x07U) ||
        ((manipulator_packet[5] & 0x01U) != 0U))
    {
        return false;  // SA 잡기 명령이 ARM 허가를 해제하지 않는지 확인한다.
    }

    user.sd = 1U;  // SD Kill을 활성화한다.
    if (!ManipulatorLink_BuildPacket(manipulator_packet,
                                     0x36U,
                                     &user,
                                     ROBOT_MODE_ARM) ||
        (manipulator_packet[4] != 0x03U) ||
        (manipulator_packet[5] != 0xACU))
    {
        return false;  // Kill 상태가 ARM 허가를 제거하는지 확인한다.
    }

    return !JetsonSpi_Process(&jetson);  // 준비되지 않은 SPI 전송을 거부하는지 확인한다.
}
