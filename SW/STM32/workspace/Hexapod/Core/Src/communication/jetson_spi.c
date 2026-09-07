#include "communication/jetson_spi.h"

#include "main.h"
#include "sensor/gps.h"

#include <stddef.h>
#include <string.h>

#define JETSON_SPI_JOINT_MIN_RAD      (-2.35619449f)
#define JETSON_SPI_JOINT_MAX_RAD      ( 2.35619449f)
#define JETSON_SPI_IMU_SCALE          10000.0f
#define JETSON_SPI_FIRST_DELTA_100US  50U
#define JETSON_SPI_MAX_DELTA_MS       25U

static JetsonSpi_Handle_t *g_jetson_spi_handle;

static uint16_t JetsonSpi_ReadU16Le(const uint8_t *source)
{
    return (uint16_t)source[0] |
           ((uint16_t)source[1] << 8U);
}

static void JetsonSpi_WriteU16Le(uint8_t *destination, uint16_t value)
{
    destination[0] = (uint8_t)(value & 0xFFU);
    destination[1] = (uint8_t)((value >> 8U) & 0xFFU);
}

static void JetsonSpi_WriteI16Le(uint8_t *destination, int16_t value)
{
    JetsonSpi_WriteU16Le(destination, (uint16_t)value);
}

static void JetsonSpi_WriteU32Le(uint8_t *destination, uint32_t value)
{
    destination[0] = (uint8_t)(value & 0xFFU);
    destination[1] = (uint8_t)((value >> 8U) & 0xFFU);
    destination[2] = (uint8_t)((value >> 16U) & 0xFFU);
    destination[3] = (uint8_t)((value >> 24U) & 0xFFU);
}

static void JetsonSpi_WriteI32Le(uint8_t *destination, int32_t value)
{
    JetsonSpi_WriteU32Le(destination, (uint32_t)value);
}

static int32_t JetsonSpi_RoundFloat(float value)
{
    if (value >= 0.0f)
    {
        return (int32_t)(value + 0.5f);
    }

    return (int32_t)(value - 0.5f);
}

/* Jetson 좌표계에 맞춰 송신할 관절각의 부호만 변환한다. */
static float JetsonSpi_GetTransmitJointAngle(uint32_t joint, float angle_rad)
{
    const uint32_t leg = joint / ROBOT_JOINTS_PER_LEG;             // 0부터 시작하는 다리 번호를 구한다.
    const uint32_t joint_in_leg = joint % ROBOT_JOINTS_PER_LEG;    // 다리 안의 관절 번호를 구한다.
    const bool invert = ((leg < 3U) && (joint_in_leg == 1U)) ||
                        ((leg >= 3U) && (joint_in_leg == 2U));      // 앞 세 다리 2번과 뒤 세 다리 3번 관절을 선택한다.

    return invert ? -angle_rad : angle_rad;                        // 원본을 바꾸지 않고 송신값만 반전한다.
}

static uint8_t JetsonSpi_EncodeJoint(float angle_rad)
{
    float normalized;
    int32_t encoded;

    if (angle_rad < JETSON_SPI_JOINT_MIN_RAD)
    {
        angle_rad = JETSON_SPI_JOINT_MIN_RAD;
    }
    if (angle_rad > JETSON_SPI_JOINT_MAX_RAD)
    {
        angle_rad = JETSON_SPI_JOINT_MAX_RAD;
    }

    normalized = (angle_rad - JETSON_SPI_JOINT_MIN_RAD) /
                 (JETSON_SPI_JOINT_MAX_RAD - JETSON_SPI_JOINT_MIN_RAD);
    encoded = JetsonSpi_RoundFloat(normalized * 255.0f);

    if (encoded < 0)
    {
        encoded = 0;
    }
    if (encoded > 255)
    {
        encoded = 255;
    }

    return (uint8_t)encoded;
}

static int16_t JetsonSpi_EncodeImu(float angle_rad)
{
    float scaled = angle_rad * JETSON_SPI_IMU_SCALE;

    if (scaled > 32767.0f)
    {
        scaled = 32767.0f;
    }
    if (scaled < -32768.0f)
    {
        scaled = -32768.0f;
    }

    return (int16_t)JetsonSpi_RoundFloat(scaled);
}

static int32_t JetsonSpi_EncodeI32(double scaled)
{
    if (scaled > 2147483647.0)
    {
        scaled = 2147483647.0;
    }
    if (scaled < -2147483648.0)
    {
        scaled = -2147483648.0;
    }

    return (int32_t)((scaled >= 0.0) ? (scaled + 0.5) : (scaled - 0.5));
}

static int16_t JetsonSpi_EncodeI16(float scaled)
{
    if (scaled > 32767.0f)
    {
        scaled = 32767.0f;
    }
    if (scaled < -32768.0f)
    {
        scaled = -32768.0f;
    }

    return (int16_t)JetsonSpi_RoundFloat(scaled);
}

static uint16_t JetsonSpi_EncodeU16(float scaled)
{
    if (scaled < 0.0f)
    {
        scaled = 0.0f;
    }
    if (scaled > 65535.0f)
    {
        scaled = 65535.0f;
    }

    return (uint16_t)JetsonSpi_RoundFloat(scaled);
}

static bool JetsonSpi_IsEmptyFrame(const uint8_t frame[JETSON_SPI_TRANSFER_SIZE])
{
    uint32_t index;

    for (index = 0U; index < JETSON_SPI_TRANSFER_SIZE; ++index)
    {
        if (frame[index] != 0U)
        {
            return false;
        }
    }

    return true;
}

void JetsonSpi_Init(JetsonSpi_Handle_t *handle,
                    SPI_HandleTypeDef *spi)
{
    if (handle == NULL)
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));
    handle->spi = spi;
    g_jetson_spi_handle = handle;
    HAL_GPIO_WritePin(DRDY_GPIO_Port, DRDY_Pin, GPIO_PIN_RESET);

    if (spi != NULL)
    {
        handle->protocol_ready = true;
    }
}

uint16_t JetsonSpi_Crc16CcittFalse(const uint8_t *data,
                                   uint32_t length)
{
    uint16_t crc = 0xFFFFU;
    uint32_t index;
    uint8_t bit;

    if (data == NULL)
    {
        return 0U;
    }

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
                crc = (uint16_t)(crc << 1U);
            }
        }
    }

    return crc;
}

bool JetsonSpi_ParseFrame(const uint8_t frame[JETSON_SPI_FRAME_SIZE],
                          JetsonSpi_ParsedPacket_t *packet)
{
    uint8_t version;
    uint16_t received_crc;
    uint16_t calculated_crc;

    if ((frame == NULL) || (packet == NULL))
    {
        return false;
    }

    if (frame[JETSON_SPI_OFFSET_MAGIC] != JETSON_SPI_MAGIC)
    {
        return false;
    }

    version = (uint8_t)(frame[JETSON_SPI_OFFSET_VERSION_TYPE] >> 4U);
    if (version != JETSON_SPI_PROTOCOL_VERSION)
    {
        return false;
    }

    received_crc = JetsonSpi_ReadU16Le(&frame[JETSON_SPI_OFFSET_CRC]);
    calculated_crc = JetsonSpi_Crc16CcittFalse(
        frame,
        JETSON_SPI_CRC_INPUT_SIZE);
    if (received_crc != calculated_crc)
    {
        return false;
    }

    packet->type = (JetsonSpi_PacketType_t)
        (frame[JETSON_SPI_OFFSET_VERSION_TYPE] & 0x0FU);
    packet->sequence = JetsonSpi_ReadU16Le(
        &frame[JETSON_SPI_OFFSET_SEQUENCE]);
    packet->delta_time_100us = frame[JETSON_SPI_OFFSET_DELTA_TIME];
    packet->flags = frame[JETSON_SPI_OFFSET_FLAGS];
    memcpy(packet->payload,
           &frame[JETSON_SPI_OFFSET_PAYLOAD],
           JETSON_SPI_PAYLOAD_SIZE);

    return true;
}

bool JetsonSpi_ParseCommandFrame(const uint8_t frame[JETSON_SPI_TRANSFER_SIZE],
                                  JetsonSpi_CommandFrame_t *command)
{
    uint8_t version;
    uint8_t type;
    uint16_t received_crc;
    uint16_t calculated_crc;

    if ((frame == NULL) || (command == NULL) ||
        (frame[JETSON_SPI_OFFSET_MAGIC] != JETSON_SPI_MAGIC))
    {
        return false;
    }

    version = (uint8_t)(frame[JETSON_SPI_OFFSET_VERSION_TYPE] >> 4U);
    type = (uint8_t)(frame[JETSON_SPI_OFFSET_VERSION_TYPE] & 0x0FU);
    if ((version != JETSON_SPI_PROTOCOL_VERSION) ||
        (type != (uint8_t)JETSON_SPI_TYPE_COMMAND))
    {
        return false;
    }

    received_crc = JetsonSpi_ReadU16Le(
        &frame[JETSON_SPI_COMMAND_OFFSET_CRC]);
    calculated_crc = JetsonSpi_Crc16CcittFalse(
        frame,
        JETSON_SPI_COMMAND_CRC_INPUT_SIZE);
    if (received_crc != calculated_crc)
    {
        return false;
    }

    command->sequence = JetsonSpi_ReadU16Le(
        &frame[JETSON_SPI_OFFSET_SEQUENCE]);
    command->delta_time_100us = frame[JETSON_SPI_OFFSET_DELTA_TIME];
    command->flags = frame[JETSON_SPI_OFFSET_FLAGS];
    memcpy(command->payload,
           &frame[JETSON_SPI_COMMAND_OFFSET_PAYLOAD],
           JETSON_SPI_COMMAND_PAYLOAD_SIZE);
    return true;
}

bool JetsonSpi_PrepareSensorFrame(JetsonSpi_Handle_t *handle,
                                  const RobotSensorSnapshot_t *snapshot,
                                  bool relay_enabled,
                                  uint32_t now_ms)
{
    uint32_t elapsed_ms;
    uint8_t delta_time_100us;
    uint16_t crc;
    uint32_t joint;
    uint32_t leg;
    uint32_t gps_age_ms;
    uint8_t contact_mask = 0U;
    uint8_t gps_flags = 0U;
    uint8_t gps_age_100ms;
    uint8_t *sensor_frame;
    uint8_t *gps_frame;

    if ((handle == NULL) || (snapshot == NULL) ||
        (handle->spi == NULL) || !handle->protocol_ready ||
        handle->tx_frame_ready || handle->transfer_active)
    {
        return false;
    }

    if (handle->transfer_count == 0U)
    {
        delta_time_100us = JETSON_SPI_FIRST_DELTA_100US;
    }
    else
    {
        elapsed_ms = now_ms - handle->last_frame_ms;
        if (elapsed_ms > JETSON_SPI_MAX_DELTA_MS)
        {
            elapsed_ms = JETSON_SPI_MAX_DELTA_MS;
        }
        delta_time_100us = (uint8_t)(elapsed_ms * 10U);
    }

    memset(handle->tx_frame, 0, sizeof(handle->tx_frame));
    sensor_frame = &handle->tx_frame[JETSON_SPI_SENSOR_FRAME_OFFSET];
    gps_frame = &handle->tx_frame[JETSON_SPI_GPS_FRAME_OFFSET];

    sensor_frame[JETSON_SPI_OFFSET_MAGIC] = JETSON_SPI_MAGIC;
    sensor_frame[JETSON_SPI_OFFSET_VERSION_TYPE] =
        JETSON_SPI_MAKE_VERSION_TYPE(JETSON_SPI_PROTOCOL_VERSION,
                                     JETSON_SPI_TYPE_SENSOR);
    JetsonSpi_WriteU16Le(&sensor_frame[JETSON_SPI_OFFSET_SEQUENCE],
                         handle->tx_sequence);
    sensor_frame[JETSON_SPI_OFFSET_DELTA_TIME] = delta_time_100us;

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        const float angle_rad = relay_enabled ?
            snapshot->joint_angle_rad[joint] : 0.0f;  // 릴레이 OFF 시 ADC 대신 0도를 선택한다.

        sensor_frame[JETSON_SPI_OFFSET_JOINTS + joint] =
            JetsonSpi_EncodeJoint(JetsonSpi_GetTransmitJointAngle(
                joint, angle_rad));  // Jetson 송신 좌표계로만 변환해 인코딩한다.
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        if (snapshot->foot_contact[leg])
        {
            contact_mask |= (uint8_t)(1U << leg);
        }
    }

    JetsonSpi_WriteI16Le(&sensor_frame[JETSON_SPI_OFFSET_IMU_ROLL],
                         JetsonSpi_EncodeImu(snapshot->imu.attitude_rad.roll));
    JetsonSpi_WriteI16Le(&sensor_frame[JETSON_SPI_OFFSET_IMU_PITCH],
                         JetsonSpi_EncodeImu(snapshot->imu.attitude_rad.pitch));
    JetsonSpi_WriteI16Le(&sensor_frame[JETSON_SPI_OFFSET_IMU_YAW],
                         JetsonSpi_EncodeImu(snapshot->imu.attitude_rad.yaw));

    sensor_frame[JETSON_SPI_OFFSET_FLAGS] =
        (uint8_t)(contact_mask & JETSON_SPI_SENSOR_FOOT_CONTACT_MASK);

    crc = JetsonSpi_Crc16CcittFalse(sensor_frame,
                                    JETSON_SPI_CRC_INPUT_SIZE);
    JetsonSpi_WriteU16Le(&sensor_frame[JETSON_SPI_OFFSET_CRC], crc);

    if ((snapshot->gps.timestamp_ms != 0U) &&
        (snapshot->gps.timestamp_ms != handle->last_gps_timestamp_ms))
    {
        handle->gps_sequence++;
        handle->last_gps_timestamp_ms = snapshot->gps.timestamp_ms;
    }

    gps_age_ms = (snapshot->gps.timestamp_ms == 0U) ? UINT32_MAX :
                 (now_ms - snapshot->gps.timestamp_ms);
    gps_age_100ms = (gps_age_ms >= 25500U) ? 255U :
                    (uint8_t)(gps_age_ms / 100U);

    if (snapshot->gps.fix_ok)
    {
        gps_flags |= JETSON_SPI_GPS_FLAG_FIX_OK;
    }
    if (snapshot->gps.valid)
    {
        gps_flags |= JETSON_SPI_GPS_FLAG_POSITION_VALID;
    }
    if (snapshot->gps.velocity_valid)
    {
        gps_flags |= JETSON_SPI_GPS_FLAG_VELOCITY_VALID;
    }
    if (snapshot->gps.time_valid)
    {
        gps_flags |= JETSON_SPI_GPS_FLAG_TIME_VALID;
    }
    if (snapshot->gps.protocol == (uint8_t)GPS_PROTOCOL_UBX)
    {
        gps_flags |= JETSON_SPI_GPS_FLAG_PROTOCOL_UBX;
    }
    if (snapshot->gps.protocol == (uint8_t)GPS_PROTOCOL_NMEA)
    {
        gps_flags |= JETSON_SPI_GPS_FLAG_PROTOCOL_NMEA;
    }

    gps_frame[JETSON_SPI_OFFSET_MAGIC] = JETSON_SPI_MAGIC;
    gps_frame[JETSON_SPI_OFFSET_VERSION_TYPE] =
        JETSON_SPI_MAKE_VERSION_TYPE(JETSON_SPI_PROTOCOL_VERSION,
                                     JETSON_SPI_TYPE_GPS);
    JetsonSpi_WriteU16Le(&gps_frame[JETSON_SPI_OFFSET_SEQUENCE],
                         handle->gps_sequence);
    gps_frame[JETSON_SPI_OFFSET_DELTA_TIME] = gps_age_100ms;
    gps_frame[JETSON_SPI_OFFSET_FLAGS] = gps_flags;
    JetsonSpi_WriteI32Le(&gps_frame[JETSON_SPI_GPS_OFFSET_LATITUDE],
                         JetsonSpi_EncodeI32(snapshot->gps.latitude_deg * 1.0e7));
    JetsonSpi_WriteI32Le(&gps_frame[JETSON_SPI_GPS_OFFSET_LONGITUDE],
                         JetsonSpi_EncodeI32(snapshot->gps.longitude_deg * 1.0e7));
    JetsonSpi_WriteI32Le(&gps_frame[JETSON_SPI_GPS_OFFSET_ALTITUDE],
                         JetsonSpi_EncodeI32((double)snapshot->gps.altitude_m * 1000.0));
    JetsonSpi_WriteU32Le(&gps_frame[JETSON_SPI_GPS_OFFSET_ITOW],
                         snapshot->gps.i_tow_ms);
    JetsonSpi_WriteI16Le(&gps_frame[JETSON_SPI_GPS_OFFSET_VELOCITY_N],
                         JetsonSpi_EncodeI16(snapshot->gps.velocity_north_mps * 100.0f));
    JetsonSpi_WriteI16Le(&gps_frame[JETSON_SPI_GPS_OFFSET_VELOCITY_E],
                         JetsonSpi_EncodeI16(snapshot->gps.velocity_east_mps * 100.0f));
    JetsonSpi_WriteU16Le(&gps_frame[JETSON_SPI_GPS_OFFSET_HACC],
                         JetsonSpi_EncodeU16(snapshot->gps.horizontal_accuracy_m * 100.0f));
    gps_frame[JETSON_SPI_GPS_OFFSET_SATELLITES] = snapshot->gps.satellites_used;
    gps_frame[JETSON_SPI_GPS_OFFSET_FIX_TYPE] = snapshot->gps.fix_type;
    crc = JetsonSpi_Crc16CcittFalse(gps_frame,
                                    JETSON_SPI_CRC_INPUT_SIZE);
    JetsonSpi_WriteU16Le(&gps_frame[JETSON_SPI_OFFSET_CRC], crc);

    handle->tx_sequence++;
    handle->last_frame_ms = now_ms;
    handle->tx_frame_ready = true;
    return true;
}

static bool JetsonSpi_FinalizeTransfer(JetsonSpi_Handle_t *handle)
{
    JetsonSpi_CommandFrame_t command;

    handle->tx_frame_ready = false;
    handle->rx_packet_valid = false;

    if (handle->transfer_error)
    {
        handle->transfer_error = false;
        handle->error_count++;
        return false;
    }

    handle->transfer_count++;

    /* Jetson이 센서 읽기만 수행하며 보낸 64바이트 0은 명령 오류로 세지 않는다. */
    if (JetsonSpi_IsEmptyFrame(handle->rx_frame))
    {
        return true;
    }

    if (!JetsonSpi_ParseCommandFrame(handle->rx_frame, &command))
    {
        handle->invalid_rx_count++;
        return true;
    }

    if (handle->has_rx_sequence)
    {
        const uint16_t gap = (uint16_t)(command.sequence -
                                        handle->last_rx_sequence);
        if (gap > 1U)
        {
            handle->sequence_gap_count += (uint32_t)(gap - 1U);
        }
    }

    handle->rx_packet.type = JETSON_SPI_TYPE_COMMAND;
    handle->rx_packet.sequence = command.sequence;
    handle->rx_packet.delta_time_100us = command.delta_time_100us;
    handle->rx_packet.flags = command.flags;
    memcpy(handle->rx_packet.payload,
           command.payload,
           JETSON_SPI_PAYLOAD_SIZE);  // 기존 조회 API에는 명령 Payload 앞 24바이트를 제공한다.
    handle->last_rx_sequence = command.sequence;
    handle->has_rx_sequence = true;
    handle->rx_packet_valid = true;
    handle->valid_rx_count++;

    handle->command = command;
    handle->command_pending = true;
    handle->command_count++;

    return true;
}

bool JetsonSpi_Process(JetsonSpi_Handle_t *handle)
{
    HAL_StatusTypeDef status;

    if ((handle == NULL) || (handle->spi == NULL) ||
        !handle->protocol_ready)
    {
        return false;
    }

    if (handle->transfer_complete)
    {
        handle->transfer_complete = false;
        return JetsonSpi_FinalizeTransfer(handle);
    }

    if (handle->transfer_active)
    {
        return true;
    }

    if (!handle->tx_frame_ready)
    {
        return false;
    }

    handle->transfer_error = false;
    status = HAL_SPI_TransmitReceive_DMA(handle->spi,
                                         handle->tx_frame,
                                         handle->rx_frame,
                                         JETSON_SPI_TRANSFER_SIZE);
    if (status != HAL_OK)
    {
        handle->error_count++;
        return false;
    }

    handle->transfer_active = true;
    HAL_GPIO_WritePin(DRDY_GPIO_Port, DRDY_Pin, GPIO_PIN_SET);
    return true;
}

void HAL_SPI_TxRxCpltCallback(SPI_HandleTypeDef *hspi)
{
    if ((g_jetson_spi_handle == NULL) ||
        (g_jetson_spi_handle->spi != hspi))
    {
        return;
    }

    HAL_GPIO_WritePin(DRDY_GPIO_Port, DRDY_Pin, GPIO_PIN_RESET);
    g_jetson_spi_handle->transfer_active = false;
    g_jetson_spi_handle->transfer_complete = true;
}

void HAL_SPI_ErrorCallback(SPI_HandleTypeDef *hspi)
{
    if ((g_jetson_spi_handle == NULL) ||
        (g_jetson_spi_handle->spi != hspi))
    {
        return;
    }

    HAL_GPIO_WritePin(DRDY_GPIO_Port, DRDY_Pin, GPIO_PIN_RESET);
    g_jetson_spi_handle->transfer_active = false;
    g_jetson_spi_handle->transfer_error = true;
    g_jetson_spi_handle->transfer_complete = true;
}

bool JetsonSpi_GetLastRxPacket(const JetsonSpi_Handle_t *handle,
                               JetsonSpi_ParsedPacket_t *packet)
{
    if ((handle == NULL) || (packet == NULL) || !handle->rx_packet_valid)
    {
        return false;
    }

    *packet = handle->rx_packet;
    return true;
}

bool JetsonSpi_TakeCommand(JetsonSpi_Handle_t *handle,
                           JetsonSpi_CommandFrame_t *command)
{
    if ((handle == NULL) || (command == NULL) || !handle->command_pending)
    {
        return false;
    }

    *command = handle->command;
    handle->command_pending = false;
    return true;
}
