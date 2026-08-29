#include "communication/jetson_spi.h"

#include "main.h"

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

static int32_t JetsonSpi_RoundFloat(float value)
{
    if (value >= 0.0f)
    {
        return (int32_t)(value + 0.5f);
    }

    return (int32_t)(value - 0.5f);
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

static bool JetsonSpi_IsEmptyFrame(const uint8_t frame[JETSON_SPI_FRAME_SIZE])
{
    uint32_t index;

    for (index = 0U; index < JETSON_SPI_FRAME_SIZE; ++index)
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

bool JetsonSpi_ParseCommandFrame(const uint8_t frame[JETSON_SPI_FRAME_SIZE],
                                 JetsonSpi_CommandFrame_t *command)
{
    JetsonSpi_ParsedPacket_t parsed;

    if ((command == NULL) || !JetsonSpi_ParseFrame(frame, &parsed) ||
        (parsed.type != JETSON_SPI_TYPE_COMMAND))
    {
        return false;
    }

    command->sequence = parsed.sequence;
    command->delta_time_100us = parsed.delta_time_100us;
    command->flags = parsed.flags;
    memcpy(command->payload, parsed.payload, JETSON_SPI_PAYLOAD_SIZE);
    return true;
}

bool JetsonSpi_PrepareSensorFrame(JetsonSpi_Handle_t *handle,
                                  const RobotSensorSnapshot_t *snapshot,
                                  uint32_t now_ms)
{
    uint32_t elapsed_ms;
    uint8_t delta_time_100us;
    uint16_t crc;
    uint32_t joint;
    uint32_t leg;
    uint8_t contact_mask = 0U;

    if ((handle == NULL) || (snapshot == NULL) ||
        (handle->spi == NULL) || !handle->protocol_ready)
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
    handle->tx_frame[JETSON_SPI_OFFSET_MAGIC] = JETSON_SPI_MAGIC;
    handle->tx_frame[JETSON_SPI_OFFSET_VERSION_TYPE] =
        JETSON_SPI_MAKE_VERSION_TYPE(JETSON_SPI_PROTOCOL_VERSION,
                                     JETSON_SPI_TYPE_SENSOR);
    JetsonSpi_WriteU16Le(&handle->tx_frame[JETSON_SPI_OFFSET_SEQUENCE],
                         handle->tx_sequence);
    handle->tx_frame[JETSON_SPI_OFFSET_DELTA_TIME] = delta_time_100us;

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        handle->tx_frame[JETSON_SPI_OFFSET_JOINTS + joint] =
            JetsonSpi_EncodeJoint(snapshot->joint_angle_rad[joint]);
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        if (snapshot->foot_contact[leg])
        {
            contact_mask |= (uint8_t)(1U << leg);
        }
    }

    JetsonSpi_WriteI16Le(&handle->tx_frame[JETSON_SPI_OFFSET_IMU_ROLL],
                         JetsonSpi_EncodeImu(snapshot->imu.attitude_rad.roll));
    JetsonSpi_WriteI16Le(&handle->tx_frame[JETSON_SPI_OFFSET_IMU_PITCH],
                         JetsonSpi_EncodeImu(snapshot->imu.attitude_rad.pitch));
    JetsonSpi_WriteI16Le(&handle->tx_frame[JETSON_SPI_OFFSET_IMU_YAW],
                         JetsonSpi_EncodeImu(snapshot->imu.attitude_rad.yaw));

    handle->tx_frame[JETSON_SPI_OFFSET_FLAGS] =
        (uint8_t)(contact_mask & JETSON_SPI_SENSOR_FOOT_CONTACT_MASK);

    crc = JetsonSpi_Crc16CcittFalse(handle->tx_frame,
                                    JETSON_SPI_CRC_INPUT_SIZE);
    JetsonSpi_WriteU16Le(&handle->tx_frame[JETSON_SPI_OFFSET_CRC], crc);

    handle->tx_sequence++;
    handle->last_frame_ms = now_ms;
    handle->tx_frame_ready = true;
    return true;
}

static bool JetsonSpi_FinalizeTransfer(JetsonSpi_Handle_t *handle)
{
    JetsonSpi_ParsedPacket_t parsed;

    handle->tx_frame_ready = false;
    handle->rx_packet_valid = false;

    if (handle->transfer_error)
    {
        handle->transfer_error = false;
        handle->error_count++;
        return false;
    }

    handle->transfer_count++;

    /* Jetson이 센서 읽기만 수행하며 보낸 32바이트 0은 명령 오류로 세지 않는다. */
    if (JetsonSpi_IsEmptyFrame(handle->rx_frame))
    {
        return true;
    }

    if (!JetsonSpi_ParseFrame(handle->rx_frame, &parsed))
    {
        handle->invalid_rx_count++;
        return true;
    }

    if (handle->has_rx_sequence)
    {
        const uint16_t gap = (uint16_t)(parsed.sequence -
                                        handle->last_rx_sequence);
        if (gap > 1U)
        {
            handle->sequence_gap_count += (uint32_t)(gap - 1U);
        }
    }

    handle->rx_packet = parsed;
    handle->last_rx_sequence = parsed.sequence;
    handle->has_rx_sequence = true;
    handle->rx_packet_valid = true;
    handle->valid_rx_count++;

    if (parsed.type == JETSON_SPI_TYPE_COMMAND)
    {
        handle->command.sequence = parsed.sequence;
        handle->command.delta_time_100us = parsed.delta_time_100us;
        handle->command.flags = parsed.flags;
        memcpy(handle->command.payload,
               parsed.payload,
               JETSON_SPI_PAYLOAD_SIZE);
        handle->command_pending = true;
        handle->command_count++;
    }

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
                                         JETSON_SPI_FRAME_SIZE);
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
