#include "sensor/imu.h"

#include <string.h>

#define IMU_PI                  3.14159265358979323846f
#define IMU_STANDARD_GRAVITY    9.80665f
#define IMU_ACCEL_FULL_SCALE_G  16.0f
#define IMU_GYRO_FULL_SCALE_DPS 2000.0f
#define IMU_DEG_TO_RAD          (IMU_PI / 180.0f)

_Static_assert((IMU_RX_BUFFER_SIZE & (IMU_RX_BUFFER_SIZE - 1U)) == 0U,
               "IMU_RX_BUFFER_SIZE must be a power of two");

static int16_t IMU_ReadI16(const uint8_t *data)
{
    return (int16_t)((uint16_t)data[0] | ((uint16_t)data[1] << 8));
}

static void IMU_QueueRxByte(IMU_Handle_t *handle, uint8_t byte)
{
    uint16_t next_head;

    next_head = (uint16_t)((handle->rx_head + 1U) & (IMU_RX_BUFFER_SIZE - 1U));
    if (next_head == handle->rx_tail)
    {
        handle->rx_overflow_count++;
        return;
    }

    handle->rx_buffer[handle->rx_head] = byte;
    handle->rx_head = next_head;
}

static bool IMU_ChecksumIsValid(const uint8_t *frame)
{
    uint8_t checksum = 0U;
    uint32_t index;

    for (index = 0U; index < (IMU_WIT_FRAME_SIZE - 1U); ++index)
    {
        checksum = (uint8_t)(checksum + frame[index]);
    }

    return checksum == frame[IMU_WIT_FRAME_SIZE - 1U];
}

static void IMU_ParseFrame(IMU_Handle_t *handle)
{
    const uint8_t *frame = handle->frame;
    int16_t raw[4];
    uint32_t axis;

    if (!IMU_ChecksumIsValid(frame))
    {
        handle->data.checksum_error_count++;
        return;
    }

    switch (frame[1])
    {
        case 0x51U: /* acceleration and temperature */
            for (axis = 0U; axis < 3U; ++axis)
            {
                raw[axis] = IMU_ReadI16(&frame[2U + (axis * 2U)]);
                handle->data.acceleration_mps2[axis]
                    = ((float)raw[axis] / 32768.0f)
                    * IMU_ACCEL_FULL_SCALE_G * IMU_STANDARD_GRAVITY;
            }
            handle->data.temperature_c = (float)IMU_ReadI16(&frame[8]) / 100.0f;
            handle->data.valid_mask |= IMU_VALID_ACCELEROMETER;
            break;

        case 0x52U: /* angular velocity and temperature */
            for (axis = 0U; axis < 3U; ++axis)
            {
                raw[axis] = IMU_ReadI16(&frame[2U + (axis * 2U)]);
                handle->data.angular_velocity_radps[axis]
                    = ((float)raw[axis] / 32768.0f)
                    * IMU_GYRO_FULL_SCALE_DPS * IMU_DEG_TO_RAD;
            }
            handle->data.temperature_c = (float)IMU_ReadI16(&frame[8]) / 100.0f;
            handle->data.valid_mask |= IMU_VALID_GYROSCOPE;
            break;

        case 0x53U: /* roll, pitch, yaw and temperature */
            for (axis = 0U; axis < 3U; ++axis)
            {
                raw[axis] = IMU_ReadI16(&frame[2U + (axis * 2U)]);
                handle->data.euler_angle_rad[axis]
                    = ((float)raw[axis] / 32768.0f) * IMU_PI;
            }
            handle->data.temperature_c = (float)IMU_ReadI16(&frame[8]) / 100.0f;
            handle->data.valid_mask |= IMU_VALID_ANGLE;
            break;

        case 0x54U: /* magnetic field; keep device counts until calibrated */
            for (axis = 0U; axis < 3U; ++axis)
            {
                handle->data.magnetic_raw[axis]
                    = IMU_ReadI16(&frame[2U + (axis * 2U)]);
            }
            handle->data.temperature_c = (float)IMU_ReadI16(&frame[8]) / 100.0f;
            handle->data.valid_mask |= IMU_VALID_MAGNETOMETER;
            break;

        case 0x59U: /* quaternion q0, q1, q2, q3 */
            for (axis = 0U; axis < 4U; ++axis)
            {
                raw[axis] = IMU_ReadI16(&frame[2U + (axis * 2U)]);
                handle->data.quaternion[axis] = (float)raw[axis] / 32768.0f;
            }
            handle->data.valid_mask |= IMU_VALID_QUATERNION;
            break;

        default:
            return;
    }

    handle->data.mcu_time_ms = HAL_GetTick();
    handle->data.frame_counter++;
}

void IMU_Init(IMU_Handle_t *handle, UART_HandleTypeDef *uart)
{
    if (handle == NULL)
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));
    handle->uart = uart;
}

HAL_StatusTypeDef IMU_Start(IMU_Handle_t *handle)
{
    if ((handle == NULL) || (handle->uart == NULL))
    {
        return HAL_ERROR;
    }

    return HAL_UART_Receive_IT(handle->uart, &handle->rx_byte, 1U);
}

void IMU_RxCpltCallback(IMU_Handle_t *handle, UART_HandleTypeDef *uart)
{
    if ((handle == NULL) || (uart != handle->uart))
    {
        return;
    }

    IMU_QueueRxByte(handle, handle->rx_byte);
    (void)HAL_UART_Receive_IT(handle->uart, &handle->rx_byte, 1U);
}

void IMU_ErrorCallback(IMU_Handle_t *handle, UART_HandleTypeDef *uart)
{
    if ((handle == NULL) || (uart != handle->uart))
    {
        return;
    }

    (void)HAL_UART_AbortReceive(handle->uart);
    (void)HAL_UART_Receive_IT(handle->uart, &handle->rx_byte, 1U);
}

uint32_t IMU_Process(IMU_Handle_t *handle)
{
    uint32_t processed = 0U;
    uint8_t byte;

    if (handle == NULL)
    {
        return 0U;
    }

    while (handle->rx_tail != handle->rx_head)
    {
        byte = handle->rx_buffer[handle->rx_tail];
        handle->rx_tail = (uint16_t)((handle->rx_tail + 1U)
                                    & (IMU_RX_BUFFER_SIZE - 1U));
        IMU_ProcessByte(handle, byte);
        processed++;
    }

    return processed;
}

void IMU_ProcessByte(IMU_Handle_t *handle, uint8_t byte)
{
    if (handle == NULL)
    {
        return;
    }

    if (handle->frame_index == 0U)
    {
        if (byte == 0x55U)
        {
            handle->frame[0] = byte;
            handle->frame_index = 1U;
        }
        return;
    }

    handle->frame[handle->frame_index++] = byte;
    if (handle->frame_index >= IMU_WIT_FRAME_SIZE)
    {
        IMU_ParseFrame(handle);
        handle->frame_index = 0U;
    }
}

bool IMU_GetLatest(const IMU_Handle_t *handle, IMU_Data_t *out)
{
    if ((handle == NULL) || (out == NULL) || (handle->data.frame_counter == 0U))
    {
        return false;
    }

    *out = handle->data;
    return true;
}

bool IMU_HasNavigationData(const IMU_Data_t *data)
{
    const uint32_t required = IMU_VALID_ACCELEROMETER
                            | IMU_VALID_GYROSCOPE
                            | IMU_VALID_ANGLE;

    return (data != NULL) && ((data->valid_mask & required) == required);
}
