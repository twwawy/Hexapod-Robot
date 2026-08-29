#include "sensor/imu.h"

#include "common/robot_config.h"

#include <math.h>
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
                    * IMU_ACCEL_FULL_SCALE_G * IMU_STANDARD_GRAVITY
                    * (float)handle->calibration.acceleration_sign[axis];
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
                    * IMU_GYRO_FULL_SCALE_DPS * IMU_DEG_TO_RAD
                    * (float)handle->calibration.angular_velocity_sign[axis];
            }
            handle->data.temperature_c = (float)IMU_ReadI16(&frame[8]) / 100.0f;
            handle->data.valid_mask |= IMU_VALID_GYROSCOPE;
            break;

        case 0x53U: /* roll, pitch, yaw and temperature */
            for (axis = 0U; axis < 3U; ++axis)
            {
                raw[axis] = IMU_ReadI16(&frame[2U + (axis * 2U)]);
                handle->data.euler_angle_rad[axis]
                    = ((float)raw[axis] / 32768.0f) * IMU_PI
                    * (float)handle->calibration.euler_angle_sign[axis]
                    - handle->calibration.euler_offset_rad[axis];
            }
            handle->data.temperature_c = (float)IMU_ReadI16(&frame[8]) / 100.0f;
            handle->data.valid_mask |= IMU_VALID_ANGLE;
            handle->data.angle_frame_counter++;  // 새 자세 표본 번호를 갱신한다.
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
    uint32_t axis;   // 보정할 축 번호를 저장한다.

    if (handle == NULL)
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));
    handle->uart = uart;

    for (axis = 0U; axis < 3U; ++axis)
    {
        handle->calibration.acceleration_sign[axis] = 1;       // 기본 축 부호를 유지한다.
        handle->calibration.angular_velocity_sign[axis] = 1;  // 기본 축 부호를 유지한다.
        handle->calibration.euler_angle_sign[axis] = 1;        // 기본 축 부호를 유지한다.
    }
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
    return (data != NULL) &&
           ((data->valid_mask & IMU_VALID_ANGLE) != 0U);  // 각도 프레임만 제어 유효값으로 인정한다.
}

/* 실측한 WT931 장착 방향과 자세 Offset을 적용한다. */
void IMU_SetCalibration(IMU_Handle_t *handle,
                        const IMU_Calibration_t *calibration)
{
    if ((handle == NULL) || (calibration == NULL))
    {
        return;
    }

    handle->calibration = *calibration;   // 이후 수신 프레임에 보정값을 적용한다.
}

/* 현재 사용 중인 WT931 보정값을 반환한다. */
void IMU_GetCalibration(const IMU_Handle_t *handle,
                        IMU_Calibration_t *calibration)
{
    if ((handle == NULL) || (calibration == NULL))
    {
        return;
    }

    *calibration = handle->calibration;   // 시험 코드에 보정값을 전달한다.
}

/* 현재 IMU 표본이 수평 영점 측정에 사용할 정지 상태인지 확인한다. */
static bool IMU_LevelCalibration_IsStationary(const IMU_Data_t *data)
{
    uint32_t axis;  // 검사할 각속도 축 번호를 저장한다.

    if ((data == NULL) ||
        ((data->valid_mask & IMU_VALID_ANGLE) == 0U) ||
        ((data->valid_mask & IMU_VALID_GYROSCOPE) == 0U) ||
        !isfinite(data->euler_angle_rad[0]) ||
        !isfinite(data->euler_angle_rad[1]))
    {
        return false;
    }

    for (axis = 0U; axis < 3U; ++axis)
    {
        if (!isfinite(data->angular_velocity_radps[axis]) ||
            (fabsf(data->angular_velocity_radps[axis]) > ROBOT_IMU_LEVEL_MAX_GYRO_RADPS))
        {
            return false;  // 한 축이라도 움직이면 현재 표본을 제외한다.
        }
    }
    return true;
}

/* 현재 정지 자세 표본 누적을 처음부터 다시 준비한다. */
static void IMU_LevelCalibration_ResetCapture(IMU_LevelCalibration_t *calibration,
                                              uint32_t now_ms)
{
    calibration->capture_start_ms = now_ms;  // 새 정지 구간 시작 시각을 저장한다.
    calibration->sample_count = 0U;          // 이전 정지 표본 수를 제거한다.
    calibration->euler_sum_rad[0] = 0.0f;    // 이전 Roll 합계를 제거한다.
    calibration->euler_sum_rad[1] = 0.0f;    // 이전 Pitch 합계를 제거한다.
}

/* Roll·Pitch 자동 영점 측정 상태를 초기화한다. */
void IMU_LevelCalibration_Init(IMU_LevelCalibration_t *calibration,
                               uint32_t now_ms)
{
    if (calibration == NULL)
    {
        return;
    }

    memset(calibration, 0, sizeof(*calibration));             // 이전 영점 측정값을 제거한다.
    calibration->state = IMU_LEVEL_CALIBRATION_WAITING;       // IMU 안정 대기부터 시작한다.
    calibration->start_ms = now_ms;                           // 전체 제한 시간 기준을 저장한다.
    calibration->capture_start_ms = now_ms;                   // 첫 정지 구간 기준을 준비한다.
}

/* 정지 자세 평균을 현재 보정표에 더해 Roll·Pitch 영점으로 적용한다. */
bool IMU_LevelCalibration_Update(IMU_LevelCalibration_t *calibration,
                                 IMU_Handle_t *imu,
                                 uint32_t now_ms)
{
    IMU_Calibration_t applied;  // 새로 적용할 IMU 보정표를 저장한다.
    float average_rad[2];       // 측정한 Roll·Pitch 평균을 저장한다.
    uint32_t axis;              // 계산할 수평 자세 축 번호를 저장한다.

    if ((calibration == NULL) || (imu == NULL))
    {
        return false;
    }
    if (calibration->state == IMU_LEVEL_CALIBRATION_COMPLETE)
    {
        return true;
    }
    if (calibration->state == IMU_LEVEL_CALIBRATION_FAILED)
    {
        return false;
    }
    if ((now_ms - calibration->start_ms) >= ROBOT_IMU_LEVEL_TIMEOUT_MS)
    {
        calibration->state = IMU_LEVEL_CALIBRATION_FAILED;  // 제한 시간 초과를 고정한다.
        return false;
    }
    if ((now_ms - calibration->start_ms) < ROBOT_IMU_LEVEL_SETTLE_MS)
    {
        return false;  // 장치 출력이 안정될 때까지 표본을 받지 않는다.
    }
    if ((imu->data.angle_frame_counter == 0U) ||
        (imu->data.angle_frame_counter == calibration->last_angle_frame_count))
    {
        return false;  // 같은 자세 프레임을 중복 합산하지 않는다.
    }

    calibration->last_angle_frame_count = imu->data.angle_frame_counter;  // 새 자세 프레임을 소비한다.
    if (!IMU_LevelCalibration_IsStationary(&imu->data))
    {
        calibration->state = IMU_LEVEL_CALIBRATION_WAITING;  // 정지 자세를 다시 기다린다.
        IMU_LevelCalibration_ResetCapture(calibration, now_ms);  // 움직인 구간의 표본을 제거한다.
        return false;
    }

    if (calibration->state != IMU_LEVEL_CALIBRATION_CAPTURING)
    {
        calibration->state = IMU_LEVEL_CALIBRATION_CAPTURING;  // 정지 표본 수집을 시작한다.
        IMU_LevelCalibration_ResetCapture(calibration, now_ms); // 새 정지 구간으로 초기화한다.
    }
    else if (calibration->sample_count > 0U)
    {
        for (axis = 0U; axis < 2U; ++axis)
        {
            const float mean = calibration->euler_sum_rad[axis] /
                               (float)calibration->sample_count;  // 현재 축 평균을 계산한다.

            if (fabsf(imu->data.euler_angle_rad[axis] - mean) >
                ROBOT_IMU_LEVEL_MAX_DEVIATION_RAD)
            {
                IMU_LevelCalibration_ResetCapture(calibration, now_ms);  // 자세가 흔들린 표본 구간을 제거한다.
                break;
            }
        }
    }

    calibration->euler_sum_rad[0] += imu->data.euler_angle_rad[0];  // 현재 Roll을 합산한다.
    calibration->euler_sum_rad[1] += imu->data.euler_angle_rad[1];  // 현재 Pitch를 합산한다.
    calibration->sample_count++;                                   // 정상 표본 수를 갱신한다.

    if ((calibration->sample_count < ROBOT_IMU_LEVEL_MIN_SAMPLES) ||
        ((now_ms - calibration->capture_start_ms) < ROBOT_IMU_LEVEL_CAPTURE_MS))
    {
        return false;
    }

    for (axis = 0U; axis < 2U; ++axis)
    {
        average_rad[axis] = calibration->euler_sum_rad[axis] /
                            (float)calibration->sample_count;  // 정지 자세 평균을 계산한다.
        if (!isfinite(average_rad[axis]) ||
            (fabsf(average_rad[axis]) > ROBOT_IMU_LEVEL_MAX_OFFSET_RAD))
        {
            calibration->state = IMU_LEVEL_CALIBRATION_FAILED;  // 기울어진 설치에서 자동 보정을 막는다.
            return false;
        }
    }

    IMU_GetCalibration(imu, &applied);  // 기존 축 방향과 영점값을 유지한다.
    for (axis = 0U; axis < 2U; ++axis)
    {
        applied.euler_offset_rad[axis] += average_rad[axis];          // 기존 영점에 측정 편차를 더한다.
        calibration->measured_offset_rad[axis] =
            applied.euler_offset_rad[axis];                           // 최종 적용 영점을 표시한다.
        calibration->measured_offset_deg[axis] =
            applied.euler_offset_rad[axis] * ROBOT_RAD_TO_DEG_F;      // 최종 영점을 deg로 표시한다.
        imu->data.euler_angle_rad[axis] -= average_rad[axis];         // 최신 자세값도 즉시 새 영점으로 맞춘다.
    }
    IMU_SetCalibration(imu, &applied);                                // 이후 자세 프레임에 새 영점을 적용한다.
    calibration->state = IMU_LEVEL_CALIBRATION_COMPLETE;              // 정상 보정 완료를 고정한다.
    return true;
}
