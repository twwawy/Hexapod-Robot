#include "measurement/imu_calibration_measurement.h"

#include "measurement/measurement_debug.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

/* WT931 수평 정지 Offset 누적 상태를 초기화한다. */
void ImuCalibrationMeasurement_Init(ImuCalibrationMeasurement_t *measurement)
{
    if (measurement != NULL)
    {
        memset(measurement, 0, sizeof(*measurement));  // 이전 자세 표본을 제거한다.
        g_measurement_debug.imu_sample_count = 0U;     // 디버거 표본 수를 초기화한다.
        g_measurement_debug.calibration.imu_calibrated = false;  // IMU 완료 상태를 초기화한다.
        g_measurement_debug.imu_euler_sum_rad[0] = 0.0;  // Roll 누적값을 초기화한다.
        g_measurement_debug.imu_euler_sum_rad[1] = 0.0;  // Pitch 누적값을 초기화한다.
        g_measurement_debug.imu_euler_sum_rad[2] = 0.0;  // Yaw 누적값을 초기화한다.
    }
}

/* 실제 WT931의 유효한 자세 표본 하나를 누적한다. */
bool ImuCalibrationMeasurement_AddSample(ImuCalibrationMeasurement_t *measurement,
                                         const IMU_Data_t *data)
{
    uint32_t axis;  // 자세 축 번호를 저장한다.

    if ((measurement == NULL) || (data == NULL) ||
        !IMU_HasNavigationData(data) || (measurement->sample_count == UINT32_MAX))
    {
        return false;
    }

    for (axis = 0U; axis < 3U; ++axis)
    {
        if (!isfinite(data->euler_angle_rad[axis]))
        {
            return false;
        }
        measurement->euler_sum_rad[axis] += data->euler_angle_rad[axis];  // 자세 Offset 표본을 누적한다.
        g_measurement_debug.imu_euler_sum_rad[axis] = measurement->euler_sum_rad[axis];  // 디버거 누적값을 갱신한다.
    }
    measurement->sample_count++;  // 정상 표본 수를 기록한다.
    g_measurement_debug.imu_sample_count = measurement->sample_count;  // 디버거 표본 수를 갱신한다.
    return true;
}

/* 확인한 축 부호와 수평 자세 평균으로 WT931 보정값을 만든다. */
bool ImuCalibrationMeasurement_Build(const ImuCalibrationMeasurement_t *measurement,
                                     const int8_t acceleration_sign[3],
                                     const int8_t angular_velocity_sign[3],
                                     const int8_t euler_angle_sign[3],
                                     IMU_Calibration_t *calibration)
{
    uint32_t axis;  // 보정할 축 번호를 저장한다.

    if ((measurement == NULL) || (acceleration_sign == NULL) ||
        (angular_velocity_sign == NULL) || (euler_angle_sign == NULL) ||
        (calibration == NULL) || (measurement->sample_count == 0U))
    {
        return false;
    }

    for (axis = 0U; axis < 3U; ++axis)
    {
        if (((acceleration_sign[axis] != 1) && (acceleration_sign[axis] != -1)) ||
            ((angular_velocity_sign[axis] != 1) && (angular_velocity_sign[axis] != -1)) ||
            ((euler_angle_sign[axis] != 1) && (euler_angle_sign[axis] != -1)))
        {
            return false;
        }

        calibration->acceleration_sign[axis] = acceleration_sign[axis];          // 가속도 장착 부호를 저장한다.
        calibration->angular_velocity_sign[axis] = angular_velocity_sign[axis];  // 각속도 장착 부호를 저장한다.
        calibration->euler_angle_sign[axis] = euler_angle_sign[axis];            // 자세 장착 부호를 저장한다.
        calibration->euler_offset_rad[axis] = (float)euler_angle_sign[axis] *
            (float)(measurement->euler_sum_rad[axis] /
                    (double)measurement->sample_count);                          // 수평 자세 Offset을 저장한다.
        g_measurement_debug.calibration.imu.acceleration_sign[axis] =
            calibration->acceleration_sign[axis];                                // 디버거 가속도 부호를 갱신한다.
        g_measurement_debug.calibration.imu.angular_velocity_sign[axis] =
            calibration->angular_velocity_sign[axis];                            // 디버거 각속도 부호를 갱신한다.
        g_measurement_debug.calibration.imu.euler_angle_sign[axis] =
            calibration->euler_angle_sign[axis];                                 // 디버거 자세 부호를 갱신한다.
        g_measurement_debug.calibration.imu.euler_offset_rad[axis] =
            calibration->euler_offset_rad[axis];                                 // 디버거 자세 Offset을 갱신한다.
    }

    g_measurement_debug.calibration.imu_calibrated = true;  // IMU 실측 완료를 표시한다.
    return true;
}
