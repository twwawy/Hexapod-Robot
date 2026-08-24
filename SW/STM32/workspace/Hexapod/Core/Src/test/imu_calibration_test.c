#include "test/imu_calibration_test.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

/* WT931 수평 정지 Offset 누적 상태를 초기화한다. */
void ImuCalibrationTest_Init(ImuCalibrationTest_t *test)
{
    if (test != NULL)
    {
        memset(test, 0, sizeof(*test));  // 이전 자세 표본을 제거한다.
    }
}

/* 실제 WT931의 유한한 자세 표본 하나를 누적한다. */
bool ImuCalibrationTest_AddSample(ImuCalibrationTest_t *test,
                                  const IMU_Data_t *data)
{
    uint32_t axis;  // 자세 축 번호를 저장한다.

    if ((test == NULL) || (data == NULL) || !IMU_HasNavigationData(data) ||
        (test->sample_count == UINT32_MAX))
    {
        return false;
    }

    for (axis = 0U; axis < 3U; ++axis)
    {
        if (!isfinite(data->euler_angle_rad[axis]))
        {
            return false;
        }
        test->euler_sum_rad[axis] += data->euler_angle_rad[axis];  // 자세 Offset 표본을 누적한다.
    }
    test->sample_count++;  // 정상 표본 수를 기록한다.
    return true;
}

/* 확인한 축 부호와 수평 자세 평균으로 WT931 보정값을 만든다. */
bool ImuCalibrationTest_Build(const ImuCalibrationTest_t *test,
                              const int8_t acceleration_sign[3],
                              const int8_t angular_velocity_sign[3],
                              const int8_t euler_angle_sign[3],
                              IMU_Calibration_t *calibration)
{
    uint32_t axis;  // 보정할 축 번호를 저장한다.

    if ((test == NULL) || (acceleration_sign == NULL) ||
        (angular_velocity_sign == NULL) || (euler_angle_sign == NULL) ||
        (calibration == NULL) || (test->sample_count == 0U))
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
            (float)(test->euler_sum_rad[axis] / (double)test->sample_count);      // 부호 적용 후 수평 자세 Offset을 저장한다.
    }

    return true;
}
