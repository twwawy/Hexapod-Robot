#include "high_control/safety.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

/* Reset 없는 두 Fault Latch를 초기화한다. */
void Safety_Init(Safety_Handle_t *handle)
{
    if (handle != NULL)
    {
        memset(handle, 0, sizeof(*handle));  // 전원 인가 시 Fault를 해제한다.
    }
}

/* IMU 자세와 여섯 IK 결과를 평가하여 Fault를 Latch한다. */
RobotSafetyOutput_t Safety_Evaluate(Safety_Handle_t *handle,
                                    const RobotImuState_t *imu,
                                    const bool ik_valid[ROBOT_LEG_COUNT])
{
    uint32_t leg;      // 검사할 다리 번호를 저장한다.
    bool imu_invalid;  // IMU 비유한 상태를 저장한다.
    bool ik_invalid = false;  // IK 실패 상태를 저장한다.

    if (handle == NULL)
    {
        RobotSafetyOutput_t invalid = {false, true};  // Handle 오류를 제어기 Fault로 만든다.
        return invalid;
    }

    imu_invalid = (imu == NULL) || !isfinite(imu->attitude_rad.roll) ||
                  !isfinite(imu->attitude_rad.pitch);  // Roll·Pitch 유한수를 검사한다.

    if (ik_valid == NULL)
    {
        ik_invalid = true;  // IK 배열 누락을 Fault로 처리한다.
    }
    else
    {
        for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
        {
            if (!ik_valid[leg])
            {
                ik_invalid = true;  // 한 다리의 IK 실패를 기록한다.
            }
        }
    }

    if (!imu_invalid &&
        ((fabsf(imu->attitude_rad.roll) >= ROBOT_ROLLOVER_LIMIT_RAD) ||
         (fabsf(imu->attitude_rad.pitch) >= ROBOT_ROLLOVER_LIMIT_RAD)))
    {
        handle->latched.rollover_fault = true;  // 전복 Fault를 영구 Latch한다.
    }

    if (imu_invalid || ik_invalid)
    {
        handle->latched.controller_fault = true;  // IMU 또는 IK Fault를 영구 Latch한다.
    }

    return handle->latched;
}
