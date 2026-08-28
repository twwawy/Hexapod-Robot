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

/* 외부에서 확인한 제어기 Fault를 복구 없이 Latch한다. */
void Safety_LatchControllerFault(Safety_Handle_t *handle)
{
    if (handle != NULL)
    {
        handle->latched.controller_fault = true;  // 상위 제어기 Fault를 영구 유지한다.
    }
}

/* IMU 자세만 평가하여 Fault를 Latch한다. */
RobotSafetyOutput_t Safety_EvaluateImu(Safety_Handle_t *handle,
                                       const RobotImuState_t *imu)
{
    bool imu_invalid;  // IMU 비유한 상태를 저장한다.

    if (handle == NULL)
    {
        RobotSafetyOutput_t invalid = {false, true};  // Handle 오류를 제어기 Fault로 만든다.
        return invalid;
    }

    imu_invalid = (imu == NULL) || !isfinite(imu->attitude_rad.roll) ||
                  !isfinite(imu->attitude_rad.pitch);  // Roll·Pitch 유한수를 검사한다.

    if (!imu_invalid &&
        ((fabsf(imu->attitude_rad.roll) >= ROBOT_ROLLOVER_LIMIT_RAD) ||
         (fabsf(imu->attitude_rad.pitch) >= ROBOT_ROLLOVER_LIMIT_RAD)))
    {
        handle->latched.rollover_fault = true;  // 전복 Fault를 영구 Latch한다.
    }

    if (imu_invalid)
    {
        Safety_LatchControllerFault(handle);  // 비유한 IMU를 즉시 영구 Fault로 올린다.
    }

    return handle->latched;
}

/* IMU와 연속 세 번의 IK 실패를 평가하여 Fault를 Latch한다. */
RobotSafetyOutput_t Safety_Evaluate(Safety_Handle_t *handle,
                                    const RobotImuState_t *imu,
                                    const bool ik_valid[ROBOT_LEG_COUNT])
{
    uint32_t leg;             // 검사할 다리 번호를 저장한다.
    bool ik_invalid = false;  // 이번 IK 실패 상태를 저장한다.

    if (handle == NULL)
    {
        RobotSafetyOutput_t invalid = {false, true};  // Handle 오류를 제어기 Fault로 만든다.
        return invalid;
    }

    (void)Safety_EvaluateImu(handle, imu);  // 현재 자세 Fault를 먼저 평가한다.

    if (ik_valid == NULL)
    {
        Safety_LatchControllerFault(handle);  // IK 결과 누락은 즉시 구조적 Fault로 올린다.
        return handle->latched;
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        if (!ik_valid[leg])
        {
            ik_invalid = true;  // 한 다리의 IK 실패를 이번 표본에 기록한다.
        }
    }

    if (ik_invalid)
    {
        if (handle->ik_failure_count < ROBOT_IK_FAULT_CONFIRM_SAMPLES)
        {
            handle->ik_failure_count++;  // 연속 IK 실패 횟수를 제한해 누적한다.
        }

        if (handle->ik_failure_count >= ROBOT_IK_FAULT_CONFIRM_SAMPLES)
        {
            Safety_LatchControllerFault(handle);  // 세 번째 연속 실패를 영구 Fault로 올린다.
        }
    }
    else
    {
        handle->ik_failure_count = 0U;  // 정상 IK 한 번에서 연속 실패를 초기화한다.
    }

    return handle->latched;
}
