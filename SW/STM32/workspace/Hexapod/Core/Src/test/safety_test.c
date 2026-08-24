#include "test/safety_test.h"

#include "high_control/safety.h"

#include <math.h>
#include <string.h>

/* Safety 경계와 Reset 없는 Fault Latch를 각각 검사한다. */
bool SafetyTest_Run(void)
{
    Safety_Handle_t handle;                    // 시험용 Fault Latch를 저장한다.
    RobotImuState_t imu;                       // 명시적인 IMU 자세를 저장한다.
    RobotSafetyOutput_t output;                // Safety 결과를 저장한다.
    bool ik_valid[ROBOT_LEG_COUNT];            // 명시적인 IK 결과를 저장한다.

    memset(&imu, 0, sizeof(imu));      // 수평 IMU를 준비한다.
    memset(ik_valid, 1, sizeof(ik_valid));  // 여섯 정상 IK를 준비한다.
    Safety_Init(&handle);              // 첫 Fault 시험을 초기화한다.
    output = Safety_Evaluate(&handle, &imu, ik_valid);  // 정상 입력을 평가한다.
    if (output.rollover_fault || output.controller_fault)
    {
        return false;
    }

    imu.attitude_rad.roll = ROBOT_ROLLOVER_LIMIT_RAD;  // 전복 경계값을 넣는다.
    output = Safety_Evaluate(&handle, &imu, ik_valid);  // 전복 Fault를 평가한다.
    imu.attitude_rad.roll = 0.0f;                       // 정상 자세로 되돌린다.
    output = Safety_Evaluate(&handle, &imu, ik_valid);  // Latch 유지 여부를 평가한다.
    if (!output.rollover_fault)
    {
        return false;
    }

    Safety_Init(&handle);                       // 제어기 Fault 시험을 새로 시작한다.
    ik_valid[2] = false;                        // 3번 다리 IK 실패를 넣는다.
    output = Safety_Evaluate(&handle, &imu, ik_valid);  // IK Fault를 평가한다.
    ik_valid[2] = true;                         // IK를 정상으로 되돌린다.
    if (!output.controller_fault ||
        !Safety_Evaluate(&handle, &imu, ik_valid).controller_fault)
    {
        return false;
    }

    Safety_Init(&handle);                       // 비유한 IMU 시험을 새로 시작한다.
    imu.attitude_rad.pitch = NAN;               // 비유한 Pitch를 넣는다.
    output = Safety_Evaluate(&handle, &imu, ik_valid);  // IMU Fault를 평가한다.
    return output.controller_fault;             // Controller Fault인지 확인한다.
}
