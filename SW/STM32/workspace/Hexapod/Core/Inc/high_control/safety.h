#ifndef SAFETY_H
#define SAFETY_H

#include "common/robot_types.h"

typedef struct
{
    RobotSafetyOutput_t latched;  // Reset 없는 Fault Latch를 저장한다.
    uint8_t ik_failure_count;     // 연속 IK 실패 횟수를 저장한다.
} Safety_Handle_t;

void Safety_Init(Safety_Handle_t *handle);  // Fault Latch를 초기화한다.

void Safety_LatchControllerFault(Safety_Handle_t *handle);  // 외부 제어기 Fault를 영구 Latch한다.

RobotSafetyOutput_t Safety_EvaluateImu(Safety_Handle_t *handle,
                                       const RobotImuState_t *imu);  // 자세 Fault만 평가한다.

RobotSafetyOutput_t Safety_Evaluate(Safety_Handle_t *handle,
                                    const RobotImuState_t *imu,
                                    const bool ik_valid[ROBOT_LEG_COUNT]);  // 자세와 IK Fault를 평가한다.

#endif
