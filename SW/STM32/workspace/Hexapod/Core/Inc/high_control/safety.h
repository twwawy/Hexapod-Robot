#ifndef SAFETY_H
#define SAFETY_H

#include "common/robot_types.h"

typedef struct
{
    RobotSafetyOutput_t latched;  // Reset 없는 Fault Latch를 저장한다.
} Safety_Handle_t;

void Safety_Init(Safety_Handle_t *handle);  // Fault Latch를 초기화한다.

RobotSafetyOutput_t Safety_Evaluate(Safety_Handle_t *handle,
                                    const RobotImuState_t *imu,
                                    const bool ik_valid[ROBOT_LEG_COUNT]);  // 자세와 IK Fault를 평가한다.

#endif
