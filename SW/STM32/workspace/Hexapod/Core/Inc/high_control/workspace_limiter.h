#ifndef WORKSPACE_LIMITER_H
#define WORKSPACE_LIMITER_H

#include "common/robot_types.h"

typedef struct
{
    RobotBodyTwist_t gait_applied;  // 마지막 정상 보행 명령을 저장한다.
} WorkspaceLimiter_Handle_t;

void WorkspaceLimiter_Init(WorkspaceLimiter_Handle_t *handle);  // 적용 명령을 0으로 초기화한다.

bool WorkspaceLimiter_AllFeetValid(const RobotVec3_t feet_body[ROBOT_LEG_COUNT],
                                   const RobotEuler_t *posture_rad);  // 자세 적용 후 여섯 발 IK를 검사한다.

RobotBodyTwist_t WorkspaceLimiter_Gait(WorkspaceLimiter_Handle_t *handle,
                                       const RobotBodyTwist_t *candidate,
                                       bool manual_enable,
                                       const RobotEuler_t *posture_rad,
                                       bool reset_command,
                                       bool *accepted);  // 한 위상의 Stance·Swing 작업공간을 미리 검사한다.

#endif
