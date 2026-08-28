#ifndef WORKSPACE_LIMITER_H
#define WORKSPACE_LIMITER_H

#include "common/robot_types.h"

typedef struct
{
    RobotBodyTwist_t gait_applied;         // 마지막 정상 보행 명령을 저장한다.
    RobotBodyTwist_t gait_pending;         // 검사 중인 보행 명령을 저장한다.
    uint8_t preview_sample;                // 다음 검사 지점 번호를 저장한다.
    uint8_t preview_swing_mask;            // 검사할 Swing 다리를 저장한다.
    bool preview_startup_phase;            // 첫 위상 검사 여부를 저장한다.
    bool preview_active;                   // 분산 검사의 진행 여부를 저장한다.
    bool phase_result_valid;               // 위상 검사 결과 존재 여부를 저장한다.
    bool phase_result_accepted;            // 위상 검사 통과 여부를 저장한다.
} WorkspaceLimiter_Handle_t;

void WorkspaceLimiter_Init(WorkspaceLimiter_Handle_t *handle);  // 적용 명령을 0으로 초기화한다.

bool WorkspaceLimiter_AllFeetValid(const RobotVec3_t feet_body[ROBOT_LEG_COUNT],
                                   const RobotEuler_t *posture_rad);  // 자세 적용 후 여섯 발 IK를 검사한다.

RobotBodyTwist_t WorkspaceLimiter_Gait(WorkspaceLimiter_Handle_t *handle,
                                       const RobotBodyTwist_t *candidate,
                                       bool manual_enable,
                                       const RobotGaitPhase_t *gait,
                                       const RobotEuler_t *posture_rad,
                                       bool reset_command,
                                       bool *accepted);  // 다음 위상 다섯 지점을 분산 검사한다.

#endif
