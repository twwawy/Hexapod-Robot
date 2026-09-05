#ifndef WORKSPACE_LIMITER_H
#define WORKSPACE_LIMITER_H

#include "common/robot_types.h"

typedef struct
{
    RobotBodyTwist_t gait_applied;         // 작업공간에 맞춘 실제 사용자 명령을 저장한다.
    RobotBodyTwist_t gait_requested;       // 마지막으로 통과한 축소 전 명령을 저장한다.
    RobotBodyTwist_t gait_pending;         // 검사 중인 사용자 명령을 저장한다.
    RobotBodyTwist_t gait_preview;         // Heading 보정을 합친 검사 명령을 저장한다.
    float gait_applied_scale;              // 통과한 명령의 공통 보폭 비율을 저장한다.
    float preview_scale;                   // 검사 중인 공통 보폭 비율을 저장한다.
    uint8_t gait_applied_step_count;       // 현재 명령을 적용한 걸음 수를 저장한다.
    uint8_t preview_reduction_count;       // 현재 검사의 보폭 축소 횟수를 저장한다.
    uint8_t preview_sample;                // 다음 검사 지점 번호를 저장한다.
    uint8_t preview_swing_mask;            // 검사할 Swing 다리를 저장한다.
    bool preview_startup_phase;            // 첫 위상 검사 여부를 저장한다.
    bool preview_reuses_applied;           // 둘째 걸음의 기존 명령 재사용 여부를 저장한다.
    bool preview_active;                   // 세 지점 검사의 진행 여부를 저장한다.
    bool phase_result_valid;               // 위상 검사 결과 존재 여부를 저장한다.
    bool phase_result_accepted;            // 위상 검사 통과 여부를 저장한다.
} WorkspaceLimiter_Handle_t;

void WorkspaceLimiter_Init(WorkspaceLimiter_Handle_t *handle);  // 적용 명령을 0으로 초기화한다.

bool WorkspaceLimiter_AllFeetValid(const RobotVec3_t feet_body[ROBOT_LEG_COUNT],
                                   const RobotEuler_t *posture_rad);  // 자세 적용 후 여섯 발 IK를 검사한다.

RobotBodyTwist_t WorkspaceLimiter_Gait(WorkspaceLimiter_Handle_t *handle,
                                       const RobotBodyTwist_t *user_candidate,
                                       float yaw_feedback_radps,
                                       bool manual_enable,
                                       const RobotGaitPhase_t *gait,
                                       const RobotEuler_t *posture_rad,
                                       bool reset_command,
                                       bool *accepted);  // 한 주기에 보폭 후보 하나의 세 지점을 검사한다.

#endif
