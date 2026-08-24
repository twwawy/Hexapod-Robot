#include "test/workspace_test.h"

#include "high_control/leg_kinematics.h"
#include "high_control/workspace_limiter.h"

#include <math.h>

/* 최대 보행 후보 채택과 과도한 발 위치의 최종 제한을 검사한다. */
bool WorkspaceTest_Run(void)
{
    WorkspaceLimiter_Handle_t limiter;           // 보행 명령 기억을 저장한다.
    RobotBodyTwist_t candidate = {0};             // 최대 동시 보행 후보를 저장한다.
    RobotBodyTwist_t applied;                     // 실제 채택한 명령을 저장한다.
    RobotEuler_t posture = {0};                   // 시험 자세를 저장한다.
    RobotVec3_t base[ROBOT_LEG_COUNT];            // 기본 발 위치를 저장한다.
    RobotVec3_t outside;                          // 작업공간 밖 발 위치를 저장한다.
    RobotVec3_t limited;                          // 최종 제한 발 위치를 저장한다.
    bool accepted;                                // 보행 채택 여부를 저장한다.
    bool was_limited;                             // 발 제한 여부를 저장한다.

    WorkspaceLimiter_Init(&limiter);  // 직전 명령을 0으로 준비한다.
    candidate.vx = ROBOT_MAX_LINEAR_SPEED_MPS;    // 합의한 최대 전진 속도를 넣는다.
    candidate.vy = ROBOT_MAX_LINEAR_SPEED_MPS;    // 합의한 최대 횡이동 속도를 넣는다.
    candidate.wz = ROBOT_MAX_YAW_RATE_RADPS;      // 합의한 최대 회전 속도를 넣는다.
    applied = WorkspaceLimiter_Gait(&limiter, &candidate, true,
                                    &posture, false, &accepted);  // 한 위상 전체를 검사한다.
    if (!isfinite(applied.vx) || !isfinite(applied.vy) || !isfinite(applied.wz))
    {
        return false;
    }

    LegKinematics_GetBaseFeet(base);  // 정상 발 기준을 읽는다.
    outside = base[0];                // 1번 다리 위치에서 시작한다.
    outside.x += 1.0f;                // 확실한 작업공간 밖 위치를 만든다.
    if (!LegKinematics_LimitFoot(0U, &outside, &limited, &was_limited) ||
        !was_limited || !LegKinematics_IsReachable(0U, &limited))
    {
        return false;
    }

    return true;
}
