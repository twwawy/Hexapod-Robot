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
    RobotGaitPhase_t gait = {0};                  // 정상 Tripod 상태를 저장한다.
    RobotVec3_t base[ROBOT_LEG_COUNT];            // 기본 발 위치를 저장한다.
    RobotVec3_t outside;                          // 작업공간 밖 발 위치를 저장한다.
    RobotVec3_t inner_body;                       // 관절 제한 안쪽 발 위치를 저장한다.
    RobotVec3_t inner_local;                      // 관절 제한 안쪽 다리 좌표를 저장한다.
    RobotVec3_t limited;                          // 최종 제한 발 위치를 저장한다.
    bool accepted;                                // 보행 채택 여부를 저장한다.
    bool was_limited;                             // 발 제한 여부를 저장한다.
    uint32_t leg;                                 // 준비할 다리 번호를 저장한다.

    WorkspaceLimiter_Init(&limiter);       // 직전 명령을 0으로 준비한다.
    gait.enabled_internal = true;          // 정상 보행 상태를 활성화한다.
    gait.next_phase_preview = true;        // 첫 위상 검사를 시작한다.
    gait.next_phase_startup = true;        // 최초 반 보폭을 선택한다.
    gait.next_phase_swing_mask = 0x15U;    // 1·3·5번 다리를 Swing으로 둔다.
    candidate.vx = 0.01f;                  // 안전한 전진 후보를 넣는다.

    for (leg = 0U; leg < ROBOT_GAIT_PREVIEW_SAMPLE_COUNT; ++leg)
    {
        applied = WorkspaceLimiter_Gait(&limiter, &candidate, true,
                                        &gait, &posture, false,
                                        &accepted);  // 5 ms마다 다음 경로 한 점을 검사한다.
        gait.next_phase_preview = false;             // 검사 시작 신호를 한 번만 유지한다.
        if ((leg + 1U) < ROBOT_GAIT_PREVIEW_SAMPLE_COUNT)
        {
            if (accepted || limiter.phase_result_valid || (applied.vx != 0.0f))
            {
                return false;
            }
        }
    }
    if (!accepted || !limiter.phase_result_valid ||
        !limiter.phase_result_accepted ||
        (fabsf(applied.vx - candidate.vx) > 1.0e-6f) ||
        !isfinite(applied.vx) || !isfinite(applied.vy) || !isfinite(applied.wz))
    {
        return false;
    }

    gait.next_phase_preview = true;      // 다음 Tripod 검사를 시작한다.
    gait.next_phase_startup = false;     // 정상 전체 보폭을 선택한다.
    gait.next_phase_swing_mask = 0x2AU;  // 2·4·6번 다리를 Swing으로 둔다.
    candidate.vx = 0.02f;                // 25 ms 시점의 후보를 넣는다.
    applied = WorkspaceLimiter_Gait(&limiter, &candidate, true,
                                    &gait, &posture, false,
                                    &accepted);  // 첫 경로 지점과 속도를 고정한다.
    gait.next_phase_preview = false;  // 남은 검사에서 새 입력 고정을 막는다.
    candidate.vx = 0.03f;             // 검사 중 들어온 최신 입력을 넣는다.
    for (leg = 1U; leg < ROBOT_GAIT_PREVIEW_SAMPLE_COUNT; ++leg)
    {
        applied = WorkspaceLimiter_Gait(&limiter, &candidate, true,
                                        &gait, &posture, false,
                                        &accepted);  // 고정 후보의 남은 네 지점을 검사한다.
    }
    if (accepted || !limiter.phase_result_accepted ||
        (fabsf(applied.vx - 0.02f) > 1.0e-6f) ||
        (fabsf(limiter.gait_pending.vx - 0.02f) > 1.0e-6f))
    {
        return false;
    }

    gait.next_phase_preview = true;  // 위험 후보의 다음 위상 검사를 시작한다.
    candidate.vx = 10.0f;            // 확실히 위험한 새 후보를 넣는다.
    applied = WorkspaceLimiter_Gait(&limiter, &candidate, true,
                                    &gait, &posture, false,
                                    &accepted);  // 첫 위험 지점에서 후보를 거부한다.
    if (accepted || !limiter.phase_result_valid ||
        limiter.phase_result_accepted ||
        (fabsf(applied.vx - 0.02f) > 1.0e-6f))
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

    inner_local.x = ROBOT_LINK_1_M +
                    fabsf(ROBOT_LINK_2_M - ROBOT_LINK_3_M) +
                    ROBOT_WORKSPACE_MARGIN_M;  // 기하학적으로만 유효한 안쪽 경계를 만든다.
    inner_local.y = 0.0f;  // Coxa 정면 방향을 선택한다.
    inner_local.z = 0.0f;  // 최대 접힘이 필요한 높이를 선택한다.
    if (!LegKinematics_LegToBody(0U, &inner_local, &inner_body) ||
        !LegKinematics_LimitFoot(0U, &inner_body, &limited, &was_limited) ||
        !was_limited || !LegKinematics_IsReachable(0U, &limited))
    {
        return false;
    }

    inner_local.x = -ROBOT_BASE_FOOT_RADIUS_M;  // Coxa 관절 범위를 넘는 뒤쪽 목표를 만든다.
    inner_local.y = 0.0f;                       // Coxa 각도를 180도로 만든다.
    inner_local.z = ROBOT_BASE_FOOT_Z_M;        // 나머지 관절은 기본 높이를 유지한다.
    if (!LegKinematics_LegToBody(0U, &inner_local, &inner_body) ||
        !LegKinematics_LimitFoot(0U, &inner_body, &limited, &was_limited) ||
        !was_limited || !LegKinematics_IsReachable(0U, &limited))
    {
        return false;
    }

    return true;
}
