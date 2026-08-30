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

    WorkspaceLimiter_Init(&limiter);       // 직전 명령을 0으로 준비한다.
    gait.enabled_internal = true;          // 정상 보행 상태를 활성화한다.
    gait.next_phase_preview = true;        // 첫 위상 검사를 시작한다.
    gait.next_phase_startup = true;        // 최초 반 보폭을 선택한다.
    gait.next_phase_swing_mask = 0x15U;    // 1·3·5번 다리를 Swing으로 둔다.
    candidate.vx = 0.01f;                  // 안전한 전진 후보를 넣는다.
    candidate.wz = 0.01f;                  // 두 걸음에 고정할 사용자 회전을 넣는다.

    applied = WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                    &gait, &posture, false,
                                    &accepted);  // 한 5 ms 주기에서 시작·중앙·끝을 검사한다.
    if (!accepted || !limiter.phase_result_valid ||
        !limiter.phase_result_accepted ||
        (fabsf(applied.vx - candidate.vx) > 1.0e-6f) ||
        !isfinite(applied.vx) || !isfinite(applied.vy) || !isfinite(applied.wz))
    {
        return false;
    }

    gait.next_phase_preview = false;  // 같은 걸음 안에서 추가 검사를 막는다.
    applied = WorkspaceLimiter_Gait(&limiter, &candidate, 0.05f, true,
                                    &gait, &posture, false,
                                    &accepted);  // 걸음 중 최신 Heading 보정을 즉시 적용한다.
    if (!accepted ||
        (fabsf(limiter.gait_applied.wz - candidate.wz) > 1.0e-6f) ||
        (fabsf(applied.wz - 0.06f) > 1.0e-6f) ||
        (limiter.gait_applied_step_count != 1U))
    {
        return false;
    }

    gait.next_phase_preview = true;      // 다음 Tripod 검사를 시작한다.
    gait.next_phase_startup = false;     // 정상 전체 보폭을 선택한다.
    gait.next_phase_swing_mask = 0x2AU;  // 2·4·6번 다리를 Swing으로 둔다.
    candidate.vx = ROBOT_MAX_LINEAR_SPEED_MPS;  // 최대 전진 후보를 넣는다.
    applied = WorkspaceLimiter_Gait(&limiter, &candidate, 0.05f, true,
                                    &gait, &posture, false,
                                    &accepted);  // 둘째 걸음에서 첫 속도를 다시 검사한다.
    if (accepted || !limiter.phase_result_accepted ||
        (fabsf(applied.vx - 0.01f) > 1.0e-6f) ||
        (fabsf(limiter.gait_preview.wz - 0.06f) > 1.0e-6f) ||
        (fabsf(limiter.gait_pending.vx - 0.01f) > 1.0e-6f) ||
        (limiter.gait_applied_step_count != 2U))
    {
        return false;
    }

    gait.next_phase_preview = true;      // 셋째 걸음의 새 속도 검사를 시작한다.
    gait.next_phase_swing_mask = 0x15U;  // 1·3·5번 다리를 다시 Swing으로 둔다.
    applied = WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                    &gait, &posture, false,
                                    &accepted);  // 두 걸음 뒤 최대 후보를 새로 채택한다.
    if (!accepted || !limiter.phase_result_accepted ||
        (fabsf(applied.vx - ROBOT_MAX_LINEAR_SPEED_MPS) > 1.0e-6f) ||
        (limiter.gait_applied_step_count != 1U))
    {
        return false;
    }

    gait.next_phase_preview = true;      // 넷째 걸음의 기존 속도 검사를 시작한다.
    gait.next_phase_swing_mask = 0x2AU;  // 2·4·6번 다리를 다시 Swing으로 둔다.
    candidate.vx = 10.0f;                // 둘째 걸음 중 위험한 새 입력을 넣는다.
    applied = WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                    &gait, &posture, false,
                                    &accepted);  // 위험 입력 대신 기존 속도를 유지한다.
    if (accepted || !limiter.phase_result_accepted ||
        (fabsf(applied.vx - ROBOT_MAX_LINEAR_SPEED_MPS) > 1.0e-6f) ||
        (limiter.gait_applied_step_count != 2U))
    {
        return false;
    }

    gait.next_phase_preview = true;      // 다섯째 걸음의 위험 후보 검사를 시작한다.
    gait.next_phase_swing_mask = 0x15U;  // 1·3·5번 다리를 다시 Swing으로 둔다.
    applied = WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                    &gait, &posture, false,
                                    &accepted);  // 두 걸음 뒤 위험 후보를 거부한다.
    if (accepted || !limiter.phase_result_valid ||
        limiter.phase_result_accepted ||
        (fabsf(applied.vx - ROBOT_MAX_LINEAR_SPEED_MPS) > 1.0e-6f))
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
