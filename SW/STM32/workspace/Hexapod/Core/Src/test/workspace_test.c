#include "test/workspace_test.h"

#include "high_control/leg_kinematics.h"
#include "high_control/workspace_limiter.h"

#include <math.h>
#include <string.h>

/* 기본 계획 공개·잔차 고정·경로 거부 후 재시도를 검사한다. */
static bool WorkspaceTest_CheckRlPlan(void)
{
    WorkspaceLimiter_Handle_t limiter;                   // RL 경로 검사를 준비한다.
    FootTrajectory_Plan_t plan;                          // 최초 공개 계획을 보존한다.
    RobotLegResidual_t residual[ROBOT_LEG_COUNT] = {0};  // 원자적으로 적용할 잔차를 준비한다.
    RobotBodyTwist_t candidate = {0};                    // 작은 전진 명령을 준비한다.
    RobotGaitPhase_t gait = {0};                         // 첫 위상 검사 요청을 준비한다.
    RobotEuler_t posture = {0};                          // 수평 자세를 준비한다.
    bool accepted;                                       // 경로 검사 결과를 저장한다.
    uint32_t cycle;                                      // 여러 제어 주기를 진행한다.

    WorkspaceLimiter_Init(&limiter);                // 기본 지지점과 검사 상태를 준비한다.
    WorkspaceLimiter_SetRlEnabled(&limiter, true);  // RL 잔차 대기를 활성화한다.
    candidate.vx = 0.005f;                          // 작업공간 안의 전진 속도를 선택한다.
    gait.waiting_start = true;                      // 잔차 전에는 발을 고정한다.
    gait.next_phase_preview = true;                 // 기본 계획 생성을 요청한다.
    gait.next_phase_startup = true;                 // 첫 반 보폭 위상을 선택한다.
    gait.next_phase_swing_mask = 0x15U;             // 1·3·5번 다리 잔차를 적용한다.
    (void)WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                &gait, &posture, false, &accepted);  // 잔차 없이도 기본 계획을 공개한다.
    if (accepted || limiter.preview_active || limiter.phase_result_valid ||
        !WorkspaceLimiter_GetRlPlan(&limiter, &plan))
    {
        return false;
    }

    gait.next_phase_preview = false;  // 같은 계획의 대기를 계속한다.
    candidate.vx = 0.02f;             // 다음 계획용 새 조종값을 넣는다.
    (void)WorkspaceLimiter_Gait(&limiter, &candidate, 0.1f, true,
                                &gait, &posture, false, &accepted);  // 현재 공개 기준의 고정을 확인한다.
    if (memcmp(plan.nominal, limiter.rl_plan.nominal, sizeof(plan.nominal)) != 0 ||
        (plan.twist.vx != limiter.rl_plan.twist.vx) ||
        WorkspaceLimiter_SubmitRlResidual(&limiter, plan.plan_id + 1U, 0x15U, residual) ||
        WorkspaceLimiter_SubmitRlResidual(&limiter, plan.plan_id, 0x2AU, residual))
    {
        return false;
    }
    residual[1].dx = NAN;  // 적용 마스크 밖의 손상 값도 검사한다.
    if (WorkspaceLimiter_SubmitRlResidual(&limiter, plan.plan_id, 0x15U, residual))
    {
        return false;
    }
    residual[1].dx = 0.01f;   // 지지발에 사용하지 않을 정상 잔차를 준비한다.
    residual[0].dx = 0.005f;  // 첫 다리 착지점의 작은 이동을 요청한다.
    residual[0].dz = 0.005f;  // 착지 높이와 Swing 높이를 구분한다.
    residual[0].dh = -0.01f;  // 첫 다리 들어 올림 높이를 낮춘다.
    if (!WorkspaceLimiter_SubmitRlResidual(&limiter, plan.plan_id, 0x15U, residual) ||
        (limiter.rl_plan.target[1].x != plan.nominal[1].x))
    {
        return false;
    }
    residual[0].dx = 0.02f;  // 검사 중 후보를 교체하려는 값을 준비한다.
    if (WorkspaceLimiter_SubmitRlResidual(&limiter, plan.plan_id, 0x15U, residual) ||
        (fabsf(limiter.rl_plan.target[0].x - plan.nominal[0].x - 0.005f) > 1.0e-6f))
    {
        return false;
    }
    posture.roll = NAN;  // 기본 속도 축소로 해결할 수 없는 경로 오류를 만든다.
    (void)WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                &gait, &posture, false, &accepted);  // 기존 기본 계획을 유지하며 후보만 거부한다.
    if (!limiter.rl_plan_rejected || limiter.phase_result_valid ||
        !limiter.rl_plan.valid || (limiter.rl_plan.plan_id != plan.plan_id) ||
        (limiter.preview_scale != 1.0f))
    {
        return false;
    }
    posture.roll = 0.0f;      // 유효한 자세로 새 후보를 검사한다.
    residual[0].dx = 0.005f;  // 같은 기준에 같은 잔차를 다시 요청한다.
    if (!WorkspaceLimiter_SubmitRlResidual(&limiter, plan.plan_id, 0x15U, residual))
    {
        return false;
    }
    for (cycle = 0U; cycle < 3U; ++cycle)
    {
        (void)WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                    &gait, &posture, false, &accepted);  // 아홉 지점을 제한된 주기당 검사량으로 확인한다.
    }
    return accepted && limiter.phase_result_valid && limiter.phase_result_accepted &&
           (fabsf(limiter.rl_plan.target[0].x - plan.nominal[0].x - 0.005f) < 1.0e-6f) &&
           (fabsf(limiter.rl_plan.height[0] - plan.nominal_height[0] + 0.01f) < 1.0e-6f);  // 재시도에도 잔차가 누적되지 않는지 확인한다.
}


/* 한 발 명령의 즉시 갱신과 실제 지지점의 경로 거부를 검사한다. */
static bool WorkspaceTest_CheckWavePreview(void)
{
    WorkspaceLimiter_Handle_t limiter;  // 실제 경로 검사 상태를 저장한다.
    RobotBodyTwist_t candidate = {0};   // 감속된 조종 후보를 저장한다.
    RobotGaitPhase_t gait = {0};        // 한 발 경로 검사 요청을 저장한다.
    RobotEuler_t posture = {0};         // 수평 자세를 준비한다.
    RobotVec3_t feet[ROBOT_LEG_COUNT];  // 실제 지지점의 시험 좌표를 저장한다.
    RobotVec3_t offset = {0};           // 기본 몸체 위치를 준비한다.
    bool accepted;                     // 조종 후보 채택 여부를 저장한다.
    uint32_t cycle;                    // 축소 검사 횟수를 제한한다.

    WorkspaceLimiter_Init(&limiter);                   // 이전 속도와 검사 결과를 제거한다.
    LegKinematics_GetBaseFeet(feet);                    // 실제 출발 지지점을 준비한다.
    WorkspaceLimiter_SetFeet(&limiter, feet, &offset);  // 실제 발 위치를 경로 검사기에 전달한다.
    candidate.vx = 0.005f;                              // 작은 전진 명령을 준비한다.
    gait.enabled_internal = true;                      // 정상 보행 검사를 활성화한다.
    gait.next_phase_pattern = ROBOT_GAIT_WAVE;         // 개별 보행의 다음 경로를 선택한다.
    gait.next_phase_preview = true;                    // 첫 발 경로를 요청한다.
    gait.next_phase_swing_mask = 1U;                   // 첫 다리만 이륙시킨다.
    (void)WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                &gait, &posture, false, &accepted);  // 첫 한 발 명령을 검사한다.
    if (!limiter.phase_result_accepted)
    {
        return false;
    }
    candidate.vx = -0.005f;                  // 다음 발에 반대 방향을 요청한다.
    gait.next_phase_swing_mask = 1U << 5U;  // 다음 다리만 이륙시킨다.
    (void)WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                &gait, &posture, false, &accepted);  // 이전 두 걸음 명령 기억 없이 검사한다.
    if (!limiter.phase_result_accepted || (limiter.gait_requested.vx != candidate.vx))
    {
        return false;
    }
    feet[2].x += 1.0f;                                  // 이동발 외의 지지점을 작업공간 밖으로 옮긴다.
    WorkspaceLimiter_SetFeet(&limiter, feet, &offset);  // 잘못된 실제 지지점을 전달한다.
    for (cycle = 0U; cycle < 10U; ++cycle)
    {
        gait.next_phase_preview = (cycle == 0U);  // 최초 요청 이후 축소 검사만 진행한다.
        (void)WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                    &gait, &posture, false, &accepted);  // 보폭 축소로 숨길 수 없는 시작점 오류를 검사한다.
    }
    return limiter.phase_result_valid && !limiter.phase_result_accepted;  // 실제 지지점 오류에서 이륙을 거부한다.
}

/* 같은 최대 입력이 감속 후 추가 대기 없이 재사용되는지 검사한다. */
static bool WorkspaceTest_CheckReducedCommand(void)
{
    WorkspaceLimiter_Handle_t limiter;  // 감속과 원본 명령 기억을 저장한다.
    RobotBodyTwist_t candidate = {0};   // 작업공간을 넘는 최대 동시 입력을 저장한다.
    RobotBodyTwist_t applied;           // 실제 적용된 감속 명령을 저장한다.
    RobotEuler_t posture = {0};         // 평지 수평 자세를 저장한다.
    RobotGaitPhase_t gait = {0};        // 검사할 Tripod 역할을 저장한다.
    float reduced_scale;               // 최초 통과한 공통 축척을 저장한다.
    bool accepted;                     // 원본 명령의 채택 여부를 저장한다.
    uint32_t cycle;                    // 감속 검사 대기 주기를 저장한다.
    uint32_t phase;                    // 반복 검사할 위상 번호를 저장한다.

    WorkspaceLimiter_Init(&limiter);                   // 이전 검사와 명령을 제거한다.
    candidate.vx = ROBOT_MAX_LINEAR_SPEED_MPS;          // 최대 전진 속도를 요청한다.
    candidate.vy = ROBOT_MAX_LATERAL_SPEED_MPS;         // 최대 횡이동 속도를 요청한다.
    candidate.wz = ROBOT_MAX_YAW_RATE_RADPS;            // 최대 회전 속도를 함께 요청한다.
    gait.enabled_internal = true;                      // 정상 보행 검사를 활성화한다.
    gait.next_phase_preview = true;                    // 최초 위상 검사를 요청한다.
    gait.next_phase_startup = true;                    // 최초 반 보폭을 선택한다.
    gait.next_phase_swing_mask = 0x15U;                 // 첫 Tripod 그룹을 선택한다.
    applied = WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                    &gait, &posture, false,
                                    &accepted);  // 불가능한 원본 명령을 한 번 검사한다.
    if (accepted || !limiter.preview_active || limiter.phase_result_valid ||
        limiter.phase_result_accepted || (applied.vx != 0.0f) ||
        (applied.vy != 0.0f) || (applied.wz != 0.0f))
    {
        return false;  // 원본 실패가 즉시 적용되거나 최종 정지로 확정되는지 확인한다.
    }

    gait.next_phase_preview = false;  // 최초 요청을 반복하지 않고 재검사를 진행한다.
    for (cycle = 0U; limiter.preview_active && (cycle < 16U); ++cycle)
    {
        applied = WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                        &gait, &posture, false,
                                        &accepted);  // 다음 제어 주기에서 감속 후보를 검사한다.
        if (accepted || (limiter.preview_active && limiter.phase_result_valid))
        {
            return false;  // 감속 중 원본 채택이나 미완료 결과 확정을 차단한다.
        }
    }
    reduced_scale = applied.vx / candidate.vx;  // 실제 속도로 공통 축척을 확인한다.
    if (limiter.preview_active || !limiter.phase_result_valid ||
        !limiter.phase_result_accepted || (reduced_scale <= 0.0f) ||
        (reduced_scale >= 1.0f) ||
        (fabsf(applied.vy - candidate.vy * reduced_scale) > 1.0e-6f) ||
        (fabsf(applied.wz - candidate.wz * reduced_scale) > 1.0e-6f) ||
        (fabsf(limiter.gait_applied_scale - reduced_scale) > 1.0e-6f) ||
        (limiter.gait_requested.vx != candidate.vx) ||
        (limiter.gait_requested.vy != candidate.vy) ||
        (limiter.gait_requested.wz != candidate.wz))
    {
        return false;  // 원본 방향을 유지한 유효 감속과 원본 기억을 확인한다.
    }

    for (phase = 1U; phase < 5U; ++phase)
    {
        const uint8_t expected_steps = ((phase % 2U) == 0U) ? 1U : 2U;  // 두 걸음 묶음의 위치를 계산한다.

        if (phase == 2U)
        {
            candidate.vx -= 0.0005f;  // 같은 조종 입력의 작은 X 위치 보정 변화를 넣는다.
            candidate.vy -= 0.0005f;  // 같은 조종 입력의 작은 Y 위치 보정 변화를 넣는다.
        }

        gait.next_phase_preview = true;                                    // 다음 위상의 검사를 요청한다.
        gait.next_phase_startup = false;                                   // 반복 전체 보폭을 선택한다.
        gait.next_phase_swing_mask = ((phase % 2U) == 0U) ? 0x15U : 0x2AU;  // 두 Tripod를 교대로 검사한다.
        applied = WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                        &gait, &posture, false,
                                        &accepted);  // 같은 입력의 기존 축척을 재검증한다.
        if (accepted || limiter.preview_active || !limiter.phase_result_valid ||
            !limiter.phase_result_accepted ||
            (limiter.gait_applied_step_count != expected_steps) ||
            (fabsf(applied.vx - candidate.vx * reduced_scale) > 1.0e-6f) ||
            (fabsf(applied.vy - candidate.vy * reduced_scale) > 1.0e-6f) ||
            (fabsf(applied.wz - candidate.wz * reduced_scale) > 1.0e-6f))
        {
            return false;  // 같은 입력에서 재감속 대기나 속도 누적 축소를 검출한다.
        }
    }

    posture.roll = NAN;                   // 저장된 축척으로도 검사 불가능한 새 자세를 넣는다.
    gait.next_phase_swing_mask = 0x2AU;  // 다음 둘째 걸음에서 현재 자세를 재검사한다.
    applied = WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                    &gait, &posture, false,
                                    &accepted);  // 캐시가 최신 IK 검사를 우회하는지 확인한다.
    gait.next_phase_preview = false;  // 같은 자세의 검사 요청을 반복하지 않는다.
    for (cycle = 0U; limiter.preview_active && (cycle < 16U); ++cycle)
    {
        if (accepted || limiter.phase_result_accepted)
        {
            return false;  // 불가능한 자세의 캐시 재사용 허가를 검출한다.
        }
        applied = WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                        &gait, &posture, false,
                                        &accepted);  // 불가능한 자세의 재검사를 제한된 주기에 종료한다.
    }
    if (accepted || limiter.preview_active || !limiter.phase_result_valid ||
        limiter.phase_result_accepted ||
        (fabsf(applied.vx - candidate.vx * reduced_scale) > 1.0e-6f) ||
        (limiter.gait_applied_step_count != 1U))
    {
        return false;  // 실패한 캐시 검사가 기존 속도와 걸음 수를 바꾸는지 확인한다.
    }

    posture.roll = 0.0f;               // 다시 검사 가능한 수평 자세를 넣는다.
    gait.next_phase_preview = true;  // 정상 자세에서 남은 둘째 걸음을 요청한다.
    applied = WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                    &gait, &posture, false,
                                    &accepted);  // 실패 전 감속 명령을 다시 검증한다.
    if (accepted || limiter.preview_active || !limiter.phase_result_accepted ||
        (limiter.gait_applied_step_count != 2U) ||
        (fabsf(applied.vx - candidate.vx * reduced_scale) > 1.0e-6f))
    {
        return false;  // 자세 복구 뒤 유효 축척을 재사용하는지 확인한다.
    }

    candidate.vx = 0.01f;                 // 새 명령 묶음에 작은 안전 속도를 넣는다.
    candidate.vy = 0.0f;                  // 횡이동 요청을 제거한다.
    candidate.wz = 0.0f;                  // 회전 요청을 제거한다.
    gait.next_phase_swing_mask = 0x15U;  // 다음 명령 묶음의 첫 Tripod를 선택한다.
    applied = WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                    &gait, &posture, false,
                                    &accepted);  // 변경된 안전 명령을 원래 크기로 검사한다.
    if (!accepted || limiter.preview_active || !limiter.phase_result_accepted ||
        (limiter.gait_applied_step_count != 1U) ||
        (fabsf(limiter.gait_applied_scale - 1.0f) > 1.0e-6f) ||
        (fabsf(applied.vx - candidate.vx) > 1.0e-6f) ||
        (applied.vy != 0.0f) || (applied.wz != 0.0f))
    {
        return false;  // 이전 축척이 새 안전 명령까지 감속하는지 확인한다.
    }

    applied = WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                    &gait, &posture, true,
                                    &accepted);  // 명시적 Reset으로 감속 기억을 제거한다.
    if (!accepted || limiter.preview_active || limiter.phase_result_valid ||
        limiter.phase_result_accepted || (limiter.gait_applied_step_count != 0U) ||
        (limiter.gait_requested.vx != 0.0f) || (limiter.gait_requested.vy != 0.0f) ||
        (limiter.gait_requested.wz != 0.0f) || (applied.vx != 0.0f) ||
        (applied.vy != 0.0f) || (applied.wz != 0.0f))
    {
        return false;  // Reset 뒤 남은 명령이나 검사 결과를 검출한다.
    }

    candidate.vx = ROBOT_MAX_LINEAR_SPEED_MPS;   // Reset 뒤 원래 최대 전진 입력을 복구한다.
    candidate.vy = ROBOT_MAX_LATERAL_SPEED_MPS;  // Reset 뒤 원래 최대 횡이동 입력을 복구한다.
    candidate.wz = ROBOT_MAX_YAW_RATE_RADPS;     // Reset 뒤 원래 최대 회전 입력을 복구한다.
    gait.next_phase_startup = true;             // Reset 뒤 첫 위상을 다시 선택한다.
    (void)WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                &gait, &posture, false,
                                &accepted);  // 같은 원본을 축척 기억 없이 다시 검사한다.
    return !accepted && limiter.preview_active && !limiter.phase_result_valid;
}

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
    uint32_t cycle;                               // 위험 후보의 재검사 주기를 저장한다.

    if (!WorkspaceTest_CheckRlPlan() ||
        !WorkspaceTest_CheckWavePreview() || !WorkspaceTest_CheckReducedCommand())
    {
        return false;
    }

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
                                    &accepted);  // 두 걸음 뒤 위험 후보의 감속 검사를 시작한다.
    gait.next_phase_preview = false;  // 같은 위험 후보를 다시 시작하지 않는다.
    for (cycle = 0U; limiter.preview_active && (cycle < 16U); ++cycle)
    {
        if (accepted || limiter.phase_result_valid || limiter.phase_result_accepted ||
            (fabsf(applied.vx - ROBOT_MAX_LINEAR_SPEED_MPS) > 1.0e-6f))
        {
            return false;  // 위험 후보 재검사 중 직전 유효 속도 유지를 확인한다.
        }
        applied = WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                        &gait, &posture, false,
                                        &accepted);  // 제한된 주기 안에서 불가능한 후보를 거부한다.
    }
    if (accepted || !limiter.phase_result_valid ||
        limiter.preview_active || limiter.phase_result_accepted ||
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
