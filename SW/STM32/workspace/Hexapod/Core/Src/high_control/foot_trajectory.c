#include "high_control/foot_trajectory.h"

#include "high_control/contact_adaptation.h"
#include "high_control/leg_kinematics.h"
#include "high_control/stance_trajectory.h"
#include "high_control/swing_trajectory.h"
#include "high_control/workspace_limiter.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define FOOT_RECOVERY_HEIGHT_M 0.10f

static const float foot_leg_angle[ROBOT_LEG_COUNT] =
{
    -45.0f * ROBOT_DEG_TO_RAD_F, -90.0f * ROBOT_DEG_TO_RAD_F,
    -135.0f * ROBOT_DEG_TO_RAD_F, 45.0f * ROBOT_DEG_TO_RAD_F,
    90.0f * ROBOT_DEG_TO_RAD_F, 135.0f * ROBOT_DEG_TO_RAD_F
};  // 여섯 다리 장착각을 저장한다.

/* 현재 Swing 다리를 비트로 변환한다. */
static uint8_t FootTrajectory_SwingMask(const RobotGaitPhase_t *gait)
{
    uint8_t mask = 0U;  // Swing 다리 비트를 준비한다.
    uint32_t leg;       // 확인할 다리 번호를 저장한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        if ((gait->state[leg] == ROBOT_LEG_SWING) ||
            (gait->state[leg] == ROBOT_LEG_LATE_LANDING) ||
            (gait->state[leg] == ROBOT_LEG_TOUCHDOWN_CANDIDATE))
        {
            mask |= (uint8_t)(1U << leg);  // 현재 공중 이동 그룹을 표시한다.
        }
    }

    return mask;
}

/* 정지 또는 모드 전환에서 위상 속도 기억을 제거한다. */
static void FootTrajectory_ClearPhaseTwist(FootTrajectory_Handle_t *handle)
{
    memset(&handle->phase_twist, 0, sizeof(handle->phase_twist));  // 고정된 위상 속도를 제거한다.
    handle->previous_swing_mask = 0U;                            // 이전 Swing 그룹을 제거한다.
    handle->phase_twist_valid = false;                           // 다음 위상을 첫 걸음으로 준비한다.
}

/* 자세 역회전이 적용된 발을 궤적 좌표로 되돌린다. */
static RobotVec3_t FootTrajectory_RotateForward(const RobotVec3_t *input,
                                                const RobotEuler_t *posture)
{
    RobotVec3_t output;                    // 궤적 좌표의 발 위치를 저장한다.
    const float cr = cosf(posture->roll);  // Roll Cosine을 계산한다.
    const float sr = sinf(posture->roll);  // Roll Sine을 계산한다.
    const float cp = cosf(posture->pitch); // Pitch Cosine을 계산한다.
    const float sp = sinf(posture->pitch); // Pitch Sine을 계산한다.
    const float cy = cosf(posture->yaw);   // Yaw Cosine을 계산한다.
    const float sy = sinf(posture->yaw);   // Yaw Sine을 계산한다.

    output.x = cy * cp * input->x +
               (cy * sp * sr - sy * cr) * input->y +
               (cy * sp * cr + sy * sr) * input->z;  // 궤적 X를 계산한다.
    output.y = sy * cp * input->x +
               (sy * sp * sr + cy * cr) * input->y +
               (sy * sp * cr - cy * sr) * input->z;  // 궤적 Y를 계산한다.
    output.z = -sp * input->x + cp * sr * input->y +
               cp * cr * input->z;                   // 궤적 Z를 계산한다.
    return output;
}

/* 주어진 Body Twist의 한 위상 발 이동량을 계산한다. */
static RobotVec3_t FootTrajectory_PhaseDisplacement(const RobotVec3_t *foot,
                                                    const RobotBodyTwist_t *twist)
{
    RobotVec3_t displacement;  // 한 위상 발 이동량을 저장한다.

    displacement.x = ROBOT_GAIT_PHASE_TIME_S *
                     (-twist->vx + twist->wz * foot->y);  // 몸체 이동 반대 X를 계산한다.
    displacement.y = ROBOT_GAIT_PHASE_TIME_S *
                     (-twist->vy - twist->wz * foot->x);  // 몸체 이동 반대 Y를 계산한다.
    displacement.z = ROBOT_GAIT_PHASE_TIME_S * (-twist->vz);  // 몸체 이동 반대 Z를 계산한다.
    return displacement;
}

/* 기본 착지 목표와 높이를 한 계획에 고정한다. */
void FootTrajectory_BuildPlan(FootTrajectory_Plan_t *plan,
                               const RobotVec3_t feet[ROBOT_LEG_COUNT],
                               const RobotVec3_t *body_offset_m,
                               const RobotBodyTwist_t *twist,
                               RobotGaitPattern_t pattern,
                               uint8_t swing_mask)
{
    RobotVec3_t base[ROBOT_LEG_COUNT];  // 몸체 보정 전 기준점을 저장한다.
    uint32_t leg;                       // 계획할 다리 번호를 저장한다.

    if ((plan == NULL) || (feet == NULL) || (body_offset_m == NULL) || (twist == NULL))
    {
        return;
    }

    memset(plan, 0, sizeof(*plan));   // 이전 잔차와 목표를 제거한다.
    LegKinematics_GetBaseFeet(base);  // 여섯 기본 발 위치를 준비한다.
    plan->twist = *twist;             // 기본 끝점을 만든 속도를 고정한다.
    plan->swing_mask = swing_mask;    // 잔차 적용 대상을 고정한다.
    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        RobotVec3_t displacement;  // 기준점의 한 위상 이동량을 저장한다.
        const float stance_phases = (pattern == ROBOT_GAIT_WAVE)
                                  ? (float)ROBOT_WAVE_STANCE_PHASES : 1.0f;  // 다음 이륙까지의 지지 구간을 선택한다.

        base[leg].x -= body_offset_m->x;                                             // 몸체 보정 X를 반영한다.
        base[leg].y -= body_offset_m->y;                                             // 몸체 보정 Y를 반영한다.
        base[leg].z -= body_offset_m->z;                                             // 몸체 보정 Z를 반영한다.
        displacement = FootTrajectory_PhaseDisplacement(&base[leg], twist);          // 실행과 같은 위상 이동량을 계산한다.
        plan->start[leg] = feet[leg];                                                // 실제 연속 시작점을 고정한다.
        plan->nominal[leg].x = base[leg].x - 0.5f * stance_phases * displacement.x;  // 기본 착지 X를 계산한다.
        plan->nominal[leg].y = base[leg].y - 0.5f * stance_phases * displacement.y;  // 기본 착지 Y를 계산한다.
        plan->nominal[leg].z = base[leg].z - 0.5f * stance_phases * displacement.z;  // 기본 착지 Z를 계산한다.
        plan->nominal_height[leg] = fminf(fmaxf(ROBOT_SWING_HEIGHT_M + body_offset_m->z,
                                                ROBOT_SWING_HEIGHT_MIN_M),
                                          ROBOT_SWING_HEIGHT_MAX_M);  // 실행과 같은 기본 높이를 계산한다.
        plan->target[leg] = plan->nominal[leg];                       // 잔차 0의 목표를 준비한다.
        plan->height[leg] = plan->nominal_height[leg];                // 잔차 0의 높이를 준비한다.
    }
    plan->valid = (swing_mask != 0U) &&
                  ((swing_mask & ~((1U << ROBOT_LEG_COUNT) - 1U)) == 0U);  // 실제 다리만 포함한 계획을 공개한다.
}

/* 검증된 계획을 현재 Swing과 분리해 다음 이륙까지 보관한다. */
bool FootTrajectory_SetPlan(FootTrajectory_Handle_t *handle,
                             const FootTrajectory_Plan_t *plan)
{
    if ((handle == NULL) || (plan == NULL) || !plan->valid)
    {
        return false;
    }
    if (handle->active_plan_valid && (handle->active_plan_id == plan->plan_id))
    {
        return true;
    }
    handle->pending_plan = *plan;  // 같은 끝점과 높이를 다음 위상에 전달한다.
    return true;
}

/* 새 이륙만 취소하고 이미 실행 중인 착지 목표를 유지한다. */
void FootTrajectory_CancelPlan(FootTrajectory_Handle_t *handle)
{
    if (handle != NULL)
    {
        handle->pending_plan.valid = false;  // 대기 후보가 뒤늦게 적용되는 일을 막는다.
    }
}


/* 한 Stance 발을 Body Twist 반대 방향으로 한 주기 이동한다. */
static void FootTrajectory_IntegrateStance(RobotVec3_t *position,
                                           const RobotBodyTwist_t *twist)
{
    const float x_dot = -twist->vx + twist->wz * position->y;  // Stance X속도를 계산한다.
    const float y_dot = -twist->vy - twist->wz * position->x;  // Stance Y속도를 계산한다.

    position->x += x_dot * ROBOT_CONTROL_PERIOD_S;  // Stance X를 적분한다.
    position->y += y_dot * ROBOT_CONTROL_PERIOD_S;  // Stance Y를 적분한다.
    position->z -= twist->vz * ROBOT_CONTROL_PERIOD_S;  // Stance Z를 적분한다.
}

/* 세 착지 오차의 중간값을 계산한다. */
static float FootTrajectory_Median3(float first, float second, float third)
{
    if (first > second)
    {
        const float temporary = first;  // 첫째 값을 임시 저장한다.
        first = second;                 // 작은 값을 앞으로 옮긴다.
        second = temporary;             // 큰 값을 뒤로 옮긴다.
    }
    if (second > third)
    {
        const float temporary = second;  // 둘째 값을 임시 저장한다.
        second = third;                  // 작은 값을 앞으로 옮긴다.
        third = temporary;               // 큰 값을 뒤로 옮긴다.
    }
    if (first > second)
    {
        second = first;  // 남은 두 값 중 큰 값을 중간값으로 둔다.
    }
    return second;
}

/* 공통 착지 Z 복구 상태를 제거한다. */
static void FootTrajectory_ClearCommonZRecovery(FootTrajectory_Handle_t *handle)
{
    if (handle == NULL)
    {
        return;
    }

    memset(handle->landing_z_error_valid, 0,
           sizeof(handle->landing_z_error_valid));  // 수집 중인 착지 오차를 제거한다.
    memset(handle->touchdown_pending, 0,
           sizeof(handle->touchdown_pending));  // 대기 중인 첫 명령 Z를 제거한다.
    memset(handle->common_z_recovery_remaining, 0,
           sizeof(handle->common_z_recovery_remaining));  // 두 Tripod의 남은 복구량을 제거한다.
    memset(handle->common_z_recovery_total, 0,
           sizeof(handle->common_z_recovery_total));  // 두 Tripod의 전체 복구량을 제거한다.
    memset(handle->common_z_recovery_progress, 0,
           sizeof(handle->common_z_recovery_progress));  // 두 Tripod의 S-curve 진행률을 제거한다.
}

/* 한 Tripod의 공통 하강 오차를 복구 대상으로 저장한다. */
static void FootTrajectory_RecordLandingZ(FootTrajectory_Handle_t *handle,
                                           uint32_t leg,
                                           float landing_error)
{
    const uint32_t group = leg % 2U;    // 현재 Tripod 번호를 선택한다.
    const uint32_t first = group;       // 현재 Tripod의 첫 다리를 선택한다.
    const uint32_t second = first + 2U; // 현재 Tripod의 둘째 다리를 선택한다.
    const uint32_t third = first + 4U;  // 현재 Tripod의 셋째 다리를 선택한다.
    float common_error;                 // 세 다리의 공통 착지 오차를 저장한다.
    float recovery;                     // 데드밴드와 이득을 적용한 복구량을 저장한다.

    handle->landing_z_error[leg] = landing_error;  // PWM 명령 FK의 착지 Z 오차를 저장한다.
    handle->landing_z_error_valid[leg] = true;     // 현재 다리 오차 수집을 표시한다.

    if (!handle->landing_z_error_valid[first] ||
        !handle->landing_z_error_valid[second] ||
        !handle->landing_z_error_valid[third])
    {
        return;
    }

    common_error = FootTrajectory_Median3(handle->landing_z_error[first],
                                           handle->landing_z_error[second],
                                           handle->landing_z_error[third]);  // 한 다리 이상값을 제외한 공통 오차를 선택한다.
    if (common_error < -ROBOT_COMMON_Z_RECOVERY_DEADBAND_M)
    {
        recovery = fminf(
            (-common_error) * ROBOT_COMMON_Z_RECOVERY_GAIN,
            ROBOT_COMMON_Z_RECOVERY_MAX_M);  // 검출된 공통 하강 오차 전체를 복구 후보로 만든다.
        if (recovery > handle->common_z_recovery_remaining[group])
        {
            handle->common_z_recovery_remaining[group] = recovery;  // 더 큰 복구량으로 잔량을 교체한다.
            handle->common_z_recovery_total[group] = recovery;      // 새 S-curve 전체량을 저장한다.
            handle->common_z_recovery_progress[group] = 0.0f;       // 새 S-curve를 처음부터 시작한다.
        }
    }
    handle->landing_z_error_valid[first] = false;   // 첫 다리 수집을 완료한다.
    handle->landing_z_error_valid[second] = false;  // 둘째 다리 수집을 완료한다.
    handle->landing_z_error_valid[third] = false;   // 셋째 다리 수집을 완료한다.
}

/* 착지한 Tripod의 공통 Z 오차를 Stance 중 천천히 제거한다. */
static void FootTrajectory_ApplyCommonZRecovery(FootTrajectory_Handle_t *handle,
                                                 const RobotGaitPhase_t *gait)
{
    uint32_t group;  // 확인할 Tripod 번호를 저장한다.

    if ((handle == NULL) || (gait == NULL))
    {
        return;
    }

    for (group = 0U; group < 2U; ++group)
    {
        const uint32_t first = group;        // 현재 Tripod의 첫 다리를 선택한다.
        const uint32_t second = first + 2U;  // 현재 Tripod의 둘째 다리를 선택한다.
        const uint32_t third = first + 4U;   // 현재 Tripod의 셋째 다리를 선택한다.
        float previous_progress;             // 이번 주기 전 S-curve 진행률을 저장한다.
        float next_progress;                 // 이번 주기 후 S-curve 진행률을 저장한다.
        float previous_curve;                // 이번 주기 전 S-curve 위치를 저장한다.
        float next_curve;                    // 이번 주기 후 S-curve 위치를 저장한다.
        float recovery_step;                 // 이번 주기 복구량을 저장한다.

        if ((handle->common_z_recovery_remaining[group] <= 0.0f) ||
            (gait->state[first] != ROBOT_LEG_STANCE) ||
            (gait->state[second] != ROBOT_LEG_STANCE) ||
            (gait->state[third] != ROBOT_LEG_STANCE))
        {
            continue;  // Swing과 지지발 재착지 중에는 복구를 보류한다.
        }

        if (handle->common_z_recovery_total[group] <= 0.0f)
        {
            handle->common_z_recovery_total[group] =
                handle->common_z_recovery_remaining[group];  // 누락된 전체량을 현재 잔량으로 복구한다.
            handle->common_z_recovery_progress[group] = 0.0f;  // S-curve를 처음부터 시작한다.
        }

        previous_progress = handle->common_z_recovery_progress[group];  // 현재 진행률을 보존한다.
        next_progress = fminf(previous_progress +
                              ROBOT_CONTROL_PERIOD_S / ROBOT_COMMON_Z_RECOVERY_TIME_S,
                              1.0f);  // Stance에서만 S-curve 시간을 진행한다.
        previous_curve = previous_progress * previous_progress *
                         (3.0f - 2.0f * previous_progress);  // 이전 Smoothstep 위치를 계산한다.
        next_curve = next_progress * next_progress *
                     (3.0f - 2.0f * next_progress);  // 다음 Smoothstep 위치를 계산한다.
        recovery_step = fminf(
            handle->common_z_recovery_remaining[group],
            handle->common_z_recovery_total[group] *
            (next_curve - previous_curve));  // 시작·끝은 느리고 중간은 빠르게 복구한다.
        handle->memory[first].z += recovery_step;                       // 첫 지지발 Z를 복구한다.
        handle->memory[second].z += recovery_step;                      // 둘째 지지발 Z를 복구한다.
        handle->memory[third].z += recovery_step;                       // 셋째 지지발 Z를 복구한다.
        handle->common_z_recovery_remaining[group] -= recovery_step;    // 적용한 복구량을 차감한다.
        handle->common_z_recovery_progress[group] = next_progress;      // 새 S-curve 진행률을 저장한다.
        if (handle->common_z_recovery_remaining[group] <= 0.0f)
        {
            handle->common_z_recovery_remaining[group] = 0.0f;  // 부동소수점 잔여량을 제거한다.
            handle->common_z_recovery_total[group] = 0.0f;      // 완료한 전체량을 제거한다.
            handle->common_z_recovery_progress[group] = 0.0f;   // 완료한 진행률을 제거한다.
        }
    }
}

/* 궤적이 비활성일 때 한 다리 메모리를 기준 위치로 되돌린다. */
static RobotVec3_t FootTrajectory_ResetLeg(FootTrajectory_Handle_t *handle,
                                           uint32_t leg,
                                           const RobotVec3_t *base)
{
    handle->memory[leg] = *base;                          // 연속 발 위치를 기준점으로 둔다.
    handle->landing_target_z[leg] = base->z;              // 정상 착지 Z를 기준점으로 둔다.
    handle->previous_state[leg] = ROBOT_LEG_STANCE;      // 이전 상태를 Stance로 둔다.
    handle->adapted_stance[leg] = false;                 // 접촉 적응 상태를 제거한다.
    handle->custom_swing[leg] = false;                   // 사용자 Swing 시작점을 제거한다.
    handle->swing_resume_progress[leg] = 0.0f;           // Swing 재개 진행률을 제거한다.
    handle->swing_resume_active[leg] = false;            // Swing 재개 상태를 제거한다.
    handle->landing_z_error_valid[leg] = false;          // 이전 착지 오차를 제거한다.
    handle->touchdown_pending[leg] = false;              // 대기 중인 실측 FK를 제거한다.
    return *base;
}

/* 한 다리의 상태 전환을 반영해 연속 발 위치를 계산한다. */
static RobotVec3_t FootTrajectory_AdaptiveLeg(FootTrajectory_Handle_t *handle,
                                              uint32_t leg,
                                              const RobotVec3_t *base,
                                              const RobotVec3_t *displacement,
                                              const RobotBodyTwist_t *twist,
                                              const RobotGaitPhase_t *gait,
                                              float swing_height,
                                              bool trajectory_enable,
                                              bool hold_stance,
                                              bool all_stance_mode)
{
    RobotVec3_t front;        // 위상 앞쪽 발 위치를 저장한다.
    RobotVec3_t rear;         // 위상 뒤쪽 발 위치를 저장한다.
    RobotVec3_t swing_start;  // 실제 Swing 시작점을 저장한다.
    RobotVec3_t output;       // 이번 발 위치를 저장한다.
    float swing_progress;     // 정지 후 다시 매핑한 Swing 진행률을 저장한다.
    const RobotLegState_t state = gait->state[leg];         // 현재 다리 상태를 선택한다.
    const RobotLegState_t previous = handle->previous_state[leg];  // 이전 다리 상태를 선택한다.
    const float progress = gait->progress[leg];             // 현재 다리 진행률을 선택한다.

    if (!trajectory_enable)
    {
        return FootTrajectory_ResetLeg(handle, leg, base);  // 비활성 발 위치를 기준점으로 되돌린다.
    }

    const float stance_phases = (gait->gait_pattern == ROBOT_GAIT_WAVE)
                              ? (float)ROBOT_WAVE_STANCE_PHASES : 1.0f;  // 재이륙까지의 지지 시간을 선택한다.

    front.x = base->x - 0.5f * stance_phases * displacement->x;  // 전체 지지 구간의 앞쪽 X를 계산한다.
    front.y = base->y - 0.5f * stance_phases * displacement->y;  // 전체 지지 구간의 앞쪽 Y를 계산한다.
    front.z = base->z - 0.5f * stance_phases * displacement->z;  // 전체 지지 구간의 앞쪽 Z를 계산한다.
    rear.x = base->x + 0.5f * displacement->x;   // 위상 뒤쪽 X를 계산한다.
    rear.y = base->y + 0.5f * displacement->y;   // 위상 뒤쪽 Y를 계산한다.
    rear.z = base->z + 0.5f * displacement->z;   // 위상 뒤쪽 Z를 계산한다.
    swing_start = gait->startup_phase ? *base : rear;  // 첫 위상만 기준점에서 Swing을 시작한다.

    if ((state == ROBOT_LEG_TOUCHDOWN_CANDIDATE) ||
        (state == ROBOT_LEG_HOLD))
    {
        output = handle->memory[leg];  // 접촉 확인과 보행 일시정지에서 현재 위치를 유지한다.
    }
    else if (state == ROBOT_LEG_RECOVERY_SWING)
    {
        if (previous != ROBOT_LEG_RECOVERY_SWING)
        {
            handle->recovery_start[leg] = handle->memory[leg];  // 복구 진입 순간 위치를 저장한다.
        }

        output = SwingTrajectory_Calculate(progress,
                                           &handle->recovery_start[leg],
                                           base,
                                           FOOT_RECOVERY_HEIGHT_M,
                                           0.0f,
                                           foot_leg_angle[leg]);  // 기준점으로 복구 Swing을 계산한다.
        handle->memory[leg] = output;       // 연속 발 위치를 갱신한다.
        handle->adapted_stance[leg] = true;// 복구 후 연속 Stance를 사용한다.
        handle->custom_swing[leg] = false; // 이전 사용자 Swing을 제거한다.
    }
    else if (state == ROBOT_LEG_LATE_LANDING)
    {
        if (!gait->late_landing_exhausted[leg])
        {
            ContactAdaptation_ApplyLateLanding(&handle->memory[leg],
                                                foot_leg_angle[leg]);  // 한계 전까지 지면을 탐색한다.
        }

        handle->adapted_stance[leg] = true;  // 착지 후 현재 위치를 지지점으로 유지한다.
        output = handle->memory[leg];         // 탐색 중 위치를 반환한다.
    }
    else if (state == ROBOT_LEG_SWING)
    {
        if (handle->active_plan.valid &&
            ((handle->active_plan.swing_mask & (uint8_t)(1U << leg)) != 0U))
        {
            front = handle->active_plan.target[leg];          // 검사에 사용한 잔차 착지점을 그대로 사용한다.
            swing_height = handle->active_plan.height[leg];  // 검사에 사용한 다리별 높이를 그대로 사용한다.
        }
        if (previous != ROBOT_LEG_SWING)
        {
            handle->landing_z_error_valid[leg] = false;  // 새 Swing의 착지 오차를 준비한다.
            handle->swing_start[leg] = handle->memory[leg];  // 실제 직전 위치에서 새 Swing을 시작한다.
            handle->custom_swing[leg] = true;                // 연속 시작점을 활성화한다.
            handle->adapted_stance[leg] = false;             // 새 Swing에서 적응 상태를 해제한다.
            handle->swing_resume_active[leg] =
                (previous == ROBOT_LEG_TOUCHDOWN_CANDIDATE);  // 접촉 후보 취소에서만 남은 Swing을 다시 매핑한다.
            handle->swing_resume_progress[leg] = progress;  // 재개 순간의 기존 위상 진행률을 저장한다.
        }

        if (handle->custom_swing[leg])
        {
            swing_start = handle->swing_start[leg];  // 이전 연속 위치를 Swing 시작점으로 사용한다.
        }

        swing_progress = progress;  // 일반 Swing 진행률을 기본값으로 사용한다.
        if (handle->swing_resume_active[leg])
        {
            const float remaining = 1.0f - handle->swing_resume_progress[leg];  // 남은 위상 비율을 계산한다.

            swing_progress = (remaining > 0.000001f) ?
                (progress - handle->swing_resume_progress[leg]) / remaining :
                1.0f;  // 정지 위치를 0으로 두고 남은 경로를 다시 매핑한다.
        }

        handle->landing_target_z[leg] = front.z;  // 이번 Swing의 정상 착지 Z를 저장한다.
        output = SwingTrajectory_Calculate(swing_progress,
                                           &swing_start,
                                           &front,
                                           swing_height,
                                           ROBOT_SWING_RADIAL_OFFSET_M,
                                           foot_leg_angle[leg]);  // 정상 Swing 궤적을 계산한다.
        if ((progress >= ROBOT_EARLY_LANDING_PROGRESS) &&
            (output.z <= (handle->landing_target_z[leg] +
                          ROBOT_SWING_LANDING_APPROACH_M)) &&
            (output.z < handle->memory[leg].z))
        {
            const float minimum_z = handle->memory[leg].z -
                ROBOT_SWING_LANDING_SPEED_MPS * ROBOT_CONTROL_PERIOD_S;  // 한 주기의 최대 하강 위치를 계산한다.

            output.z = fmaxf(output.z, minimum_z);  // 지면 접근 중 하강 충격을 제한한다.
        }
        handle->memory[leg] = output;  // 연속 발 위치를 갱신한다.
    }
    else if (state == ROBOT_LEG_STANCE)
    {
        if (all_stance_mode)
        {
            FootTrajectory_IntegrateStance(&handle->memory[leg], twist);  // 보정 모드 발 고정을 적분한다.
        }
        else if (!hold_stance)
        {
            if ((previous == ROBOT_LEG_LATE_LANDING) ||
                (previous == ROBOT_LEG_TOUCHDOWN_CANDIDATE))
            {
                handle->adapted_stance[leg] = true;  // 확인된 접촉 위치를 새 지지점으로 둔다.
            }
            else if (previous == ROBOT_LEG_SWING)
            {
                handle->adapted_stance[leg] = true;  // 실제 Swing 끝점에서 새 Stance를 시작한다.
            }
            else if (gait->gait_pattern == ROBOT_GAIT_WAVE)
            {
                const float phase_delta = fmaxf(progress - handle->previous_progress[leg], 0.0f);  // 멈춘 위상의 중복 적분을 막는다.
                const float duration_s = fminf(phase_delta * ROBOT_GAIT_PHASE_TIME_S,
                                               ROBOT_CONTROL_PERIOD_S) *
                                         (gait->startup_phase ? 0.5f : 1.0f);  // 첫 순회에서 아직 들지 않은 발의 이동량을 줄인다.

                handle->memory[leg] = StanceTrajectory_Advance(&handle->memory[leg],
                                                               twist, duration_s);  // 다섯 지지발을 현재 위치에서 연속 이동한다.
                handle->adapted_stance[leg] = true;  // 다음 위상에서도 실제 지지점을 이어간다.
            }
            else if (handle->adapted_stance[leg])
            {
                FootTrajectory_IntegrateStance(&handle->memory[leg], twist);  // 적응 지지점을 Body 반대로 이동한다.
            }
            else if (gait->startup_phase)
            {
                handle->memory[leg] = StanceTrajectory_Interpolate(progress, base, &rear);  // 첫 Stance를 반 보폭으로 시작한다.
            }
            else
            {
                handle->memory[leg] = StanceTrajectory_Interpolate(progress, &front, &rear);  // 정상 Stance를 계산한다.
            }
        }

        output = handle->memory[leg];  // 현재 Stance 위치를 반환한다.
    }
    else
    {
        output = handle->memory[leg];  // 알 수 없는 상태에서 직전 위치를 유지한다.
    }

    handle->previous_progress[leg] = progress;  // 다음 주기의 실제 위상 진행량을 준비한다.
    if (state != ROBOT_LEG_HOLD)
    {
        handle->previous_state[leg] = state;  // HOLD 전 상태를 보존해 기존 궤적을 이어간다.
    }
    return output;
}

/* 기본 발 위치와 내부 연속 상태를 초기화한다. */
void FootTrajectory_Init(FootTrajectory_Handle_t *handle)
{
    RobotVec3_t base[ROBOT_LEG_COUNT];  // 기본 발 위치를 저장한다.
    uint32_t leg;                       // 초기화할 다리 번호를 저장한다.

    if (handle == NULL)
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));  // 이전 궤적 상태를 제거한다.
    LegKinematics_GetBaseFeet(base);     // 기본 발 위치를 계산한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        handle->memory[leg] = base[leg];                    // 발 메모리를 기본 위치로 둔다.
        handle->landing_target_z[leg] = base[leg].z;        // 정상 착지 Z를 기본 위치로 둔다.
        handle->previous_state[leg] = ROBOT_LEG_STANCE;     // 이전 상태를 Stance로 둔다.
    }

    handle->initialized = true;  // 궤적 초기화를 표시한다.
}

/* 접촉 위치를 궤적 메모리와 지형 측정 대기에 저장한다. */
static void FootTrajectory_SaveTouchdown(FootTrajectory_Handle_t *handle,
                                         uint8_t leg,
                                         const RobotVec3_t *trajectory_foot)
{
    handle->memory[leg] = *trajectory_foot;                  // 접촉 위치를 다음 목표로 고정한다.
    handle->adapted_stance[leg] = true;                      // 접촉 위치 기반 Stance를 준비한다.
    handle->landing_z_error[leg] = trajectory_foot->z -
                                   handle->landing_target_z[leg];  // 접촉 후보의 원시 Z 오차를 임시 저장한다.
    handle->landing_z_error_valid[leg] = false;              // 접촉 확인 전 오차를 보류한다.
    handle->touchdown_pending[leg] = true;                   // 접촉 확인 후 Z 반영을 예약한다.
}

/* 접촉 순간 Body 발 위치를 궤적 좌표로 바꿔 고정한다. */
bool FootTrajectory_LatchTouchdown(FootTrajectory_Handle_t *handle,
                                   uint8_t leg,
                                   const RobotVec3_t *commanded_foot_body,
                                   const RobotEuler_t *posture_rad)
{
    RobotVec3_t commanded_trajectory;  // 자세를 제거한 PWM 명령 발 위치를 저장한다.

    if ((handle == NULL) || (commanded_foot_body == NULL) ||
        (posture_rad == NULL) || (leg >= ROBOT_LEG_COUNT) ||
        !handle->initialized)
    {
        return false;
    }

    commanded_trajectory = FootTrajectory_RotateForward(commanded_foot_body,
                                                         posture_rad);  // PWM 명령 발을 궤적 좌표로 변환한다.
    FootTrajectory_SaveTouchdown(handle,
                                 leg,
                                 &commanded_trajectory);  // 변환한 접촉 위치를 고정한다.
    return true;
}

/* 접촉 순간의 직전 궤적 명령 위치를 고정한다. */
bool FootTrajectory_LatchCommandedTouchdown(FootTrajectory_Handle_t *handle,
                                            uint8_t leg)
{
    RobotVec3_t commanded_trajectory;  // 직전 궤적 명령을 보존한다.

    if ((handle == NULL) || (leg >= ROBOT_LEG_COUNT) || !handle->initialized)
    {
        return false;
    }

    commanded_trajectory = handle->memory[leg];  // 접촉 전 마지막 명령 위치를 복사한다.
    FootTrajectory_SaveTouchdown(handle,
                                 leg,
                                 &commanded_trajectory);  // 지연 없는 궤적 위치를 고정한다.
    return true;
}

/* 접촉 확인 후 고정한 궤적으로 두 Tripod의 공통 Z 오차를 수집한다. */
void FootTrajectory_UpdateCommandedLanding(FootTrajectory_Handle_t *handle,
                                            const RobotGaitPhase_t *gait,
                                            bool common_z_enable)
{
    uint32_t leg;  // 확인할 다리 번호를 저장한다.

    if ((handle == NULL) || (gait == NULL))
    {
        return;
    }

    if (!common_z_enable || (gait->gait_pattern == ROBOT_GAIT_WAVE))
    {
        FootTrajectory_ClearCommonZRecovery(handle);  // 비활성 시 측정과 복구 잔량을 모두 제거한다.
        return;
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        if (!handle->touchdown_pending[leg])
        {
            continue;
        }

        if (gait->state[leg] == ROBOT_LEG_TOUCHDOWN_CANDIDATE)
        {
            continue;  // 접촉 확정까지 처음 고정한 위치를 보존한다.
        }

        handle->touchdown_pending[leg] = false;  // 후보 취소 또는 확정 측정을 완료한다.
        if (gait->state[leg] != ROBOT_LEG_STANCE)
        {
            continue;
        }

        FootTrajectory_RecordLandingZ(handle, leg,
                                      handle->landing_z_error[leg]);  // 첫 접촉 순간의 명령 Z 오차를 수집한다.
    }
}

/* Body Twist와 다리 상태로 여섯 발의 연속 목표를 계산한다. */
RobotFootTargets_t FootTrajectory_Step(FootTrajectory_Handle_t *handle,
                                       const RobotBodyTwist_t *twist,
                                       const RobotDroneOutput_t *drone,
                                       const RobotGaitPhase_t *gait,
                                       const RobotEuler_t *posture_rad)
{
    RobotFootTargets_t output;             // 이번 발 목표를 저장한다.
    RobotBodyTwist_t applied_twist;        // 실제 궤적에 사용할 Twist를 저장한다.
    RobotVec3_t base[ROBOT_LEG_COUNT];     // 기본 발 위치를 저장한다.
    RobotVec3_t corrected[ROBOT_LEG_COUNT];// 몸체 보정 후 기준 위치를 저장한다.
    bool all_stance_mode;                  // 보정 이동 모드를 저장한다.
    bool manual_stance_hold;               // 수동 정지 Stance 유지를 저장한다.
    bool trajectory_enable;                // 발 궤적 활성화를 저장한다.
    bool hold_stance;                      // 특수 착지 Stance 유지를 저장한다.
    bool common_z_enable;                  // 정상 보행 공통 Z 복구 여부를 저장한다.
    float swing_height;                    // 현재 Swing 높이를 저장한다.
    uint32_t leg;                          // 계산할 다리 번호를 저장한다.

    memset(&output, 0, sizeof(output));  // 기본 출력을 0으로 준비한다.

    if ((handle == NULL) || (twist == NULL) || (drone == NULL) ||
        (gait == NULL) || (posture_rad == NULL))
    {
        return output;
    }

    if (!handle->initialized)
    {
        FootTrajectory_Init(handle);  // 누락된 초기화를 보완한다.
    }

    if (drone->hold_feet)
    {
        memcpy(output.foot, handle->memory, sizeof(output.foot));  // RL 종료 후 실제 접촉 발 위치를 보존한다.
        output.command_accepted = true;                            // 연속 위치 유지 명령을 허가한다.
        return output;
    }

    if (gait->late_landing_hold)
    {
        FootTrajectory_ClearPhaseTwist(handle);  // 중단한 위상의 속도 기억을 제거한다.
        for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
        {
            output.foot[leg] = handle->memory[leg];  // 탐색 한계 순간의 모든 발 목표를 유지한다.
        }
        output.command_accepted = true;  // 위치 유지 명령을 정상 출력으로 표시한다.
        return output;
    }

    LegKinematics_GetBaseFeet(base);  // 기본 발 위치를 계산한다.
    applied_twist = *twist;           // 상위 적용 Twist를 복사한다.
    output.command_accepted = true;   // 기본적으로 보정 명령을 채택한다.
    if ((drone->locomotion_enable || drone->manual_enable) &&
        (drone->tripod_mode == ROBOT_TRIPOD_NORMAL) &&
        gait->enabled_internal)
    {
        const uint8_t swing_mask = FootTrajectory_SwingMask(gait);                // 현재 Swing 그룹을 읽는다.
        const uint8_t new_swing_mask = swing_mask & ~handle->previous_swing_mask; // 새로 공중에 든 다리를 찾는다.

        if (gait->support_recovery_active && handle->phase_twist_valid)
        {
            applied_twist = handle->phase_twist;  // 지지발 재착지 중 기존 위상 속도를 보존한다.
        }
        else if (new_swing_mask != 0U)
        {
            handle->active_plan.valid = false;  // 이전 다리 잔차를 새 위상에 재사용하지 않는다.
            if (handle->pending_plan.valid &&
                (handle->pending_plan.swing_mask == swing_mask))
            {
                handle->active_plan = handle->pending_plan;       // 검증한 끝점과 높이를 한 번 고정한다.
                handle->pending_plan.valid = false;               // 이번 위상 후보의 재사용을 막는다.
                handle->active_plan_id = handle->active_plan.plan_id;  // 실제 이륙 계획 번호를 기록한다.
                handle->active_plan_mask = swing_mask;            // 실제 잔차 적용 다리를 기록한다.
                handle->active_plan_valid = true;                 // 실제 적용 이력을 활성화한다.
                applied_twist = handle->active_plan.twist;        // 기본 계획과 같은 속도로 움직인다.
            }
            handle->pending_plan.valid = false;  // 위상이 달라진 오래된 후보도 폐기한다.
            handle->phase_twist = applied_twist;  // 착륙 시 세 지점 검증을 마친 속도를 현재 위상에 고정한다.
            handle->phase_twist_valid = true;     // 현재 위상 속도 고정을 활성화한다.
        }

        if (handle->phase_twist_valid)
        {
            applied_twist = handle->phase_twist;  // 위상 중 입력 변화로 보폭이 바뀌지 않게 한다.
        }

        if (!gait->support_recovery_active)
        {
            handle->previous_swing_mask = swing_mask;  // 재착지 밖에서 다음 위상 전환을 준비한다.
        }
    }
    else
    {
        FootTrajectory_ClearPhaseTwist(handle);  // 정지와 특수 모드에서 다음 첫 걸음을 준비한다.
    }

    manual_stance_hold = (drone->locomotion_enable || drone->manual_enable) && !gait->enabled_internal &&
                         (drone->tripod_mode == ROBOT_TRIPOD_NORMAL);  // 입력 유지 중 IK 거부도 정지로 처리한다.
    all_stance_mode = drone->correction_enable && drone->body_control_enable &&
                      !gait->enabled_internal && !drone->tripod_enable &&
                      (drone->tripod_mode == ROBOT_TRIPOD_NORMAL);  // 명시적 보정 모드 조건을 계산한다.
    trajectory_enable = drone->body_control_enable || drone->tripod_enable;  // 궤적 활성 조건을 계산한다.
    hold_stance = (drone->tripod_mode != ROBOT_TRIPOD_NORMAL) ||
                  gait->waiting_start || manual_stance_hold;  // 특수 착지·수동 정지 중 Stance를 유지한다.
    common_z_enable = (drone->locomotion_enable || drone->manual_enable) && gait->enabled_internal &&
                      !gait->waiting_start &&
                      (gait->gait_pattern == ROBOT_GAIT_TRIPOD) &&
                      (drone->tripod_mode == ROBOT_TRIPOD_NORMAL);  // 실제 보행 중에만 공통 Z를 적용한다.

    if (manual_stance_hold)
    {
        memset(&applied_twist, 0, sizeof(applied_twist));  // 정지 중 직전 보폭과 Yaw 보정 적용을 차단한다.
    }

    if (common_z_enable)
    {
        FootTrajectory_ApplyCommonZRecovery(handle, gait);  // 이전 Tripod의 공통 하강량을 복구한다.
    }
    else if (!(drone->locomotion_enable || drone->manual_enable) || (gait->gait_pattern == ROBOT_GAIT_WAVE) ||
             (drone->tripod_mode != ROBOT_TRIPOD_NORMAL))
    {
        FootTrajectory_ClearCommonZRecovery(handle);  // 다른 모드의 착지 오차를 제거한다.
    }

    if (drone->reset_command || drone->stand_enable ||
        drone->landing_enable || drone->kill_enable)
    {
        memset(&handle->body_offset_m, 0, sizeof(handle->body_offset_m));  // 명시적 초기화 상태에서 보정 Offset을 제거한다.
    }
    else if (all_stance_mode)
    {
        RobotVec3_t candidate = handle->body_offset_m;  // 새 보정 Offset 후보를 저장한다.
        RobotVec3_t candidate_feet[ROBOT_LEG_COUNT];    // 새 보정 발 위치를 저장한다.

        candidate.x += applied_twist.vx * ROBOT_CONTROL_PERIOD_S;  // 몸체 X Offset을 적분한다.
        candidate.y += applied_twist.vy * ROBOT_CONTROL_PERIOD_S;  // 몸체 Y Offset을 적분한다.
        candidate.z += applied_twist.vz * ROBOT_CONTROL_PERIOD_S;  // 몸체 Z Offset을 적분한다.

        for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
        {
            candidate_feet[leg].x = base[leg].x - candidate.x;  // 후보 발 X를 계산한다.
            candidate_feet[leg].y = base[leg].y - candidate.y;  // 후보 발 Y를 계산한다.
            candidate_feet[leg].z = base[leg].z - candidate.z;  // 후보 발 Z를 계산한다.
        }

        if (WorkspaceLimiter_AllFeetValid(candidate_feet, posture_rad))
        {
            handle->body_offset_m = candidate;  // 전체 IK가 유효하면 보정 Offset을 채택한다.
        }
        else
        {
            output.command_accepted = false;  // 유효하지 않은 보정 명령을 거부한다.
            applied_twist.vx = 0.0f;          // 이번 X 보정 적분을 차단한다.
            applied_twist.vy = 0.0f;          // 이번 Y 보정 적분을 차단한다.
            applied_twist.vz = 0.0f;          // 이번 Z 보정 적분을 차단한다.
        }
    }

    swing_height = fminf(fmaxf(ROBOT_SWING_HEIGHT_M + handle->body_offset_m.z,
                                ROBOT_SWING_HEIGHT_MIN_M),
                          ROBOT_SWING_HEIGHT_MAX_M);  // 몸체 Z Offset에 맞춰 Swing 높이를 계산한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        corrected[leg].x = base[leg].x - handle->body_offset_m.x;  // 보정 기준 X를 계산한다.
        corrected[leg].y = base[leg].y - handle->body_offset_m.y;  // 보정 기준 Y를 계산한다.
        corrected[leg].z = base[leg].z - handle->body_offset_m.z;  // 보정 기준 Z를 계산한다.
    }

    if (!drone->body_control_enable || (drone->tripod_mode != ROBOT_TRIPOD_NORMAL))
    {
        memset(&applied_twist, 0, sizeof(applied_twist));  // 비정상 보행 모드에서 Twist를 제거한다.
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        const RobotVec3_t displacement = FootTrajectory_PhaseDisplacement(&corrected[leg],
                                                                           &applied_twist);  // 한 위상 이동량을 계산한다.
        output.foot[leg] = FootTrajectory_AdaptiveLeg(handle,
                                                       leg,
                                                       &corrected[leg],
                                                       &displacement,
                                                       &applied_twist,
                                                       gait,
                                                       swing_height,
                                                       trajectory_enable,
                                                       hold_stance,
                                                       all_stance_mode);  // 다리별 연속 궤적을 계산한다.
    }

    return output;
}
