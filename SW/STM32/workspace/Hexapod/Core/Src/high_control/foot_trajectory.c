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

/* 궤적이 비활성일 때 한 다리 메모리를 기준 위치로 되돌린다. */
static RobotVec3_t FootTrajectory_ResetLeg(FootTrajectory_Handle_t *handle,
                                           uint32_t leg,
                                           const RobotVec3_t *base)
{
    handle->memory[leg] = *base;                          // 연속 발 위치를 기준점으로 둔다.
    handle->previous_state[leg] = ROBOT_LEG_STANCE;      // 이전 상태를 Stance로 둔다.
    handle->adapted_stance[leg] = false;                 // 접촉 적응 상태를 제거한다.
    handle->custom_swing[leg] = false;                   // 사용자 Swing 시작점을 제거한다.
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
    const RobotLegState_t state = gait->state[leg];         // 현재 다리 상태를 선택한다.
    const RobotLegState_t previous = handle->previous_state[leg];  // 이전 다리 상태를 선택한다.
    const float progress = gait->progress[leg];             // 현재 다리 진행률을 선택한다.

    if (!trajectory_enable)
    {
        return FootTrajectory_ResetLeg(handle, leg, base);  // 비활성 발 위치를 기준점으로 되돌린다.
    }

    front.x = base->x - 0.5f * displacement->x;  // 위상 앞쪽 X를 계산한다.
    front.y = base->y - 0.5f * displacement->y;  // 위상 앞쪽 Y를 계산한다.
    front.z = base->z - 0.5f * displacement->z;  // 위상 앞쪽 Z를 계산한다.
    rear.x = base->x + 0.5f * displacement->x;   // 위상 뒤쪽 X를 계산한다.
    rear.y = base->y + 0.5f * displacement->y;   // 위상 뒤쪽 Y를 계산한다.
    rear.z = base->z + 0.5f * displacement->z;   // 위상 뒤쪽 Z를 계산한다.
    swing_start = gait->startup_phase ? *base : rear;  // 첫 위상만 기준점에서 Swing을 시작한다.

    if (state == ROBOT_LEG_RECOVERY_SWING)
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
        if (previous == ROBOT_LEG_LATE_LANDING)
        {
            ContactAdaptation_ApplyLateLanding(&handle->memory[leg],
                                                foot_leg_angle[leg]);  // 두 번째 Late 주기부터 지면을 탐색한다.
        }

        handle->adapted_stance[leg] = true;  // 착지 후 현재 위치를 지지점으로 유지한다.
        output = handle->memory[leg];         // 탐색 중 위치를 반환한다.
    }
    else if (state == ROBOT_LEG_SWING)
    {
        if (previous != ROBOT_LEG_SWING)
        {
            if (handle->adapted_stance[leg])
            {
                handle->swing_start[leg] = handle->memory[leg];  // 적응 Stance 위치에서 새 Swing을 시작한다.
                handle->custom_swing[leg] = true;                // 사용자 시작점을 활성화한다.
            }
            else
            {
                handle->swing_start[leg] = swing_start;  // 정상 위상 시작점을 저장한다.
                handle->custom_swing[leg] = false;       // 정상 시작점을 사용한다.
            }

            handle->adapted_stance[leg] = false;  // 새 Swing에서 적응 상태를 해제한다.
        }

        if (handle->custom_swing[leg])
        {
            swing_start = handle->swing_start[leg];  // 이전 연속 위치를 Swing 시작점으로 사용한다.
        }

        output = SwingTrajectory_Calculate(progress,
                                           &swing_start,
                                           &front,
                                           swing_height,
                                           ROBOT_SWING_RADIAL_OFFSET_M,
                                           foot_leg_angle[leg]);  // 정상 Swing 궤적을 계산한다.
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
            if (previous == ROBOT_LEG_LATE_LANDING)
            {
                handle->adapted_stance[leg] = true;  // Late 착지 위치를 새 지지점으로 둔다.
            }
            else if (ContactAdaptation_IsEarlyLanding(previous, state, progress))
            {
                if (handle->custom_swing[leg])
                {
                    swing_start = handle->swing_start[leg];  // 사용자 Swing 시작점을 유지한다.
                }

                handle->memory[leg] = SwingTrajectory_Calculate(progress,
                                                                 &swing_start,
                                                                 &front,
                                                                 swing_height,
                                                                 ROBOT_SWING_RADIAL_OFFSET_M,
                                                                 foot_leg_angle[leg]);  // 접촉 순간 Swing 위치를 고정한다.
                handle->adapted_stance[leg] = true;  // 조기 착지 지지점을 표시한다.
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

    handle->previous_state[leg] = state;  // 다음 상태 전환을 위해 저장한다.
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
        handle->previous_state[leg] = ROBOT_LEG_STANCE;     // 이전 상태를 Stance로 둔다.
    }

    handle->initialized = true;  // 궤적 초기화를 표시한다.
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
    bool trajectory_enable;                // 발 궤적 활성화를 저장한다.
    bool hold_stance;                      // 특수 착지 Stance 유지를 저장한다.
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

    LegKinematics_GetBaseFeet(base);  // 기본 발 위치를 계산한다.
    applied_twist = *twist;           // 상위 적용 Twist를 복사한다.
    output.command_accepted = true;   // 기본적으로 보정 명령을 채택한다.
    all_stance_mode = drone->body_control_enable && !drone->tripod_enable &&
                      (drone->tripod_mode == ROBOT_TRIPOD_NORMAL);  // 보정 모드 조건을 계산한다.
    trajectory_enable = drone->body_control_enable || drone->tripod_enable;  // 궤적 활성 조건을 계산한다.
    hold_stance = (drone->tripod_mode != ROBOT_TRIPOD_NORMAL);               // 특수 착지 Stance 유지를 계산한다.

    if (!drone->body_control_enable)
    {
        memset(&handle->body_offset_m, 0, sizeof(handle->body_offset_m));  // 비활성 상태에서 보정 Offset을 제거한다.
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
