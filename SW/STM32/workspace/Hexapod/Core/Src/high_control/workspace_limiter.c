#include "high_control/workspace_limiter.h"

#include "high_control/leg_kinematics.h"
#include "high_control/stance_trajectory.h"
#include "high_control/swing_trajectory.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static const float workspace_leg_angle_rad[ROBOT_LEG_COUNT] =
{
    -45.0f * ROBOT_DEG_TO_RAD_F,   // 1번 다리 장착각을 저장한다.
    -90.0f * ROBOT_DEG_TO_RAD_F,   // 2번 다리 장착각을 저장한다.
    -135.0f * ROBOT_DEG_TO_RAD_F,  // 3번 다리 장착각을 저장한다.
    45.0f * ROBOT_DEG_TO_RAD_F,    // 4번 다리 장착각을 저장한다.
    90.0f * ROBOT_DEG_TO_RAD_F,    // 5번 다리 장착각을 저장한다.
    135.0f * ROBOT_DEG_TO_RAD_F    // 6번 다리 장착각을 저장한다.
};

/* Body 발 벡터에 자세의 역회전을 적용한다. */
static RobotVec3_t WorkspaceLimiter_RotateInverse(const RobotVec3_t *input,
                                                  const RobotEuler_t *posture)
{
    RobotVec3_t output;                      // 역회전한 발 위치를 저장한다.
    const float cr = cosf(posture->roll);   // Roll Cosine을 계산한다.
    const float sr = sinf(posture->roll);   // Roll Sine을 계산한다.
    const float cp = cosf(posture->pitch);  // Pitch Cosine을 계산한다.
    const float sp = sinf(posture->pitch);  // Pitch Sine을 계산한다.
    const float cy = cosf(posture->yaw);    // Yaw Cosine을 계산한다.
    const float sy = sinf(posture->yaw);    // Yaw Sine을 계산한다.
    const float r11 = cy * cp;              // 회전행렬 1행 1열을 계산한다.
    const float r12 = cy * sp * sr - sy * cr;
    const float r13 = cy * sp * cr + sy * sr;
    const float r21 = sy * cp;
    const float r22 = sy * sp * sr + cy * cr;
    const float r23 = sy * sp * cr - cy * sr;
    const float r31 = -sp;
    const float r32 = cp * sr;
    const float r33 = cp * cr;

    output.x = r11 * input->x + r21 * input->y + r31 * input->z;  // 역회전 X를 계산한다.
    output.y = r12 * input->x + r22 * input->y + r32 * input->z;  // 역회전 Y를 계산한다.
    output.z = r13 * input->x + r23 * input->y + r33 * input->z;  // 역회전 Z를 계산한다.
    return output;
}

/* 두 보행 명령이 같은지 확인한다. */
static bool WorkspaceLimiter_TwistEqual(const RobotBodyTwist_t *first,
                                        const RobotBodyTwist_t *second)
{
    return (first->vx == second->vx) &&
           (first->vy == second->vy) &&
           (first->vz == second->vz) &&
           (first->wz == second->wz);  // 네 축 명령을 함께 비교한다.
}

/* 새 보행 명령의 분산 검사를 시작한다. */
static void WorkspaceLimiter_StartPreview(WorkspaceLimiter_Handle_t *handle,
                                          const RobotBodyTwist_t *candidate,
                                          const RobotGaitPhase_t *gait)
{
    handle->gait_pending = *candidate;                         // 25 ms 시점의 후보를 고정한다.
    handle->preview_sample = 0U;                               // 첫 경로 지점부터 시작한다.
    handle->preview_swing_mask = gait->next_phase_swing_mask;  // 다음 Tripod 역할을 고정한다.
    handle->preview_startup_phase = gait->next_phase_startup;  // 첫 위상 여부를 저장한다.
    handle->preview_active = true;                             // 분산 검사를 활성화한다.
    handle->phase_result_valid = false;                        // 이전 위상 검사 결과를 제거한다.
    handle->phase_result_accepted = false;                     // 새 후보를 미통과 상태로 둔다.
}

/* 한 검사 지점의 실제 Tripod 발 목표 여섯 개를 함께 검사한다. */
static bool WorkspaceLimiter_PreviewPoint(const WorkspaceLimiter_Handle_t *handle,
                                          const RobotEuler_t *posture)
{
    RobotVec3_t base[ROBOT_LEG_COUNT];  // 기본 발 위치를 저장한다.
    RobotVec3_t feet[ROBOT_LEG_COUNT];  // 검사할 여섯 발 위치를 저장한다.
    const float progress =
        (float)handle->preview_sample /
        (float)(ROBOT_GAIT_PREVIEW_SAMPLE_COUNT - 1U);  // 시작·중앙·끝을 포함한 다섯 지점을 만든다.
    uint32_t leg;  // 계산할 다리 번호를 저장한다.

    LegKinematics_GetBaseFeet(base);  // 기본 발 위치를 계산한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        RobotVec3_t displacement;  // 후보 명령의 한 위상 이동량을 저장한다.
        RobotVec3_t front;         // Swing 착지점을 저장한다.
        RobotVec3_t rear;          // Swing 시작점을 저장한다.
        const bool swing = (handle->preview_swing_mask &
                            (uint8_t)(1U << leg)) != 0U;  // 현재 다리 역할을 선택한다.

        displacement.x = ROBOT_GAIT_PHASE_TIME_S *
                         (-handle->gait_pending.vx +
                          handle->gait_pending.wz * base[leg].y);  // Stance X 이동량을 계산한다.
        displacement.y = ROBOT_GAIT_PHASE_TIME_S *
                         (-handle->gait_pending.vy -
                          handle->gait_pending.wz * base[leg].x);  // Stance Y 이동량을 계산한다.
        displacement.z = ROBOT_GAIT_PHASE_TIME_S *
                         (-handle->gait_pending.vz);               // Stance Z 이동량을 계산한다.
        front.x = base[leg].x - 0.5f * displacement.x;             // 앞쪽 X 끝점을 계산한다.
        front.y = base[leg].y - 0.5f * displacement.y;             // 앞쪽 Y 끝점을 계산한다.
        front.z = base[leg].z - 0.5f * displacement.z;             // 앞쪽 Z 끝점을 계산한다.
        rear.x = base[leg].x + 0.5f * displacement.x;              // 뒤쪽 X 끝점을 계산한다.
        rear.y = base[leg].y + 0.5f * displacement.y;              // 뒤쪽 Y 끝점을 계산한다.
        rear.z = base[leg].z + 0.5f * displacement.z;              // 뒤쪽 Z 끝점을 계산한다.

        if (swing)
        {
            const RobotVec3_t *start = handle->preview_startup_phase
                                     ? &base[leg]
                                     : &rear;  // 첫 위상 Swing 시작점을 선택한다.
            feet[leg] = SwingTrajectory_Calculate(progress,
                                                   start,
                                                   &front,
                                                   ROBOT_SWING_HEIGHT_M,
                                                   ROBOT_SWING_RADIAL_OFFSET_M,
                                                   workspace_leg_angle_rad[leg]);  // Swing 목표를 계산한다.
        }
        else if (handle->preview_startup_phase)
        {
            feet[leg] = StanceTrajectory_Interpolate(progress,
                                                      &base[leg],
                                                      &rear);  // 첫 위상 Stance 목표를 계산한다.
        }
        else
        {
            feet[leg] = StanceTrajectory_Interpolate(progress,
                                                      &front,
                                                      &rear);  // 정상 Stance 목표를 계산한다.
        }
    }

    return WorkspaceLimiter_AllFeetValid(feet, posture);  // 여섯 IK를 한 번씩만 검사한다.
}

/* 마지막 적용 보행 명령을 0으로 초기화한다. */
void WorkspaceLimiter_Init(WorkspaceLimiter_Handle_t *handle)
{
    if (handle != NULL)
    {
        memset(handle, 0, sizeof(*handle));  // 이전 적용 명령을 제거한다.
    }
}

/* 자세 역회전 후 여섯 발의 IK 가능 여부를 검사한다. */
bool WorkspaceLimiter_AllFeetValid(const RobotVec3_t feet_body[ROBOT_LEG_COUNT],
                                   const RobotEuler_t *posture_rad)
{
    uint32_t leg;  // 검사할 다리 번호를 저장한다.

    if ((feet_body == NULL) || (posture_rad == NULL))
    {
        return false;
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        const RobotVec3_t rotated = WorkspaceLimiter_RotateInverse(&feet_body[leg], posture_rad);  // 자세 후보를 적용한다.

        if (!LegKinematics_IsReachable((uint8_t)leg, &rotated))
        {
            return false;  // 한 다리라도 IK 불가능하면 전체 후보를 거부한다.
        }
    }

    return true;
}

/* 25 ms 전에 고정한 다음 위상 후보를 다섯 주기에 분산 검사한다. */
RobotBodyTwist_t WorkspaceLimiter_Gait(WorkspaceLimiter_Handle_t *handle,
                                       const RobotBodyTwist_t *candidate,
                                       bool manual_enable,
                                       const RobotGaitPhase_t *gait,
                                       const RobotEuler_t *posture_rad,
                                       bool reset_command,
                                       bool *accepted)
{
    RobotBodyTwist_t zero = {0.0f, 0.0f, 0.0f, 0.0f};  // 오류 시 0 명령을 준비한다.

    if ((handle == NULL) || (candidate == NULL) || (gait == NULL) ||
        (posture_rad == NULL) || (accepted == NULL))
    {
        return zero;
    }

    *accepted = WorkspaceLimiter_TwistEqual(candidate,
                                             &handle->gait_applied);  // 현재 입력의 적용 여부를 표시한다.

    if (reset_command)
    {
        memset(handle, 0, sizeof(*handle));  // Reset에서 보행 검사 상태를 제거한다.
    }
    else if (manual_enable)
    {
        if (gait->next_phase_preview)
        {
            WorkspaceLimiter_StartPreview(handle, candidate, gait);  // 25 ms 시점의 후보를 검사에 고정한다.
            *accepted = false;                                       // 다섯 지점 통과 전까지 적용을 보류한다.
        }

        if (handle->preview_active)
        {
            if (!WorkspaceLimiter_PreviewPoint(handle, posture_rad))
            {
                handle->preview_active = false;        // 위험 지점에서 분산 검사를 종료한다.
                handle->phase_result_valid = true;     // 검사 실패 결과를 확정한다.
                handle->phase_result_accepted = false; // 다음 위상 진입을 차단한다.
                *accepted = false;                     // 직전 유효 명령을 유지한다.
            }
            else
            {
                handle->preview_sample++;  // 다음 5 ms 검사 지점으로 이동한다.
                *accepted = false;         // 전체 경로 통과 전까지 적용을 보류한다.

                if (handle->preview_sample >= ROBOT_GAIT_PREVIEW_SAMPLE_COUNT)
                {
                    handle->gait_applied = handle->gait_pending;  // 검증한 25 ms 시점 후보를 예약한다.
                    handle->preview_active = false;               // 다섯 지점 검사를 완료한다.
                    handle->phase_result_valid = true;            // 검사 완료 결과를 확정한다.
                    handle->phase_result_accepted = true;         // 다음 위상 진입을 허용한다.
                    *accepted = WorkspaceLimiter_TwistEqual(candidate,
                                                            &handle->gait_applied);  // 현재 후보 채택 여부를 표시한다.
                }
            }
        }
    }
    else
    {
        handle->gait_applied = *candidate;      // 보정 모드는 발 Offset 단계에서 검사한다.
        handle->preview_active = false;         // 수동 보행 검사를 중단한다.
        handle->phase_result_valid = false;     // 수동 위상 검사 결과를 제거한다.
        handle->phase_result_accepted = false;  // 수동 위상 허가를 제거한다.
        *accepted = true;                       // 보정 명령은 발 Offset 단계에서 검사한다.
    }

    return handle->gait_applied;
}
