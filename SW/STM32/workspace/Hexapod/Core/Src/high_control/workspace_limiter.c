#include "high_control/workspace_limiter.h"

#include "high_control/leg_kinematics.h"
#include "high_control/stance_trajectory.h"
#include "high_control/swing_trajectory.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

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

/* 한 위상의 Stance와 Swing 전체 샘플을 미리 검사한다. */
static bool WorkspaceLimiter_PreviewGait(const RobotBodyTwist_t *candidate,
                                         const RobotEuler_t *posture)
{
    static const float leg_angle[ROBOT_LEG_COUNT] =
    {
        -45.0f * ROBOT_DEG_TO_RAD_F, -90.0f * ROBOT_DEG_TO_RAD_F,
        -135.0f * ROBOT_DEG_TO_RAD_F, 45.0f * ROBOT_DEG_TO_RAD_F,
        90.0f * ROBOT_DEG_TO_RAD_F, 135.0f * ROBOT_DEG_TO_RAD_F
    };  // 여섯 다리 장착각을 저장한다.
    RobotVec3_t base[ROBOT_LEG_COUNT];  // 기본 발 위치를 저장한다.
    uint32_t leg;                       // 검사할 다리 번호를 저장한다.

    LegKinematics_GetBaseFeet(base);  // 기본 발 위치를 계산한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        RobotVec3_t displacement;  // 한 위상 Stance 이동량을 저장한다.
        RobotVec3_t front;         // Swing 착지점을 저장한다.
        RobotVec3_t rear;          // Swing 시작점을 저장한다.
        uint32_t sample;           // 궤적 샘플 번호를 저장한다.

        displacement.x = ROBOT_GAIT_PHASE_TIME_S *
                         (-candidate->vx + candidate->wz * base[leg].y);  // Stance X 이동량을 계산한다.
        displacement.y = ROBOT_GAIT_PHASE_TIME_S *
                         (-candidate->vy - candidate->wz * base[leg].x);  // Stance Y 이동량을 계산한다.
        displacement.z = ROBOT_GAIT_PHASE_TIME_S * (-candidate->vz);      // Stance Z 이동량을 계산한다.
        front.x = base[leg].x - 0.5f * displacement.x;                    // 앞쪽 X 끝점을 계산한다.
        front.y = base[leg].y - 0.5f * displacement.y;                    // 앞쪽 Y 끝점을 계산한다.
        front.z = base[leg].z - 0.5f * displacement.z;                    // 앞쪽 Z 끝점을 계산한다.
        rear.x = base[leg].x + 0.5f * displacement.x;                     // 뒤쪽 X 끝점을 계산한다.
        rear.y = base[leg].y + 0.5f * displacement.y;                     // 뒤쪽 Y 끝점을 계산한다.
        rear.z = base[leg].z + 0.5f * displacement.z;                     // 뒤쪽 Z 끝점을 계산한다.

        for (sample = 0U; sample <= 20U; ++sample)
        {
            const float progress = (float)sample / 20.0f;  // 현재 미리보기 진행률을 계산한다.
            RobotVec3_t feet[ROBOT_LEG_COUNT];             // 현재 검사할 발 배열을 저장한다.
            RobotVec3_t stance = StanceTrajectory_Interpolate(progress, &front, &rear);  // Stance 샘플을 계산한다.
            RobotVec3_t swing = SwingTrajectory_Calculate(progress, &rear, &front,
                                                           ROBOT_SWING_HEIGHT_M,
                                                           ROBOT_SWING_RADIAL_OFFSET_M,
                                                           leg_angle[leg]);  // Swing 샘플을 계산한다.

            memcpy(feet, base, sizeof(feet));  // 다른 다리는 기본 위치로 채운다.
            feet[leg] = stance;                // 현재 다리에 Stance 샘플을 넣는다.
            if (!WorkspaceLimiter_AllFeetValid(feet, posture))
            {
                return false;  // Stance 경계 이탈을 거부한다.
            }

            feet[leg] = swing;  // 현재 다리에 Swing 샘플을 넣는다.
            if (!WorkspaceLimiter_AllFeetValid(feet, posture))
            {
                return false;  // Swing 경계 이탈을 거부한다.
            }
        }
    }

    return true;
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

/* 수동 보행 후보의 전체 Stance·Swing 경로를 검사하고 직전 유효값을 유지한다. */
RobotBodyTwist_t WorkspaceLimiter_Gait(WorkspaceLimiter_Handle_t *handle,
                                       const RobotBodyTwist_t *candidate,
                                       bool manual_enable,
                                       const RobotEuler_t *posture_rad,
                                       bool reset_command,
                                       bool *accepted)
{
    RobotBodyTwist_t zero = {0.0f, 0.0f, 0.0f, 0.0f};  // 오류 시 0 명령을 준비한다.

    if ((handle == NULL) || (candidate == NULL) ||
        (posture_rad == NULL) || (accepted == NULL))
    {
        return zero;
    }

    *accepted = true;  // 기본적으로 후보 채택을 표시한다.

    if (reset_command)
    {
        handle->gait_applied = zero;  // Reset에서 보행 명령을 제거한다.
    }
    else if (manual_enable)
    {
        if (WorkspaceLimiter_PreviewGait(candidate, posture_rad))
        {
            handle->gait_applied = *candidate;  // 전체 궤적이 유효하면 새 명령을 채택한다.
        }
        else
        {
            *accepted = false;  // 유효하지 않으면 직전 명령을 유지한다.
        }
    }
    else
    {
        handle->gait_applied = *candidate;  // 보정 모드는 발 Offset 단계에서 검사한다.
    }

    return handle->gait_applied;
}
