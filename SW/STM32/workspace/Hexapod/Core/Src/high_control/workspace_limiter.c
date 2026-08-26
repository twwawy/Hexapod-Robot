#include "high_control/workspace_limiter.h"

#include "high_control/leg_kinematics.h"

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

/* 수동 보행 후보를 적용하고 최종 발 목표 검사를 다음 제어 단계에 맡긴다. */
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
        handle->gait_applied = *candidate;  // 현재 주기의 보행 명령을 적용한다.
    }
    else
    {
        handle->gait_applied = *candidate;  // 보정 모드는 발 Offset 단계에서 검사한다.
    }

    return handle->gait_applied;
}
