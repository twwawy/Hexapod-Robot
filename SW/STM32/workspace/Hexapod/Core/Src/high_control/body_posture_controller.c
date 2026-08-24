#include "high_control/body_posture_controller.h"

#include "high_control/workspace_limiter.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define POSTURE_KP             2.0f
#define POSTURE_KI             0.0f
#define POSTURE_RATE_MAX       (15.0f * ROBOT_DEG_TO_RAD_F)
#define POSTURE_INTEGRAL_LIMIT 0.50f

/* 실수 값을 지정한 범위로 제한한다. */
static float BodyPosture_Clamp(float value, float minimum, float maximum)
{
    return fminf(fmaxf(value, minimum), maximum);  // 최소·최대 범위를 적용한다.
}

/* 각도를 -pi~pi 범위로 정리한다. */
static float BodyPosture_WrapPi(float angle)
{
    while (angle > ROBOT_PI_F)
    {
        angle -= 2.0f * ROBOT_PI_F;  // 양의 범위를 한 바퀴 줄인다.
    }
    while (angle < -ROBOT_PI_F)
    {
        angle += 2.0f * ROBOT_PI_F;  // 음의 범위를 한 바퀴 늘린다.
    }
    return angle;
}

/* 현재값을 목표값으로 최대 한 주기만 이동한다. */
static float BodyPosture_MoveToward(float current, float target, float maximum_step)
{
    const float difference = BodyPosture_Clamp(target - current,
                                                -maximum_step,
                                                maximum_step);  // 한 주기 변화량을 제한한다.
    return current + difference;
}

/* Body 발 위치에 자세의 역회전을 적용한다. */
static RobotVec3_t BodyPosture_RotateInverse(const RobotVec3_t *input,
                                             const RobotEuler_t *posture)
{
    RobotVec3_t output;                      // 역회전한 발 위치를 저장한다.
    const float cr = cosf(posture->roll);   // Roll Cosine을 계산한다.
    const float sr = sinf(posture->roll);   // Roll Sine을 계산한다.
    const float cp = cosf(posture->pitch);  // Pitch Cosine을 계산한다.
    const float sp = sinf(posture->pitch);  // Pitch Sine을 계산한다.
    const float cy = cosf(posture->yaw);    // Yaw Cosine을 계산한다.
    const float sy = sinf(posture->yaw);    // Yaw Sine을 계산한다.
    const float r11 = cy * cp;
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

/* 자세 PI 명령과 적분 상태를 초기화한다. */
void BodyPostureController_Init(BodyPostureController_Handle_t *handle)
{
    if (handle != NULL)
    {
        memset(handle, 0, sizeof(*handle));  // 이전 자세 제어 상태를 제거한다.
    }
}

/* Roll·Pitch·보정 Yaw PI 후보를 검사하고 발 위치에 역회전한다. */
BodyPostureController_Output_t BodyPostureController_Step(
    BodyPostureController_Handle_t *handle,
    const RobotVec3_t feet_body[ROBOT_LEG_COUNT],
    const RobotDroneOutput_t *drone,
    const RobotEuler_t *measured_rad,
    bool reset_command)
{
    BodyPostureController_Output_t output;  // 이번 자세 제어 출력을 저장한다.
    RobotEuler_t candidate;                 // 새 자세 명령 후보를 저장한다.
    RobotEuler_t integral_candidate;        // 새 적분 후보를 저장한다.
    uint32_t leg;                           // 회전할 다리 번호를 저장한다.

    memset(&output, 0, sizeof(output));  // 기본 출력을 0으로 준비한다.

    if ((handle == NULL) || (feet_body == NULL) ||
        (drone == NULL) || (measured_rad == NULL))
    {
        return output;
    }

    output.accepted = true;            // 기본적으로 후보 채택을 표시한다.
    candidate = handle->command_rad;   // 직전 정상 자세에서 후보를 시작한다.
    integral_candidate = handle->integral;  // 직전 적분에서 후보를 시작한다.

    if (reset_command || !drone->posture_enable)
    {
        memset(&handle->command_rad, 0, sizeof(handle->command_rad));  // 자세 명령을 제거한다.
        memset(&handle->integral, 0, sizeof(handle->integral));        // 자세 적분을 제거한다.
        handle->correction_yaw_base_rad = measured_rad->yaw;           // 현재 Yaw를 보정 기준으로 저장한다.
    }
    else if (drone->manual_enable)
    {
        const float roll_error = BodyPosture_WrapPi(
            drone->posture_reference_rad.roll - measured_rad->roll);  // Roll 오차를 계산한다.
        const float pitch_error = BodyPosture_WrapPi(
            drone->posture_reference_rad.pitch - measured_rad->pitch);  // Pitch 오차를 계산한다.

        if (!handle->previous_manual)
        {
            memset(&handle->integral, 0, sizeof(handle->integral));  // 수동 진입 시 적분을 초기화한다.
            integral_candidate = handle->integral;
        }

        integral_candidate.roll = BodyPosture_Clamp(
            handle->integral.roll + roll_error * ROBOT_CONTROL_PERIOD_S,
            -POSTURE_INTEGRAL_LIMIT,
            POSTURE_INTEGRAL_LIMIT);  // Roll 적분 후보를 계산한다.
        integral_candidate.pitch = BodyPosture_Clamp(
            handle->integral.pitch + pitch_error * ROBOT_CONTROL_PERIOD_S,
            -POSTURE_INTEGRAL_LIMIT,
            POSTURE_INTEGRAL_LIMIT);  // Pitch 적분 후보를 계산한다.
        candidate.roll = BodyPosture_Clamp(
            handle->command_rad.roll +
            BodyPosture_Clamp(POSTURE_KP * roll_error + POSTURE_KI * integral_candidate.roll,
                              -POSTURE_RATE_MAX, POSTURE_RATE_MAX) * ROBOT_CONTROL_PERIOD_S,
            -ROBOT_MAX_ROLL_RAD,
            ROBOT_MAX_ROLL_RAD);  // Roll 명령 후보를 계산한다.
        candidate.pitch = BodyPosture_Clamp(
            handle->command_rad.pitch +
            BodyPosture_Clamp(POSTURE_KP * pitch_error + POSTURE_KI * integral_candidate.pitch,
                              -POSTURE_RATE_MAX, POSTURE_RATE_MAX) * ROBOT_CONTROL_PERIOD_S,
            -ROBOT_MAX_PITCH_RAD,
            ROBOT_MAX_PITCH_RAD);  // Pitch 명령 후보를 계산한다.
        candidate.yaw = BodyPosture_MoveToward(handle->command_rad.yaw,
                                                0.0f,
                                                POSTURE_RATE_MAX * ROBOT_CONTROL_PERIOD_S);  // 수동 모드 Yaw 오버레이를 0으로 복귀시킨다.

        if (WorkspaceLimiter_AllFeetValid(feet_body, &candidate))
        {
            handle->command_rad = candidate;      // 정상 자세 후보를 채택한다.
            handle->integral.roll = integral_candidate.roll;    // Roll 적분 후보를 채택한다.
            handle->integral.pitch = integral_candidate.pitch;  // Pitch 적분 후보를 채택한다.
        }
        else
        {
            output.accepted = false;  // 작업공간 밖 후보에서 직전 명령을 유지한다.
        }

        handle->integral.yaw = 0.0f;  // 수동 모드 Yaw 적분을 제거한다.
    }
    else if (drone->correction_enable)
    {
        float yaw_target;  // 보정 Yaw 절대 목표를 저장한다.
        float yaw_error;   // 보정 Yaw 오차를 저장한다.

        if (!handle->previous_correction)
        {
            handle->correction_yaw_base_rad = measured_rad->yaw;  // 보정 진입 Heading을 저장한다.
            memset(&handle->integral, 0, sizeof(handle->integral));// 보정 진입 적분을 제거한다.
            integral_candidate = handle->integral;
        }

        yaw_target = BodyPosture_WrapPi(handle->correction_yaw_base_rad +
                                         drone->posture_reference_rad.yaw);  // 상대 Yaw 목표를 만든다.
        yaw_error = BodyPosture_WrapPi(yaw_target - measured_rad->yaw);      // Yaw 오차를 계산한다.
        integral_candidate.yaw = BodyPosture_Clamp(
            handle->integral.yaw + yaw_error * ROBOT_CONTROL_PERIOD_S,
            -POSTURE_INTEGRAL_LIMIT,
            POSTURE_INTEGRAL_LIMIT);  // Yaw 적분 후보를 계산한다.
        candidate.yaw = BodyPosture_Clamp(
            handle->command_rad.yaw +
            BodyPosture_Clamp(POSTURE_KP * yaw_error + POSTURE_KI * integral_candidate.yaw,
                              -POSTURE_RATE_MAX, POSTURE_RATE_MAX) * ROBOT_CONTROL_PERIOD_S,
            -ROBOT_MAX_CORRECTION_YAW_RAD,
            ROBOT_MAX_CORRECTION_YAW_RAD);  // 보정 Yaw 명령 후보를 계산한다.
        candidate.roll = BodyPosture_MoveToward(handle->command_rad.roll,
                                                 0.0f,
                                                 POSTURE_RATE_MAX * ROBOT_CONTROL_PERIOD_S);  // Roll을 0으로 복귀시킨다.
        candidate.pitch = BodyPosture_MoveToward(handle->command_rad.pitch,
                                                  0.0f,
                                                  POSTURE_RATE_MAX * ROBOT_CONTROL_PERIOD_S); // Pitch를 0으로 복귀시킨다.

        if (WorkspaceLimiter_AllFeetValid(feet_body, &candidate))
        {
            handle->command_rad = candidate;                // 정상 자세 후보를 채택한다.
            handle->integral.yaw = integral_candidate.yaw;  // Yaw 적분 후보를 채택한다.
        }
        else
        {
            output.accepted = false;  // 작업공간 밖 후보에서 직전 명령을 유지한다.
        }

        handle->integral.roll = 0.0f;   // 보정 모드 Roll 적분을 제거한다.
        handle->integral.pitch = 0.0f;  // 보정 모드 Pitch 적분을 제거한다.
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        output.targets.foot[leg] = BodyPosture_RotateInverse(&feet_body[leg],
                                                              &handle->command_rad);  // 실제 자세 명령을 발 위치에 적용한다.
    }

    output.targets.command_accepted = output.accepted;  // 자세 채택 상태를 발 출력에 기록한다.
    output.command_rad = handle->command_rad;           // 실제 자세 명령을 반환한다.
    handle->previous_manual = drone->manual_enable;     // 다음 수동 진입 검출을 위해 저장한다.
    handle->previous_correction = drone->correction_enable;  // 다음 보정 진입 검출을 위해 저장한다.
    return output;
}
