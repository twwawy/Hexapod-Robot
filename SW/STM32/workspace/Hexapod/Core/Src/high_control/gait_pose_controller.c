#include "high_control/gait_pose_controller.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define GAIT_POSITION_KP           1.0f
#define GAIT_POSITION_KI           0.0f
#define GAIT_YAW_KP                2.0f
#define GAIT_YAW_KI                0.0f
#define GAIT_POSITION_FEEDBACK_MAX 0.05f
#define GAIT_YAW_FEEDBACK_MAX      (15.0f * ROBOT_DEG_TO_RAD_F)
#define GAIT_POSITION_INTEGRAL_MAX 0.20f
#define GAIT_YAW_INTEGRAL_MAX      0.50f
#define GAIT_LINEAR_STEP_MAX       (0.5f * ROBOT_CONTROL_PERIOD_S)
#define GAIT_YAW_STEP_MAX          (90.0f * ROBOT_DEG_TO_RAD_F * ROBOT_CONTROL_PERIOD_S)

/* 실수 값을 지정한 범위로 제한한다. */
static float GaitPoseController_Clamp(float value, float minimum, float maximum)
{
    return fminf(fmaxf(value, minimum), maximum);  // 최소·최대 범위를 적용한다.
}

/* 각도를 -pi~pi 범위로 정리한다. */
static float GaitPoseController_WrapPi(float angle)
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

/* 한 명령의 주기당 변화량을 제한한다. */
static float GaitPoseController_RateLimit(float target, float previous, float maximum_step)
{
    const float delta = GaitPoseController_Clamp(target - previous,
                                                  -maximum_step,
                                                  maximum_step);  // 변화량을 제한한다.
    return previous + delta;
}

/* 위치·Heading PI 상태를 초기화한다. */
void GaitPoseController_Init(GaitPoseController_Handle_t *handle)
{
    if (handle != NULL)
    {
        memset(handle, 0, sizeof(*handle));  // 모든 누적 상태를 제거한다.
    }
}

/* 사용자 명령에 Position과 Heading PI를 더해 Body Twist를 계산한다. */
GaitPoseController_Output_t GaitPoseController_Step(
    GaitPoseController_Handle_t *handle,
    bool reset_command,
    const RobotDroneOutput_t *drone,
    const RobotVec3_t *body_position_world,
    uint8_t valid_leg_count,
    float yaw_measured_rad)
{
    GaitPoseController_Output_t output;  // 이번 제어 출력을 저장한다.
    const bool manual = (drone != NULL) && drone->manual_enable;          // 수동 모드 상태를 저장한다.
    const bool correction = (drone != NULL) && drone->correction_enable;  // 보정 모드 상태를 저장한다.

    memset(&output, 0, sizeof(output));  // 기본 출력을 0으로 준비한다.

    if ((handle == NULL) || (drone == NULL) || (body_position_world == NULL))
    {
        return output;
    }

    if (reset_command || (manual && !handle->previous_manual))
    {
        handle->x_reference_m = body_position_world->x;  // 현재 X를 새 기준으로 잡는다.
        handle->y_reference_m = body_position_world->y;  // 현재 Y를 새 기준으로 잡는다.
        handle->x_integral = 0.0f;                       // X 적분을 초기화한다.
        handle->y_integral = 0.0f;                       // Y 적분을 초기화한다.
        handle->yaw_integral = 0.0f;                     // Yaw 적분을 초기화한다.
    }

    if (manual)
    {
        if (valid_leg_count > 0U)
        {
            float error_x;  // World X 위치 오차를 저장한다.
            float error_y;  // World Y 위치 오차를 저장한다.

            handle->x_reference_m +=
                (cosf(drone->posture_reference_rad.yaw) * drone->vx_user_mps -
                 sinf(drone->posture_reference_rad.yaw) * drone->vy_user_mps) *
                ROBOT_CONTROL_PERIOD_S;  // 사용자 속도로 World X 기준을 적분한다.
            handle->y_reference_m +=
                (sinf(drone->posture_reference_rad.yaw) * drone->vx_user_mps +
                 cosf(drone->posture_reference_rad.yaw) * drone->vy_user_mps) *
                ROBOT_CONTROL_PERIOD_S;  // 사용자 속도로 World Y 기준을 적분한다.

            error_x = handle->x_reference_m - body_position_world->x;  // X 위치 오차를 계산한다.
            error_y = handle->y_reference_m - body_position_world->y;  // Y 위치 오차를 계산한다.
            handle->x_integral = GaitPoseController_Clamp(
                handle->x_integral + error_x * ROBOT_CONTROL_PERIOD_S,
                -GAIT_POSITION_INTEGRAL_MAX,
                GAIT_POSITION_INTEGRAL_MAX);  // X 적분을 제한한다.
            handle->y_integral = GaitPoseController_Clamp(
                handle->y_integral + error_y * ROBOT_CONTROL_PERIOD_S,
                -GAIT_POSITION_INTEGRAL_MAX,
                GAIT_POSITION_INTEGRAL_MAX);  // Y 적분을 제한한다.
            output.feedback_world.x = GaitPoseController_Clamp(
                GAIT_POSITION_KP * error_x + GAIT_POSITION_KI * handle->x_integral,
                -GAIT_POSITION_FEEDBACK_MAX,
                GAIT_POSITION_FEEDBACK_MAX);  // X Feedback을 계산한다.
            output.feedback_world.y = GaitPoseController_Clamp(
                GAIT_POSITION_KP * error_y + GAIT_POSITION_KI * handle->y_integral,
                -GAIT_POSITION_FEEDBACK_MAX,
                GAIT_POSITION_FEEDBACK_MAX);  // Y Feedback을 계산한다.
        }
        else
        {
            handle->x_reference_m = body_position_world->x;  // 추정 불가 시 현재 X를 유지한다.
            handle->y_reference_m = body_position_world->y;  // 추정 불가 시 현재 Y를 유지한다.
            handle->x_integral = 0.0f;                       // X 적분을 제거한다.
            handle->y_integral = 0.0f;                       // Y 적분을 제거한다.
        }

        {
            const float cosine = cosf(yaw_measured_rad);  // Body 변환용 Cosine을 계산한다.
            const float sine = sinf(yaw_measured_rad);    // Body 변환용 Sine을 계산한다.
            const float feedback_x_body = cosine * output.feedback_world.x +
                                          sine * output.feedback_world.y;  // X Feedback을 Body로 변환한다.
            const float feedback_y_body = -sine * output.feedback_world.x +
                                           cosine * output.feedback_world.y;  // Y Feedback을 Body로 변환한다.
            const float yaw_error = GaitPoseController_WrapPi(
                drone->posture_reference_rad.yaw - yaw_measured_rad);  // Heading 오차를 계산한다.

            handle->yaw_integral = GaitPoseController_Clamp(
                handle->yaw_integral + yaw_error * ROBOT_CONTROL_PERIOD_S,
                -GAIT_YAW_INTEGRAL_MAX,
                GAIT_YAW_INTEGRAL_MAX);  // Heading 적분을 제한한다.
            output.yaw_feedback_radps = GaitPoseController_Clamp(
                GAIT_YAW_KP * yaw_error + GAIT_YAW_KI * handle->yaw_integral,
                -GAIT_YAW_FEEDBACK_MAX,
                GAIT_YAW_FEEDBACK_MAX);  // Heading Feedback을 계산한다.
            output.twist.vx = GaitPoseController_Clamp(
                drone->vx_user_mps + feedback_x_body,
                -ROBOT_MAX_LINEAR_SPEED_MPS,
                ROBOT_MAX_LINEAR_SPEED_MPS);  // X속도 후보를 계산한다.
            output.twist.vy = GaitPoseController_Clamp(
                drone->vy_user_mps + feedback_y_body,
                -ROBOT_MAX_LINEAR_SPEED_MPS,
                ROBOT_MAX_LINEAR_SPEED_MPS);  // Y속도 후보를 계산한다.
            output.twist.wz = GaitPoseController_Clamp(
                drone->wz_user_radps + output.yaw_feedback_radps,
                -ROBOT_MAX_YAW_RATE_RADPS,
                ROBOT_MAX_YAW_RATE_RADPS);  // Yaw속도 후보를 계산한다.
        }
    }
    else if (correction)
    {
        output.twist.vx = GaitPoseController_Clamp(drone->correction_velocity_mps.x,
                                                   -ROBOT_MAX_LINEAR_SPEED_MPS,
                                                   ROBOT_MAX_LINEAR_SPEED_MPS);  // 보정 X속도를 전달한다.
        output.twist.vy = GaitPoseController_Clamp(drone->correction_velocity_mps.y,
                                                   -ROBOT_MAX_LINEAR_SPEED_MPS,
                                                   ROBOT_MAX_LINEAR_SPEED_MPS);  // 보정 Y속도를 전달한다.
        output.twist.vz = GaitPoseController_Clamp(drone->correction_velocity_mps.z,
                                                   -ROBOT_MAX_CORRECTION_SPEED_MPS,
                                                   ROBOT_MAX_CORRECTION_SPEED_MPS);  // 보정 Z속도를 전달한다.
        handle->x_integral = 0.0f;    // 보정 모드에서 위치 적분을 제거한다.
        handle->y_integral = 0.0f;    // 보정 모드에서 위치 적분을 제거한다.
        handle->yaw_integral = 0.0f;  // 보정 모드에서 Heading 적분을 제거한다.
    }
    else
    {
        handle->x_reference_m = body_position_world->x;  // 비활성 시 현재 X를 저장한다.
        handle->y_reference_m = body_position_world->y;  // 비활성 시 현재 Y를 저장한다.
        handle->x_integral = 0.0f;                       // X 적분을 제거한다.
        handle->y_integral = 0.0f;                       // Y 적분을 제거한다.
        handle->yaw_integral = 0.0f;                     // Yaw 적분을 제거한다.
    }

    if (manual || correction)
    {
        output.twist.vx = GaitPoseController_RateLimit(output.twist.vx,
                                                        handle->previous.vx,
                                                        GAIT_LINEAR_STEP_MAX);  // X속도 변화를 제한한다.
        output.twist.vy = GaitPoseController_RateLimit(output.twist.vy,
                                                        handle->previous.vy,
                                                        GAIT_LINEAR_STEP_MAX);  // Y속도 변화를 제한한다.
        output.twist.vz = GaitPoseController_RateLimit(output.twist.vz,
                                                        handle->previous.vz,
                                                        GAIT_LINEAR_STEP_MAX);  // Z속도 변화를 제한한다.
        output.twist.wz = GaitPoseController_RateLimit(output.twist.wz,
                                                        handle->previous.wz,
                                                        GAIT_YAW_STEP_MAX);  // Yaw속도 변화를 제한한다.
        handle->previous = output.twist;  // 다음 주기 Rate Limit 상태를 저장한다.
    }
    else
    {
        memset(&handle->previous, 0, sizeof(handle->previous));  // 비활성 명령을 즉시 0으로 만든다.
    }

    output.x_reference_m = handle->x_reference_m;  // 현재 X 기준을 반환한다.
    output.y_reference_m = handle->y_reference_m;  // 현재 Y 기준을 반환한다.
    handle->previous_manual = manual;               // 수동 모드 전환 상태를 저장한다.
    return output;
}
