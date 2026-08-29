#include "high_control/drone_controller.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

/* 실수 값을 지정한 범위로 제한한다. */
static float DroneController_Clamp(float value, float minimum, float maximum)
{
    return fminf(fmaxf(value, minimum), maximum);  // 최소·최대 범위를 적용한다.
}

/* 각도를 -pi~pi 범위로 정리한다. */
static float DroneController_WrapPi(float angle)
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

/* -1000~1000 짐벌에 Dead Zone을 적용해 -1~1로 변환한다. */
static float DroneController_Normalize(int16_t input, int16_t deadband)
{
    const float limited = DroneController_Clamp((float)input, -1000.0f, 1000.0f);  // raw 입력 범위를 제한한다.

    if (fabsf(limited) <= (float)deadband)
    {
        return 0.0f;
    }

    return copysignf((fabsf(limited) - (float)deadband) /
                     (1000.0f - (float)deadband), limited);  // Dead Zone 밖을 다시 정규화한다.
}

/* Drone Controller의 상태와 Heading 기준을 초기화한다. */
void DroneController_Init(DroneController_Handle_t *handle)
{
    if (handle == NULL)
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));                       // 이전 제어 상태를 제거한다.
    handle->previous_mode = ROBOT_MODE_LANDING;              // 시작 모드를 착지로 둔다.
    handle->landing_state = DRONE_LANDING_LANDED;            // 착지 세부 상태를 완료로 둔다.
}

/* Priority 출력을 서기·착지·보행·보정 명령으로 변환한다. */
RobotDroneOutput_t DroneController_Step(DroneController_Handle_t *handle,
                                        const RobotPriorityOutput_t *priority,
                                        const bool contact[ROBOT_LEG_COUNT],
                                        float yaw_measured_rad)
{
    RobotDroneOutput_t output;    // 이번 제어 출력을 저장한다.
    bool all_contact = true;      // 전체 접촉 상태를 저장한다.
    bool contact_135 = true;      // 1·3·5 접촉 상태를 저장한다.
    bool contact_246 = true;      // 2·4·6 접촉 상태를 저장한다.
    bool lateral_mode;            // S1 횡이동 모드를 저장한다.
    bool mode_changed;            // 제어 모드 변경 여부를 저장한다.
    bool s1_changed;              // S1 이동 방식 변경 여부를 저장한다.
    uint32_t leg;                 // 접촉을 확인할 다리 번호를 저장한다.

    memset(&output, 0, sizeof(output));  // 기본 출력을 0으로 준비한다.

    if ((handle == NULL) || (priority == NULL) || (contact == NULL))
    {
        output.kill_enable = true;  // 필수 입력 누락 시 출력을 차단한다.
        return output;
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        all_contact = all_contact && contact[leg];  // 전체 접촉을 누적한다.

        if ((leg % 2U) == 0U)
        {
            contact_135 = contact_135 && contact[leg];  // 1·3·5 접촉을 누적한다.
        }
        else
        {
            contact_246 = contact_246 && contact[leg];  // 2·4·6 접촉을 누적한다.
        }
    }

    lateral_mode = (priority->s1 != 0U);                                      // S1 이동 방식을 해석한다.
    mode_changed = (priority->active_mode != handle->previous_mode);           // 모드 전환을 검출한다.
    s1_changed = (priority->active_mode == ROBOT_MODE_MANUAL) &&
                 (lateral_mode != handle->previous_s1);                        // 수동 S1 전환을 검출한다.
    output.reset_command = priority->reset_command;                             // Reset 요청을 전달한다.

    if (mode_changed)
    {
        handle->throttle_filter = 0.0f;  // 새 모드 계산 전에 Throttle을 초기화한다.
        handle->yaw_filter = 0.0f;       // 새 모드 계산 전에 Yaw를 초기화한다.
        handle->roll_filter = 0.0f;      // 새 모드 계산 전에 Roll을 초기화한다.
        handle->pitch_filter = 0.0f;     // 새 모드 계산 전에 Pitch를 초기화한다.
    }
    else if (s1_changed)
    {
        handle->yaw_filter = 0.0f;       // Yaw 회전과 횡이동 사이의 잔류값을 제거한다.
    }

    if ((priority->active_mode == ROBOT_MODE_MANUAL) ||
        (priority->active_mode == ROBOT_MODE_CORRECTION))
    {
        const float alpha = expf(-2.0f * ROBOT_PI_F *
                                 ROBOT_STICK_FILTER_HZ * ROBOT_CONTROL_PERIOD_S);  // 5 Hz LPF 계수를 계산한다.
        const float throttle = DroneController_Normalize(priority->throttle,
                                                          ROBOT_THROTTLE_DEADBAND); // Throttle을 정규화한다.
        const float yaw = DroneController_Normalize(priority->yaw,
                                                     ROBOT_STICK_DEADBAND);          // Yaw를 정규화한다.
        const float roll = DroneController_Normalize(priority->roll,
                                                      ROBOT_STICK_DEADBAND);         // Roll을 정규화한다.
        const float pitch = DroneController_Normalize(priority->pitch,
                                                       ROBOT_STICK_DEADBAND);        // Pitch를 정규화한다.

        handle->throttle_filter = alpha * handle->throttle_filter + (1.0f - alpha) * throttle;  // Throttle LPF를 갱신한다.
        handle->yaw_filter = alpha * handle->yaw_filter + (1.0f - alpha) * yaw;                  // Yaw LPF를 갱신한다.
        handle->roll_filter = alpha * handle->roll_filter + (1.0f - alpha) * roll;               // Roll LPF를 갱신한다.
        handle->pitch_filter = alpha * handle->pitch_filter + (1.0f - alpha) * pitch;            // Pitch LPF를 갱신한다.
    }
    else
    {
        handle->throttle_filter = 0.0f;  // 비동작 모드 Throttle을 제거한다.
        handle->yaw_filter = 0.0f;       // 비동작 모드 Yaw를 제거한다.
        handle->roll_filter = 0.0f;      // 비동작 모드 Roll을 제거한다.
        handle->pitch_filter = 0.0f;     // 비동작 모드 Pitch를 제거한다.
    }

    if (priority->active_mode == ROBOT_MODE_STAND)
    {
        output.stand_enable = true;  // 서기 궤적을 활성화한다.

        if (handle->previous_mode != ROBOT_MODE_STAND)
        {
            handle->stand_settle_time_s = 0.0f;  // 새 서기 안정 시간을 초기화한다.
            handle->stand_complete = false;      // 새 서기 완료 상태를 초기화한다.
        }

        if (handle->posture_memory < 1.0f)
        {
            handle->posture_memory = fminf(handle->posture_memory +
                                            ROBOT_CONTROL_PERIOD_S / ROBOT_STAND_TIME_S,
                                            1.0f);  // 5.6초 자세 진행률을 누적한다.
            handle->stand_settle_time_s = 0.0f;      // 상승 중 안정 시간을 초기화한다.
        }
        else
        {
            handle->stand_settle_time_s = fminf(handle->stand_settle_time_s +
                                                 ROBOT_CONTROL_PERIOD_S,
                                                 ROBOT_SETTLING_TIME_S);  // 자세 안정 시간을 누적한다.
            handle->stand_complete =
                (handle->stand_settle_time_s >= ROBOT_SETTLING_TIME_S);   // 안정 완료를 판단한다.
        }

        output.stand_done = handle->stand_complete;  // Priority에 서기 완료를 전달한다.
    }
    else if (priority->active_mode == ROBOT_MODE_LANDING)
    {
        output.landing_enable = true;  // 착지 궤적을 활성화한다.
        handle->stand_complete = false;// 다음 서기를 위해 완료를 해제한다.

        if (handle->previous_mode != ROBOT_MODE_LANDING)
        {
            handle->landing_state_time_s = 0.0f;  // 새 착지 상태 시간을 초기화한다.

            if (handle->posture_memory <= 0.0f)
            {
                handle->landing_state = DRONE_LANDING_LANDED;  // 이미 낮으면 완료로 둔다.
            }
            else if (handle->gait_was_active)
            {
                handle->landing_state = DRONE_LANDING_ALL_FEET;  // 보행 후 전체 발을 내린다.
            }
            else
            {
                handle->landing_state = DRONE_LANDING_LOWERING;  // 정지 상태에서 바로 하강한다.
            }
        }

        switch (handle->landing_state)
        {
            case DRONE_LANDING_LANDED:
                handle->posture_memory = 0.0f;       // 완전 착지 자세를 유지한다.
                handle->landing_state_time_s = 0.0f; // 착지 상태 시간을 초기화한다.
                handle->gait_was_active = false;     // 보행 이력을 제거한다.
                output.landing_done = true;          // Priority에 착지 완료를 전달한다.
                break;

            case DRONE_LANDING_ALL_FEET:
                output.tripod_enable = true;                         // Tripod 착지를 활성화한다.
                output.tripod_mode = ROBOT_TRIPOD_LAND_ALL;          // 전체 발 착지를 선택한다.
                if (all_contact)
                {
                    handle->landing_state = DRONE_LANDING_RECOVERY_135;  // 1·3·5 복구로 이동한다.
                    handle->landing_state_time_s = 0.0f;                  // 복구 시간을 초기화한다.
                }
                break;

            case DRONE_LANDING_RECOVERY_135:
                output.tripod_enable = true;                            // Tripod 복구를 활성화한다.
                output.tripod_mode = ROBOT_TRIPOD_RECOVERY_135;         // 1·3·5 복구를 선택한다.
                handle->landing_state_time_s = fminf(handle->landing_state_time_s +
                                                      ROBOT_CONTROL_PERIOD_S,
                                                      ROBOT_RECOVERY_TIME_S);  // 복구 진행 시간을 누적한다.
                output.recovery_progress = handle->landing_state_time_s / ROBOT_RECOVERY_TIME_S;  // 복구 진행률을 계산한다.
                if ((handle->landing_state_time_s >= ROBOT_RECOVERY_TIME_S) && contact_135)
                {
                    handle->landing_state = DRONE_LANDING_RECOVERY_246;  // 2·4·6 복구로 이동한다.
                    handle->landing_state_time_s = 0.0f;                  // 복구 시간을 초기화한다.
                }
                break;

            case DRONE_LANDING_RECOVERY_246:
                output.tripod_enable = true;                            // Tripod 복구를 활성화한다.
                output.tripod_mode = ROBOT_TRIPOD_RECOVERY_246;         // 2·4·6 복구를 선택한다.
                handle->landing_state_time_s = fminf(handle->landing_state_time_s +
                                                      ROBOT_CONTROL_PERIOD_S,
                                                      ROBOT_RECOVERY_TIME_S);  // 복구 진행 시간을 누적한다.
                output.recovery_progress = handle->landing_state_time_s / ROBOT_RECOVERY_TIME_S;  // 복구 진행률을 계산한다.
                if ((handle->landing_state_time_s >= ROBOT_RECOVERY_TIME_S) && contact_246)
                {
                    handle->landing_state = DRONE_LANDING_LOWERING;  // 몸체 하강으로 이동한다.
                    handle->landing_state_time_s = 0.0f;             // 하강 시간을 초기화한다.
                }
                break;

            case DRONE_LANDING_LOWERING:
                handle->posture_memory = fmaxf(handle->posture_memory -
                                                ROBOT_CONTROL_PERIOD_S / ROBOT_LANDING_TIME_S,
                                                0.0f);  // 5.6초 동안 자세를 내린다.
                if (handle->posture_memory <= 0.0f)
                {
                    handle->landing_state = DRONE_LANDING_SETTLING;  // 안정 상태로 이동한다.
                    handle->landing_state_time_s = 0.0f;             // 안정 시간을 초기화한다.
                }
                break;

            case DRONE_LANDING_SETTLING:
                handle->landing_state_time_s = fminf(handle->landing_state_time_s +
                                                      ROBOT_CONTROL_PERIOD_S,
                                                      ROBOT_SETTLING_TIME_S);  // 착지 안정 시간을 누적한다.
                if (handle->landing_state_time_s >= ROBOT_SETTLING_TIME_S)
                {
                    handle->landing_state = DRONE_LANDING_LANDED;  // 완전 착지로 이동한다.
                    handle->landing_state_time_s = 0.0f;           // 상태 시간을 초기화한다.
                    output.landing_done = true;                    // Priority에 완료를 전달한다.
                }
                break;

            default:
                handle->landing_state = DRONE_LANDING_LANDED;  // 잘못된 착지 상태를 복구한다.
                handle->landing_state_time_s = 0.0f;
                break;
        }
    }
    else if ((priority->active_mode == ROBOT_MODE_READY) ||
             (priority->active_mode == ROBOT_MODE_MANUAL) ||
             (priority->active_mode == ROBOT_MODE_CORRECTION))
    {
        handle->posture_memory = 1.0f;  // 서 있는 자세를 유지한다.
        output.stand_done = true;       // READY 상태의 서기 완료를 유지한다.
    }
    else if ((priority->active_mode == ROBOT_MODE_FAULT) ||
             (priority->active_mode == ROBOT_MODE_KILL))
    {
        output.kill_enable = true;      // Fault와 Kill에서 릴레이 차단을 요청한다.
        output.reset_command = false;   // Fault 상태 Reset 명령을 제거한다.
    }

    if (priority->active_mode == ROBOT_MODE_MANUAL)
    {
        bool motion_request;  // 보행 시작 조건을 저장한다.

        output.manual_enable = true;       // 수동 제어를 활성화한다.
        output.body_control_enable = true; // 위치·Heading PI를 활성화한다.
        output.posture_enable = true;      // 자세 PI를 활성화한다.
        output.vx_user_mps = ROBOT_MAX_LINEAR_SPEED_MPS * handle->throttle_filter;  // 전후 속도를 계산한다.

        if (lateral_mode)
        {
            output.vy_user_mps = -ROBOT_MAX_LATERAL_SPEED_MPS * handle->yaw_filter;  // Yaw 짐벌을 반대 방향 횡이동으로 사용한다.
            output.wz_user_radps = 0.0f;                                           // 사용자 회전을 차단한다.
        }
        else
        {
            output.vy_user_mps = 0.0f;                                             // 횡이동을 차단한다.
            output.wz_user_radps = -ROBOT_MAX_YAW_RATE_RADPS * handle->yaw_filter;  // Yaw 회전 방향을 반전한다.
        }

        output.posture_reference_rad.roll = ROBOT_MAX_ROLL_RAD * handle->roll_filter;    // Roll 목표를 계산한다.
        output.posture_reference_rad.pitch = ROBOT_MAX_PITCH_RAD * handle->pitch_filter; // Pitch 목표를 계산한다.
        motion_request = (fabsf(output.vx_user_mps) >= ROBOT_GAIT_LINEAR_THRESHOLD_MPS) ||
                         (fabsf(output.vy_user_mps) >= ROBOT_GAIT_LINEAR_THRESHOLD_MPS) ||
                         (fabsf(output.wz_user_radps) >= ROBOT_GAIT_YAW_THRESHOLD_RADPS);  // 세 보행 시작 조건을 검사한다.

        if (motion_request)
        {
            output.tripod_enable = true;                   // 정상 보행을 활성화한다.
            output.tripod_mode = ROBOT_TRIPOD_NORMAL;      // 정상 Tripod 모드를 선택한다.
            handle->gait_was_active = true;                // 착지용 보행 이력을 저장한다.
        }
    }

    if (priority->active_mode == ROBOT_MODE_CORRECTION)
    {
        output.correction_enable = true;      // 보정 제어를 활성화한다.
        output.body_control_enable = true;    // 몸체 제어를 활성화한다.
        output.posture_enable = true;         // 자세 제어를 활성화한다.
        output.correction_velocity_mps.x = ROBOT_MAX_CORRECTION_SPEED_MPS * handle->pitch_filter;    // Pitch로 보정 X속도를 만든다.
        output.correction_velocity_mps.y = -ROBOT_MAX_CORRECTION_SPEED_MPS * handle->roll_filter;    // Roll 보정 방향을 반전한다.
        output.correction_velocity_mps.z = ROBOT_MAX_CORRECTION_SPEED_MPS * handle->throttle_filter; // Throttle로 보정 Z속도를 만든다.
        output.posture_reference_rad.yaw = -ROBOT_MAX_CORRECTION_YAW_RAD * handle->yaw_filter;        // Yaw 보정 방향을 반전한다.
    }

    if (output.reset_command)
    {
        handle->yaw_reference_memory = DroneController_WrapPi(yaw_measured_rad);  // Reset 시 현재 Heading을 저장한다.
        handle->gait_was_active = false;                                          // 이전 보행 이력을 제거한다.
    }
    else if (priority->active_mode == ROBOT_MODE_MANUAL)
    {
        if ((handle->previous_mode != ROBOT_MODE_MANUAL) || s1_changed)
        {
            handle->yaw_reference_memory = DroneController_WrapPi(yaw_measured_rad);  // 모드 전환 Heading을 저장한다.
        }
        else
        {
            handle->yaw_reference_memory = DroneController_WrapPi(
                handle->yaw_reference_memory + output.wz_user_radps * ROBOT_CONTROL_PERIOD_S);  // 사용자 회전을 Heading에 적분한다.
        }

        output.posture_reference_rad.yaw = handle->yaw_reference_memory;  // 수동 Heading 기준을 전달한다.
    }

    output.posture_progress = handle->posture_memory;  // 현재 자세 진행률을 반환한다.
    handle->previous_mode = priority->active_mode;     // 다음 주기 모드 전환을 위해 저장한다.
    handle->previous_s1 = lateral_mode;                // 다음 주기 S1 전환을 위해 저장한다.
    return output;
}
