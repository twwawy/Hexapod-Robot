#include "high_control/control_priority.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* 네 짐벌이 READY 중립 범위인지 확인한다. */
static bool ControlPriority_SticksNeutral(const RobotUserCommand_t *user)
{
    return (user != NULL) &&
           (abs(user->throttle) <= ROBOT_THROTTLE_DEADBAND) &&
           (abs(user->yaw) <= ROBOT_STICK_DEADBAND) &&
           (abs(user->roll) <= ROBOT_STICK_DEADBAND) &&
           (abs(user->pitch) <= ROBOT_STICK_DEADBAND);  // 네 축의 중립을 함께 검사한다.
}

/* Control Priority 상태를 완전 착지로 초기화한다. */
void ControlPriority_Init(ControlPriority_Handle_t *handle)
{
    if (handle == NULL)
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));               // 이전 상태를 제거한다.
    handle->supervisor = CONTROL_SUPERVISOR_LANDED;   // 시작 상태를 LANDED로 둔다.
}

/* 사용자 스위치와 Fault 우선순위로 현재 모드를 결정한다. */
RobotPriorityOutput_t ControlPriority_Step(ControlPriority_Handle_t *handle,
                                           const RobotUserCommand_t *user,
                                           bool stand_done,
                                           bool landing_done,
                                           const RobotSafetyOutput_t *safety)
{
    RobotPriorityOutput_t output;  // 현재 Priority 출력을 저장한다.
    bool stand_request;            // SB 서기 요청을 저장한다.
    bool manual_request;           // SC 수동 요청을 저장한다.
    bool correction_request;       // SC 보정 요청을 저장한다.
    bool kill_request;             // SD Kill 요청을 저장한다.

    memset(&output, 0, sizeof(output));  // 기본 출력을 LANDING과 0으로 준비한다.
    output.active_mode = ROBOT_MODE_LANDING;

    if ((handle == NULL) || (user == NULL) || (safety == NULL))
    {
        output.active_mode = ROBOT_MODE_FAULT;  // 필수 입력 누락을 Fault로 처리한다.
        return output;
    }

    stand_request = (user->sb == 1U);       // SB 중간 위치를 서기로 해석한다.
    manual_request = (user->sc == 0U);      // SC 첫 위치를 수동으로 해석한다.
    correction_request = (user->sc == 2U);  // SC 끝 위치를 보정으로 해석한다.
    kill_request = (user->sd != 0U);        // SD 활성화를 Kill로 해석한다.

    if (kill_request)
    {
        handle->supervisor = CONTROL_SUPERVISOR_KILL;  // Kill을 가장 먼저 선택한다.
        handle->stand_command_armed = false;            // 서기 재입력을 차단한다.
    }
    else if (safety->rollover_fault || safety->controller_fault)
    {
        handle->supervisor = CONTROL_SUPERVISOR_FAULT;  // Safety Fault를 선택한다.
        handle->stand_command_armed = false;             // 서기 재입력을 차단한다.
    }
    else
    {
        switch (handle->supervisor)
        {
            case CONTROL_SUPERVISOR_KILL:
            case CONTROL_SUPERVISOR_FAULT:
                handle->supervisor = CONTROL_SUPERVISOR_LANDING;  // Fault 해제 경로는 착지로 둔다.
                break;

            case CONTROL_SUPERVISOR_LANDED:
                if (!stand_request)
                {
                    handle->stand_command_armed = true;  // SB 해제 후 새 서기 요청을 허가한다.
                }
                else if (handle->stand_command_armed)
                {
                    handle->supervisor = CONTROL_SUPERVISOR_STANDING;  // 새 서기를 시작한다.
                    handle->stand_command_armed = false;                // 중복 서기를 막는다.
                }
                break;

            case CONTROL_SUPERVISOR_STANDING:
                if (!stand_request)
                {
                    handle->supervisor = CONTROL_SUPERVISOR_LANDING;  // 서기 중 해제 시 착지한다.
                }
                else if (stand_done)
                {
                    handle->supervisor = CONTROL_SUPERVISOR_READY;    // 서기 완료 후 READY로 간다.
                }
                break;

            case CONTROL_SUPERVISOR_READY:
                if (!stand_request)
                {
                    handle->supervisor = CONTROL_SUPERVISOR_LANDING;  // SB 해제 시 착지한다.
                    handle->stand_command_armed = false;               // 다음 서기를 다시 무장시킨다.
                }
                break;

            case CONTROL_SUPERVISOR_LANDING:
                if (landing_done)
                {
                    handle->supervisor = CONTROL_SUPERVISOR_LANDED;  // 착지 완료 후 LANDED로 간다.
                }
                break;

            default:
                handle->supervisor = CONTROL_SUPERVISOR_LANDING;  // 잘못된 상태를 착지로 복구한다.
                handle->stand_command_armed = false;
                break;
        }
    }

    if (handle->supervisor != CONTROL_SUPERVISOR_READY)
    {
        handle->motion_armed = false;  // READY 밖에서는 동작 허가를 해제한다.
        handle->neutral_time_s = 0.0f; // 중립 유지 시간을 초기화한다.
    }
    else if (!handle->motion_armed)
    {
        if (ControlPriority_SticksNeutral(user))
        {
            handle->neutral_time_s = fminf(handle->neutral_time_s + ROBOT_CONTROL_PERIOD_S,
                                           0.20f);  // 중립 유지 시간을 누적한다.
            handle->motion_armed = (handle->neutral_time_s >= 0.20f);  // 0.2초 후 동작을 허가한다.
        }
        else
        {
            handle->neutral_time_s = 0.0f;  // 중립이 깨지면 다시 측정한다.
        }
    }

    switch (handle->supervisor)
    {
        case CONTROL_SUPERVISOR_STANDING: output.active_mode = ROBOT_MODE_STAND; break;
        case CONTROL_SUPERVISOR_READY: output.active_mode = ROBOT_MODE_READY; break;
        case CONTROL_SUPERVISOR_FAULT: output.active_mode = ROBOT_MODE_FAULT; break;
        case CONTROL_SUPERVISOR_KILL: output.active_mode = ROBOT_MODE_KILL; break;
        default: output.active_mode = ROBOT_MODE_LANDING; break;  // LANDED와 LANDING을 착지 출력으로 묶는다.
    }

    if ((handle->supervisor == CONTROL_SUPERVISOR_READY) && handle->motion_armed)
    {
        if (correction_request)
        {
            output.active_mode = ROBOT_MODE_CORRECTION;  // 보정 모드를 선택한다.
        }
        else if (manual_request)
        {
            output.active_mode = ROBOT_MODE_MANUAL;      // 수동 모드를 선택한다.
        }

        output.reset_command = (user->se != 0U);         // READY 동작 중 Reset을 허가한다.
    }

    if ((output.active_mode == ROBOT_MODE_MANUAL) ||
        (output.active_mode == ROBOT_MODE_CORRECTION))
    {
        output.throttle = user->throttle;  // 동작 모드에서 Throttle을 전달한다.
        output.yaw = user->yaw;            // 동작 모드에서 Yaw를 전달한다.
        output.roll = user->roll;          // 동작 모드에서 Roll을 전달한다.
        output.pitch = user->pitch;        // 동작 모드에서 Pitch를 전달한다.
        output.sa = (user->sa != 0U) ? 1U : 0U;  // SA를 논리값으로 전달한다.
        output.s1 = (user->s1 != 0U) ? 1U : 0U;  // S1 이동 방식을 전달한다.
    }

    return output;
}
