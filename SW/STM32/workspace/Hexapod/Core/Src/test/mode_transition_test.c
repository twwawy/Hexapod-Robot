#include "test/mode_transition_test.h"

#include "high_control/control_priority.h"
#include "high_control/drone_controller.h"

#include <math.h>
#include <string.h>

/* SB 첫 위치와 SC 가운데가 현재 동작 모드를 유지하는지 검사한다. */
static bool ModeTransitionTest_CheckSwitchFunctions(void)
{
    ControlPriority_Handle_t control;  // 시험용 우선순위 상태를 저장한다.
    RobotUserCommand_t user;           // 명시적 스위치 입력을 저장한다.
    RobotSafetyOutput_t safety;        // 정상 Safety 입력을 저장한다.
    RobotPriorityOutput_t output;      // 우선순위 결과를 저장한다.
    uint32_t cycle;                    // READY 중립 유지 횟수를 저장한다.

    memset(&user, 0, sizeof(user));        // 모든 사용자 입력을 해제한다.
    memset(&safety, 0, sizeof(safety));    // Fault 없는 상태를 준비한다.
    ControlPriority_Init(&control);        // 우선순위 상태를 착지로 초기화한다.
    (void)ControlPriority_Step(&control, &user, false, false, &safety);  // 새 서기 요청을 무장한다.
    user.sb = 1U;                          // 서기 유지 위치를 선택한다.
    (void)ControlPriority_Step(&control, &user, false, false, &safety);  // 서기를 시작한다.
    (void)ControlPriority_Step(&control, &user, true, false, &safety);   // READY로 전환한다.

    for (cycle = 0U; cycle < 40U; ++cycle)
    {
        output = ControlPriority_Step(&control, &user, true, false, &safety);  // 중립 유지로 동작을 허가한다.
    }
    if (output.active_mode != ROBOT_MODE_MANUAL)
    {
        return false;
    }

    user.sc = 1U;  // SC 무기능 가운데 위치를 선택한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // 직전 조종 모드를 유지한다.
    if ((output.active_mode != ROBOT_MODE_MANUAL) || output.reset_command)
    {
        return false;
    }

    user.sc = 2U;  // SC 끝의 보정 모드를 선택한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // 보정 전환을 적용한다.
    if ((output.active_mode != ROBOT_MODE_CORRECTION) || output.reset_command)
    {
        return false;
    }

    user.sc = 1U;  // SC 무기능 가운데 위치로 돌아간다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // 직전 보정 모드를 유지한다.
    if ((output.active_mode != ROBOT_MODE_CORRECTION) || output.reset_command)
    {
        return false;
    }

    user.sb = 0U;  // SB 첫 위치를 선택한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // SB 무기능 위치를 적용한다.
    if (output.active_mode != ROBOT_MODE_CORRECTION)
    {
        return false;
    }

    user.sb = 1U;  // SB 서기 유지 위치로 돌아간다.
    user.se = 1U;  // SE의 명령 없는 READY를 요청한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // SE 기능을 적용한다.
    if ((output.active_mode != ROBOT_MODE_READY) || output.reset_command)
    {
        return false;
    }

    user.se = 0U;  // SE READY를 해제한다.
    user.sc = 1U;  // SC 무기능 가운데 위치를 선택한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // SE 해제 후 직전 보정을 복원한다.
    if ((output.active_mode != ROBOT_MODE_CORRECTION) || output.reset_command)
    {
        return false;
    }

    user.sb = 2U;  // SB 끝의 착지를 선택한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // 착지 전환을 적용한다.
    return output.active_mode == ROBOT_MODE_LANDING;  // SB 끝 착지가 유지되는지 확인한다.
}

/* 짐벌 기능이 바뀌는 모드와 S1 전환에서 잔류 명령을 검사한다. */
bool ModeTransitionTest_Run(void)
{
    DroneController_Handle_t controller;       // 시험용 Drone Controller를 저장한다.
    RobotPriorityOutput_t priority;             // 명시적 모드와 짐벌 입력을 저장한다.
    RobotDroneOutput_t output;                  // 제어 결과를 저장한다.
    bool contact[ROBOT_LEG_COUNT];              // 명시적 접촉 상태를 저장한다.
    uint32_t cycle;                             // 필터 진행 횟수를 저장한다.

    if (!ModeTransitionTest_CheckSwitchFunctions())
    {
        return false;
    }

    memset(&priority, 0, sizeof(priority));  // 시험 입력을 중립으로 만든다.
    memset(contact, 1, sizeof(contact));     // 모든 발 접촉을 명시한다.
    DroneController_Init(&controller);       // 필터 상태를 초기화한다.

    priority.active_mode = ROBOT_MODE_MANUAL;  // 수동 모드로 전환한다.
    priority.throttle = 1000;                  // 최대 전진 입력을 넣는다.
    priority.yaw = 1000;                       // 최대 회전 입력을 넣는다.
    for (cycle = 0U; cycle < 30U; ++cycle)
    {
        output = DroneController_Step(&controller, &priority, contact, 0.0f);  // 필터에 명령을 누적한다.
    }
    if ((output.vx_user_mps <= 0.0f) || (output.wz_user_radps >= 0.0f))
    {
        return false;
    }

    priority.active_mode = ROBOT_MODE_CORRECTION;  // 짐벌 기능을 보정으로 바꾼다.
    output = DroneController_Step(&controller, &priority, contact, 0.0f);  // 새 모드 첫 출력을 계산한다.
    if ((fabsf(output.correction_velocity_mps.x) > 0.01f) ||
        (fabsf(output.correction_velocity_mps.z) > 0.01f))
    {
        return false;
    }

    priority.active_mode = ROBOT_MODE_MANUAL;  // 수동으로 다시 전환한다.
    priority.s1 = 0U;                          // Yaw 회전 기능을 선택한다.
    priority.throttle = 0;                     // 전진 입력을 제거한다.
    (void)DroneController_Step(&controller, &priority, contact, 0.0f);  // 수동 첫 출력을 계산한다.
    priority.s1 = 1U;                          // 같은 짐벌을 횡이동으로 바꾼다.
    output = DroneController_Step(&controller, &priority, contact, 0.0f);  // S1 전환 첫 출력을 계산한다.
    return (output.vy_user_mps < 0.0f) &&
           (fabsf(output.vy_user_mps) < 0.06f);  // 반대 방향 횡이동과 S1 전환 연속성을 확인한다.
}
