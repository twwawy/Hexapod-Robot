#include "test/mode_transition_test.h"

#include "high_control/drone_controller.h"

#include <math.h>
#include <string.h>

/* 짐벌 기능이 바뀌는 모드와 SA 전환에서 잔류 명령을 검사한다. */
bool ModeTransitionTest_Run(void)
{
    DroneController_Handle_t controller;       // 시험용 Drone Controller를 저장한다.
    RobotPriorityOutput_t priority;             // 명시적 모드와 짐벌 입력을 저장한다.
    RobotDroneOutput_t output;                  // 제어 결과를 저장한다.
    bool contact[ROBOT_LEG_COUNT];              // 명시적 접촉 상태를 저장한다.
    uint32_t cycle;                             // 필터 진행 횟수를 저장한다.

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
    if ((output.vx_user_mps <= 0.0f) || (output.wz_user_radps <= 0.0f))
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
    priority.sa = 0U;                          // Yaw 회전 기능을 선택한다.
    priority.throttle = 0;                     // 전진 입력을 제거한다.
    (void)DroneController_Step(&controller, &priority, contact, 0.0f);  // 수동 첫 출력을 계산한다.
    priority.sa = 1U;                          // 같은 짐벌을 횡이동으로 바꾼다.
    output = DroneController_Step(&controller, &priority, contact, 0.0f);  // SA 전환 첫 출력을 계산한다.
    return fabsf(output.vy_user_mps) < 0.06f;  // 이전 회전값이 최대 횡이동으로 튀지 않는지 확인한다.
}
