#include "test/mode_transition_test.h"

#include "high_control/body_posture_controller.h"
#include "high_control/control_priority.h"
#include "high_control/drone_controller.h"
#include "high_control/foot_trajectory.h"

#include <math.h>
#include <string.h>

/* SC 모드 선택과 SA 그리퍼 명령이 분리되는지 검사한다. */
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

    user.sa = 1U;  // SA 그리퍼 놓기 명령을 선택한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // SA가 모드를 바꾸지 않는지 확인한다.
    if (output.active_mode != ROBOT_MODE_MANUAL)
    {
        return false;
    }

    user.sc = 1U;  // SC 가운데의 매니퓰레이터 모드를 선택한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // ARM 모드로 전환한다.
    if (output.active_mode != ROBOT_MODE_ARM)
    {
        return false;
    }

    user.sa = 0U;  // SA 그리퍼 잡기 명령을 선택한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // 잡기 중 ARM 모드를 유지한다.
    if ((output.active_mode != ROBOT_MODE_ARM) || output.reset_command)
    {
        return false;
    }

    user.sc = 0U;  // SC 첫 위치의 수동 모드를 선택한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // 수동 모드로 돌아간다.
    if (output.active_mode != ROBOT_MODE_MANUAL)
    {
        return false;
    }

    user.sc = 2U;  // SC 끝의 보정 모드를 선택한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // 보정 전환을 적용한다.
    if ((output.active_mode != ROBOT_MODE_CORRECTION) || output.reset_command)
    {
        return false;
    }

    user.sc = 1U;  // SC 가운데의 매니퓰레이터 모드로 돌아간다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // ARM 모드로 전환한다.
    if ((output.active_mode != ROBOT_MODE_ARM) || output.reset_command)
    {
        return false;
    }

    user.sb = 0U;  // SB 첫 위치를 선택한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // SB 무기능 위치를 적용한다.
    if (output.active_mode != ROBOT_MODE_ARM)
    {
        return false;
    }

    user.sb = 1U;  // SB 서기 유지 위치로 돌아간다.
    user.sc = 2U;  // SE 해제 후 복원할 보정 모드를 선택한다.
    user.se = 1U;  // SE의 명령 없는 READY를 요청한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // SE 기능을 적용한다.
    if ((output.active_mode != ROBOT_MODE_READY) || output.reset_command)
    {
        return false;
    }

    user.se = 0U;  // SE READY를 해제한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // SE 해제 후 직전 보정을 복원한다.
    if ((output.active_mode != ROBOT_MODE_CORRECTION) || output.reset_command)
    {
        return false;
    }

    user.sb = 2U;  // SB 끝의 착지를 선택한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // 착지 전환을 적용한다.
    return output.active_mode == ROBOT_MODE_LANDING;  // SB 끝 착지가 유지되는지 확인한다.
}

/* 보정 자세가 ARM을 거친 뒤 수동 모드에서도 유지되는지 검사한다. */
static bool ModeTransitionTest_CheckCorrectionMemory(void)
{
    FootTrajectory_Handle_t trajectory;  // 보정 Offset 상태를 저장한다.
    RobotBodyTwist_t twist;               // 보정 이동 명령을 저장한다.
    RobotDroneOutput_t drone;             // 모드별 궤적 허가를 저장한다.
    RobotGaitPhase_t gait;                 // 전체 Stance 상태를 저장한다.
    RobotEuler_t posture;                  // 수평 자세를 저장한다.
    RobotVec3_t corrected_offset;          // 보정 직후 Offset을 저장한다.
    uint32_t leg;                          // 초기화할 다리 번호를 저장한다.

    memset(&twist, 0, sizeof(twist));      // 보정 명령을 0으로 준비한다.
    memset(&drone, 0, sizeof(drone));      // 모드 출력을 0으로 준비한다.
    memset(&gait, 0, sizeof(gait));        // 전체 Stance를 준비한다.
    memset(&posture, 0, sizeof(posture));  // 수평 자세를 준비한다.
    FootTrajectory_Init(&trajectory);      // 기본 발 위치를 준비한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        gait.state[leg] = ROBOT_LEG_STANCE;  // 여섯 발을 지지 상태로 둔다.
    }

    twist.vx = 0.02f;                   // X축 보정 속도를 입력한다.
    twist.vy = -0.01f;                  // Y축 보정 속도를 입력한다.
    twist.vz = 0.01f;                   // Z축 보정 속도를 입력한다.
    drone.correction_enable = true;     // 보정 모드를 활성화한다.
    drone.body_control_enable = true;   // 몸체 Offset 적분을 허가한다.
    drone.tripod_mode = ROBOT_TRIPOD_NORMAL;  // 정상 보정 경로를 선택한다.
    (void)FootTrajectory_Step(&trajectory, &twist, &drone,
                              &gait, &posture);  // 보정 Offset을 한 주기 누적한다.
    corrected_offset = trajectory.body_offset_m;  // 보정 결과를 저장한다.

    memset(&twist, 0, sizeof(twist));  // ARM에서 이동 명령을 제거한다.
    memset(&drone, 0, sizeof(drone));  // ARM의 6족 자세 고정 출력을 준비한다.
    (void)FootTrajectory_Step(&trajectory, &twist, &drone,
                              &gait, &posture);  // ARM 모드 한 주기를 계산한다.
    if ((fabsf(trajectory.body_offset_m.x - corrected_offset.x) > 0.0000001f) ||
        (fabsf(trajectory.body_offset_m.y - corrected_offset.y) > 0.0000001f) ||
        (fabsf(trajectory.body_offset_m.z - corrected_offset.z) > 0.0000001f))
    {
        return false;
    }

    drone.manual_enable = true;            // 수동 모드를 활성화한다.
    drone.body_control_enable = true;      // 몸체 제어를 다시 활성화한다.
    drone.tripod_mode = ROBOT_TRIPOD_NORMAL;  // 정상 보행 경로를 선택한다.
    (void)FootTrajectory_Step(&trajectory, &twist, &drone,
                              &gait, &posture);  // 수동 모드 첫 주기를 계산한다.
    return (fabsf(trajectory.body_offset_m.x - corrected_offset.x) <= 0.0000001f) &&
           (fabsf(trajectory.body_offset_m.y - corrected_offset.y) <= 0.0000001f) &&
           (fabsf(trajectory.body_offset_m.z - corrected_offset.z) <= 0.0000001f);  // 세 축 보정값 유지를 확인한다.
}

/* 보정 각도가 ARM과 보정 재진입 사이에서 유지되는지 검사한다. */
static bool ModeTransitionTest_CheckCorrectionPostureMemory(void)
{
    BodyPostureController_Handle_t controller;  // 자세 제어 상태를 저장한다.
    BodyPostureController_Output_t output;      // 자세 제어 결과를 저장한다.
    FootTrajectory_Handle_t trajectory;         // 시험용 기본 발 위치를 저장한다.
    RobotDroneOutput_t drone;                    // 모드별 자세 허가를 저장한다.
    RobotEuler_t measured;                       // 시험용 실측 자세를 저장한다.
    RobotEuler_t corrected_command;              // 보정 직후 각도를 저장한다.
    uint32_t cycle;                              // 보정 진행 주기를 저장한다.

    memset(&drone, 0, sizeof(drone));            // 모드 출력을 0으로 준비한다.
    memset(&measured, 0, sizeof(measured));      // 수평 자세를 준비한다.
    BodyPostureController_Init(&controller);     // 자세 제어 상태를 초기화한다.
    FootTrajectory_Init(&trajectory);            // 유효한 기본 발 위치를 준비한다.
    drone.posture_enable = true;                  // 보정 자세 제어를 활성화한다.
    drone.correction_enable = true;               // 보정 모드를 활성화한다.
    drone.posture_reference_rad.yaw = 10.0f * ROBOT_DEG_TO_RAD_F;  // Yaw 보정 목표를 입력한다.

    for (cycle = 0U; cycle < 20U; ++cycle)
    {
        output = BodyPostureController_Step(&controller,
                                             trajectory.memory,
                                             &drone,
                                             &measured,
                                             false);  // 보정 각도를 누적한다.
    }
    corrected_command = output.command_rad;  // 보정 결과를 저장한다.
    if (fabsf(corrected_command.yaw) <= 0.0000001f)
    {
        return false;
    }

    memset(&drone, 0, sizeof(drone));  // ARM의 자세 고정 출력을 준비한다.
    output = BodyPostureController_Step(&controller,
                                         trajectory.memory,
                                         &drone,
                                         &measured,
                                         false);  // ARM 모드 한 주기를 계산한다.
    if ((fabsf(output.command_rad.roll - corrected_command.roll) > 0.0000001f) ||
        (fabsf(output.command_rad.pitch - corrected_command.pitch) > 0.0000001f) ||
        (fabsf(output.command_rad.yaw - corrected_command.yaw) > 0.0000001f))
    {
        return false;
    }

    measured.yaw = corrected_command.yaw;  // ARM에서 유지된 실제 Heading을 반영한다.
    drone.posture_enable = true;            // 보정 자세 제어를 다시 활성화한다.
    drone.correction_enable = true;         // 보정 모드로 다시 진입한다.
    output = BodyPostureController_Step(&controller,
                                         trajectory.memory,
                                         &drone,
                                         &measured,
                                         false);  // 보정 재진입 첫 주기를 계산한다.
    return (fabsf(output.command_rad.roll - corrected_command.roll) <= 0.0000001f) &&
           (fabsf(output.command_rad.pitch - corrected_command.pitch) <= 0.0000001f) &&
           (fabsf(output.command_rad.yaw - corrected_command.yaw) <= 0.0000001f);  // 세 축 각도 유지를 확인한다.
}

/* 짐벌 기능이 바뀌는 모드와 S1 전환에서 잔류 명령을 검사한다. */
bool ModeTransitionTest_Run(void)
{
    DroneController_Handle_t controller;       // 시험용 Drone Controller를 저장한다.
    RobotPriorityOutput_t priority;             // 명시적 모드와 짐벌 입력을 저장한다.
    RobotDroneOutput_t output;                  // 제어 결과를 저장한다.
    bool contact[ROBOT_LEG_COUNT];              // 명시적 접촉 상태를 저장한다.
    uint32_t cycle;                             // 필터 진행 횟수를 저장한다.

    if (!ModeTransitionTest_CheckSwitchFunctions() ||
        !ModeTransitionTest_CheckCorrectionMemory() ||
        !ModeTransitionTest_CheckCorrectionPostureMemory())
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
