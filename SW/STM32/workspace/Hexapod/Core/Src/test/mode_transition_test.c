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
    user.connected = true;                // 정상 조종기 연결을 준비한다.
    user.motion_armed = true;             // 입력 전처리의 동작 허가를 준비한다.
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
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // 강화학습 진입 중립 확인을 시작한다.
    if (output.active_mode != ROBOT_MODE_READY)
    {
        return false;
    }

    for (cycle = 0U; cycle < 41U; ++cycle)
    {
        output = ControlPriority_Step(&control, &user, true, false, &safety);  // 새 진입 중립 시간을 누적한다.
    }
    if (output.active_mode != ROBOT_MODE_RL)
    {
        return false;
    }

    user.sb = 1U;  // SB 서기 유지 위치로 돌아간다.
    user.sc = 2U;  // SE 해제 후 복원할 보정 모드를 선택한다.
    user.se = 1U;  // SE의 기본 자세 READY를 요청한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // SE 기본 자세 복귀를 적용한다.
    if ((output.active_mode != ROBOT_MODE_READY) || !output.reset_command)
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

/* 강화학습 진입 허가와 상위 안전 요청의 우선순위를 검사한다. */
static bool ModeTransitionTest_CheckRlPriority(void)
{
    ControlPriority_Handle_t control;                 // 강화학습 우선순위 상태를 저장한다.
    RobotUserCommand_t user;                          // 조종기 입력과 연결 상태를 저장한다.
    RobotSafetyOutput_t safety;                       // 시험할 안전 요청을 저장한다.
    RobotPriorityOutput_t output;                     // 모드 선택 결과를 저장한다.
    uint32_t cycle;                                   // 중립 유지 횟수를 저장한다.

    memset(&user, 0, sizeof(user));      // SB 첫 위치와 중립을 준비한다.
    memset(&safety, 0, sizeof(safety));  // 정상 안전 상태를 준비한다.
    ControlPriority_Init(&control);      // 부팅 직후 착지 상태를 준비한다.
    user.connected = true;               // 정상 조종기 연결을 준비한다.
    user.motion_armed = true;            // 입력 동작 허가를 준비한다.

    for (cycle = 0U; cycle < 50U; ++cycle)
    {
        output = ControlPriority_Step(&control, &user, false, true, &safety);  // 부팅 직후 SB 첫 위치를 유지한다.
        if (output.active_mode != ROBOT_MODE_LANDING)
        {
            return false;
        }
    }

    user.sb = 1U;                                                           // 명시적으로 서기를 요청한다.
    output = ControlPriority_Step(&control, &user, false, false, &safety);  // 서기 상태로 진입한다.
    if (output.active_mode != ROBOT_MODE_STAND)
    {
        return false;
    }

    user.sb = 0U;                                                           // 서기 완료 전에 강화학습을 선택한다.
    output = ControlPriority_Step(&control, &user, false, false, &safety);  // 서기 완료까지 기다린다.
    if (output.active_mode != ROBOT_MODE_STAND)
    {
        return false;
    }

    user.connected = false;                                                    // 조종기 연결이 끊긴 상태를 준비한다.
    for (cycle = 0U; cycle < 50U; ++cycle)
    {
        output = ControlPriority_Step(&control, &user, true, false, &safety);  // 연결 없이 서기 완료를 전달한다.
    }
    if ((output.active_mode != ROBOT_MODE_READY) || control.motion_armed)
    {
        return false;
    }

    user.connected = true;                                                 // 조종기 연결을 복구한다.
    user.motion_armed = false;                                             // 입력 무장은 아직 차단한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // 연결만으로 진입하지 않는지 확인한다.
    if ((output.active_mode != ROBOT_MODE_READY) || control.motion_armed)
    {
        return false;
    }

    user.motion_armed = true;                                                  // 입력 무장을 허가한다.
    user.throttle = 1000;                                                      // 중립이 아닌 전진 입력을 준비한다.
    for (cycle = 0U; cycle < 50U; ++cycle)
    {
        output = ControlPriority_Step(&control, &user, true, false, &safety);  // 비중립 진입 차단을 확인한다.
    }
    if ((output.active_mode != ROBOT_MODE_READY) || control.motion_armed)
    {
        return false;
    }

    user.throttle = 0;                                                         // 새 진입을 위해 중립으로 돌린다.
    for (cycle = 0U; cycle < 41U; ++cycle)
    {
        output = ControlPriority_Step(&control, &user, true, false, &safety);  // 필요한 중립 시간을 유지한다.
    }
    if (output.active_mode != ROBOT_MODE_RL)
    {
        return false;
    }

    user.throttle = -800;                                                          // 부호가 있는 후진 입력을 준비한다.
    user.yaw = 600;                                                                // 양의 회전 짐벌을 준비한다.
    user.roll = 1000;                                                              // 강화학습에서 제외할 Roll을 준비한다.
    user.pitch = -1000;                                                            // 강화학습에서 제외할 Pitch를 준비한다.
    user.s1 = ROBOT_WALK_WAVE_TURN;                                                // 강화학습에서 제외할 Wave 선택을 준비한다.
    for (user.sc = 0U; user.sc <= 2U; ++user.sc)
    {
        output = ControlPriority_Step(&control, &user, true, false, &safety);      // SB가 세 SC 선택보다 우선하는지 확인한다.
        if ((output.active_mode != ROBOT_MODE_RL) || (output.throttle != -800) ||
            (output.yaw != 600) || (output.roll != 0) || (output.pitch != 0) ||
            (output.s1 != ROBOT_WALK_TRIPOD_TURN))
        {
            return false;
        }
    }

    user.se = 1U;                                                           // 강화학습 중 READY를 요청한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);   // READY가 강화학습을 차단하는지 확인한다.
    if ((output.active_mode != ROBOT_MODE_READY) || !output.reset_command)
    {
        return false;
    }

    user.se = 0U;                                                          // READY 요청을 해제한다.
    ControlPriority_DisarmMotion(&control);                                // 정책 오류 후 입력 허가를 제거한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // 비중립 상태의 재개를 차단한다.
    if ((output.active_mode != ROBOT_MODE_READY) || control.motion_armed)
    {
        return false;
    }

    user.sb = 2U;                                                          // 강화학습 선택 후 착지를 요청한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // 착지가 중립 허가보다 우선하는지 확인한다.
    if (output.active_mode != ROBOT_MODE_LANDING)
    {
        return false;
    }

    user.sb = 0U;                                                          // 강화학습을 다시 선택한다.
    safety.rollover_fault = true;                                          // 전도 Fault를 입력한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // Fault가 강화학습을 차단하는지 확인한다.
    if (output.active_mode != ROBOT_MODE_FAULT)
    {
        return false;
    }

    user.sd = 1U;                                                          // Fault와 함께 Kill을 요청한다.
    output = ControlPriority_Step(&control, &user, true, false, &safety);  // Kill의 최상위 우선순위를 확인한다.
    return output.active_mode == ROBOT_MODE_KILL;                          // 최종 차단 모드를 확인한다.
}

/* 강화학습의 두 축 속도와 고정 보행 및 필터 전환을 검사한다. */
static bool ModeTransitionTest_CheckRlDrone(void)
{
    DroneController_Handle_t controller;           // 시험용 필터와 모드 상태를 저장한다.
    RobotPriorityOutput_t priority;                // 직접 전달할 모드 입력을 저장한다.
    RobotDroneOutput_t output;                     // 강화학습 제어 결과를 저장한다.
    bool contact[ROBOT_LEG_COUNT];                 // 여섯 발의 정상 접촉을 저장한다.
    float previous_wz_radps;                       // S1 변경 직전 회전 속도를 저장한다.
    uint32_t cycle;                                // 필터 진행 주기를 저장한다.

    memset(&priority, 0, sizeof(priority));                                 // 중립 입력을 준비한다.
    memset(contact, 1, sizeof(contact));                                    // 전체 접촉을 준비한다.
    DroneController_Init(&controller);                                      // 이전 제어 상태를 제거한다.
    priority.active_mode = ROBOT_MODE_MANUAL;                               // 기존 수동 필터를 준비한다.
    priority.throttle = 1000;                                               // 전진 잔류값을 만든다.
    priority.yaw = -1000;                                                   // 회전 잔류값을 만든다.
    priority.roll = 1000;                                                   // Roll 잔류값을 만든다.
    priority.pitch = -1000;                                                 // Pitch 잔류값을 만든다.
    for (cycle = 0U; cycle < 40U; ++cycle)
    {
        (void)DroneController_Step(&controller, &priority, contact, 0.2f);  // 수동 필터를 누적한다.
    }

    priority.active_mode = ROBOT_MODE_RL;                                               // 강화학습 모드로 전환한다.
    priority.throttle = 0;                                                              // 새 전진 입력을 제거한다.
    priority.yaw = 0;                                                                   // 새 회전 입력을 제거한다.
    output = DroneController_Step(&controller, &priority, contact, 0.2f);               // 강화학습 첫 출력을 계산한다.
    if (!output.rl_enable || !output.locomotion_enable || output.manual_enable ||
        !output.body_control_enable || !output.posture_enable || !output.stand_done ||
        (output.vx_user_mps != 0.0f) || (output.wz_user_radps != 0.0f) ||
        (output.posture_reference_rad.roll != 0.0f) ||
        (output.posture_reference_rad.pitch != 0.0f) || output.tripod_enable)
    {
        return false;
    }

    priority.throttle = -1000;                                                 // 최대 후진 입력을 준비한다.
    priority.yaw = 1000;                                                       // 반전되는 회전 입력을 준비한다.
    priority.s1 = ROBOT_WALK_TRIPOD_LATERAL;                                   // 제외할 횡이동 선택을 준비한다.
    for (cycle = 0U; cycle < 40U; ++cycle)
    {
        output = DroneController_Step(&controller, &priority, contact, 0.2f);  // 강화학습 속도를 필터링한다.
    }
    if ((output.vx_user_mps >= -ROBOT_GAIT_LINEAR_THRESHOLD_MPS) ||
        (output.wz_user_radps >= -ROBOT_GAIT_YAW_THRESHOLD_RADPS) ||
        (output.vy_user_mps != 0.0f) || !output.tripod_enable ||
        (output.gait_pattern != ROBOT_GAIT_TRIPOD) ||
        (output.posture_reference_rad.roll != 0.0f) ||
        (output.posture_reference_rad.pitch != 0.0f))
    {
        return false;
    }

    previous_wz_radps = output.wz_user_radps;                                        // 고정 패턴 확인용 속도를 저장한다.
    priority.s1 = ROBOT_WALK_WAVE_TURN;                                              // Wave 선택으로 변경한다.
    output = DroneController_Step(&controller, &priority, contact, 0.2f);            // S1 변경 후에도 Tripod를 유지한다.
    if ((output.gait_pattern != ROBOT_GAIT_TRIPOD) ||
        (output.wz_user_radps > previous_wz_radps) || (output.vy_user_mps != 0.0f))
    {
        return false;
    }

    memset(&priority, 0, sizeof(priority));                                          // 모든 짐벌을 중립으로 만든다.
    priority.active_mode = ROBOT_MODE_MANUAL;                                        // 수동 모드로 복귀한다.
    output = DroneController_Step(&controller, &priority, contact, 0.2f);            // 강화학습 잔류 필터를 제거한다.
    return output.manual_enable && output.locomotion_enable && !output.rl_enable &&
           (output.vx_user_mps == 0.0f) && (output.wz_user_radps == 0.0f) &&
           (output.posture_reference_rad.roll == 0.0f) &&
           (output.posture_reference_rad.pitch == 0.0f);                             // 모드 복귀 시 이전 속도·자세 입력 차단을 확인한다.
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
    DroneController_Handle_t controller;        // 시험용 Drone Controller를 저장한다.
    DroneController_Handle_t fresh_controller;  // 이전 모드 이력이 없는 비교 제어기를 저장한다.
    RobotPriorityOutput_t priority;             // 명시적 모드와 짐벌 입력을 저장한다.
    RobotDroneOutput_t output;                  // 제어 결과를 저장한다.
    RobotDroneOutput_t fresh_output;            // 새 제어기의 동일 입력 결과를 저장한다.
    bool contact[ROBOT_LEG_COUNT];              // 명시적 접촉 상태를 저장한다.
    uint32_t cycle;                             // 필터 진행 횟수를 저장한다.

    if (!ModeTransitionTest_CheckSwitchFunctions() ||
        !ModeTransitionTest_CheckRlPriority() ||
        !ModeTransitionTest_CheckRlDrone() ||
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
    DroneController_Init(&fresh_controller);  // 이전 모드의 필터가 없는 기준을 준비한다.
    fresh_output = DroneController_Step(&fresh_controller, &priority, contact, 0.0f);  // 동일한 첫 보정 입력을 계산한다.
    if ((fabsf(output.correction_velocity_mps.x - fresh_output.correction_velocity_mps.x) > 0.0000001f) ||
        (fabsf(output.correction_velocity_mps.y - fresh_output.correction_velocity_mps.y) > 0.0000001f) ||
        (fabsf(output.correction_velocity_mps.z - fresh_output.correction_velocity_mps.z) > 0.0000001f) ||
        (fabsf(output.posture_reference_rad.yaw - fresh_output.posture_reference_rad.yaw) > 0.0000001f))
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
