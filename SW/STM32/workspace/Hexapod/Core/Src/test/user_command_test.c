#include "test/user_command_test.h"

#include "common/robot_calibration.h"
#include "high_control/control_priority.h"
#include "high_control/drone_controller.h"
#include "user_command/user_command.h"

#include <math.h>
#include <string.h>

/* 비대칭 중립값과 양방향 보정에서 S1의 세 구간 경계를 검사한다. */
static bool UserCommandTest_CheckS1Boundaries(void)
{
    static const uint16_t position[] = {0U, 332U, 333U, 500U, 666U, 667U, 1000U};  // 양 끝과 소수 경계의 앞뒤를 정의한다.
    static const uint8_t expected[] =
    {
        ROBOT_WALK_TRIPOD_TURN, ROBOT_WALK_TRIPOD_TURN,                       // 33.3% 미만의 선택을 정의한다.
        ROBOT_WALK_WAVE_TURN, ROBOT_WALK_WAVE_TURN, ROBOT_WALK_WAVE_TURN,     // 두 경계를 포함한 중앙 선택을 정의한다.
        ROBOT_WALK_TRIPOD_LATERAL, ROBOT_WALK_TRIPOD_LATERAL                  // 66.6% 초과의 선택을 정의한다.
    };
    UserCommand_ChannelCalibration_t calibration =
        {100U, 423U, 1100U, 1, {0U, 0U, 1U}, true};  // 중립 보정과 기존 스위치 표에 의존하지 않는 입력을 준비한다.
    UserCommand_Handle_t handle;                    // 시험용 채널 변환 상태를 저장한다.
    uint16_t raw[ROBOT_CRSF_CHANNEL_COUNT] = {0};  // 명시적인 채널 입력을 준비한다.
    uint32_t direction;                            // 정방향과 반전 보정의 시험 번호를 저장한다.
    uint32_t index;                                // 현재 경계 시험 번호를 저장한다.

    UserCommand_Init(&handle);  // 독립적인 경계 시험 상태를 초기화한다.
    for (direction = 0U; direction < 2U; ++direction)
    {
        calibration.direction = (direction == 0U) ? 1 : -1;  // 두 채널 방향을 차례로 선택한다.
        if (!UserCommand_SetCalibration(&handle, 9U, &calibration))
        {
            return false;
        }

        for (index = 0U; index < (sizeof(position) / sizeof(position[0])); ++index)
        {
            raw[9] = (uint16_t)(calibration.raw_min + ((direction == 0U) ?
                position[index] : 1000U - position[index]));  // 동일한 물리 위치의 정방향·반전 raw를 만든다.
            UserCommand_UpdateChannels(&handle, raw, index);  // 경계 위치를 실제 입력 경로로 변환한다.
            if (handle.command.s1 != expected[index])
            {
                return false;
            }
        }

        raw[9] = 0U;                                     // 보정 최소값보다 낮은 입력을 준비한다.
        UserCommand_UpdateChannels(&handle, raw, 10U);  // 범위를 벗어난 최소 입력을 변환한다.
        if (handle.command.s1 != ((direction == 0U) ? ROBOT_WALK_TRIPOD_TURN : ROBOT_WALK_TRIPOD_LATERAL))
        {
            return false;
        }
        raw[9] = UINT16_MAX;                              // 보정 최대값보다 높은 입력을 준비한다.
        UserCommand_UpdateChannels(&handle, raw, 11U);  // 범위를 벗어난 최대 입력을 변환한다.
        if (handle.command.s1 != ((direction == 0U) ? ROBOT_WALK_TRIPOD_LATERAL : ROBOT_WALK_TRIPOD_TURN))
        {
            return false;
        }
    }
    return true;
}

/* 세 보행 선택의 전달과 회전·횡이동 명령 및 전환 초기화를 검사한다. */
static bool UserCommandTest_CheckWalkingSelection(void)
{
    static const uint8_t mode[] =
    {
        ROBOT_WALK_TRIPOD_TURN, ROBOT_WALK_TRIPOD_LATERAL, ROBOT_WALK_WAVE_TURN,
        ROBOT_WALK_TRIPOD_LATERAL, ROBOT_WALK_TRIPOD_TURN, ROBOT_WALK_WAVE_TURN  // 각 조합의 구간 전환을 검사한다.
    };
    ControlPriority_Handle_t control;           // 시험용 우선순위 상태를 저장한다.
    DroneController_Handle_t controller;        // 시험용 조종 변환 상태를 저장한다.
    RobotUserCommand_t user = {0};              // 명시적인 사용자 입력을 준비한다.
    RobotSafetyOutput_t safety = {0};           // Fault가 없는 시험 조건을 준비한다.
    RobotPriorityOutput_t priority;             // 실제 우선순위 전달 결과를 저장한다.
    RobotDroneOutput_t output;                  // 보행 종류와 조종 속도를 저장한다.
    bool contact[ROBOT_LEG_COUNT];              // 명시적인 지면 접촉을 저장한다.
    uint32_t index;                             // 현재 보행 선택의 시험 번호를 저장한다.
    uint32_t cycle;                             // 입력 필터 안정 시간을 계산한다.

    ControlPriority_Init(&control);                   // 우선순위 상태를 초기화한다.
    DroneController_Init(&controller);                // 조종 변환 상태를 초기화한다.
    memset(contact, 1, sizeof(contact));               // 모든 발의 지면 접촉을 준비한다.
    control.supervisor = CONTROL_SUPERVISOR_READY;    // 서기가 완료된 시험 조건을 선택한다.
    control.motion_armed = true;                      // 입력 전달 시험의 동작을 허가한다.
    user.sb = 1U;                                     // 서기 유지 상태를 선택한다.
    user.throttle = 1000;                             // 최대 전진 명령을 입력한다.
    user.yaw = 1000;                                  // 최대 회전 또는 횡이동 명령을 입력한다.

    for (index = 0U; index < (sizeof(mode) / sizeof(mode[0])); ++index)
    {
        const bool lateral = (mode[index] == ROBOT_WALK_TRIPOD_LATERAL);  // 이번 구간의 짐벌 기능을 저장한다.
        const RobotGaitPattern_t pattern = (mode[index] == ROBOT_WALK_WAVE_TURN) ?
            ROBOT_GAIT_WAVE : ROBOT_GAIT_TRIPOD;  // 이번 구간에서 기대하는 보행 종류를 선택한다.

        user.s1 = mode[index];  // 다음 보행 선택을 사용자 입력에 적용한다.
        priority = ControlPriority_Step(&control, &user, true, false, &safety);  // 세 값의 보존 여부를 검사한다.
        output = DroneController_Step(&controller, &priority, contact, 0.4f);  // 전환 직후의 조종 명령을 계산한다.
        if ((priority.active_mode != ROBOT_MODE_MANUAL) || (priority.s1 != mode[index]) ||
            (output.gait_pattern != pattern) || !output.tripod_enable ||
            (fabsf(output.posture_reference_rad.yaw - 0.4f) > 0.00001f) ||
            (fabsf(output.vy_user_mps) >= 0.25f * ROBOT_MAX_LATERAL_SPEED_MPS) ||
            (fabsf(output.wz_user_radps) >= 0.25f * ROBOT_MAX_YAW_RATE_RADPS))
        {
            return false;
        }

        for (cycle = 0U; cycle < 200U; ++cycle)
        {
            output = DroneController_Step(&controller, &priority, contact, 0.4f);  // 보행 명령을 필터 정상상태까지 유지한다.
        }
        if ((fabsf(output.vx_user_mps - ROBOT_MAX_LINEAR_SPEED_MPS) > 0.00001f) ||
            (fabsf(output.vy_user_mps - (lateral ? -ROBOT_MAX_LATERAL_SPEED_MPS : 0.0f)) > 0.00001f) ||
            (fabsf(output.wz_user_radps - (lateral ? 0.0f : -ROBOT_MAX_YAW_RATE_RADPS)) > 0.00001f))
        {
            return false;
        }
    }
    priority.throttle = -1000;  // 중앙 한 발 보행의 후진 명령을 입력한다.
    priority.yaw = -1000;       // 중앙 한 발 보행의 반대 회전을 입력한다.
    for (cycle = 0U; cycle < 200U; ++cycle)
    {
        output = DroneController_Step(&controller, &priority, contact, 0.4f);  // 반대 방향 명령을 안정시킨다.
    }
    return (output.gait_pattern == ROBOT_GAIT_WAVE) &&
           (fabsf(output.vx_user_mps + ROBOT_MAX_LINEAR_SPEED_MPS) <= 0.00001f) &&
           (fabsf(output.vy_user_mps) <= 0.00001f) &&
           (fabsf(output.wz_user_radps - ROBOT_MAX_YAW_RATE_RADPS) <= 0.00001f);  // 한 발 보행의 후진과 양방향 회전을 확인한다.
}

/* 중앙 CRSF 보정값과 연결 끊김·중립 재허가를 검사한다. */
bool UserCommandTest_Run(void)
{
    UserCommand_Handle_t handle;                         // 시험용 변환 상태를 저장한다.
    RobotUserCommand_t command;                          // 변환 결과를 저장한다.
    uint16_t raw[ROBOT_CRSF_CHANNEL_COUNT];              // 명시적인 CRSF 채널 입력을 저장한다.
    uint32_t channel;                                    // 채널 초기화 번호를 저장한다.

    if (!UserCommandTest_CheckS1Boundaries() || !UserCommandTest_CheckWalkingSelection())
    {
        return false;
    }

    UserCommand_Init(&handle);  // 사용자 명령 상태를 준비한다.
    for (channel = 0U; channel < USER_COMMAND_USED_CHANNELS; ++channel)
    {
        if (!g_robot_calibration.crsf[channel].calibrated ||
            !UserCommand_SetCalibration(&handle, (uint8_t)channel,
                                        &g_robot_calibration.crsf[channel]))
        {
            return false;
        }
    }
    for (channel = 0U; channel < ROBOT_CRSF_CHANNEL_COUNT; ++channel)
    {
        raw[channel] = (channel < USER_COMMAND_USED_CHANNELS) ?
            g_robot_calibration.crsf[channel].raw_center : 992U;  // 실측 중립값으로 모든 채널을 준비한다.
    }

    UserCommand_UpdateChannels(&handle, raw, 1000U);  // 중립 프레임을 전달한다.
    UserCommand_UpdateTimeout(&handle, 1000U);        // 중립 시간을 시작한다.
    UserCommand_UpdateChannels(&handle, raw, 1200U);  // 연결을 유지한 중립 프레임을 전달한다.
    UserCommand_UpdateTimeout(&handle, 1200U);        // 0.2초 재허가를 완료한다.
    (void)UserCommand_Get(&handle, &command);          // 허가된 명령을 읽는다.
    if (!command.connected || !command.motion_armed || (command.s1 != ROBOT_WALK_WAVE_TURN))
    {
        return false;
    }

    raw[2] = (g_robot_calibration.crsf[2].direction > 0) ?
        g_robot_calibration.crsf[2].raw_max : g_robot_calibration.crsf[2].raw_min;  // CH3를 논리 최대로 둔다.
    raw[3] = (g_robot_calibration.crsf[3].direction > 0) ?
        g_robot_calibration.crsf[3].raw_min : g_robot_calibration.crsf[3].raw_max;  // CH4를 논리 최소로 둔다.
    raw[9] = (g_robot_calibration.crsf[9].direction > 0) ?
        g_robot_calibration.crsf[9].raw_max : g_robot_calibration.crsf[9].raw_min;  // CH10을 오른쪽으로 둔다.
    UserCommand_UpdateChannels(&handle, raw, 1205U);  // 최대값 시험 프레임을 전달한다.
    (void)UserCommand_Get(&handle, &command);          // 변환한 명령을 읽는다.
    if ((command.throttle != 1000) || (command.yaw != -1000) || (command.s1 != ROBOT_WALK_TRIPOD_LATERAL))
    {
        return false;
    }

    raw[6] = g_robot_calibration.crsf[6].raw_max;  // CH7 SC를 끝 위치로 둔다.
    raw[8] = g_robot_calibration.crsf[8].raw_min;  // CH9 SE를 해제한다.
    UserCommand_UpdateChannels(&handle, raw, 1210U);  // SC·SE 채널을 전달한다.
    (void)UserCommand_Get(&handle, &command);          // 스위치 변환 결과를 읽는다.
    if ((command.sc != 2U) || (command.se != 0U))
    {
        return false;
    }

    raw[6] = g_robot_calibration.crsf[6].raw_min;  // CH7 SC를 첫 위치로 둔다.
    raw[8] = g_robot_calibration.crsf[8].raw_max;  // CH9 SE를 누른다.
    UserCommand_UpdateChannels(&handle, raw, 1215U);  // 반대 SC·SE 상태를 전달한다.
    (void)UserCommand_Get(&handle, &command);          // 스위치 변환 결과를 읽는다.
    if ((command.sc != 0U) || (command.se != 1U))
    {
        return false;
    }

    UserCommand_UpdateTimeout(&handle, 1316U);  // 마지막 프레임 100 ms 이후를 검사한다.
    (void)UserCommand_Get(&handle, &command);    // Failsafe 명령을 읽는다.
    return !command.connected && !command.motion_armed &&
           (command.throttle == 0) && (command.yaw == 0);  // 연결 끊김 시 중립인지 확인한다.
}
