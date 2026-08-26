#include "test/user_command_test.h"

#include "common/robot_calibration.h"
#include "user_command/user_command.h"

#include <string.h>

/* 중앙 CRSF 보정값과 연결 끊김·중립 재허가를 검사한다. */
bool UserCommandTest_Run(void)
{
    UserCommand_Handle_t handle;                         // 시험용 변환 상태를 저장한다.
    RobotUserCommand_t command;                          // 변환 결과를 저장한다.
    uint16_t raw[ROBOT_CRSF_CHANNEL_COUNT];              // 명시적인 CRSF 채널 입력을 저장한다.
    uint32_t channel;                                    // 채널 초기화 번호를 저장한다.

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
    if (!command.connected || !command.motion_armed || (command.s1 != 0U))
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
    if ((command.throttle != 1000) || (command.yaw != -1000) || (command.s1 != 1U))
    {
        return false;
    }

    UserCommand_UpdateTimeout(&handle, 1306U);  // 마지막 프레임 100 ms 이후를 검사한다.
    (void)UserCommand_Get(&handle, &command);    // Failsafe 명령을 읽는다.
    return !command.connected && !command.motion_armed &&
           (command.throttle == 0) && (command.yaw == 0);  // 연결 끊김 시 중립인지 확인한다.
}
