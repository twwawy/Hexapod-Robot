#include "test/user_command_test.h"

#include "user_command/user_command.h"

#include <string.h>

/* 이상적인 CRSF 범위와 연결 끊김·중립 재허가를 검사한다. */
bool UserCommandTest_Run(void)
{
    UserCommand_Handle_t handle;                         // 시험용 변환 상태를 저장한다.
    RobotUserCommand_t command;                          // 변환 결과를 저장한다.
    uint16_t raw[ROBOT_CRSF_CHANNEL_COUNT];              // 명시적인 CRSF 채널 입력을 저장한다.
    uint32_t channel;                                    // 채널 초기화 번호를 저장한다.

    UserCommand_Init(&handle);  // 이상적인 보정표를 준비한다.
    for (channel = 0U; channel < ROBOT_CRSF_CHANNEL_COUNT; ++channel)
    {
        raw[channel] = 992U;  // 모든 채널을 중립에 둔다.
    }

    UserCommand_UpdateChannels(&handle, raw, 1000U);  // 중립 프레임을 전달한다.
    UserCommand_UpdateTimeout(&handle, 1000U);        // 중립 시간을 시작한다.
    UserCommand_UpdateTimeout(&handle, 1200U);        // 0.2초 재허가를 완료한다.
    (void)UserCommand_Get(&handle, &command);          // 허가된 명령을 읽는다.
    if (!command.connected || !command.motion_armed)
    {
        return false;
    }

    raw[2] = 1811U;                                    // CH3 Throttle을 최대값으로 둔다.
    raw[3] = 172U;                                     // CH4 Yaw를 최소값으로 둔다.
    UserCommand_UpdateChannels(&handle, raw, 1205U);  // 최대값 시험 프레임을 전달한다.
    (void)UserCommand_Get(&handle, &command);          // 변환한 명령을 읽는다.
    if ((command.throttle != 1000) || (command.yaw != -1000))
    {
        return false;
    }

    UserCommand_UpdateTimeout(&handle, 1306U);  // 마지막 프레임 100 ms 이후를 검사한다.
    (void)UserCommand_Get(&handle, &command);    // Failsafe 명령을 읽는다.
    return !command.connected && !command.motion_armed &&
           (command.throttle == 0) && (command.yaw == 0);  // 연결 끊김 시 중립인지 확인한다.
}
