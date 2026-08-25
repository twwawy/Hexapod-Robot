#include "user_command/user_command.h"

#include <stddef.h>
#include <string.h>

#define CRSF_IDEAL_MIN_RAW    172U
#define CRSF_IDEAL_CENTER_RAW 992U
#define CRSF_IDEAL_MAX_RAW    1811U

/* 정수 입력을 지정한 범위로 제한한다. */
static int32_t UserCommand_ClampI32(int32_t value, int32_t minimum, int32_t maximum)
{
    if (value < minimum)
    {
        return minimum;
    }
    if (value > maximum)
    {
        return maximum;
    }
    return value;
}

/* 한 CRSF 채널을 -1000~1000으로 정규화한다. */
static int16_t UserCommand_Normalize(const UserCommand_ChannelCalibration_t *calibration,
                                     uint16_t raw)
{
    int32_t normalized;   // 정규화한 채널 값을 저장한다.

    if ((calibration == NULL) ||
        (calibration->raw_min >= calibration->raw_center) ||
        (calibration->raw_center >= calibration->raw_max))
    {
        return 0;
    }

    if (raw >= calibration->raw_center)
    {
        normalized = ((int32_t)raw - (int32_t)calibration->raw_center) * 1000 /
                     ((int32_t)calibration->raw_max - (int32_t)calibration->raw_center);  // 양수 구간을 변환한다.
    }
    else
    {
        normalized = -((int32_t)calibration->raw_center - (int32_t)raw) * 1000 /
                     ((int32_t)calibration->raw_center - (int32_t)calibration->raw_min);  // 음수 구간을 변환한다.
    }

    normalized = UserCommand_ClampI32(normalized, -1000, 1000);  // 이상적인 출력 범위로 제한한다.
    normalized *= (calibration->direction < 0) ? -1 : 1;         // 실측 채널 방향을 적용한다.
    return (int16_t)normalized;
}

/* 한 CRSF 채널을 2단 또는 3단 스위치 상태로 변환한다. */
static uint8_t UserCommand_MapSwitch(const UserCommand_ChannelCalibration_t *calibration,
                                     uint16_t raw,
                                     bool three_position)
{
    const uint16_t low_boundary = (uint16_t)(((uint32_t)calibration->raw_min +
                                               calibration->raw_center) / 2U);  // Low와 Mid 경계를 계산한다.
    const uint16_t high_boundary = (uint16_t)(((uint32_t)calibration->raw_center +
                                                calibration->raw_max) / 2U);    // Mid와 High 경계를 계산한다.
    uint8_t physical_position;   // 물리 스위치 위치를 저장한다.

    if (!three_position)
    {
        physical_position = (raw >= calibration->raw_center) ? 2U : 0U;  // 2단 스위치를 Low/High로 나눈다.
    }
    else if (raw < low_boundary)
    {
        physical_position = 0U;  // Low 위치를 선택한다.
    }
    else if (raw > high_boundary)
    {
        physical_position = 2U;  // High 위치를 선택한다.
    }
    else
    {
        physical_position = 1U;  // Mid 위치를 선택한다.
    }

    return calibration->switch_map[physical_position];  // 실측 논리값으로 변환한다.
}

/* S1 위치를 중앙 기준의 두 이동 방식으로 변환한다. */
static uint8_t UserCommand_MapS1Mode(const UserCommand_ChannelCalibration_t *calibration,
                                     uint16_t raw)
{
    bool right_side;  // 보정 방향을 적용한 오른쪽 위치를 저장한다.

    if (calibration == NULL)
    {
        return 0U;
    }

    if (raw == calibration->raw_center)
    {
        return 0U;  // 정확한 중앙값은 기본 회전 방식으로 둔다.
    }

    right_side = (raw > calibration->raw_center);  // 중앙보다 큰 쪽을 오른쪽으로 해석한다.
    if (calibration->direction < 0)
    {
        right_side = !right_side;  // 반전 채널이면 물리 방향을 바로잡는다.
    }

    return right_side ? 1U : 0U;
}

/* 현재 네 짐벌이 중립 범위인지 확인한다. */
static bool UserCommand_IsNeutral(const RobotUserCommand_t *command)
{
    return (command != NULL) &&
           (command->throttle >= -ROBOT_THROTTLE_DEADBAND) &&
           (command->throttle <= ROBOT_THROTTLE_DEADBAND) &&
           (command->yaw >= -ROBOT_STICK_DEADBAND) &&
           (command->yaw <= ROBOT_STICK_DEADBAND) &&
           (command->roll >= -ROBOT_STICK_DEADBAND) &&
           (command->roll <= ROBOT_STICK_DEADBAND) &&
           (command->pitch >= -ROBOT_STICK_DEADBAND) &&
           (command->pitch <= ROBOT_STICK_DEADBAND);  // 네 축의 중립을 함께 확인한다.
}

/* 이상적인 CRSF 범위를 사용하는 초기 보정 테이블을 준비한다. */
void UserCommand_Init(UserCommand_Handle_t *handle)
{
    uint32_t channel;   // 초기화할 채널 번호를 저장한다.

    if (handle == NULL)
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));  // 이전 사용자 입력 상태를 제거한다.

    for (channel = 0U; channel < USER_COMMAND_USED_CHANNELS; ++channel)
    {
        handle->channel[channel].raw_min = CRSF_IDEAL_MIN_RAW;        // 이상적인 최소값을 넣는다.
        handle->channel[channel].raw_center = CRSF_IDEAL_CENTER_RAW;  // 이상적인 중립값을 넣는다.
        handle->channel[channel].raw_max = CRSF_IDEAL_MAX_RAW;        // 이상적인 최대값을 넣는다.
        handle->channel[channel].direction = 1;                       // 기본 채널 방향을 유지한다.
        handle->channel[channel].switch_map[0] = 0U;                  // Low 위치를 0으로 둔다.
        handle->channel[channel].switch_map[1] = 1U;                  // Mid 위치를 1로 둔다.
        handle->channel[channel].switch_map[2] = 2U;                  // High 위치를 2로 둔다.
        handle->channel[channel].calibrated = false;                  // 실측 전 상태로 표시한다.
    }
}

/* 한 CRSF 채널의 실측 보정값을 설정한다. */
bool UserCommand_SetCalibration(UserCommand_Handle_t *handle,
                                uint8_t channel,
                                const UserCommand_ChannelCalibration_t *calibration)
{
    if ((handle == NULL) || (calibration == NULL) ||
        (channel >= USER_COMMAND_USED_CHANNELS) ||
        (calibration->raw_min >= calibration->raw_center) ||
        (calibration->raw_center >= calibration->raw_max))
    {
        return false;
    }

    handle->channel[channel] = *calibration;  // 선택한 채널 테이블을 갱신한다.
    return true;
}

/* 정상 RC 채널을 프로젝트 사용자 명령으로 변환한다. */
void UserCommand_UpdateChannels(UserCommand_Handle_t *handle,
                                const uint16_t raw[ROBOT_CRSF_CHANNEL_COUNT],
                                uint32_t now_ms)
{
    if ((handle == NULL) || (raw == NULL))
    {
        return;
    }

    handle->command.roll = UserCommand_Normalize(&handle->channel[0], raw[0]);          // CH1 Roll을 변환한다.
    handle->command.pitch = UserCommand_Normalize(&handle->channel[1], raw[1]);         // CH2 Pitch를 변환한다.
    handle->command.throttle = UserCommand_Normalize(&handle->channel[2], raw[2]);      // CH3 Throttle을 변환한다.
    handle->command.yaw = UserCommand_Normalize(&handle->channel[3], raw[3]);           // CH4 Yaw를 변환한다.
    handle->command.sa = UserCommand_MapSwitch(&handle->channel[4], raw[4], false);     // CH5 SA를 변환한다.
    handle->command.sb = UserCommand_MapSwitch(&handle->channel[5], raw[5], true);      // CH6 SB를 변환한다.
    handle->command.sc = UserCommand_MapSwitch(&handle->channel[6], raw[6], true);      // CH7 SC를 변환한다.
    handle->command.sd = UserCommand_MapSwitch(&handle->channel[7], raw[7], false);     // CH8 SD를 변환한다.
    handle->command.se = UserCommand_MapSwitch(&handle->channel[8], raw[8], false);     // CH9 SE를 변환한다.
    handle->command.s1 = UserCommand_MapS1Mode(&handle->channel[9], raw[9]);             // CH10 S1 이동 방식을 변환한다.
    handle->command.timestamp_ms = now_ms;                                              // 정상 프레임 시각을 기록한다.
    handle->command.connected = true;                                                   // CRSF 연결을 표시한다.
}

/* CRSF Timeout과 재연결 중립 유지 조건을 갱신한다. */
void UserCommand_UpdateTimeout(UserCommand_Handle_t *handle,
                               uint32_t now_ms)
{
    if (handle == NULL)
    {
        return;
    }

    if (!handle->command.connected ||
        ((now_ms - handle->command.timestamp_ms) > ROBOT_CRSF_TIMEOUT_MS))
    {
        handle->command.connected = false;      // 연결 끊김을 표시한다.
        handle->command.motion_armed = false;   // 사용자 동작 허가를 해제한다.
        handle->neutral_timing = false;         // 재허가 시간 측정을 초기화한다.
        handle->command.throttle = 0;           // 이동 명령을 제거한다.
        handle->command.yaw = 0;                // 회전 명령을 제거한다.
        handle->command.roll = 0;               // Roll 명령을 제거한다.
        handle->command.pitch = 0;              // Pitch 명령을 제거한다.
        return;
    }

    if (!handle->command.motion_armed && UserCommand_IsNeutral(&handle->command))
    {
        if (!handle->neutral_timing)
        {
            handle->neutral_start_ms = now_ms;  // 중립 유지 시작 시각을 기록한다.
            handle->neutral_timing = true;      // 중립 시간 측정을 시작한다.
        }
        else if ((now_ms - handle->neutral_start_ms) >= ROBOT_CRSF_REARM_MS)
        {
            handle->command.motion_armed = true;  // 중립 유지 후 입력을 허가한다.
        }
    }
    else if (!UserCommand_IsNeutral(&handle->command))
    {
        handle->neutral_timing = false;  // 중립이 깨지면 측정을 다시 시작한다.
    }
}

/* 현재 연결과 재허가 조건을 반영한 사용자 명령을 반환한다. */
bool UserCommand_Get(const UserCommand_Handle_t *handle,
                     RobotUserCommand_t *command)
{
    if ((handle == NULL) || (command == NULL))
    {
        return false;
    }

    *command = handle->command;  // 최신 사용자 입력을 복사한다.

    if (!command->connected || !command->motion_armed)
    {
        command->throttle = 0;  // 허가 전 Throttle을 차단한다.
        command->yaw = 0;       // 허가 전 Yaw를 차단한다.
        command->roll = 0;      // 허가 전 Roll을 차단한다.
        command->pitch = 0;     // 허가 전 Pitch를 차단한다.
    }

    return command->connected;
}
