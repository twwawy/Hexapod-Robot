#ifndef USER_COMMAND_H
#define USER_COMMAND_H

#include "common/robot_types.h"

#include <stdbool.h>
#include <stdint.h>

#define USER_COMMAND_USED_CHANNELS 10U

typedef struct
{
    uint16_t raw_min;      // 채널 최소값을 저장한다.
    uint16_t raw_center;   // 채널 중립값을 저장한다.
    uint16_t raw_max;      // 채널 최대값을 저장한다.
    int8_t direction;      // 채널 방향을 저장한다.
    uint8_t switch_map[3]; // 스위치 Low·Mid·High 논리값을 저장한다.
    bool calibrated;      // 실측 완료 여부를 저장한다.
} UserCommand_ChannelCalibration_t;

typedef struct
{
    UserCommand_ChannelCalibration_t channel[USER_COMMAND_USED_CHANNELS];  // CH1~CH10 보정값을 저장한다.
    RobotUserCommand_t command;                                           // 최근 사용자 명령을 저장한다.
    uint32_t neutral_start_ms;                                            // 재연결 중립 시작 시각을 저장한다.
    bool neutral_timing;                                                  // 중립 시간 측정 여부를 저장한다.
} UserCommand_Handle_t;

void UserCommand_Init(UserCommand_Handle_t *handle);  // 이상적인 CRSF 기본 테이블을 준비한다.

bool UserCommand_SetCalibration(UserCommand_Handle_t *handle,
                                uint8_t channel,
                                const UserCommand_ChannelCalibration_t *calibration);  // 한 채널 보정값을 갱신한다.

void UserCommand_UpdateChannels(UserCommand_Handle_t *handle,
                                const uint16_t raw[ROBOT_CRSF_CHANNEL_COUNT],
                                uint32_t now_ms);  // 정상 CRSF 채널을 프로젝트 명령으로 변환한다.

void UserCommand_UpdateTimeout(UserCommand_Handle_t *handle,
                               uint32_t now_ms);   // CRSF 연결 끊김과 재허가를 갱신한다.

bool UserCommand_Get(const UserCommand_Handle_t *handle,
                     RobotUserCommand_t *command);  // 현재 안전한 사용자 명령을 반환한다.

#endif
