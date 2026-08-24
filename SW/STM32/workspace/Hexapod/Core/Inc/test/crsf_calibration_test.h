#ifndef CRSF_CALIBRATION_TEST_H
#define CRSF_CALIBRATION_TEST_H

#include "user_command/user_command.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    uint16_t minimum[USER_COMMAND_USED_CHANNELS];  // 채널별 관측 최소 raw를 저장한다.
    uint16_t center[USER_COMMAND_USED_CHANNELS];   // 채널별 중립 raw를 저장한다.
    uint16_t maximum[USER_COMMAND_USED_CHANNELS];  // 채널별 관측 최대 raw를 저장한다.
    bool center_captured;                          // 중립 측정 여부를 저장한다.
} CrsfCalibrationTest_t;

void CrsfCalibrationTest_Init(CrsfCalibrationTest_t *test);  // 실제 CRSF 범위 측정을 초기화한다.
void CrsfCalibrationTest_Update(CrsfCalibrationTest_t *test,
                                const uint16_t raw[ROBOT_CRSF_CHANNEL_COUNT]);  // 채널 최소·최대를 갱신한다.
void CrsfCalibrationTest_CaptureCenter(CrsfCalibrationTest_t *test,
                                       const uint16_t raw[ROBOT_CRSF_CHANNEL_COUNT]);  // 현재 중립과 스위치 중심을 기록한다.
bool CrsfCalibrationTest_Build(const CrsfCalibrationTest_t *test,
                               const int8_t direction[USER_COMMAND_USED_CHANNELS],
                               UserCommand_ChannelCalibration_t table[USER_COMMAND_USED_CHANNELS]);  // 실측 채널표를 만든다.

#endif
