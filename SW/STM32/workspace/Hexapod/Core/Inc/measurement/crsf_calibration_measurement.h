#ifndef CRSF_CALIBRATION_MEASUREMENT_H
#define CRSF_CALIBRATION_MEASUREMENT_H

#include "user_command/user_command.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    uint16_t minimum[USER_COMMAND_USED_CHANNELS];  // 채널별 관측 최소 raw를 저장한다.
    uint16_t center[USER_COMMAND_USED_CHANNELS];   // 채널별 중립 raw를 저장한다.
    uint16_t maximum[USER_COMMAND_USED_CHANNELS];  // 채널별 관측 최대 raw를 저장한다.
    bool center_captured;                          // 중립 측정 여부를 저장한다.
} CrsfCalibrationMeasurement_t;

void CrsfCalibrationMeasurement_Init(CrsfCalibrationMeasurement_t *measurement);  // 실제 CRSF 범위 측정을 초기화한다.
void CrsfCalibrationMeasurement_Update(CrsfCalibrationMeasurement_t *measurement,
                                       const uint16_t raw[ROBOT_CRSF_CHANNEL_COUNT]);  // 채널 최소·최대를 갱신한다.
void CrsfCalibrationMeasurement_CaptureCenter(CrsfCalibrationMeasurement_t *measurement,
                                              const uint16_t raw[ROBOT_CRSF_CHANNEL_COUNT]);  // 현재 중립을 기록한다.
bool CrsfCalibrationMeasurement_Build(const CrsfCalibrationMeasurement_t *measurement,
                                      const int8_t direction[USER_COMMAND_USED_CHANNELS],
                                      const uint8_t switch_map[USER_COMMAND_USED_CHANNELS][3],
                                      UserCommand_ChannelCalibration_t table[USER_COMMAND_USED_CHANNELS]);  // 실측 채널표를 만든다.

#endif
