#include "measurement/crsf_calibration_measurement.h"

#include "measurement/measurement_debug.h"

#include <stddef.h>
#include <string.h>

/* 실제 CRSF 채널 범위 기록을 초기화한다. */
void CrsfCalibrationMeasurement_Init(CrsfCalibrationMeasurement_t *measurement)
{
    uint32_t channel;  // 초기화할 채널 번호를 저장한다.

    if (measurement == NULL)
    {
        return;
    }

    memset(measurement, 0, sizeof(*measurement));  // 이전 채널 기록을 제거한다.
    g_measurement_debug.crsf_center_captured = false;  // 디버거 중립 기록을 초기화한다.
    for (channel = 0U; channel < USER_COMMAND_USED_CHANNELS; ++channel)
    {
        measurement->minimum[channel] = UINT16_MAX;  // 첫 raw를 최소값으로 받을 준비를 한다.
        g_measurement_debug.crsf_minimum[channel] = UINT16_MAX;          // 디버거 최소값을 준비한다.
        g_measurement_debug.crsf_center[channel] = 0U;                   // 디버거 중립값을 초기화한다.
        g_measurement_debug.crsf_maximum[channel] = 0U;                  // 디버거 최대값을 초기화한다.
        g_measurement_debug.calibration.crsf[channel].calibrated = false; // CRSF 완료를 초기화한다.
    }
}

/* CH1~CH10의 실제 최소·최대 raw를 갱신한다. */
void CrsfCalibrationMeasurement_Update(CrsfCalibrationMeasurement_t *measurement,
                                       const uint16_t raw[ROBOT_CRSF_CHANNEL_COUNT])
{
    uint32_t channel;  // 측정할 채널 번호를 저장한다.

    if ((measurement == NULL) || (raw == NULL))
    {
        return;
    }

    for (channel = 0U; channel < USER_COMMAND_USED_CHANNELS; ++channel)
    {
        if (raw[channel] < measurement->minimum[channel])
        {
            measurement->minimum[channel] = raw[channel];  // 새 최소 raw를 저장한다.
            g_measurement_debug.crsf_minimum[channel] = raw[channel];  // 디버거 최소값을 갱신한다.
        }
        if (raw[channel] > measurement->maximum[channel])
        {
            measurement->maximum[channel] = raw[channel];  // 새 최대 raw를 저장한다.
            g_measurement_debug.crsf_maximum[channel] = raw[channel];  // 디버거 최대값을 갱신한다.
        }
    }
}

/* 네 짐벌 중립과 현재 스위치 위치의 실제 raw를 저장한다. */
void CrsfCalibrationMeasurement_CaptureCenter(CrsfCalibrationMeasurement_t *measurement,
                                              const uint16_t raw[ROBOT_CRSF_CHANNEL_COUNT])
{
    uint32_t channel;  // 기록할 CRSF 채널 번호를 저장한다.

    if ((measurement != NULL) && (raw != NULL))
    {
        memcpy(measurement->center, raw, sizeof(measurement->center));  // CH1~CH10 현재 raw를 복사한다.
        measurement->center_captured = true;                            // 중심 위치 측정을 표시한다.
        for (channel = 0U; channel < USER_COMMAND_USED_CHANNELS; ++channel)
        {
            g_measurement_debug.crsf_center[channel] = raw[channel];  // 디버거 중립값을 갱신한다.
        }
        g_measurement_debug.crsf_center_captured = true;  // 디버거 중립 기록을 표시한다.
    }
}

/* 관측 범위와 방향·스위치 논리값으로 CRSF 보정표를 만든다. */
bool CrsfCalibrationMeasurement_Build(const CrsfCalibrationMeasurement_t *measurement,
                                      const int8_t direction[USER_COMMAND_USED_CHANNELS],
                                      const uint8_t switch_map[USER_COMMAND_USED_CHANNELS][3],
                                      UserCommand_ChannelCalibration_t table[USER_COMMAND_USED_CHANNELS])
{
    uint32_t channel;  // 보정할 채널 번호를 저장한다.

    if ((measurement == NULL) || (direction == NULL) ||
        (switch_map == NULL) || (table == NULL) || !measurement->center_captured)
    {
        return false;
    }

    for (channel = 0U; channel < USER_COMMAND_USED_CHANNELS; ++channel)
    {
        const uint16_t center = (channel < 4U) ? measurement->center[channel] :
            (uint16_t)(((uint32_t)measurement->minimum[channel] +
                        measurement->maximum[channel]) / 2U);  // 스위치는 관측 범위 중간을 경계 기준으로 둔다.

        if ((measurement->minimum[channel] >= center) ||
            (center >= measurement->maximum[channel]) ||
            ((direction[channel] != 1) && (direction[channel] != -1)))
        {
            return false;
        }

        table[channel].raw_min = measurement->minimum[channel];       // 실측 최소 raw를 저장한다.
        table[channel].raw_center = center;                            // 짐벌 중립 또는 스위치 경계를 저장한다.
        table[channel].raw_max = measurement->maximum[channel];       // 실측 최대 raw를 저장한다.
        table[channel].direction = direction[channel];                // 실측 방향을 저장한다.
        memcpy(table[channel].switch_map, switch_map[channel],
               sizeof(table[channel].switch_map));                    // 스위치 논리 대응을 저장한다.
        table[channel].calibrated = true;                              // 실측 완료를 표시한다.
        g_measurement_debug.calibration.crsf[channel] = table[channel]; // 디버거 최종 CRSF 값을 갱신한다.
    }

    return true;
}
