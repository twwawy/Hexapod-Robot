#include "measurement/adc_mapping_measurement.h"

#include "measurement/measurement_debug.h"

#include <stddef.h>
#include <string.h>

/* ADC 입력 배치를 모두 미확인 상태로 초기화한다. */
void AdcMappingMeasurement_Init(AdcMappingMeasurement_t *measurement)
{
    uint32_t leg;    // 초기화할 다리 번호를 저장한다.
    uint32_t input;  // 초기화할 입력 번호를 저장한다.

    if (measurement != NULL)
    {
        memset(measurement, 0, sizeof(*measurement));  // 이전 배선 기록을 제거한다.
        g_measurement_debug.calibration.adc_mapping_calibrated = false;  // 매핑 완료 상태를 초기화한다.
        for (leg = 0U; leg < MCP3008_LEG_COUNT; ++leg)
        {
            for (input = 0U; input < MCP3008_LEG_INPUT_COUNT; ++input)
            {
                g_measurement_debug.adc_mapping_recorded[leg][input] = false;  // 입력별 확인을 초기화한다.
            }
        }
    }
}

/* 움직이거나 누른 센서의 실제 MCP3008 채널을 기록한다. */
bool AdcMappingMeasurement_Record(AdcMappingMeasurement_t *measurement,
                                  uint8_t leg,
                                  MCP3008_LegInput_t input,
                                  uint8_t device,
                                  uint8_t channel)
{
    if ((measurement == NULL) || (leg >= MCP3008_LEG_COUNT) ||
        ((uint32_t)input >= MCP3008_LEG_INPUT_COUNT) ||
        (device >= MCP3008_DEVICE_COUNT) || (channel >= MCP3008_CHANNEL_COUNT))
    {
        return false;
    }

    measurement->table[leg][input].device = device;    // 확인한 장치 번호를 저장한다.
    measurement->table[leg][input].channel = channel;  // 확인한 채널 번호를 저장한다.
    measurement->recorded[leg][input] = true;          // 해당 입력 확인을 표시한다.
    g_measurement_debug.calibration.adc[leg][input].device = device;    // 디버거 장치 번호를 갱신한다.
    g_measurement_debug.calibration.adc[leg][input].channel = channel;  // 디버거 채널 번호를 갱신한다.
    g_measurement_debug.adc_mapping_recorded[leg][input] = true;        // 디버거 확인 상태를 갱신한다.
    g_measurement_debug.calibration.adc_mapping_calibrated =
        AdcMappingMeasurement_IsComplete(measurement);                  // 24개 매핑 완료 여부를 갱신한다.
    return true;
}

/* 24개 입력이 중복 없이 모두 기록됐는지 확인한다. */
bool AdcMappingMeasurement_IsComplete(const AdcMappingMeasurement_t *measurement)
{
    bool used[MCP3008_DEVICE_COUNT][MCP3008_CHANNEL_COUNT] = {{false}};  // 채널 중복을 표시한다.
    uint32_t leg;    // 검사할 다리 번호를 저장한다.
    uint32_t input;  // 검사할 입력 번호를 저장한다.

    if (measurement == NULL)
    {
        return false;
    }

    for (leg = 0U; leg < MCP3008_LEG_COUNT; ++leg)
    {
        for (input = 0U; input < MCP3008_LEG_INPUT_COUNT; ++input)
        {
            const MCP3008_InputMapping_t *mapping = &measurement->table[leg][input];  // 검사할 채널을 선택한다.

            if (!measurement->recorded[leg][input] ||
                used[mapping->device][mapping->channel])
            {
                return false;
            }
            used[mapping->device][mapping->channel] = true;  // 사용한 채널을 표시한다.
        }
    }

    return true;
}
