#include "test/sensor_test.h"

#include <stddef.h>
#include <string.h>

/* 실제 센서 raw 최소·최대 기록을 준비한다. */
void SensorTest_Init(SensorTest_Handle_t *handle,
                     SensorManager_Handle_t *sensors)
{
    uint32_t device;  // ADC 장치 번호를 저장한다.
    uint32_t channel; // ADC 채널 번호를 저장한다.

    if (handle == NULL)
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));  // 이전 측정값을 제거한다.
    handle->sensors = sensors;           // 실제 센서 통합기를 연결한다.
    for (device = 0U; device < MCP3008_DEVICE_COUNT; ++device)
    {
        for (channel = 0U; channel < MCP3008_CHANNEL_COUNT; ++channel)
        {
            handle->adc_min[device][channel] = MCP3008_MAX_RAW_VALUE;  // 최소값 시작점을 최대 raw로 둔다.
        }
    }
}

/* GPS·WT931·MCP3008의 실제 최신값과 raw 범위를 기록한다. */
bool SensorTest_Process(SensorTest_Handle_t *handle)
{
    uint32_t device;  // ADC 장치 번호를 저장한다.
    uint32_t channel; // ADC 채널 번호를 저장한다.

    if ((handle == NULL) || (handle->sensors == NULL))
    {
        return false;
    }

    if (!SensorManager_Update(handle->sensors))
    {
        handle->read_error_count++;  // 실제 ADC 읽기 실패를 기록한다.
        return false;
    }

    (void)SensorManager_GetSnapshot(handle->sensors, &handle->latest);  // 최신 스냅샷을 보관한다.
    for (device = 0U; device < MCP3008_DEVICE_COUNT; ++device)
    {
        for (channel = 0U; channel < MCP3008_CHANNEL_COUNT; ++channel)
        {
            const uint16_t raw = handle->sensors->adc.raw[device][channel];  // 실제 채널 raw를 읽는다.
            if (raw < handle->adc_min[device][channel])
            {
                handle->adc_min[device][channel] = raw;  // 새 최소값을 저장한다.
            }
            if (raw > handle->adc_max[device][channel])
            {
                handle->adc_max[device][channel] = raw;  // 새 최대값을 저장한다.
            }
        }
    }

    handle->sample_count++;  // 정상 측정 횟수를 기록한다.
    return true;
}
