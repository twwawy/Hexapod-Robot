#include "measurement/sensor_raw_measurement.h"

#include "measurement/measurement_debug.h"

#include <stddef.h>
#include <string.h>

/* 센서 통신과 ADC raw 범위 기록을 초기화한다. */
void SensorRawMeasurement_Init(SensorRawMeasurement_t *measurement,
                               SensorManager_Handle_t *sensors)
{
    uint32_t device;   // ADC 장치 번호를 저장한다.
    uint32_t channel;  // ADC 채널 번호를 저장한다.

    if (measurement == NULL)
    {
        return;
    }

    memset(measurement, 0, sizeof(*measurement));  // 이전 측정값을 제거한다.
    measurement->sensors = sensors;                // 실제 센서 통합기를 연결한다.
    g_measurement_debug.sensor_sample_count = 0U;  // 전역 정상 측정 횟수를 초기화한다.
    g_measurement_debug.sensor_error_count = 0U;   // 전역 오류 횟수를 초기화한다.
    for (device = 0U; device < MCP3008_DEVICE_COUNT; ++device)
    {
        for (channel = 0U; channel < MCP3008_CHANNEL_COUNT; ++channel)
        {
            measurement->adc_min[device][channel] = MCP3008_MAX_RAW_VALUE;  // 첫 최소값을 받을 준비를 한다.
            g_measurement_debug.adc_raw[device][channel] = 0U;               // 최근 raw를 초기화한다.
            g_measurement_debug.adc_min[device][channel] = MCP3008_MAX_RAW_VALUE; // 전역 최소값을 준비한다.
            g_measurement_debug.adc_max[device][channel] = 0U;               // 전역 최대값을 초기화한다.
        }
    }
}

/* GPS·WT931·MCP3008의 실제 최신값과 raw 범위를 기록한다. */
bool SensorRawMeasurement_Sample(SensorRawMeasurement_t *measurement)
{
    uint32_t device;   // ADC 장치 번호를 저장한다.
    uint32_t channel;  // ADC 채널 번호를 저장한다.

    if ((measurement == NULL) || (measurement->sensors == NULL))
    {
        return false;
    }

    if (!SensorManager_Update(measurement->sensors))
    {
        measurement->read_error_count++;  // 실제 ADC 읽기 실패를 기록한다.
        g_measurement_debug.sensor_error_count = measurement->read_error_count;  // 디버거 오류 횟수를 갱신한다.
        return false;
    }

    (void)SensorManager_GetSnapshot(measurement->sensors, &measurement->latest);  // 최신 스냅샷을 보관한다.
    g_measurement_debug.latest_sensor = measurement->latest;                      // 디버거 최신 센서값을 갱신한다.
    for (device = 0U; device < MCP3008_DEVICE_COUNT; ++device)
    {
        for (channel = 0U; channel < MCP3008_CHANNEL_COUNT; ++channel)
        {
            const uint16_t raw = measurement->sensors->adc.raw[device][channel];  // 실제 채널 raw를 읽는다.

            g_measurement_debug.adc_raw[device][channel] = raw;  // 디버거 최근 raw를 갱신한다.

            if (raw < measurement->adc_min[device][channel])
            {
                measurement->adc_min[device][channel] = raw;  // 새 최소값을 저장한다.
                g_measurement_debug.adc_min[device][channel] = raw;  // 디버거 최소값을 갱신한다.
            }
            if (raw > measurement->adc_max[device][channel])
            {
                measurement->adc_max[device][channel] = raw;  // 새 최대값을 저장한다.
                g_measurement_debug.adc_max[device][channel] = raw;  // 디버거 최대값을 갱신한다.
            }
        }
    }

    measurement->sample_count++;  // 정상 측정 횟수를 기록한다.
    g_measurement_debug.sensor_sample_count = measurement->sample_count;  // 디버거 정상 측정 횟수를 갱신한다.
    return true;
}
