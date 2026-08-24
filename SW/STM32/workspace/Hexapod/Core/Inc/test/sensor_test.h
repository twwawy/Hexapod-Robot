#ifndef SENSOR_TEST_H
#define SENSOR_TEST_H

#include "sensor/sensor_manager.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    SensorManager_Handle_t *sensors;                     // 실제 센서 통합기를 참조한다.
    RobotSensorSnapshot_t latest;                        // 최근 실제 센서값을 저장한다.
    uint16_t adc_min[MCP3008_DEVICE_COUNT][MCP3008_CHANNEL_COUNT];  // 채널별 최소 raw를 저장한다.
    uint16_t adc_max[MCP3008_DEVICE_COUNT][MCP3008_CHANNEL_COUNT];  // 채널별 최대 raw를 저장한다.
    uint32_t sample_count;                               // 정상 ADC 읽기 횟수를 저장한다.
    uint32_t read_error_count;                           // ADC 읽기 실패 횟수를 저장한다.
} SensorTest_Handle_t;

void SensorTest_Init(SensorTest_Handle_t *handle,
                     SensorManager_Handle_t *sensors);  // 실제 센서 기록 상태를 초기화한다.

bool SensorTest_Process(SensorTest_Handle_t *handle);   // 실제 센서값과 ADC 범위를 한 번 기록한다.

#endif
