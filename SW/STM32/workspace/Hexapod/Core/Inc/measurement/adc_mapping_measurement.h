#ifndef ADC_MAPPING_MEASUREMENT_H
#define ADC_MAPPING_MEASUREMENT_H

#include "sensor/mcp3008.h"

#include <stdbool.h>

typedef struct
{
    MCP3008_InputMapping_t table[MCP3008_LEG_COUNT][MCP3008_LEG_INPUT_COUNT];  // 확인한 입력 배치를 저장한다.
    bool recorded[MCP3008_LEG_COUNT][MCP3008_LEG_INPUT_COUNT];                 // 입력별 확인 여부를 저장한다.
} AdcMappingMeasurement_t;

void AdcMappingMeasurement_Init(AdcMappingMeasurement_t *measurement);  // 미확인 ADC 배치를 초기화한다.
bool AdcMappingMeasurement_Record(AdcMappingMeasurement_t *measurement,
                                  uint8_t leg,
                                  MCP3008_LegInput_t input,
                                  uint8_t device,
                                  uint8_t channel);  // 움직인 센서의 실제 채널을 기록한다.
bool AdcMappingMeasurement_IsComplete(const AdcMappingMeasurement_t *measurement);  // 24개 입력 확인 여부를 검사한다.

#endif
