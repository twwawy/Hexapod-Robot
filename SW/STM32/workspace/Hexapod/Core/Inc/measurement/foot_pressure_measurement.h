#ifndef FOOT_PRESSURE_MEASUREMENT_H
#define FOOT_PRESSURE_MEASUREMENT_H

#include "sensor/foot_pressure.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    uint32_t unloaded_sum[ROBOT_PRESSURE_COUNT];  // 무부하 raw 합계를 저장한다.
    uint32_t loaded_sum[ROBOT_PRESSURE_COUNT];    // 접촉 raw 합계를 저장한다.
    uint16_t unloaded_count;                      // 무부하 표본 수를 저장한다.
    uint16_t loaded_count;                        // 접촉 표본 수를 저장한다.
} FootPressureMeasurement_t;

void FootPressureMeasurement_Init(FootPressureMeasurement_t *measurement);  // 압력 표본 누적을 초기화한다.
void FootPressureMeasurement_AddUnloaded(FootPressureMeasurement_t *measurement,
                                         const uint16_t raw[ROBOT_PRESSURE_COUNT]);  // 무부하 표본을 추가한다.
void FootPressureMeasurement_AddLoaded(FootPressureMeasurement_t *measurement,
                                       const uint16_t raw[ROBOT_PRESSURE_COUNT]);  // 접촉 표본을 추가한다.
bool FootPressureMeasurement_Build(const FootPressureMeasurement_t *measurement,
                                   FootPressure_Calibration_t table[ROBOT_PRESSURE_COUNT]);  // Hysteresis 임계값을 만든다.

#endif
