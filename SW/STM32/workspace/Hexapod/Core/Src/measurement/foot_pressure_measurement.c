#include "measurement/foot_pressure_measurement.h"

#include "measurement/measurement_debug.h"

#include <stddef.h>
#include <string.h>

/* 압력센서 무부하·접촉 표본 누적값을 초기화한다. */
void FootPressureMeasurement_Init(FootPressureMeasurement_t *measurement)
{
    uint32_t leg;  // 초기화할 압력센서 번호를 저장한다.

    if (measurement != NULL)
    {
        memset(measurement, 0, sizeof(*measurement));  // 이전 압력 표본을 제거한다.
        g_measurement_debug.pressure_unloaded_count = 0U;  // 디버거 무부하 표본 수를 초기화한다.
        g_measurement_debug.pressure_loaded_count = 0U;    // 디버거 접촉 표본 수를 초기화한다.
        for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
        {
            g_measurement_debug.pressure_unloaded_sum[leg] = 0U;             // 무부하 누적값을 초기화한다.
            g_measurement_debug.pressure_loaded_sum[leg] = 0U;               // 접촉 누적값을 초기화한다.
            g_measurement_debug.calibration.pressure[leg].calibrated = false; // 압력 보정 완료를 초기화한다.
        }
    }
}

/* 여섯 압력센서의 무부하 실제 raw 표본을 누적한다. */
void FootPressureMeasurement_AddUnloaded(FootPressureMeasurement_t *measurement,
                                         const uint16_t raw[ROBOT_PRESSURE_COUNT])
{
    uint32_t leg;  // 압력센서 번호를 저장한다.

    if ((measurement == NULL) || (raw == NULL) ||
        (measurement->unloaded_count == UINT16_MAX))
    {
        return;
    }

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        measurement->unloaded_sum[leg] += raw[leg];  // 무부하 raw를 누적한다.
        g_measurement_debug.pressure_unloaded_sum[leg] =
            measurement->unloaded_sum[leg];          // 디버거 무부하 누적값을 갱신한다.
    }
    measurement->unloaded_count++;  // 무부하 표본 수를 갱신한다.
    g_measurement_debug.pressure_unloaded_count = measurement->unloaded_count;  // 디버거 표본 수를 갱신한다.
}

/* 여섯 압력센서의 접촉 실제 raw 표본을 누적한다. */
void FootPressureMeasurement_AddLoaded(FootPressureMeasurement_t *measurement,
                                       const uint16_t raw[ROBOT_PRESSURE_COUNT])
{
    uint32_t leg;  // 압력센서 번호를 저장한다.

    if ((measurement == NULL) || (raw == NULL) ||
        (measurement->loaded_count == UINT16_MAX))
    {
        return;
    }

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        measurement->loaded_sum[leg] += raw[leg];  // 접촉 raw를 누적한다.
        g_measurement_debug.pressure_loaded_sum[leg] =
            measurement->loaded_sum[leg];          // 디버거 접촉 누적값을 갱신한다.
    }
    measurement->loaded_count++;  // 접촉 표본 수를 갱신한다.
    g_measurement_debug.pressure_loaded_count = measurement->loaded_count;  // 디버거 표본 수를 갱신한다.
}

/* 두 평균 사이의 40%·60% 지점으로 접촉 Hysteresis를 만든다. */
bool FootPressureMeasurement_Build(const FootPressureMeasurement_t *measurement,
                                   FootPressure_Calibration_t table[ROBOT_PRESSURE_COUNT])
{
    uint32_t leg;  // 보정표를 만들 센서 번호를 저장한다.

    if ((measurement == NULL) || (table == NULL) ||
        (measurement->unloaded_count == 0U) || (measurement->loaded_count == 0U))
    {
        return false;
    }

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        const uint16_t unloaded = (uint16_t)(measurement->unloaded_sum[leg] /
                                             measurement->unloaded_count);  // 무부하 평균을 계산한다.
        const uint16_t loaded = (uint16_t)(measurement->loaded_sum[leg] /
                                           measurement->loaded_count);      // 접촉 평균을 계산한다.
        const int32_t difference = (int32_t)loaded - (int32_t)unloaded;      // 접촉 변화 방향을 계산한다.

        if ((difference > -10) && (difference < 10))
        {
            return false;
        }

        table[leg].active_high = (difference > 0);  // 접촉 시 raw 증가 여부를 저장한다.
        if (difference > 0)
        {
            table[leg].release_threshold = (uint16_t)((int32_t)unloaded + difference * 2 / 5);  // 해제 임계값을 만든다.
            table[leg].contact_threshold = (uint16_t)((int32_t)unloaded + difference * 3 / 5);  // 접촉 임계값을 만든다.
        }
        else
        {
            table[leg].contact_threshold = (uint16_t)((int32_t)unloaded + difference * 3 / 5);  // 반전 접촉 임계값을 만든다.
            table[leg].release_threshold = (uint16_t)((int32_t)unloaded + difference * 2 / 5);  // 반전 해제 임계값을 만든다.
        }
        table[leg].calibrated = true;  // 실측 완료를 표시한다.
        g_measurement_debug.calibration.pressure[leg] = table[leg];  // 디버거 최종 임계값을 갱신한다.
    }

    return true;
}
