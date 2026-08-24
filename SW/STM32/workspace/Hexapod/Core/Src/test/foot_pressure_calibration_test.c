#include "test/foot_pressure_calibration_test.h"

#include <stddef.h>
#include <string.h>

/* 압력센서 무부하·접촉 표본 누적값을 초기화한다. */
void FootPressureCalibrationTest_Init(FootPressureCalibrationTest_t *test)
{
    if (test != NULL)
    {
        memset(test, 0, sizeof(*test));  // 이전 압력 표본을 제거한다.
    }
}

/* 여섯 압력센서의 무부하 실제 raw 표본을 누적한다. */
void FootPressureCalibrationTest_AddUnloaded(FootPressureCalibrationTest_t *test,
                                             const uint16_t raw[ROBOT_PRESSURE_COUNT])
{
    uint32_t leg;  // 압력센서 번호를 저장한다.

    if ((test == NULL) || (raw == NULL) || (test->unloaded_count == UINT16_MAX))
    {
        return;
    }

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        test->unloaded_sum[leg] += raw[leg];  // 무부하 raw를 누적한다.
    }
    test->unloaded_count++;                   // 무부하 표본 수를 갱신한다.
}

/* 여섯 압력센서의 접촉 실제 raw 표본을 누적한다. */
void FootPressureCalibrationTest_AddLoaded(FootPressureCalibrationTest_t *test,
                                           const uint16_t raw[ROBOT_PRESSURE_COUNT])
{
    uint32_t leg;  // 압력센서 번호를 저장한다.

    if ((test == NULL) || (raw == NULL) || (test->loaded_count == UINT16_MAX))
    {
        return;
    }

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        test->loaded_sum[leg] += raw[leg];  // 접촉 raw를 누적한다.
    }
    test->loaded_count++;                   // 접촉 표본 수를 갱신한다.
}

/* 두 평균 사이의 40%·60% 지점으로 접촉 Hysteresis를 만든다. */
bool FootPressureCalibrationTest_Build(const FootPressureCalibrationTest_t *test,
                                       FootPressure_Calibration_t table[ROBOT_PRESSURE_COUNT])
{
    uint32_t leg;  // 보정표를 만들 센서 번호를 저장한다.

    if ((test == NULL) || (table == NULL) ||
        (test->unloaded_count == 0U) || (test->loaded_count == 0U))
    {
        return false;
    }

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        const uint16_t unloaded = (uint16_t)(test->unloaded_sum[leg] / test->unloaded_count);  // 무부하 평균을 계산한다.
        const uint16_t loaded = (uint16_t)(test->loaded_sum[leg] / test->loaded_count);        // 접촉 평균을 계산한다.
        const int32_t difference = (int32_t)loaded - (int32_t)unloaded;                        // 접촉 변화 방향을 계산한다.

        if ((difference > -10) && (difference < 10))
        {
            return false;
        }

        table[leg].active_high = (difference > 0);  // 접촉 시 raw 증가 여부를 저장한다.
        if (difference > 0)
        {
            table[leg].release_threshold = (uint16_t)((int32_t)unloaded + difference * 2 / 5);  // 낮은 해제 임계값을 만든다.
            table[leg].contact_threshold = (uint16_t)((int32_t)unloaded + difference * 3 / 5);  // 높은 접촉 임계값을 만든다.
        }
        else
        {
            table[leg].contact_threshold = (uint16_t)((int32_t)unloaded + difference * 3 / 5);  // 낮은 접촉 임계값을 만든다.
            table[leg].release_threshold = (uint16_t)((int32_t)unloaded + difference * 2 / 5);  // 높은 해제 임계값을 만든다.
        }
        table[leg].calibrated = true;  // 실측 완료를 표시한다.
    }

    return true;
}
