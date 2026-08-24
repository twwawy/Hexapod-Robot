#include "measurement/measurement_runner.h"

#include "measurement/measurement_debug.h"

#include <stddef.h>
#include <string.h>

/* 실측 순서를 첫 센서 Raw 단계로 초기화한다. */
void MeasurementRunner_Init(MeasurementRunner_t *runner)
{
    if (runner != NULL)
    {
        memset(runner, 0, sizeof(*runner));             // 이전 완료 기록을 제거한다.
        runner->stage = MEASUREMENT_STAGE_SENSOR_RAW;   // 첫 실측 단계부터 시작한다.
        MeasurementDebug_Reset();                       // 전역 디버그 실측값을 함께 초기화한다.
        g_measurement_debug.current_stage = (uint32_t)runner->stage;  // Live Expressions 단계를 갱신한다.
    }
}

/* 현재 실측 단계를 반환한다. */
MeasurementStage_t MeasurementRunner_GetStage(const MeasurementRunner_t *runner)
{
    return (runner == NULL) ? MEASUREMENT_STAGE_COUNT : runner->stage;  // 잘못된 Handle은 범위 밖으로 표시한다.
}

/* 사용자가 확인한 현재 단계를 완료하고 다음 단계로 이동한다. */
bool MeasurementRunner_CompleteCurrent(MeasurementRunner_t *runner)
{
    if ((runner == NULL) || (runner->stage >= MEASUREMENT_STAGE_COMPLETE))
    {
        return false;
    }

    runner->completed[runner->stage] = true;                         // 현재 단계 확인을 기록한다.
    g_measurement_debug.stage_completed[runner->stage] = true;       // 전역 디버그 완료 상태를 갱신한다.
    runner->stage = (MeasurementStage_t)((uint32_t)runner->stage + 1U); // 정해진 다음 단계로 이동한다.
    g_measurement_debug.current_stage = (uint32_t)runner->stage;      // Live Expressions 단계를 갱신한다.
    return true;
}
