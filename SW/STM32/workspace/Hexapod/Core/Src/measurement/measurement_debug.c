#include "measurement/measurement_debug.h"

#if defined(__GNUC__)
volatile MeasurementDebug_t g_measurement_debug
    __attribute__((used, section(".bss.measurement_debug"))) = {0};  // 디버그 전용 BSS에 유지한다.
#else
volatile MeasurementDebug_t g_measurement_debug;  // Live Expressions에서 유지할 실측 결과를 정의한다.
#endif

/* 전역 실측 결과를 모두 0으로 초기화한다. */
void MeasurementDebug_Reset(void)
{
    const MeasurementDebug_t empty = {0};  // 초기 상태 전체를 준비한다.

    g_measurement_debug = empty;  // 이전 실측값을 한 번에 제거한다.
}
