#include "app/pressure_load_calibration.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

#define PRESSURE_LOAD_WAIT_UNLOADED_MS       500U  // 시작 직후 센서 안정화 시간을 정의한다.
#define PRESSURE_LOAD_CAPTURE_UNLOADED_MS   1000U  // 시작 자세 평균 측정 시간을 정의한다.
#define PRESSURE_LOAD_SETTLE_LOADED_MS       500U  // 완전 기립 후 진동 안정화 시간을 정의한다.
#define PRESSURE_LOAD_CAPTURE_LOADED_MS     1000U  // 체중 부하 평균 측정 시간을 정의한다.
#define PRESSURE_LOAD_MINIMUM_DIFFERENCE      15   // 보정에 필요한 최소 raw 변화량을 정의한다.
#define PRESSURE_LOAD_RELEASE_PERCENT          10   // 해제 임계값의 부하 비율을 정의한다.
#define PRESSURE_LOAD_CONTACT_PERCENT          30   // 접촉 임계값의 부하 비율을 정의한다.

volatile PressureLoadCalibrationDebug_t g_pressure_load_calibration;  // 자동 압력 보정 결과를 저장한다.

/* 현재 측정 단계와 시작 시각을 함께 갱신한다. */
static void PressureLoadCalibration_SetPhase(PressureLoadCalibration_Handle_t *handle,
                                             PressureLoadCalibration_Phase_t phase,
                                             uint32_t now_ms)
{
    handle->phase = phase;                         // 새 측정 단계를 저장한다.
    handle->phase_start_ms = now_ms;               // 새 단계 기준 시각을 저장한다.
    g_pressure_load_calibration.phase = phase;     // Live Expressions 단계를 갱신한다.
}

/* 최근 raw와 누적 평균을 Live Expressions에 갱신한다. */
static void PressureLoadCalibration_UpdateDebug(
    const PressureLoadCalibration_Handle_t *handle,
    const uint16_t raw[ROBOT_PRESSURE_COUNT])
{
    uint32_t leg;  // 갱신할 다리 번호를 저장한다.

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        g_pressure_load_calibration.current_raw[leg] = raw[leg];  // 최근 센서값을 표시한다.
        if (handle->unloaded_count > 0U)
        {
            g_pressure_load_calibration.unloaded_average[leg] =
                (uint16_t)(handle->unloaded_sum[leg] /
                           handle->unloaded_count);                 // 시작 자세 평균을 표시한다.
        }
        if (handle->loaded_count > 0U)
        {
            g_pressure_load_calibration.loaded_average[leg] =
                (uint16_t)(handle->loaded_sum[leg] /
                           handle->loaded_count);                   // 기립 자세 평균을 표시한다.
        }
        g_pressure_load_calibration.difference_raw[leg] =
            (int16_t)((int32_t)g_pressure_load_calibration.loaded_average[leg] -
                      (int32_t)g_pressure_load_calibration.unloaded_average[leg]);  // 체중 부하 변화를 표시한다.
    }

    g_pressure_load_calibration.unloaded_count = handle->unloaded_count;  // 무부하 표본 수를 표시한다.
    g_pressure_load_calibration.loaded_count = handle->loaded_count;      // 부하 표본 수를 표시한다.
}

/* 두 자세 평균으로 여섯 센서의 가벼운 접촉 임계값을 만든다. */
static bool PressureLoadCalibration_BuildAndApply(
    PressureLoadCalibration_Handle_t *handle,
    const uint16_t raw[ROBOT_PRESSURE_COUNT],
    FootPressure_Handle_t *pressure)
{
    FootPressure_Calibration_t table[ROBOT_PRESSURE_COUNT];  // 검증 후 함께 적용할 임계값을 저장한다.
    bool contact[ROBOT_PRESSURE_COUNT];                      // 새 표의 즉시 접촉 결과를 저장한다.
    uint32_t leg;                                            // 계산할 다리 번호를 저장한다.
    uint32_t sample;                                         // 접촉 확인 표본을 저장한다.

    if ((handle->unloaded_count == 0U) || (handle->loaded_count == 0U))
    {
        return false;
    }

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        const int32_t unloaded =
            (int32_t)(handle->unloaded_sum[leg] / handle->unloaded_count);  // 시작 자세 평균을 계산한다.
        const int32_t loaded =
            (int32_t)(handle->loaded_sum[leg] / handle->loaded_count);      // 기립 자세 평균을 계산한다.
        const int32_t difference = loaded - unloaded;                       // 실제 체중 변화 방향을 계산한다.

        if ((difference > -PRESSURE_LOAD_MINIMUM_DIFFERENCE) &&
            (difference < PRESSURE_LOAD_MINIMUM_DIFFERENCE))
        {
            g_pressure_load_calibration.error_leg = (uint8_t)(leg + 1U);  // 변화가 부족한 다리를 표시한다.
            return false;
        }

        table[leg].release_threshold = (uint16_t)(unloaded +
            difference * PRESSURE_LOAD_RELEASE_PERCENT / 100);             // 가벼운 부하 전 해제점을 만든다.
        table[leg].contact_threshold = (uint16_t)(unloaded +
            difference * PRESSURE_LOAD_CONTACT_PERCENT / 100);             // 가벼운 접촉 진입점을 만든다.
        table[leg].active_high = (difference > 0);                           // 센서 증가 방향을 저장한다.
        table[leg].calibrated = true;                                        // 실제 체중 실측 완료를 표시한다.
    }

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        if (!FootPressure_SetCalibration(pressure, (uint8_t)leg, &table[leg]))
        {
            g_pressure_load_calibration.error_leg = (uint8_t)(leg + 1U);  // 적용 실패 다리를 표시한다.
            return false;
        }
        g_pressure_load_calibration.table[leg] = table[leg];               // 중앙 표 복사용 값을 저장한다.
    }

    for (sample = 0U; sample < ROBOT_PRESSURE_CONTACT_CONFIRM_SAMPLES; ++sample)
    {
        FootPressure_Update(pressure, raw, contact);  // 현재 기립값을 5 ms 연속 접촉으로 확인한다.
    }
    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        g_pressure_load_calibration.loaded_contact_detected[leg] = contact[leg];  // 적용 직후 접촉을 표시한다.
    }

    g_pressure_load_calibration.applied = true;  // 런타임 적용 완료를 표시한다.
    return true;
}

/* 기립 전 자동 압력 측정을 초기화한다. */
void PressureLoadCalibration_Init(PressureLoadCalibration_Handle_t *handle,
                                  uint32_t now_ms)
{
    if (handle == NULL)
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));                                // 이전 누적값을 제거한다.
    memset((void *)&g_pressure_load_calibration, 0,
           sizeof(g_pressure_load_calibration));                       // 이전 디버그 결과를 제거한다.
    handle->start_ms = now_ms;                                         // 전체 측정 기준 시각을 저장한다.
    handle->initialized = true;                                        // 자동 측정을 허가한다.
    PressureLoadCalibration_SetPhase(
        handle, PRESSURE_LOAD_CALIBRATION_WAIT_UNLOADED, now_ms);       // 센서 안정화부터 시작한다.
}

/* 시작 자세와 완전 기립 자세의 압력 평균을 자동으로 측정한다. */
void PressureLoadCalibration_Update(PressureLoadCalibration_Handle_t *handle,
                                    const uint16_t raw[ROBOT_PRESSURE_COUNT],
                                    bool robot_ready,
                                    uint32_t now_ms,
                                    FootPressure_Handle_t *pressure)
{
    uint32_t leg;  // 누적할 다리 번호를 저장한다.

    if ((handle == NULL) || (raw == NULL) || (pressure == NULL) ||
        !handle->initialized ||
        (handle->phase == PRESSURE_LOAD_CALIBRATION_COMPLETE) ||
        (handle->phase == PRESSURE_LOAD_CALIBRATION_FAILED))
    {
        return;
    }

    PressureLoadCalibration_UpdateDebug(handle, raw);  // 최근 압력값과 평균을 표시한다.

    switch (handle->phase)
    {
        case PRESSURE_LOAD_CALIBRATION_WAIT_UNLOADED:
            if ((now_ms - handle->phase_start_ms) >= PRESSURE_LOAD_WAIT_UNLOADED_MS)
            {
                PressureLoadCalibration_SetPhase(
                    handle, PRESSURE_LOAD_CALIBRATION_CAPTURE_UNLOADED, now_ms);  // 시작 자세 측정을 시작한다.
            }
            break;

        case PRESSURE_LOAD_CALIBRATION_CAPTURE_UNLOADED:
            if (handle->unloaded_count < UINT16_MAX)
            {
                for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
                {
                    handle->unloaded_sum[leg] += raw[leg];  // 시작 자세 raw를 누적한다.
                }
                handle->unloaded_count++;                  // 시작 자세 표본 수를 갱신한다.
            }
            if ((now_ms - handle->phase_start_ms) >= PRESSURE_LOAD_CAPTURE_UNLOADED_MS)
            {
                PressureLoadCalibration_SetPhase(
                    handle, PRESSURE_LOAD_CALIBRATION_WAIT_READY, now_ms);  // 완전 기립을 기다린다.
            }
            break;

        case PRESSURE_LOAD_CALIBRATION_WAIT_READY:
            if (robot_ready)
            {
                PressureLoadCalibration_SetPhase(
                    handle, PRESSURE_LOAD_CALIBRATION_SETTLE_LOADED, now_ms);  // 기립 흔들림을 먼저 기다린다.
            }
            break;

        case PRESSURE_LOAD_CALIBRATION_SETTLE_LOADED:
            if (!robot_ready)
            {
                PressureLoadCalibration_SetPhase(
                    handle, PRESSURE_LOAD_CALIBRATION_WAIT_READY, now_ms);  // READY가 풀리면 다시 기다린다.
            }
            else if ((now_ms - handle->phase_start_ms) >= PRESSURE_LOAD_SETTLE_LOADED_MS)
            {
                PressureLoadCalibration_SetPhase(
                    handle, PRESSURE_LOAD_CALIBRATION_CAPTURE_LOADED, now_ms);  // 체중 부하 측정을 시작한다.
            }
            break;

        case PRESSURE_LOAD_CALIBRATION_CAPTURE_LOADED:
            if (!robot_ready)
            {
                memset(handle->loaded_sum, 0, sizeof(handle->loaded_sum));  // 불완전한 부하 표본을 제거한다.
                handle->loaded_count = 0U;                                 // 부하 표본 수를 초기화한다.
                PressureLoadCalibration_SetPhase(
                    handle, PRESSURE_LOAD_CALIBRATION_WAIT_READY, now_ms);  // READY가 풀리면 다시 측정한다.
                break;
            }
            if (handle->loaded_count < UINT16_MAX)
            {
                for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
                {
                    handle->loaded_sum[leg] += raw[leg];  // 기립 체중 부하 raw를 누적한다.
                }
                handle->loaded_count++;                  // 기립 자세 표본 수를 갱신한다.
            }
            if ((now_ms - handle->phase_start_ms) >= PRESSURE_LOAD_CAPTURE_LOADED_MS)
            {
                PressureLoadCalibration_UpdateDebug(handle, raw);  // 최종 평균과 변화량을 표시한다.
                if (PressureLoadCalibration_BuildAndApply(handle, raw, pressure))
                {
                    g_pressure_load_calibration.complete = true;  // 전체 성공을 표시한다.
                    PressureLoadCalibration_SetPhase(
                        handle, PRESSURE_LOAD_CALIBRATION_COMPLETE, now_ms);  // 자동 착지를 허가한다.
                }
                else
                {
                    g_pressure_load_calibration.failed = true;  // 측정 실패를 표시한다.
                    PressureLoadCalibration_SetPhase(
                        handle, PRESSURE_LOAD_CALIBRATION_FAILED, now_ms);  // 실패 후에도 자동 착지를 허가한다.
                }
            }
            break;

        default:
            g_pressure_load_calibration.failed = true;  // 알 수 없는 단계를 실패로 표시한다.
            PressureLoadCalibration_SetPhase(
                handle, PRESSURE_LOAD_CALIBRATION_FAILED, now_ms);  // 실패 후 자동 착지로 넘긴다.
            break;
    }
}

/* 자동 압력 측정이 성공 또는 실패로 끝났는지 반환한다. */
bool PressureLoadCalibration_IsFinished(void)
{
    return g_pressure_load_calibration.complete ||
           g_pressure_load_calibration.failed;  // 완료 후 기립 유지가 끝나도록 한다.
}
