#include "sensor/foot_pressure.h"

#include <stddef.h>

/* 측정 전에는 접촉으로 오인하지 않는 기본 임계값을 준비한다. */
void FootPressure_Init(FootPressure_Handle_t *handle)
{
    uint32_t leg;   // 초기화할 다리 번호를 저장한다.

    if (handle == NULL)
    {
        return;
    }

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        handle->table[leg].contact_threshold = 1023U;  // 실측 전 접촉 오인을 막는다.
        handle->table[leg].release_threshold = 1022U;  // 기본 Hysteresis를 만든다.
        handle->table[leg].active_high = true;         // 압력 증가를 기본 방향으로 둔다.
        handle->table[leg].calibrated = false;         // 실측 전 상태로 표시한다.
        handle->contact[leg] = false;                  // 초기 접촉을 해제한다.
    }
}

/* 한 압력센서의 실측 임계값을 설정한다. */
bool FootPressure_SetCalibration(FootPressure_Handle_t *handle,
                                 uint8_t leg,
                                 const FootPressure_Calibration_t *calibration)
{
    if ((handle == NULL) || (calibration == NULL) || (leg >= ROBOT_PRESSURE_COUNT))
    {
        return false;
    }

    if (calibration->active_high &&
        (calibration->release_threshold >= calibration->contact_threshold))
    {
        return false;
    }

    if (!calibration->active_high &&
        (calibration->release_threshold <= calibration->contact_threshold))
    {
        return false;
    }

    handle->table[leg] = *calibration;   // 선택한 압력센서 테이블을 갱신한다.
    handle->contact[leg] = false;        // 새 임계값 적용 전에 상태를 초기화한다.
    return true;
}

/* 여섯 압력센서의 Hysteresis 접촉 상태를 갱신한다. */
void FootPressure_Update(FootPressure_Handle_t *handle,
                         const uint16_t raw[ROBOT_PRESSURE_COUNT],
                         bool contact[ROBOT_PRESSURE_COUNT])
{
    uint32_t leg;   // 갱신할 다리 번호를 저장한다.

    if ((handle == NULL) || (raw == NULL) || (contact == NULL))
    {
        return;
    }

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        const FootPressure_Calibration_t *calibration = &handle->table[leg];  // 현재 센서 보정값을 선택한다.

        if (calibration->active_high)
        {
            if (!handle->contact[leg] && (raw[leg] >= calibration->contact_threshold))
            {
                handle->contact[leg] = true;   // 접촉 진입 임계값을 통과한다.
            }
            else if (handle->contact[leg] && (raw[leg] <= calibration->release_threshold))
            {
                handle->contact[leg] = false;  // 접촉 해제 임계값을 통과한다.
            }
        }
        else
        {
            if (!handle->contact[leg] && (raw[leg] <= calibration->contact_threshold))
            {
                handle->contact[leg] = true;   // 반전 센서의 접촉을 검출한다.
            }
            else if (handle->contact[leg] && (raw[leg] >= calibration->release_threshold))
            {
                handle->contact[leg] = false;  // 반전 센서의 해제를 검출한다.
            }
        }

        contact[leg] = handle->contact[leg];   // 최신 접촉 상태를 반환한다.
    }
}
