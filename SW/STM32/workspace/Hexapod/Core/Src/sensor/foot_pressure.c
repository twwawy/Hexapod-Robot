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
        handle->contact_count[leg] = 0U;               // 접촉 표본 수를 제거한다.
        handle->contact_window_count[leg] = 0U;        // 접촉 검사 시간을 제거한다.
        handle->release_count[leg] = 0U;               // 해제 확인 횟수를 제거한다.
        handle->raw_contact[leg] = false;              // 초기 접촉 후보를 해제한다.
        handle->contact[leg] = false;                  // 초기 확정 접촉을 해제한다.
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

    handle->table[leg] = *calibration;  // 선택한 압력센서 테이블을 갱신한다.
    handle->contact_count[leg] = 0U;         // 새 임계값의 접촉 표본을 준비한다.
    handle->contact_window_count[leg] = 0U;  // 새 임계값의 검사 시간을 준비한다.
    handle->release_count[leg] = 0U;         // 새 임계값의 해제 확인을 준비한다.
    handle->raw_contact[leg] = false;        // 새 임계값의 접촉 후보를 초기화한다.
    handle->contact[leg] = false;            // 새 임계값의 확정 접촉을 초기화한다.
    return true;
}

/* 여섯 압력센서의 10 ms 다수결 접촉 상태를 갱신한다. */
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
        const bool contact_entry = calibration->active_high ?
            (raw[leg] >= calibration->contact_threshold) :
            (raw[leg] <= calibration->contact_threshold);  // 새 접촉 검사 시작 여부를 계산한다.
        const bool contact_evidence = calibration->active_high ?
            (raw[leg] > calibration->release_threshold) :
            (raw[leg] < calibration->release_threshold);   // 검사 창의 유효 접촉 표본을 계산한다.

        if (!handle->contact[leg])
        {
            handle->release_count[leg] = 0U;  // 미접촉 상태의 해제 횟수를 제거한다.

            if (!handle->raw_contact[leg] && contact_entry)
            {
                handle->raw_contact[leg] = true;        // 10 ms 접촉 검사를 시작한다.
                handle->contact_count[leg] = 0U;        // 새 검사 창의 접촉 표본을 준비한다.
                handle->contact_window_count[leg] = 0U; // 새 검사 창의 시간을 준비한다.
            }

            if (handle->raw_contact[leg])
            {
                if (handle->contact_window_count[leg] <
                    ROBOT_PRESSURE_CONTACT_CONFIRM_SAMPLES)
                {
                    handle->contact_window_count[leg]++;  // 고정 상태의 검사 시간을 누적한다.
                }

                if (contact_evidence &&
                    (handle->contact_count[leg] < ROBOT_PRESSURE_CONTACT_CONFIRM_SAMPLES))
                {
                    handle->contact_count[leg]++;  // 유효 접촉 표본을 누적한다.
                }

                if (handle->contact_window_count[leg] >=
                    ROBOT_PRESSURE_CONTACT_CONFIRM_SAMPLES)
                {
                    handle->contact[leg] =
                        (handle->contact_count[leg] >=
                         ROBOT_PRESSURE_CONTACT_ACCEPT_SAMPLES);  // 8개 이상이면 접촉을 확정한다.
                    handle->raw_contact[leg] = handle->contact[leg];  // 실패한 후보 고정을 해제한다.
                    handle->contact_count[leg] = 0U;                  // 완료한 접촉 표본을 제거한다.
                    handle->contact_window_count[leg] = 0U;           // 완료한 검사 시간을 제거한다.
                }
            }
        }
        else
        {
            handle->contact_count[leg] = 0U;         // 확정 상태의 접촉 표본을 제거한다.
            handle->contact_window_count[leg] = 0U;  // 확정 상태의 검사 시간을 제거한다.

            if (handle->raw_contact[leg] && !contact_evidence)
            {
                handle->raw_contact[leg] = false;  // 해제 임계값 통과를 검출한다.
            }
            else if (!handle->raw_contact[leg] && contact_entry)
            {
                handle->raw_contact[leg] = true;  // 접촉 임계값 복귀를 검출한다.
            }

            if (handle->raw_contact[leg])
            {
                handle->release_count[leg] = 0U;  // 접촉 유지에서 해제 횟수를 제거한다.
            }
            else
            {
                if (handle->release_count[leg] < ROBOT_PRESSURE_RELEASE_CONFIRM_SAMPLES)
                {
                    handle->release_count[leg]++;  // 해제 연속 표본을 누적한다.
                }
                if (handle->release_count[leg] >= ROBOT_PRESSURE_RELEASE_CONFIRM_SAMPLES)
                {
                    handle->contact[leg] = false;  // 10 ms 연속 해제를 확정한다.
                    handle->release_count[leg] = 0U;  // 완료한 해제 횟수를 제거한다.
                }
            }
        }

        contact[leg] = handle->contact[leg];  // 시간 확인을 마친 접촉을 반환한다.
    }
}
