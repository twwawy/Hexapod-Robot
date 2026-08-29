#ifndef FOOT_PRESSURE_H
#define FOOT_PRESSURE_H

#include "common/robot_types.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    uint16_t contact_threshold;   // 접촉 진입 임계값을 저장한다.
    uint16_t release_threshold;   // 접촉 해제 임계값을 저장한다.
    bool active_high;             // 압력 증가 방향을 저장한다.
    bool calibrated;              // 실측 완료 여부를 저장한다.
} FootPressure_Calibration_t;

typedef struct
{
    FootPressure_Calibration_t table[ROBOT_PRESSURE_COUNT];  // 센서별 보정 테이블을 저장한다.
    uint8_t contact_count[ROBOT_PRESSURE_COUNT];             // 접촉 연속 표본 수를 저장한다.
    uint8_t release_count[ROBOT_PRESSURE_COUNT];             // 해제 연속 표본 수를 저장한다.
    bool raw_contact[ROBOT_PRESSURE_COUNT];                  // Hysteresis 직후 상태를 저장한다.
    bool contact[ROBOT_PRESSURE_COUNT];                      // 시간 확인을 마친 상태를 저장한다.
} FootPressure_Handle_t;

void FootPressure_Init(FootPressure_Handle_t *handle);   // 기본 임계값과 상태를 준비한다.

bool FootPressure_SetCalibration(FootPressure_Handle_t *handle,
                                 uint8_t leg,
                                 const FootPressure_Calibration_t *calibration);  // 한 센서 임계값을 갱신한다.

void FootPressure_Update(FootPressure_Handle_t *handle,
                         const uint16_t raw[ROBOT_PRESSURE_COUNT],
                         bool contact[ROBOT_PRESSURE_COUNT]);  // 시간 확인 접촉 상태를 갱신한다.

#endif
