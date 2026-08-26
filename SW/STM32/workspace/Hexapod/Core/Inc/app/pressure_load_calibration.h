#ifndef PRESSURE_LOAD_CALIBRATION_H
#define PRESSURE_LOAD_CALIBRATION_H

#include "sensor/foot_pressure.h"

#include <stdbool.h>
#include <stdint.h>

typedef enum
{
    PRESSURE_LOAD_CALIBRATION_WAIT_UNLOADED = 0,  // 시작 자세의 안정화를 기다린다.
    PRESSURE_LOAD_CALIBRATION_CAPTURE_UNLOADED,   // 시작 자세의 무부하값을 측정한다.
    PRESSURE_LOAD_CALIBRATION_WAIT_READY,         // 완전한 기립 상태를 기다린다.
    PRESSURE_LOAD_CALIBRATION_SETTLE_LOADED,      // 기립 직후 흔들림이 끝나길 기다린다.
    PRESSURE_LOAD_CALIBRATION_CAPTURE_LOADED,     // 실제 체중 부하값을 측정한다.
    PRESSURE_LOAD_CALIBRATION_COMPLETE,           // 임계값 생성과 적용을 완료한다.
    PRESSURE_LOAD_CALIBRATION_FAILED              // 측정 실패 후 착지를 기다린다.
} PressureLoadCalibration_Phase_t;

typedef struct
{
    uint32_t start_ms;                              // 전체 측정 시작 시각을 저장한다.
    uint32_t phase_start_ms;                        // 현재 단계 시작 시각을 저장한다.
    uint32_t unloaded_sum[ROBOT_PRESSURE_COUNT];    // 시작 자세 raw 합계를 저장한다.
    uint32_t loaded_sum[ROBOT_PRESSURE_COUNT];      // 기립 자세 raw 합계를 저장한다.
    uint16_t unloaded_count;                        // 시작 자세 표본 수를 저장한다.
    uint16_t loaded_count;                          // 기립 자세 표본 수를 저장한다.
    PressureLoadCalibration_Phase_t phase;          // 현재 자동 측정 단계를 저장한다.
    bool initialized;                               // 측정 준비 여부를 저장한다.
} PressureLoadCalibration_Handle_t;

typedef struct
{
    PressureLoadCalibration_Phase_t phase;                       // Live Expressions용 현재 단계를 저장한다.
    uint16_t current_raw[ROBOT_PRESSURE_COUNT];                   // 최근 압력 raw를 저장한다.
    uint16_t unloaded_average[ROBOT_PRESSURE_COUNT];              // 시작 자세 평균을 저장한다.
    uint16_t loaded_average[ROBOT_PRESSURE_COUNT];                // 기립 자세 평균을 저장한다.
    int16_t difference_raw[ROBOT_PRESSURE_COUNT];                 // 실제 체중 부하 변화량을 저장한다.
    FootPressure_Calibration_t table[ROBOT_PRESSURE_COUNT];       // 중앙 표에 복사할 최종값을 저장한다.
    bool loaded_contact_detected[ROBOT_PRESSURE_COUNT];           // 적용 직후 접촉 판정 결과를 저장한다.
    uint16_t unloaded_count;                                     // 시작 자세 표본 수를 표시한다.
    uint16_t loaded_count;                                       // 기립 자세 표본 수를 표시한다.
    uint8_t error_leg;                                           // 실패한 다리 번호 1~6을 저장한다.
    bool complete;                                               // 여섯 센서 보정 완료를 표시한다.
    bool failed;                                                 // 측정 또는 적용 실패를 표시한다.
    bool applied;                                                // 런타임 임계값 적용을 표시한다.
} PressureLoadCalibrationDebug_t;

extern volatile PressureLoadCalibrationDebug_t g_pressure_load_calibration;  // 자동 압력 보정 결과를 공개한다.

void PressureLoadCalibration_Init(PressureLoadCalibration_Handle_t *handle,
                                  uint32_t now_ms);  // 기립 전 자동 압력 측정을 준비한다.

void PressureLoadCalibration_Update(PressureLoadCalibration_Handle_t *handle,
                                    const uint16_t raw[ROBOT_PRESSURE_COUNT],
                                    bool robot_ready,
                                    uint32_t now_ms,
                                    FootPressure_Handle_t *pressure);  // 현재 자세에 맞는 표본을 모으고 임계값을 적용한다.

bool PressureLoadCalibration_IsFinished(void);  // 성공 또는 실패로 측정이 끝났는지 반환한다.

#endif
