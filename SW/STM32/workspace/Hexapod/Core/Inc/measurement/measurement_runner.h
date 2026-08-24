#ifndef MEASUREMENT_RUNNER_H
#define MEASUREMENT_RUNNER_H

#include <stdbool.h>
#include <stdint.h>

typedef enum
{
    MEASUREMENT_STAGE_SENSOR_RAW = 0,      // 센서 통신과 전체 ADC raw를 확인한다.
    MEASUREMENT_STAGE_IMU,                 // WT931 축과 Offset을 측정한다.
    MEASUREMENT_STAGE_ADC_MAPPING,         // 24개 ADC 입력 배치를 확인한다.
    MEASUREMENT_STAGE_RELAY,               // 릴레이와 다리 대응을 확인한다.
    MEASUREMENT_STAGE_SERVO,               // 서보 방향과 Pulse 범위를 측정한다.
    MEASUREMENT_STAGE_JOINT_FEEDBACK,      // 관절센서 각도와 ADC 값을 측정한다.
    MEASUREMENT_STAGE_FOOT_PRESSURE,       // 압력센서 무부하와 접촉값을 측정한다.
    MEASUREMENT_STAGE_CRSF,                // 실제 조종기 채널 범위를 측정한다.
    MEASUREMENT_STAGE_COMPLETE,            // 모든 실측 완료를 나타낸다.
    MEASUREMENT_STAGE_COUNT
} MeasurementStage_t;

typedef struct
{
    MeasurementStage_t stage;                              // 현재 실측 단계를 저장한다.
    bool completed[MEASUREMENT_STAGE_COMPLETE];             // 단계별 사용자 확인을 저장한다.
} MeasurementRunner_t;

void MeasurementRunner_Init(MeasurementRunner_t *runner);  // 첫 센서 Raw 단계로 초기화한다.
MeasurementStage_t MeasurementRunner_GetStage(const MeasurementRunner_t *runner);  // 현재 단계를 반환한다.
bool MeasurementRunner_CompleteCurrent(MeasurementRunner_t *runner);  // 현재 단계를 완료하고 다음으로 이동한다.

#endif
