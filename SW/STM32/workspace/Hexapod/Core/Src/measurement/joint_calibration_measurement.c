#include "measurement/joint_calibration_measurement.h"

#include "measurement/measurement_debug.h"

#include <stddef.h>
#include <string.h>

/* 관절 ADC 세 자세의 측정 상태를 초기화한다. */
void JointCalibrationMeasurement_Init(JointCalibrationMeasurement_t *measurement)
{
    uint32_t joint;  // 초기화할 관절 번호를 저장한다.

    if (measurement != NULL)
    {
        memset(measurement, 0, sizeof(*measurement));  // 이전 관절 측정값을 제거한다.
        g_measurement_debug.joint_minimum_captured = false;  // 최소 자세 완료를 초기화한다.
        g_measurement_debug.joint_zero_captured = false;     // 영점 자세 완료를 초기화한다.
        g_measurement_debug.joint_maximum_captured = false;  // 최대 자세 완료를 초기화한다.
        for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
        {
            g_measurement_debug.calibration.joint[joint].calibrated = false;  // 관절별 완료를 초기화한다.
        }
    }
}

/* 관절별 최소 자세의 실제 ADC를 기록한다. */
void JointCalibrationMeasurement_CaptureMinimum(JointCalibrationMeasurement_t *measurement,
                                                const uint16_t raw[ROBOT_JOINT_COUNT])
{
    uint32_t joint;  // 기록할 관절 번호를 저장한다.

    if ((measurement != NULL) && (raw != NULL))
    {
        memcpy(measurement->minimum, raw, sizeof(measurement->minimum));  // 최소 자세 raw를 복사한다.
        measurement->minimum_captured = true;                             // 최소 자세 기록을 표시한다.
        for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
        {
            g_measurement_debug.joint_minimum_raw[joint] = raw[joint];  // 디버거 최소 자세 raw를 갱신한다.
        }
        g_measurement_debug.joint_minimum_captured = true;  // 디버거 최소 자세 완료를 표시한다.
    }
}

/* 관절별 영점 자세의 실제 ADC를 기록한다. */
void JointCalibrationMeasurement_CaptureZero(JointCalibrationMeasurement_t *measurement,
                                             const uint16_t raw[ROBOT_JOINT_COUNT])
{
    uint32_t joint;  // 기록할 관절 번호를 저장한다.

    if ((measurement != NULL) && (raw != NULL))
    {
        memcpy(measurement->zero, raw, sizeof(measurement->zero));  // 영점 자세 raw를 복사한다.
        measurement->zero_captured = true;                          // 영점 자세 기록을 표시한다.
        for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
        {
            g_measurement_debug.joint_zero_raw[joint] = raw[joint];  // 디버거 영점 자세 raw를 갱신한다.
        }
        g_measurement_debug.joint_zero_captured = true;  // 디버거 영점 자세 완료를 표시한다.
    }
}

/* 관절별 최대 자세의 실제 ADC를 기록한다. */
void JointCalibrationMeasurement_CaptureMaximum(JointCalibrationMeasurement_t *measurement,
                                                const uint16_t raw[ROBOT_JOINT_COUNT])
{
    uint32_t joint;  // 기록할 관절 번호를 저장한다.

    if ((measurement != NULL) && (raw != NULL))
    {
        memcpy(measurement->maximum, raw, sizeof(measurement->maximum));  // 최대 자세 raw를 복사한다.
        measurement->maximum_captured = true;                             // 최대 자세 기록을 표시한다.
        for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
        {
            g_measurement_debug.joint_maximum_raw[joint] = raw[joint];  // 디버거 최대 자세 raw를 갱신한다.
        }
        g_measurement_debug.joint_maximum_captured = true;  // 디버거 최대 자세 완료를 표시한다.
    }
}

/* 세 자세의 ADC와 실제 각도로 관절별 보정표를 만든다. */
bool JointCalibrationMeasurement_Build(const JointCalibrationMeasurement_t *measurement,
                                       const float angle_min_rad[ROBOT_JOINT_COUNT],
                                       const float angle_zero_rad[ROBOT_JOINT_COUNT],
                                       const float angle_max_rad[ROBOT_JOINT_COUNT],
                                       const int8_t direction[ROBOT_JOINT_COUNT],
                                       JointFeedback_Calibration_t table[ROBOT_JOINT_COUNT])
{
    uint32_t joint;  // 보정할 관절 번호를 저장한다.

    if ((measurement == NULL) || (angle_min_rad == NULL) ||
        (angle_zero_rad == NULL) || (angle_max_rad == NULL) ||
        (direction == NULL) || (table == NULL) ||
        !measurement->minimum_captured || !measurement->zero_captured ||
        !measurement->maximum_captured)
    {
        return false;
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        uint16_t raw_min = measurement->minimum[joint];   // 최소 자세 raw를 임시 저장한다.
        uint16_t raw_max = measurement->maximum[joint];   // 최대 자세 raw를 임시 저장한다.

        if (raw_min > raw_max)
        {
            const uint16_t swap = raw_min;  // ADC 숫자 순서를 바꿀 임시값을 저장한다.

            raw_min = raw_max;              // 작은 ADC 값을 최소 필드로 옮긴다.
            raw_max = swap;                 // 큰 ADC 값을 최대 필드로 옮긴다.
        }

        if ((raw_min >= measurement->zero[joint]) ||
            (measurement->zero[joint] >= raw_max) ||
            (angle_min_rad[joint] >= angle_zero_rad[joint]) ||
            (angle_zero_rad[joint] >= angle_max_rad[joint]) ||
            ((direction[joint] != 1) && (direction[joint] != -1)))
        {
            return false;
        }

        table[joint].raw_min = raw_min;                              // 작은 ADC 끝점을 저장한다.
        table[joint].raw_zero = measurement->zero[joint];            // 영점 ADC를 저장한다.
        table[joint].raw_max = raw_max;                              // 큰 ADC 끝점을 저장한다.
        table[joint].angle_min_rad = angle_min_rad[joint];           // 실제 최소 관절각을 저장한다.
        table[joint].angle_zero_rad = angle_zero_rad[joint];         // 실제 영점 관절각을 저장한다.
        table[joint].angle_max_rad = angle_max_rad[joint];           // 실제 최대 관절각을 저장한다.
        table[joint].direction = direction[joint];                   // ADC 증가 방향을 적용한다.
        table[joint].calibrated = true;                              // 실측 완료를 표시한다.
        g_measurement_debug.calibration.joint[joint] = table[joint]; // 디버거 최종 보정값을 갱신한다.
    }

    return true;
}
