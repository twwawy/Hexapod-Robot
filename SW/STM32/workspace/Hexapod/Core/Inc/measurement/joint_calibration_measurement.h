#ifndef JOINT_CALIBRATION_MEASUREMENT_H
#define JOINT_CALIBRATION_MEASUREMENT_H

#include "sensor/joint_feedback.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    uint16_t minimum[ROBOT_JOINT_COUNT];  // 최소 자세의 관절 ADC를 저장한다.
    uint16_t zero[ROBOT_JOINT_COUNT];     // 영점 자세의 관절 ADC를 저장한다.
    uint16_t maximum[ROBOT_JOINT_COUNT];  // 최대 자세의 관절 ADC를 저장한다.
    bool minimum_captured;                // 최소 자세 기록 여부를 저장한다.
    bool zero_captured;                   // 영점 자세 기록 여부를 저장한다.
    bool maximum_captured;                // 최대 자세 기록 여부를 저장한다.
} JointCalibrationMeasurement_t;

void JointCalibrationMeasurement_Init(JointCalibrationMeasurement_t *measurement);  // 관절 ADC 측정을 초기화한다.
void JointCalibrationMeasurement_CaptureMinimum(JointCalibrationMeasurement_t *measurement,
                                                const uint16_t raw[ROBOT_JOINT_COUNT]);  // 최소 자세 raw를 기록한다.
void JointCalibrationMeasurement_CaptureZero(JointCalibrationMeasurement_t *measurement,
                                             const uint16_t raw[ROBOT_JOINT_COUNT]);  // 영점 자세 raw를 기록한다.
void JointCalibrationMeasurement_CaptureMaximum(JointCalibrationMeasurement_t *measurement,
                                                const uint16_t raw[ROBOT_JOINT_COUNT]);  // 최대 자세 raw를 기록한다.
bool JointCalibrationMeasurement_Build(const JointCalibrationMeasurement_t *measurement,
                                       const float angle_min_rad[ROBOT_JOINT_COUNT],
                                       const float angle_zero_rad[ROBOT_JOINT_COUNT],
                                       const float angle_max_rad[ROBOT_JOINT_COUNT],
                                       const int8_t direction[ROBOT_JOINT_COUNT],
                                       JointFeedback_Calibration_t table[ROBOT_JOINT_COUNT]);  // 관절별 보정표를 만든다.

#endif
