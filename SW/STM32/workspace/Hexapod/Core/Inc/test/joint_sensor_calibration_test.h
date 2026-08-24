#ifndef JOINT_SENSOR_CALIBRATION_TEST_H
#define JOINT_SENSOR_CALIBRATION_TEST_H

#include "sensor/joint_feedback.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    uint16_t minimum[ROBOT_JOINT_COUNT];  // 관절별 관측 최소 raw를 저장한다.
    uint16_t zero[ROBOT_JOINT_COUNT];     // 관절별 중립 raw를 저장한다.
    uint16_t maximum[ROBOT_JOINT_COUNT];  // 관절별 관측 최대 raw를 저장한다.
    bool zero_captured;                   // 중립 측정 완료 여부를 저장한다.
} JointSensorCalibrationTest_t;

void JointSensorCalibrationTest_Init(JointSensorCalibrationTest_t *test);  // raw 범위 기록을 초기화한다.
void JointSensorCalibrationTest_Update(JointSensorCalibrationTest_t *test,
                                       const uint16_t raw[ROBOT_JOINT_COUNT]);  // 실제 raw 범위를 갱신한다.
void JointSensorCalibrationTest_CaptureZero(JointSensorCalibrationTest_t *test,
                                            const uint16_t raw[ROBOT_JOINT_COUNT]);  // 현재 자세를 중립으로 기록한다.
bool JointSensorCalibrationTest_Build(const JointSensorCalibrationTest_t *test,
                                      const int8_t direction[ROBOT_JOINT_COUNT],
                                      JointFeedback_Calibration_t table[ROBOT_JOINT_COUNT]);  // 실측 보정표를 만든다.

#endif
