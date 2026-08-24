#ifndef IMU_CALIBRATION_TEST_H
#define IMU_CALIBRATION_TEST_H

#include "sensor/imu.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    double euler_sum_rad[3];  // 수평 정지 자세의 각도 합계를 저장한다.
    uint32_t sample_count;    // 정상 자세 표본 수를 저장한다.
} ImuCalibrationTest_t;

void ImuCalibrationTest_Init(ImuCalibrationTest_t *test);  // WT931 Offset 측정을 초기화한다.
bool ImuCalibrationTest_AddSample(ImuCalibrationTest_t *test,
                                  const IMU_Data_t *data);  // 수평 정지 자세 표본을 추가한다.
bool ImuCalibrationTest_Build(const ImuCalibrationTest_t *test,
                              const int8_t acceleration_sign[3],
                              const int8_t angular_velocity_sign[3],
                              const int8_t euler_angle_sign[3],
                              IMU_Calibration_t *calibration);  // 장착 부호와 평균 Offset을 만든다.

#endif
